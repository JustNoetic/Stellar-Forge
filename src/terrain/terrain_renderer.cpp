#include "terrain/terrain_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include "core/path_utils.hpp"
#include "core/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace sf::terrain {

namespace {

class SelectionSource final : public TerrainSource {
public:
    SelectionSource(const HeightSource& source, std::unordered_map<uint64_t, SurfaceTile>& cache) : source_(source), cache_(cache) {}
    SourceDomain domain() const noexcept override { return source_.domain(); }
    SurfaceTile describe(TileId tile, double polar, double amplitude) const override {
        if (auto entry = cache_.find(tile.key()); entry != cache_.end()) return entry->second;
        auto value = source_.describe(tile, polar, amplitude);
        if (cache_.size() == MAX_SOURCE_RECORDS) cache_.clear();
        cache_.emplace(tile.key(), value);
        return value;
    }

private:
    const HeightSource& source_;
    std::unordered_map<uint64_t, SurfaceTile>& cache_;
};

}

static_assert(sizeof(PatchGPU) == 352);
static_assert(offsetof(PatchGPU, address) == 192);
static_assert(sizeof(CachedPatchGPU) == 128);

void TerrainRenderer::init(rhi::VkContext& ctx, bool load_config) {
    std::vector<uint32_t> indices;
    for (uint32_t mask = 0; mask < 16; ++mask) {
        auto grid = grid_indices(mask);
        indices.insert(indices.end(), grid.begin(), grid.end());
    }
    indices_ = ctx.create_device_buffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data());
    for (auto& buffer : patches_)
        buffer = ctx.create_host_buffer(MAX_PATCHES * sizeof(PatchGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (auto& buffer : cached_patches_)
        buffer = ctx.create_host_buffer(MAX_PATCHES * sizeof(CachedPatchGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rhi::GraphicsPipelineDesc desc;
    desc.vert_path = "terrain/terrain_reference.vert";
    desc.frag_path = "celestial/sphere.frag";
    desc.macros = {{"TERRAIN_SURFACE", "1"}};
    desc.bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    reference_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    desc.vert_path = "terrain/terrain.vert";
    desc.bindings.push_back({4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1});
    pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    desc.macros.push_back({"TERRAIN_PRECISE_PROJECTION", "1"});
    precise_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    desc.macros = {{"TERRAIN_SURFACE", "1"}, {"TERRAIN_CACHED_FP64", "1"}};
    cached_fp64_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    cache_.init(ctx);
    auto config_path = paths::external_path("data", "terrain.json");
    if (load_config && std::filesystem::exists(config_path)) {
        auto config = paths::read_json(config_path);
        if (!config) throw std::runtime_error("Cannot read data/terrain.json");
        set_height_source(config->at("body").get<std::string>(),
            HeightSource::open(paths::external_path(config->at("dataset").get<std::string>())));
    }
}

void TerrainRenderer::shutdown(rhi::VkContext& ctx) {
    streams_.clear();
    rhi::ShaderCache::destroy_pipeline(ctx, pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, precise_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, reference_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, cached_fp64_pipeline_);
    for (auto& buffer : patches_) ctx.destroy_buffer(buffer);
    for (auto& buffer : cached_patches_) ctx.destroy_buffer(buffer);
    cache_.shutdown(ctx);
    metadata_.clear();
    ctx.destroy_buffer(indices_);
}

void TerrainRenderer::set_height_source(std::string body, std::shared_ptr<const HeightSource> source) {
    if (body.empty() && !source) {
        streams_.clear();
        selections_.clear();
        return;
    }
    auto found = std::find_if(streams_.begin(), streams_.end(), [&](const StreamBinding& entry) { return entry.body == body; });
    if (!source) {
        if (found != streams_.end()) streams_.erase(found);
        selections_.clear();
        return;
    }
    if (found != streams_.end() && found->stream && found->stream->source().domain() == source->domain()) return;
    log::info("Terrain source '{}' bound to '{}' (identity {}, revision {})", source->info().name, body,
              source->domain().identity, source->domain().revision);
    if (found != streams_.end()) {
        found->stream = std::make_unique<HeightStream>(std::move(source));
        found->reported_error.clear();
    } else {
        streams_.push_back({std::move(body), std::make_unique<HeightStream>(std::move(source)), {}});
    }
    selections_.clear();
}

float TerrainRenderer::relief_extent_au(uint32_t body) const noexcept {
    for (const auto& [index, extent] : relief_extents_) if (index == body) return extent;
    return 0.0f;
}

bool TerrainRenderer::covers(uint32_t body) const noexcept {
    return body < covered_.size() && covered_[body] != 0;
}

void TerrainRenderer::prepare(rhi::VkContext& ctx, const Camera& camera, const ephemeris::SystemSnapshot& system,
                              const rendering::RenderSettings& settings, VkExtent2D extent, uint32_t body_count) {
    prepared_.clear();
    covered_.assign(body_count, 0);
    relief_extents_.clear();
    stats_ = {};
    if (selection_generation_ != system.generation()) {
        selections_.clear();
        selection_generation_ = system.generation();
    }
    std::vector<GeometryKey> retained_keys;
    std::vector<std::shared_ptr<const HeightGeometry>> retained_heights;
    std::vector<HeightStream*> retained_streams;
    double tan_y = std::tan(camera.fov_deg * DEG_TO_RAD * 0.5);
    double focal = std::max(1u, extent.height) / (2.0 * tan_y);
    auto eye = camera.eye_pos_render();
    auto world_to_view = camera.view_rotation_double();
    SelectionView body_view;
    body_view.eye = {0, 0, 0};
    body_view.view_from_local = world_to_view;
    body_view.tan_half_fov_y = tan_y;
    body_view.tan_half_fov_x = tan_y*std::max(1u, extent.width)/std::max(1u, extent.height);
    std::vector<std::pair<double, uint32_t>> candidates;
    for (uint32_t i = 0; i < body_count; ++i) {
        const auto& body = system.body(i);
        if (body.definition.is_star || body.definition.is_spacecraft) continue;
        double radius = body.definition.radius_au();
        if (radius > 0.0) {
            auto center = camera_relative(body.render.position, eye);
            if (!visible({center, radius*(1+.01*(1+RELIEF_ROUNDING))}, body_view)) continue;
            double projected = radius*focal/std::max(glm::length(center), 1e-15);
            if (projected < TERRAIN_MIN_PROJECTED_RADIUS_PX) continue;
            covered_[i] = 1;
            candidates.emplace_back(-projected, i);
        }
    }
    std::sort(candidates.begin(), candidates.end());
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
        uint32_t i = candidates[candidate_index].second;
        uint32_t reserved = uint32_t(candidates.size()-candidate_index-1)*6;
        uint32_t remaining = MAX_PATCHES-uint32_t(prepared_.size())-reserved;
        const auto& body = system.body(i);
        double radius = body.definition.radius_au();
        HeightStream* active_stream = nullptr;
        if (settings.terrain_imported_enabled) {
            for (auto& binding : streams_) {
                if (binding.body == body.definition.name && binding.stream) {
                    active_stream = binding.stream.get();
                    break;
                }
            }
        }
        const HeightSource* source = active_stream ? &active_stream->source() : nullptr;
        if (source && !source->compatible(radius*AU_TO_KM*1000.0, std::clamp(1.0-body.definition.oblateness, .05, 1.0))) {
            source = nullptr;
            active_stream = nullptr;
        }
        auto center = camera_relative(body.render.position, eye);
        auto rotation = body_rotation(body.render.pole, body.render.spin_angle_rad);
        SelectionView view;
        view.eye = glm::transpose(rotation) * (-center / radius);
        view.view_from_local = world_to_view * rotation;
        view.tan_half_fov_y = tan_y;
        view.tan_half_fov_x = tan_y * std::max(1u, extent.width) / std::max(1u, extent.height);
        view.focal_pixels = focal;
        view.target_error_pixels = settings.terrain_error_pixels;
        view.polar_ratio = std::clamp(1.0 - body.definition.oblateness, 0.05, 1.0);
        bool synthetic = settings.terrain_enabled && !body.definition.is_black_hole &&
            body.definition.type != "Gas Giant" && body.definition.type != "Ice Giant";
        view.height_amplitude = synthetic ? std::min(double(settings.terrain_height_km)/(radius*AU_TO_KM), 0.01) : 0.0;
        view.max_patches = std::min(remaining, settings.terrain_patch_limit);
        auto imported_key = [&](TileId tile) {
            return GeometryKey{system.generation(), body.revisions.geometry, tile.key(), i, view.polar_ratio, 0.0, source->domain()};
        };
        auto& entry = selections_[i];
        bool selection_changed = false;
        if (source) {
            view.height_amplitude = std::max(std::abs(source->info().minimum_m), std::abs(source->info().maximum_m)) / source->info().equatorial_radius_m;
            uint64_t bounds_revision = source->bounds_revision();
            bool changed = entry.generation != system.generation() || entry.geometry_revision != body.revisions.geometry ||
                entry.bounds_revision != bounds_revision || entry.source != source->domain() || !(entry.view == view);
            if (entry.bounds_revision != bounds_revision || entry.generation != system.generation() ||
                entry.geometry_revision != body.revisions.geometry || entry.source != source->domain()) entry.descriptions.clear();
            SelectionSource selection_source(*source, entry.descriptions);
            if (changed) { entry.desired_valid = false; entry.prefetch.clear(); }
            std::unordered_set<uint64_t> wanted;
            auto request = [&](TileId tile) {
                if (cache_.find(imported_key(tile)) < 0) { wanted.insert(tile.key()); active_stream->request(tile); }
            };
            bool roots_ready = true;
            for (uint32_t face = 0; face < 6; ++face) {
                if (cache_.find(imported_key({face})) < 0) { request({face}); roots_ready = false; }
            }
            if (roots_ready && !entry.desired_valid) {
                entry.desired = select(view, selection_source, {{}, [&](TileId tile) { entry.prefetch.push_back(tile); }});
                std::sort(entry.prefetch.begin(), entry.prefetch.end(), [](TileId a, TileId b) { return a.key() < b.key(); });
                entry.prefetch.erase(std::unique(entry.prefetch.begin(), entry.prefetch.end()), entry.prefetch.end());
                entry.desired_valid = true;
                stats_.selection_nodes += entry.desired.visited;
            }
            bool desired_ready = roots_ready && entry.desired_valid;
            bool settled = !changed && entry.residency_revision == cache_.revision() && !entry.selection.loading_limited;
            if (roots_ready && !settled) for (auto tile : entry.prefetch) {
                if (cache_.find(imported_key(tile)) < 0) { desired_ready = false; request(tile); }
            }
            if (changed || entry.residency_revision != cache_.revision()) {
                entry.selection = desired_ready ? entry.desired : roots_ready ? select(view, selection_source, {
                    [&](TileId tile) { return cache_.find(imported_key(tile)) >= 0; }, request}) : Selection{};
                if (!roots_ready) entry.selection.loading_limited = true;
                if (!desired_ready) stats_.selection_nodes += entry.selection.visited;
                ++stats_.selection_rebuilds;
                selection_changed = true;
            }
            entry.generation = system.generation();
            entry.geometry_revision = body.revisions.geometry;
            entry.bounds_revision = bounds_revision;
            entry.residency_revision = cache_.revision();
            entry.source = source->domain();
            entry.view = view;
            auto retain = [&](TileId tile) {
                auto key = imported_key(tile);
                if (cache_.find(key) >= 0) {
                    retained_keys.push_back(key);
                    retained_heights.push_back({});
                    retained_streams.push_back(active_stream);
                }
            };
            for (uint32_t face = 0; face < 6; ++face) retain({face});
            if (selection_changed) {
                std::unordered_set<uint64_t> ancestors;
                entry.ancestors.clear();
                for (const auto& patch : entry.selection.patches) {
                    auto tile = patch.tile;
                    while (tile.level) {
                        tile = tile.parent();
                        if (!ancestors.insert(tile.key()).second) break;
                        entry.ancestors.push_back(tile);
                    }
                }
            }
            for (auto tile : entry.ancestors) retain(tile);
            for (auto& [tile, geometry] : active_stream->completed()) {
                if (wanted.contains(tile.key())) {
                    retained_keys.push_back(imported_key(tile));
                    retained_heights.push_back(std::move(geometry));
                    retained_streams.push_back(active_stream);
                } else active_stream->consumed(tile);
            }
            if (entry.selection.loading_limited) ++stats_.loading_bodies;
        } else if (entry.generation != system.generation() || entry.geometry_revision != body.revisions.geometry ||
                   entry.source != SourceDomain{} || !(entry.view == view)) {
            auto builds = metadata_.builds();
            entry.selection = select(view, &metadata_);
            stats_.metadata_builds += uint32_t(metadata_.builds() - builds);
            stats_.selection_nodes += entry.selection.visited;
            entry.view = view;
            entry.generation = system.generation();
            entry.geometry_revision = body.revisions.geometry;
            entry.source = {};
            ++stats_.selection_rebuilds;
            selection_changed = true;
        }
        Selection bootstrap;
        if (source && entry.selection.patches.empty() && entry.selection.loading_limited) {
            view.height_amplitude = 0.0;
            bootstrap = select(view);
            source = nullptr;
            selection_changed = true;
        }
        const auto& selection = bootstrap.patches.empty() ? entry.selection : bootstrap;
        if (selection.patches.empty()) continue;
        ++stats_.drawn_bodies;
        if (selection_changed) {
            entry.placements.clear();
            entry.placements.reserve(selection.patches.size());
            for (const auto& patch : selection.patches) {
                double width = std::ldexp(2.0, -int(patch.tile.level));
                PatchPlacement placement;
                placement.anchor = ellipsoid_point(patch.tile.face, -1.0+(patch.tile.x+.5)*width,
                    -1.0+(patch.tile.y+.5)*width, view.polar_ratio);
                placement.key = source ? imported_key(patch.tile)
                    : GeometryKey{system.generation(), body.revisions.geometry, patch.tile.key(), i, view.polar_ratio, view.height_amplitude};
                placement.bake.anchor_polar_ratio = {glm::vec3(placement.anchor), float(view.polar_ratio)};
                placement.bake.shape.x = float(view.height_amplitude);
                placement.bake.address = {patch.tile.face, patch.tile.level, patch.tile.x, patch.tile.y};
                entry.placements.push_back(placement);
            }
        }
        relief_extents_.emplace_back(i, source ? float(std::max(0.0, source->info().maximum_m) / (AU_TO_KM * 1000.0))
                                             : float(view.height_amplitude * radius));
        stats_.budget_limited |= selection.budget_limited;
        stats_.max_error_pixels = std::max(stats_.max_error_pixels, selection.max_error_pixels);
        stats_.deepest_level = std::max(stats_.deepest_level, selection.deepest_level);
        if (source) stats_.imported_patches += uint32_t(selection.patches.size());
        else if (view.height_amplitude == 0.0) stats_.smooth_patches += uint32_t(selection.patches.size());
        uint32_t draw_flags = uint32_t(settings.terrain_debug_tiles) | (uint32_t(settings.terrain_cached_fp64) << 1)
            | (uint32_t(settings.terrain_reference_fp64) << 2);
        if (!selection_changed && entry.draw_flags == draw_flags && entry.draw_center == center &&
            entry.draw_rotation == rotation && entry.draw_view == world_to_view) {
            prepared_.insert(prepared_.end(), entry.draws.begin(), entry.draws.end());
            continue;
        }
        entry.draws.clear();
        entry.draws.reserve(selection.patches.size());
        stats_.patch_updates += uint32_t(selection.patches.size());
        entry.draw_flags = draw_flags;
        entry.draw_center = center;
        entry.draw_rotation = rotation;
        entry.draw_view = world_to_view;
        auto eye_rotation = world_to_view * rotation;
        std::array<double, MAX_LEVEL+1> morph_errors{};
        if (source) for (uint32_t level = 0; level <= MAX_LEVEL; ++level)
            morph_errors[level] = source->morph_error(level)*focal/view.target_error_pixels;
        bool distant = glm::length(center) > 4.0*radius;
        double morph_error = error_coefficient(view.polar_ratio, view.height_amplitude)
            * focal/(GRID_CELLS*GRID_CELLS*view.target_error_pixels);
        for (size_t patch_index = 0; patch_index < selection.patches.size(); ++patch_index) {
            const auto& patch = selection.patches[patch_index];
            const auto& placement = entry.placements[patch_index];
            auto anchor = placement.anchor;
            DrawPatch draw;
            auto& gpu = draw.reference;
            gpu.anchor_radius = {center + rotation * (anchor * radius), radius};
            gpu.anchor_polar_ratio = {anchor, view.polar_ratio};
            gpu.rotation_height = {rotation[0], view.height_amplitude};
            gpu.rotation_error = {rotation[1], morph_error};
            gpu.rotation_reserved = {rotation[2], 0.0};
            gpu.eye = {view.eye, 0.0};
            gpu.address = {patch.tile.face, patch.tile.level, patch.tile.x, patch.tile.y};
            gpu.metadata = {i, patch.coarse_edges, settings.terrain_debug_tiles ? 1u : 0u, 0u};
            auto eye_anchor = world_to_view * glm::dvec3(gpu.anchor_radius);
            auto high = glm::vec3(eye_anchor);
            gpu.eye_anchor = {eye_anchor, distant ? 1.0 : 0.0};
            gpu.eye_rotation0 = {eye_rotation[0], 0.0};
            gpu.eye_rotation1 = {eye_rotation[1], 0.0};
            gpu.eye_rotation2 = {eye_rotation[2], 0.0};
            draw.cached.anchor_high = {high, float(radius)};
            draw.cached.anchor_low = {glm::vec3(eye_anchor - glm::dvec3(high)), 0.0f};
            draw.cached.rotation0 = {glm::vec3(eye_rotation[0]), 0.0f};
            draw.cached.rotation1 = {glm::vec3(eye_rotation[1]), 0.0f};
            draw.cached.rotation2 = {glm::vec3(eye_rotation[2]), 0.0f};
            if (source) {
                draw.cached.rotation0.w = float(morph_errors[patch.tile.level ? patch.tile.level-1 : 0]);
                draw.cached.rotation1.w = float(morph_errors[patch.tile.level > 1 ? patch.tile.level-2 : 0]);
                draw.cached.rotation2.w = 1.0f;
            }
            draw.cached.eye = {glm::vec3(view.eye - anchor), float(gpu.rotation_error.w)};
            draw.cached.address = gpu.address;
            draw.cached.metadata = {i, 0u, patch.coarse_edges, settings.terrain_debug_tiles ? 1u : 0u};
            draw.key = placement.key;
            draw.bake = placement.bake;
            draw.mode = distant ? 2u : 1u;
            if (settings.terrain_cached_fp64) draw.mode = 3u;
            if (settings.terrain_reference_fp64 && !source) draw.mode = 0;
            entry.draws.push_back(draw);
        }
        prepared_.insert(prepared_.end(), entry.draws.begin(), entry.draws.end());
    }
    if ((!settings.terrain_reference_fp64 && !prepared_.empty()) || !retained_keys.empty()) {
        std::vector<GeometryKey> keys;
        std::vector<GeometryBakeGPU> jobs;
        std::vector<std::shared_ptr<const HeightGeometry>> heights;
        keys.reserve(prepared_.size());
        jobs.reserve(prepared_.size());
        for (const auto& draw : prepared_) {
            keys.push_back(draw.key);
            jobs.push_back(draw.bake);
            heights.push_back({});
        }
        keys.insert(keys.end(), retained_keys.begin(), retained_keys.end());
        heights.insert(heights.end(), retained_heights.begin(), retained_heights.end());
        jobs.resize(keys.size());
        auto plan = cache_.prepare(ctx, keys, jobs, heights);
        for (uint32_t index = 0; index < retained_keys.size(); ++index) {
            if (retained_heights[index] && plan.slots[prepared_.size()+index] >= 0 && retained_streams[index]) {
                retained_streams[index]->consumed(TileId::from_key(retained_keys[index].tile));
            }
        }
        stats_.cache_hits = plan.hits;
        stats_.cache_misses = plan.misses;
        stats_.cache_bakes = uint32_t(plan.bakes.size());
        for (uint32_t index = 0; index < prepared_.size(); ++index) {
            if (plan.slots[index] < 0) prepared_[index].mode = 0;
            else prepared_[index].cached.metadata.y = uint32_t(plan.slots[index]);
        }
    }
    std::stable_sort(prepared_.begin(), prepared_.end(), [](const DrawPatch& a, const DrawPatch& b) {
        return a.mode != b.mode ? a.mode < b.mode : a.reference.metadata.y < b.reference.metadata.y;
    });
    stats_.patches = uint32_t(prepared_.size());
    stats_.metadata_entries = uint32_t(metadata_.size());
    for (auto& binding : streams_) {
        if (!binding.stream) continue;
        auto s = binding.stream->stats();
        stats_.stream.pending += s.pending;
        stats_.stream.ready += s.ready;
        stats_.stream.failed += s.failed;
        stats_.stream.builds += s.builds;
        stats_.stream.payload_reads += s.payload_reads;
        if (stats_.stream.error.empty() && !s.error.empty()) stats_.stream.error = s.error;
        if (!s.error.empty() && binding.reported_error != s.error) {
            binding.reported_error = s.error;
            log::error("Terrain source '{}': {}", binding.body, binding.reported_error);
        }
    }
    auto* references = static_cast<PatchGPU*>(patches_[ctx.frame_index()].mapped);
    auto* cached = static_cast<CachedPatchGPU*>(cached_patches_[ctx.frame_index()].mapped);
    for (uint32_t index = 0; index < prepared_.size(); ++index) {
        if (prepared_[index].mode == 0) {
            std::memcpy(references + index, &prepared_[index].reference, sizeof(PatchGPU));
            ++stats_.reference_patches;
        } else {
            std::memcpy(cached + index, &prepared_[index].cached, sizeof(CachedPatchGPU));
            ++stats_.cached_patches;
        }
    }
    if (stats_.reference_patches) ctx.flush_buffer(patches_[ctx.frame_index()], 0, stats_.reference_patches*sizeof(PatchGPU));
    if (stats_.cached_patches) ctx.flush_buffer(cached_patches_[ctx.frame_index()], stats_.reference_patches*sizeof(CachedPatchGPU),
                                              stats_.cached_patches*sizeof(CachedPatchGPU));
}

void TerrainRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, std::span<const rhi::PushDescriptorWrite> scene) {
    if (prepared_.empty()) return;
    std::vector<rhi::PushDescriptorWrite> writes(scene.begin(), scene.end());
    vkCmdBindIndexBuffer(cmd, indices_.handle, 0, VK_INDEX_TYPE_UINT32);
    constexpr uint32_t index_count = GRID_CELLS * GRID_CELLS * 6;
    for (uint32_t first = 0; first < prepared_.size();) {
        uint32_t mask = prepared_[first].reference.metadata.y;
        uint32_t mode = prepared_[first].mode;
        uint32_t end = first + 1;
        while (end < prepared_.size() && prepared_[end].mode == mode && prepared_[end].reference.metadata.y == mask) ++end;
        auto& pipeline = mode == 0 ? reference_pipeline_ : mode == 1 ? pipeline_ : mode == 2 ? precise_pipeline_ : cached_fp64_pipeline_;
        writes.resize(scene.size());
        if (mode == 0) {
            writes.push_back(rhi::PushDescriptorWrite::storage_buffer(3, patches_[ctx.frame_index()], 0, prepared_.size() * sizeof(PatchGPU)));
        } else {
            writes.push_back(rhi::PushDescriptorWrite::storage_buffer(3, cached_patches_[ctx.frame_index()], 0, prepared_.size() * sizeof(CachedPatchGPU)));
            writes.push_back(rhi::PushDescriptorWrite::storage_buffer(4, cache_.buffer()));
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
        ctx.push_descriptors(cmd, pipeline, writes);
        vkCmdDrawIndexed(cmd, index_count, end - first, mask * index_count, 0, first);
        first = end;
    }
}

}
