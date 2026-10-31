#include "terrain/terrain_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include "core/path_utils.hpp"
#include "core/log.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace sf::terrain {

namespace {

class SelectionSource final : public TerrainSource {
public:
    SelectionSource(const HeightSource& source, std::unordered_map<uint64_t, SurfaceTile>& cache) : source_(source), cache_(cache) {}
    SourceDomain domain() const noexcept override { return source_.domain(); }
    uint64_t bounds_revision() const noexcept override { return source_.bounds_revision(); }
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
static_assert(sizeof(CachedPatchGPU) == 144);

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
        {25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT, 1},
        {28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {29, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {31, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    reference_pipeline_ =rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
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

std::shared_ptr<const HeightSource> TerrainRenderer::height_source(std::string_view body) const noexcept {
    for (const auto& binding : streams_) {
        if (binding.body == body && binding.stream) return binding.stream->share_source();
    }
    return nullptr;
}

std::vector<std::pair<std::string, std::shared_ptr<const HeightSource>>> TerrainRenderer::height_sources() const {
    std::vector<std::pair<std::string, std::shared_ptr<const HeightSource>>> sources;
    for (const auto& binding : streams_) if (binding.stream) sources.emplace_back(binding.body, binding.stream->share_source());
    return sources;
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
                              const rendering::RenderSettings& settings, VkExtent2D extent, uint32_t body_count,
                              const rendering::RefractionHost* refraction) {
    prepared_.clear();
    batch_count_ = 0;
    covered_.assign(body_count, 0);
    relief_extents_.clear();
    stats_ = {};
    std::erase_if(selections_, [&](const auto& item) {
        return item.first >= system.size() || item.second.identity != system.body(item.first).revisions.identity;
    });
    auto& retained_keys = retained_keys_;
    auto& retained_heights = retained_heights_;
    auto& retained_streams = retained_streams_;
    retained_keys.clear();
    retained_heights.clear();
    retained_streams.clear();
    double tan_y = std::tan(camera.fov_deg * DEG_TO_RAD * 0.5);
    double focal = std::max(1u, extent.height) / (2.0 * tan_y);
    int refraction_body = refraction ? refraction->body : -1;
    double refraction_pad = refraction_body >= 0 ? refraction->max_displacement : 0.0;
    auto padded = [&](double tangent) { return refraction_pad > 0.0 ? std::tan(std::min(std::atan(tangent)+refraction_pad, 1.5)) : tangent; };
    auto eye = camera.eye_pos_render();
    auto world_to_view = camera.view_rotation_double();
    SelectionView body_view;
    body_view.eye = {0, 0, 0};
    body_view.view_from_local = world_to_view;
    body_view.tan_half_fov_y = padded(tan_y);
    body_view.tan_half_fov_x = padded(tan_y*std::max(1u, extent.width)/std::max(1u, extent.height));
    auto& candidates = candidates_;
    candidates.clear();
    for (uint32_t i = 0; i < body_count; ++i) {
        const auto& body = system.body(i);
        if (body.definition.is_star || body.definition.is_spacecraft) continue;
        double radius = body.opaque_radius_au();
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
        double radius = body.opaque_radius_au();
        HeightStream* active_stream = nullptr;
        if (settings.terrain_imported_enabled && !body.definition.atmosphere.pressure_reference) {
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
        double ocean_floor_m = settings.oceans_enabled && body.definition.ocean.enabled
            ? body.definition.ocean.sea_level_km * 1000.0 : -std::numeric_limits<double>::infinity();
        if (active_stream && active_stream->floor_m() != ocean_floor_m) {
            active_stream->set_floor_m(ocean_floor_m);
            selections_.erase(i);
        }
        double ocean_key = std::isfinite(ocean_floor_m) ? 1.0 + ocean_floor_m : 0.0;
        auto center = camera_relative(body.render.position, eye);
        auto rotation = body_rotation(body.render.pole, body.render.spin_angle_rad);
        SelectionView view;
        view.eye = glm::transpose(rotation) * (-center / radius);
        view.view_from_local = world_to_view * rotation;
        view.tan_half_fov_y = padded(tan_y);
        view.tan_half_fov_x = padded(tan_y * std::max(1u, extent.width) / std::max(1u, extent.height));
        view.focal_pixels = focal;
        view.target_error_pixels = settings.terrain_error_pixels;
        view.polar_ratio = std::clamp(1.0 - body.definition.oblateness, 0.05, 1.0);
        if (int(i) == refraction_body) {
            view.occluder_scale = std::clamp(refraction->lowest_visible_chord_km/std::max(refraction->ground_radius_km, 1e-9), 0.0, 1.0);
        }
        bool synthetic = settings.terrain_enabled && !body.definition.atmosphere.pressure_reference && !body.definition.is_black_hole &&
            body.definition.type != "Gas Giant" && body.definition.type != "Ice Giant";
        view.height_amplitude = synthetic ? std::min(double(settings.terrain_height_km)/(radius*AU_TO_KM), 0.01) : 0.0;
        view.max_patches = std::min(remaining, settings.terrain_patch_limit);
        auto imported_key = [&](TileId tile) {
            return GeometryKey{body.revisions.identity, body.revisions.geometry, tile.key(), i, view.polar_ratio, ocean_key, source->domain()};
        };
        auto& entry = selections_[i];
        bool selection_changed = false;
        bool view_changed = !(entry.view == view);
        if (source) {
            view.height_amplitude = std::max(std::abs(source->info().minimum_m), std::abs(source->info().maximum_m)) / source->info().equatorial_radius_m;
            uint64_t bounds_revision = source->bounds_revision();
            view_changed = !(entry.view == view);
            bool invalidated = entry.identity != body.revisions.identity || entry.geometry_revision != body.revisions.geometry ||
                entry.bounds_revision != bounds_revision || entry.source != source->domain();
            bool changed = invalidated || view_changed;
            if (invalidated) { entry.descriptions.clear(); entry.plan.invalidate(); }
            SelectionSource selection_source(*source, entry.descriptions);
            if (changed) entry.desired_valid = false;
            std::unordered_set<uint64_t> wanted;
            auto request = [&](TileId tile) {
                if (cache_.find(imported_key(tile)) < 0) { wanted.insert(tile.key()); active_stream->request(tile); }
            };
            bool dependencies_ready = !invalidated && entry.residency_revision == cache_.revision() &&
                entry.selection_from_desired && !entry.selection.loading_limited;
            bool roots_ready = dependencies_ready;
            if (!roots_ready) {
                roots_ready = true;
                for (uint32_t face = 0; face < 6; ++face) {
                    if (cache_.find(imported_key({face})) < 0) { request({face}); roots_ready = false; }
                }
            }
            bool desired_rebuilt = false;
            if (roots_ready && !entry.desired_valid) {
                prefetch_scratch_.clear();
                desired_rebuilt = update_selection(entry.desired, entry.plan, view, selection_source,
                    {{}, [&](TileId tile) { prefetch_scratch_.push_back(tile); }}, &selection_scratch_);
                stats_.selection_checks += entry.plan.checks();
                if (desired_rebuilt) {
                    entry.prefetch.swap(prefetch_scratch_);
                    std::sort(entry.prefetch.begin(), entry.prefetch.end(), [](TileId a, TileId b) { return a.key() < b.key(); });
                    entry.prefetch.erase(std::unique(entry.prefetch.begin(), entry.prefetch.end()), entry.prefetch.end());
                    stats_.selection_nodes += entry.desired.visited;
                    ++stats_.selection_rebuilds;
                } else ++stats_.selection_reuses;
                entry.desired_valid = true;
            }
            bool desired_ready = roots_ready && entry.desired_valid;
            if (roots_ready && (!dependencies_ready || desired_rebuilt)) for (auto tile : entry.prefetch) {
                if (cache_.find(imported_key(tile)) < 0) { desired_ready = false; request(tile); }
            }
            if (changed || entry.residency_revision != cache_.revision()) {
                if (desired_ready) {
                    if (desired_rebuilt || !entry.selection_from_desired) {
                        entry.selection = entry.desired;
                        selection_changed = true;
                    } else entry.selection.max_error_pixels = entry.desired.max_error_pixels;
                } else {
                    entry.selection = roots_ready ? select(view, selection_source, {
                        [&](TileId tile) { return cache_.find(imported_key(tile)) >= 0; }, request}, &selection_scratch_) : Selection{};
                    stats_.selection_nodes += entry.selection.visited;
                    if (roots_ready) ++stats_.selection_rebuilds;
                    selection_changed = true;
                }
                entry.selection_from_desired = desired_ready;
                if (!roots_ready) entry.selection.loading_limited = true;
            }
            entry.identity = body.revisions.identity;
            entry.geometry_revision = body.revisions.geometry;
            entry.bounds_revision = bounds_revision;
            entry.residency_revision = cache_.revision();
            entry.source = source->domain();
            entry.view = view;
            if (selection_changed || entry.protected_revision != cache_.revision()) {
                ++stats_.dependency_rebuilds;
                entry.protected_keys.clear();
                auto retain = [&](TileId tile) {
                    auto key = imported_key(tile);
                    auto leaf = std::lower_bound(entry.selection.patches.begin(), entry.selection.patches.end(), tile.key(),
                        [](const Patch& patch, uint64_t value) { return patch.tile.key() < value; });
                    if (leaf != entry.selection.patches.end() && leaf->tile == tile) return;
                    if (cache_.find(key) >= 0) entry.protected_keys.push_back(key);
                };
                for (uint32_t face = 0; face < 6; ++face) retain({face});
                for (auto tile : entry.selection.required_tiles) if (tile.level) retain(tile);
                entry.protected_revision = cache_.revision();
            }
            retained_keys.insert(retained_keys.end(), entry.protected_keys.begin(), entry.protected_keys.end());
            retained_heights.resize(retained_keys.size());
            retained_streams.insert(retained_streams.end(), entry.protected_keys.size(), active_stream);
            for (auto& [tile, geometry] : active_stream->completed()) {
                if (wanted.contains(tile.key())) {
                    retained_keys.push_back(imported_key(tile));
                    retained_heights.push_back(std::move(geometry));
                    retained_streams.push_back(active_stream);
                } else active_stream->consumed(tile);
            }
            if (entry.selection.loading_limited) ++stats_.loading_bodies;
        } else if (entry.identity != body.revisions.identity || entry.geometry_revision != body.revisions.geometry ||
                   entry.source != SourceDomain{} || !(entry.view == view)) {
            auto builds = metadata_.builds();
            if (entry.identity != body.revisions.identity || entry.geometry_revision != body.revisions.geometry || entry.source != SourceDomain{})
                entry.plan.invalidate();
            selection_changed = update_selection(entry.selection, entry.plan, view, ProceduralSource{&metadata_}, {}, &selection_scratch_);
            stats_.selection_checks += entry.plan.checks();
            if (selection_changed) { ++stats_.selection_rebuilds; stats_.selection_nodes += entry.selection.visited; }
            else ++stats_.selection_reuses;
            stats_.metadata_builds += uint32_t(metadata_.builds() - builds);
            entry.view = view;
            entry.identity = body.revisions.identity;
            entry.geometry_revision = body.revisions.geometry;
            entry.source = {};
            entry.selection_from_desired = false;
        }
        Selection bootstrap;
        if (source && entry.selection.patches.empty() && entry.selection.loading_limited) {
            view.height_amplitude = 0.0;
            bootstrap = select(view, nullptr, &selection_scratch_);
            source = nullptr;
            selection_changed = true;
        }
        const auto& selection = bootstrap.patches.empty() ? entry.selection : bootstrap;
        if (selection.patches.empty()) continue;
        ++stats_.drawn_bodies;
        if (selection_changed) {
            entry.placements.resize(selection.patches.size());
            for (size_t index = 0; index < selection.patches.size(); ++index) {
                const auto& patch = selection.patches[index];
                auto key = source ? imported_key(patch.tile)
                    : GeometryKey{body.revisions.identity, body.revisions.geometry, patch.tile.key(), i, view.polar_ratio, view.height_amplitude};
                auto& placement = entry.placements[index];
                if (placement.key == key) continue;
                ++stats_.placement_updates;
                double width = std::ldexp(2.0, -int(patch.tile.level));
                placement.anchor = ellipsoid_point(patch.tile.face, -1.0+(patch.tile.x+.5)*width,
                    -1.0+(patch.tile.y+.5)*width, view.polar_ratio);
                placement.key = key;
                placement.bake.anchor_polar_ratio = {glm::vec3(placement.anchor), float(view.polar_ratio)};
                placement.bake.shape.x = float(view.height_amplitude);
                placement.bake.address = {patch.tile.face, patch.tile.level, patch.tile.x, patch.tile.y};
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
        if (!selection_changed && !view_changed && entry.draw_flags == draw_flags && entry.draw_center == center &&
            entry.draw_rotation == rotation && entry.draw_view == world_to_view) {
            for (const auto& draw : entry.draws) prepared_.push_back({&draw, draw.mode});
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
            draw.cached.anchor_offset = {glm::vec3(rotation * (anchor * radius)), 0.0f};
            draw.key = placement.key;
            draw.bake = placement.bake;
            draw.mode = distant ? 2u : 1u;
            if (settings.terrain_cached_fp64) draw.mode = 3u;
            if (settings.terrain_reference_fp64 && !source) draw.mode = 0;
            entry.draws.push_back(draw);
        }
        for (const auto& draw : entry.draws) prepared_.push_back({&draw, draw.mode});
    }
    if (std::any_of(prepared_.begin(), prepared_.end(), [](const PreparedPatch& patch) { return patch.mode != 0; }) || !retained_keys.empty()) {
        auto& keys = request_keys_;
        auto& jobs = request_jobs_;
        auto& heights = request_heights_;
        keys.clear();
        jobs.clear();
        heights.clear();
        keys.reserve(prepared_.size());
        jobs.reserve(prepared_.size());
        for (const auto& draw : prepared_) {
            keys.push_back(draw.draw->key);
            jobs.push_back(draw.draw->bake);
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
            else prepared_[index].slot = uint32_t(plan.slots[index]);
        }
    }
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
    std::array<uint32_t, 64> counts{}, offsets{};
    for (const auto& patch : prepared_) ++counts[patch.mode*16 + patch.draw->reference.metadata.y];
    uint32_t first = 0;
    for (uint32_t group = 0; group < counts.size(); ++group) {
        offsets[group] = first;
        if (counts[group]) batches_[batch_count_++] = {first, counts[group], group/16, group%16};
        first += counts[group];
    }
    for (const auto& patch : prepared_) {
        uint32_t index = offsets[patch.mode*16 + patch.draw->reference.metadata.y]++;
        if (patch.mode == 0) {
            std::memcpy(references + index, &patch.draw->reference, sizeof(PatchGPU));
            ++stats_.reference_patches;
        } else {
            std::memcpy(cached + index, &patch.draw->cached, sizeof(CachedPatchGPU));
            cached[index].metadata.y = patch.slot;
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
    for (uint32_t index = 0; index < batch_count_; ++index) {
        auto [first, count, mode, mask] = batches_[index];
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
        vkCmdDrawIndexed(cmd, index_count, count, mask * index_count, 0, first);
    }
}

double TerrainRenderer::minimum_height_km(const ephemeris::BodyDefinition& body, const rendering::RenderSettings& settings) const noexcept {
    if (body.atmosphere.pressure_reference) return 0.0;
    double minimum = std::min(0.0, body.height_range_km.x);
    if (settings.terrain_imported_enabled) {
        for (const auto& binding : streams_) {
            if (binding.body != body.name || !binding.stream) continue;
            const auto& source = binding.stream->source();
            if (source.compatible(body.radius_au()*AU_TO_KM*1000.0, std::clamp(1.0-body.oblateness, 0.05, 1.0))) {
                double lowest = std::min(minimum, source.info().minimum_m*0.001);
                if (settings.oceans_enabled && body.ocean.enabled) lowest = std::max(lowest, std::min(minimum, body.ocean.sea_level_km));
                return lowest;
            }
        }
    }
    if (settings.terrain_enabled && !body.is_black_hole && body.type != "Gas Giant" && body.type != "Ice Giant")
        minimum = std::min(minimum, -std::min(double(settings.terrain_height_km), body.radius_au()*AU_TO_KM*0.01));
    return minimum;
}

}
