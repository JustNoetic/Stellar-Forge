#include "terrain/terrain_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace sf::terrain {

static_assert(sizeof(PatchGPU) == 352);
static_assert(offsetof(PatchGPU, address) == 192);
static_assert(sizeof(CachedPatchGPU) == 128);

void TerrainRenderer::init(rhi::VkContext& ctx) {
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
}

void TerrainRenderer::shutdown(rhi::VkContext& ctx) {
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

bool TerrainRenderer::covers(uint32_t body) const noexcept {
    return std::find(bodies_.begin(), bodies_.end(), body) != bodies_.end();
}

void TerrainRenderer::prepare(rhi::VkContext& ctx, const Camera& camera, const ephemeris::SystemSnapshot& system,
                              const rendering::RenderSettings& settings, VkExtent2D extent, uint32_t body_count) {
    prepared_.clear();
    bodies_.clear();
    stats_ = {};
    if (!settings.terrain_enabled) return;
    double tan_y = std::tan(camera.fov_deg * DEG_TO_RAD * 0.5);
    double focal = std::max(1u, extent.height) / (2.0 * tan_y);
    auto eye = camera.eye_pos_render();
    auto world_to_view = camera.view_rotation_double();
    std::vector<std::pair<double, uint32_t>> candidates;
    for (uint32_t i = 0; i < body_count; ++i) {
        const auto& body = system.body(i);
        if (body.definition.is_star || body.definition.is_black_hole || body.definition.is_spacecraft) continue;
        if (body.definition.type == "Gas Giant" || body.definition.type == "Ice Giant") continue;
        double radius = body.definition.radius_au();
        double distance = glm::length(camera_relative(body.render.position, eye));
        double projected = radius * focal / std::max(distance, 1e-15);
        if (radius > 0.0 && projected >= 8.0) candidates.emplace_back(-projected, i);
    }
    std::sort(candidates.begin(), candidates.end());
    for (const auto& [priority, i] : candidates) {
        uint32_t remaining = MAX_PATCHES - uint32_t(prepared_.size());
        if (remaining < 6) {
            stats_.budget_limited = true;
            continue;
        }
        const auto& body = system.body(i);
        double radius = body.definition.radius_au();
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
        view.height_amplitude = std::min(double(settings.terrain_height_km) / (radius * AU_TO_KM), 0.01);
        view.max_patches = std::min(remaining, settings.terrain_patch_limit);
        auto& entry = selections_[i];
        if (entry.generation != system.generation() || entry.geometry_revision != body.revisions.geometry || !(entry.view == view)) {
            auto builds = metadata_.builds();
            entry.selection = select(view, &metadata_);
            stats_.metadata_builds += uint32_t(metadata_.builds() - builds);
            stats_.selection_nodes += entry.selection.visited;
            entry.view = view;
            entry.generation = system.generation();
            entry.geometry_revision = body.revisions.geometry;
            ++stats_.selection_rebuilds;
        }
        const auto& selection = entry.selection;
        bodies_.push_back(i);
        stats_.budget_limited |= selection.budget_limited;
        stats_.deepest_level = std::max(stats_.deepest_level, selection.deepest_level);
        for (const auto& patch : selection.patches) {
            double width = std::ldexp(2.0, -int(patch.tile.level));
            auto anchor = ellipsoid_point(patch.tile.face, -1.0 + (patch.tile.x + 0.5) * width,
                                          -1.0 + (patch.tile.y + 0.5) * width, view.polar_ratio);
            DrawPatch draw;
            auto& gpu = draw.reference;
            gpu.anchor_radius = {center + rotation * (anchor * radius), radius};
            gpu.anchor_polar_ratio = {anchor, view.polar_ratio};
            gpu.rotation_height = {rotation[0], view.height_amplitude};
            gpu.rotation_error = {rotation[1], error_coefficient(view.polar_ratio, view.height_amplitude)
                * focal / (GRID_CELLS * GRID_CELLS * view.target_error_pixels)};
            gpu.rotation_reserved = {rotation[2], 0.0};
            gpu.eye = {view.eye, 0.0};
            gpu.address = {patch.tile.face, patch.tile.level, patch.tile.x, patch.tile.y};
            gpu.metadata = {i, patch.coarse_edges, settings.terrain_debug_tiles ? 1u : 0u, 0u};
            auto eye_anchor = world_to_view * (center + rotation * (anchor * radius));
            auto high = glm::vec3(eye_anchor);
            auto eye_rotation = world_to_view * rotation;
            gpu.eye_anchor = {eye_anchor, glm::length(center) > 4.0 * radius ? 1.0 : 0.0};
            gpu.eye_rotation0 = {eye_rotation[0], 0.0};
            gpu.eye_rotation1 = {eye_rotation[1], 0.0};
            gpu.eye_rotation2 = {eye_rotation[2], 0.0};
            draw.cached.anchor_high = {high, float(radius)};
            draw.cached.anchor_low = {glm::vec3(eye_anchor - glm::dvec3(high)), 0.0f};
            draw.cached.rotation0 = {glm::vec3(eye_rotation[0]), 0.0f};
            draw.cached.rotation1 = {glm::vec3(eye_rotation[1]), 0.0f};
            draw.cached.rotation2 = {glm::vec3(eye_rotation[2]), 0.0f};
            draw.cached.eye = {glm::vec3(view.eye - anchor), float(gpu.rotation_error.w)};
            draw.cached.address = gpu.address;
            draw.cached.metadata = {i, 0u, patch.coarse_edges, settings.terrain_debug_tiles ? 1u : 0u};
            draw.key = {system.generation(), body.revisions.geometry, patch.tile.key(), i, view.polar_ratio, view.height_amplitude};
            draw.bake.anchor_polar_ratio = {anchor, view.polar_ratio};
            draw.bake.shape.x = view.height_amplitude;
            draw.bake.address = gpu.address;
            draw.mode = glm::length(center) > 4.0 * radius ? 2u : 1u;
            if (settings.terrain_cached_fp64) draw.mode = 3u;
            if (settings.terrain_reference_fp64) draw.mode = 0;
            prepared_.push_back(draw);
        }
    }
    if (!settings.terrain_reference_fp64 && !prepared_.empty()) {
        std::vector<GeometryKey> keys;
        std::vector<GeometryBakeGPU> jobs;
        keys.reserve(prepared_.size());
        jobs.reserve(prepared_.size());
        for (const auto& draw : prepared_) {
            keys.push_back(draw.key);
            jobs.push_back(draw.bake);
        }
        auto plan = cache_.prepare(ctx, keys, jobs);
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
    auto* references = static_cast<PatchGPU*>(patches_[ctx.frame_index()].mapped);
    auto* cached = static_cast<CachedPatchGPU*>(cached_patches_[ctx.frame_index()].mapped);
    for (uint32_t index = 0; index < prepared_.size(); ++index) {
        std::memcpy(references + index, &prepared_[index].reference, sizeof(PatchGPU));
        std::memcpy(cached + index, &prepared_[index].cached, sizeof(CachedPatchGPU));
        if (prepared_[index].mode == 0) ++stats_.reference_patches;
        else ++stats_.cached_patches;
    }
    if (!prepared_.empty()) {
        ctx.flush_buffer(patches_[ctx.frame_index()], 0, prepared_.size() * sizeof(PatchGPU));
        ctx.flush_buffer(cached_patches_[ctx.frame_index()], 0, prepared_.size() * sizeof(CachedPatchGPU));
    }
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
