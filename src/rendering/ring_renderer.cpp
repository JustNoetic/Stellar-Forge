#include "rendering/ring_renderer.hpp"
#include "core/constants.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>

namespace sf::rendering {

static bool ring_host(const ephemeris::BodyDefinition& def) {
    return !def.rings.empty() && !def.is_star && !def.is_black_hole && !def.is_spacecraft;
}

void RingRenderer::init(rhi::VkContext& ctx) {
    mesh_ = build_annulus_mesh(ctx, RING_ANGULAR_SEGMENTS, RING_RADIAL_SUBDIVISIONS);

    std::vector<RingHostGPU> empty_hosts(MAX_RING_HOSTS);
    for (auto& buf : host_ssbos_) {
        buf = ctx.create_host_buffer(MAX_RING_HOSTS * sizeof(RingHostGPU),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, empty_hosts.data());
    }

    gradient_atlas_ = ctx.create_image_2d(
        RING_ATLAS_WIDTH, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, RING_PROFILE_MIP_LEVELS, RING_ATLAS_ROWS);

    shadow_lut_ = ctx.create_image_2d(
        RING_SHADOW_LUT_WIDTH, RING_SHADOW_LUT_ANGLES, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, MAX_RING_HOSTS);

    std::vector<glm::vec4> zero_atlas(RING_ATLAS_WIDTH * RING_ATLAS_ROWS, glm::vec4(0.0f));
    ctx.upload_image_2d(gradient_atlas_, zero_atlas.data(),
                        zero_atlas.size() * sizeof(glm::vec4),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    std::vector<float> zero_lut(static_cast<size_t>(RING_SHADOW_LUT_WIDTH) * RING_SHADOW_LUT_ANGLES * MAX_RING_HOSTS, 0.0f);
    ctx.upload_image_2d(shadow_lut_, zero_lut.data(),
                        zero_lut.size() * sizeof(float),
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkDeviceSize rebuild_bytes = VkDeviceSize(RING_ATLAS_WIDTH) * RING_ATLAS_ROWS * 2 * sizeof(glm::vec4) +
        VkDeviceSize(RING_SHADOW_LUT_WIDTH) * RING_SHADOW_LUT_ANGLES * MAX_RING_HOSTS * sizeof(float);
    staging_.init(ctx, rebuild_bytes * rhi::FRAMES_IN_FLIGHT);

    rhi::GraphicsPipelineDesc desc{};
    desc.vert_path = "celestial/ring.vert";
    desc.frag_path = "celestial/ring.frag";
    desc.vertex_bindings = {{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};
    desc.vertex_attributes = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}};
    desc.bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {22, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {23, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {24, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    desc.push_constant_size = sizeof(uint32_t) * 4;
    desc.depth_test = true;
    desc.depth_write = false;
    desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    desc.premultiplied_blend = true;
    desc.capture_statistics = ctx.pipeline_statistics_enabled();
    desc.cull_mode = VK_CULL_MODE_NONE;
    for (uint32_t i = 0; i < pipelines_.size(); ++i) {
        uint32_t light_mode = i / 4;
        desc.macros = {{"RING_DRAW_CAPACITY", std::to_string(1u << (i % 4))},
                       {"RING_FINITE_SOURCE", light_mode == 1 ? "1" : "0"},
                       {"RING_SINGLE_STAR", light_mode >= 2 ? "1" : "0"},
                       {"RING_FIXED_SIDE", light_mode == 2 ? "1" : light_mode == 3 ? "0" : "-1"}};
        pipelines_[i] = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    }
    if (ctx.fragment_stores_enabled()) {
        desc.bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT});
        desc.bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT});
        for (uint32_t i = 0; i < transparency_pipelines_.size(); ++i) {
            uint32_t light_mode = i / 4;
            desc.macros = {{"RING_DRAW_CAPACITY", std::to_string(1u << (i % 4))},
                {"RING_FINITE_SOURCE", light_mode == 1 ? "1" : "0"},
                {"RING_SINGLE_STAR", light_mode >= 2 ? "1" : "0"},
                {"RING_FIXED_SIDE", light_mode == 2 ? "1" : light_mode == 3 ? "0" : "-1"},
                {"TRANSPARENCY_CAPTURE", "1"}};
            transparency_pipelines_[i] = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
        }
    }
    for (auto& atlas : ringshine_atlases_) {
        atlas = ctx.create_image_2d(
            RINGSHINE_TILE_WIDTH * 16, RINGSHINE_TILE_HEIGHT * MAX_RING_HOSTS,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    }
    ctx.immediate_submit([&](VkCommandBuffer imm) {
        for (auto& atlas : ringshine_atlases_) {
            ctx.transition_image(imm, atlas, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(imm, atlas.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
            ctx.transition_image(imm, atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });

    rhi::GraphicsPipelineDesc bake_desc{};
    bake_desc.vert_path = "post/fullscreen.vert";
    bake_desc.frag_path = "post/ringshine_map.frag";
    bake_desc.bindings = {
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };
    bake_desc.push_constant_size = 80;
    bake_desc.color_formats = {VK_FORMAT_R32G32B32A32_SFLOAT};
    bake_desc.depth_format = VK_FORMAT_UNDEFINED;
    bake_desc.depth_test = false;
    bake_desc.depth_write = false;
    bake_desc.cull_mode = VK_CULL_MODE_NONE;
    ringshine_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, bake_desc);
    for (auto& buffer : ringshine_radial_buffers_) {
        buffer = ctx.create_host_buffer(MAX_RING_HOSTS * 4096 * sizeof(glm::vec2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    hosts_gpu_.reserve(MAX_RING_HOSTS);
    host_draw_ranges_.reserve(MAX_RING_HOSTS);
    force_rebuild_ = true;
}

void RingRenderer::shutdown(rhi::VkContext& ctx) {
    cached_hosts_.clear();
    staging_.shutdown(ctx);
    rhi::ShaderCache::destroy_pipeline(ctx, ringshine_pipeline_);
    for (auto& pipeline : pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    for (auto& pipeline : transparency_pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    ctx.destroy_image(shadow_lut_);
    ctx.destroy_image(gradient_atlas_);
    for (auto& atlas : ringshine_atlases_) ctx.destroy_image(atlas);
    for (auto& buffer : ringshine_radial_buffers_) ctx.destroy_buffer(buffer);
    for (auto& buf : host_ssbos_) {
        ctx.destroy_buffer(buf);
    }
    ctx.destroy_buffer(mesh_.indices);
    ctx.destroy_buffer(mesh_.vertices);
}

RingRenderer::AnnulusMesh RingRenderer::build_annulus_mesh(rhi::VkContext& ctx, uint32_t angular_segments, uint32_t radial_subdivisions) {
    std::vector<glm::vec3> verts;
    std::vector<uint32_t> indices;
    verts.reserve(static_cast<size_t>(angular_segments + 1) * (radial_subdivisions + 1));
    indices.reserve(static_cast<size_t>(angular_segments) * radial_subdivisions * 6);

    for (uint32_t i = 0; i <= angular_segments; ++i) {
        double theta = TWO_PI * double(i) / double(angular_segments);
        float ct = static_cast<float>(std::cos(theta));
        float st = static_cast<float>(std::sin(theta));
        for (uint32_t j = 0; j <= radial_subdivisions; ++j) {
            float u = float(j) / float(radial_subdivisions);
            verts.push_back({u, ct, st});
        }
    }

    uint32_t stride = radial_subdivisions + 1;
    for (uint32_t j = 0; j < radial_subdivisions; ++j) {
        for (uint32_t i = 0; i < angular_segments; ++i) {
            uint32_t p00 = i * stride + j;
            uint32_t p01 = p00 + 1;
            uint32_t p10 = (i + 1) * stride + j;
            uint32_t p11 = p10 + 1;
            indices.push_back(p00);
            indices.push_back(p01);
            indices.push_back(p10);
            indices.push_back(p10);
            indices.push_back(p01);
            indices.push_back(p11);
        }
    }

    AnnulusMesh mesh;
    mesh.index_count = static_cast<uint32_t>(indices.size());
    mesh.vertices = ctx.create_device_buffer(verts.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, verts.data());
    mesh.indices = ctx.create_device_buffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data());
    return mesh;
}

void RingRenderer::rebuild_profiles(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& sys, uint32_t body_count) {
    host_mappings_.clear();
    uint32_t next_seg_row = MAX_RING_HOSTS;

    std::vector<glm::vec4> atlas_rows;
    std::vector<uint32_t> atlas_row_indices;
    std::vector<std::vector<float>> lut_layers;

    for (uint32_t bi = 0; bi < body_count && host_mappings_.size() < MAX_RING_HOSTS; ++bi) {
        const auto& def = sys.body(bi).definition;
        if (!ring_host(def)) continue;

        bool wants_texture = std::any_of(def.rings.begin(), def.rings.end(),
                                         [](const ephemeris::RingLayerDefinition& r) { return r.is_textured; });
        std::span<const glm::vec4> raw_tex{};
        if (wants_texture) {
            const auto& source_system = sys.definition().body_system(bi);
            std::string cache_key = source_system + "/" + def.name;
            auto it = cached_raw_textures_.find(cache_key);
            if (it == cached_raw_textures_.end()) {
                std::vector<glm::vec4> loaded_tex;
                if (auto path = ephemeris::find_ring_texture_path(source_system, def.name)) {
                    if (auto decoded = load_ring_png_1d(*path, RING_ATLAS_WIDTH)) {
                        loaded_tex = std::move(*decoded);
                    }
                }
                it = cached_raw_textures_.emplace(std::move(cache_key), std::move(loaded_tex)).first;
            }
            raw_tex = it->second;
        }

        CachedHostMapping mapping{};
        mapping.body_index = bi;
        mapping.unified_row = static_cast<uint32_t>(host_mappings_.size());
        mapping.min_r_ratio = def.rings.front().inner;
        mapping.max_r_ratio = def.rings.front().outer;

        uint32_t seg_limit = std::min<uint32_t>(static_cast<uint32_t>(def.rings.size()), MAX_RING_SEGMENTS);
        std::vector<std::vector<glm::vec4>> seg_profiles;
        std::vector<RingSegmentProfileInput> seg_inputs;
        seg_profiles.reserve(seg_limit);
        seg_inputs.reserve(seg_limit);

        for (uint32_t s = 0; s < seg_limit && next_seg_row < RING_ATLAS_ROWS; ++s) {
            const auto& layer = def.rings[s];
            mapping.min_r_ratio = std::min(mapping.min_r_ratio, layer.inner);
            mapping.max_r_ratio = std::max(mapping.max_r_ratio, layer.outer);

            uint32_t row_idx = next_seg_row++;
            mapping.segment_rows.push_back(row_idx);

            auto prof = generate_ring_segment_profile(layer, raw_tex, RING_ATLAS_WIDTH);
            atlas_row_indices.push_back(row_idx);
            atlas_rows.insert(atlas_rows.end(), prof.begin(), prof.end());
            seg_profiles.push_back(std::move(prof));
        }

        for (size_t s = 0; s < seg_profiles.size(); ++s) {
            const auto& layer = def.rings[s];
            seg_inputs.push_back({layer.inner, layer.outer, layer.opacity, seg_profiles[s]});
        }

        mapping.draw_ranges = build_ring_draw_ranges(std::span(def.rings.data(), mapping.segment_rows.size()),
                                                    mapping.min_r_ratio, mapping.max_r_ratio, def.radius_au());
        if (def.atmosphere.enabled)
            mapping.atmosphere_materials = atmosphere_ring_materials(std::span(def.rings.data(),seg_profiles.size()),seg_profiles);
        auto unified = bake_unified_shadow_profile(seg_inputs, mapping.min_r_ratio, mapping.max_r_ratio, RING_ATLAS_WIDTH);
        atlas_row_indices.push_back(mapping.unified_row);
        atlas_rows.insert(atlas_rows.end(), unified.begin(), unified.end());

        lut_layers.push_back(bake_ring_shadow_lut_layer(unified, RING_ATLAS_WIDTH, RING_SHADOW_LUT_ANGLES));
        host_mappings_.push_back(std::move(mapping));
    }

    if (!atlas_row_indices.empty()) {
        VkImageSubresourceRange atlas_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, RING_PROFILE_MIP_LEVELS, 0, RING_ATLAS_ROWS};
        ctx.transition_image_range(cmd, gradient_atlas_, atlas_range,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        for (size_t i = 0; i < atlas_row_indices.size(); ++i) {
            uint32_t row = atlas_row_indices[i];
            std::span<const glm::vec4> row_span(atlas_rows.data() + i * RING_ATLAS_WIDTH, RING_ATLAS_WIDTH);
            auto mips = build_ring_profile_mips(row_span);
            for (uint32_t mip = 0; mip < mips.size(); ++mip) {
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, row, 1};
                copy.imageExtent = {static_cast<uint32_t>(mips[mip].size()), 1, 1};
                staging_.image(cmd, ctx, gradient_atlas_, copy, std::as_bytes(std::span(mips[mip])));
            }
        }

        ctx.transition_image_range(cmd, gradient_atlas_, atlas_range,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    if (!lut_layers.empty()) {
        VkImageSubresourceRange lut_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, static_cast<uint32_t>(lut_layers.size())};
        ctx.transition_image_range(cmd, shadow_lut_, lut_range,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        for (size_t layer_idx = 0; layer_idx < lut_layers.size(); ++layer_idx) {
            std::span<const float> lut_span(lut_layers[layer_idx]);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, static_cast<uint32_t>(layer_idx), 1};
            copy.imageExtent = {RING_SHADOW_LUT_WIDTH, RING_SHADOW_LUT_ANGLES, 1};
            staging_.image(cmd, ctx, shadow_lut_, copy, std::as_bytes(lut_span));
        }

        ctx.transition_image_range(cmd, shadow_lut_, lut_range,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    ringshine_profile_serial_++;
    cached_hosts_ = host_keys_;
    force_rebuild_ = false;
}

void RingRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                           const ephemeris::SystemSnapshot& sys, uint32_t body_count, int refraction_body) {
    refraction_body_ = refraction_body;
    host_keys_.clear();
    for (uint32_t bi = 0; bi < body_count && host_keys_.size() < MAX_RING_HOSTS; ++bi) {
        auto body = sys.body(bi);
        if (ring_host(body.definition)) host_keys_.push_back({bi, body.revisions.identity, body.revisions.definition});
    }
    if (force_rebuild_ || host_keys_ != cached_hosts_) rebuild_profiles(cmd, ctx, sys, body_count);

    hosts_gpu_.clear();
    host_draw_ranges_.clear();
    RenderPosition eye_pos = cam.eye_pos_render();

    for (const auto& mapping : host_mappings_) {
        if (mapping.body_index >= body_count || mapping.segment_rows.empty()) continue;
        auto body = sys.body(mapping.body_index);
        const auto& def = body.definition;
        double r_eq_au = def.radius_au();
        if (r_eq_au <= 0.0) continue;

        glm::dvec3 pole_d = glm::length(body.render.pole) > 1e-9
            ? glm::normalize(body.render.pole)
            : glm::dvec3(0.0, 1.0, 0.0);
        glm::dvec3 ref_d = (std::abs(pole_d.z) < 0.99) ? glm::dvec3(0.0, 0.0, 1.0) : glm::dvec3(1.0, 0.0, 0.0);
        glm::dvec3 tangent_d = glm::normalize(glm::cross(pole_d, ref_d));
        glm::dvec3 bitangent_d = glm::normalize(glm::cross(pole_d, tangent_d));

        glm::vec3 center_eye_rel = glm::vec3(camera_relative(body.render.position, eye_pos));
        float min_r_au = static_cast<float>(mapping.min_r_ratio * r_eq_au);
        float max_r_au = static_cast<float>(mapping.max_r_ratio * r_eq_au);

        RingHostGPU host{};
        host.center_min_r = glm::vec4(center_eye_rel, min_r_au);
        host.pole_max_r = glm::vec4(glm::vec3(pole_d), max_r_au);
        host.tangent_radius = glm::vec4(glm::vec3(tangent_d), static_cast<float>(r_eq_au));
        host.bitangent_oblateness = glm::vec4(glm::vec3(bitangent_d), static_cast<float>(std::clamp(def.oblateness, 0.0, 0.95)));

        uint32_t seg_count = static_cast<uint32_t>(mapping.segment_rows.size());
        for (uint32_t s = 0; s < seg_count; ++s) {
            const auto& layer = def.rings[s];
            RingSegmentGPU seg{};
            seg.bounds_opacity_scatter = glm::vec4(
                static_cast<float>(layer.inner * r_eq_au),
                static_cast<float>(layer.outer * r_eq_au),
                layer.opacity,
                layer.scatter
            );
            seg.phase_row = glm::vec4(
                layer.asymmetry,
                layer.backscatter,
                layer.unlit_factor,
                static_cast<float>(mapping.segment_rows[s])
            );
            seg.color_flags = glm::vec4(layer.color_rgb, layer.is_textured ? 1.0f : 0.0f);
            host.segments[s] = seg;
        }

        host.counts_indices = glm::uvec4(mapping.body_index, mapping.unified_row, seg_count, 0u);
        hosts_gpu_.push_back(host);
        host_draw_ranges_.push_back(mapping.draw_ranges);
    }

    active_host_count_ = static_cast<uint32_t>(hosts_gpu_.size());
    uint32_t f_idx = ctx.frame_index();
    if (!hosts_gpu_.empty()) {
        std::memcpy(host_ssbos_[f_idx].mapped, hosts_gpu_.data(), hosts_gpu_.size() * sizeof(RingHostGPU));
        ctx.flush_buffer(host_ssbos_[f_idx], 0, hosts_gpu_.size() * sizeof(RingHostGPU));
    }
}

void RingRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const RenderSettings& settings,
                          std::span<const rhi::PushDescriptorWrite> scene_writes, std::span<const glm::vec4> stars,
                          std::span<const rhi::PushDescriptorWrite> transparency_writes) {
    if (!settings.show_rings || active_host_count_ == 0) return;

    auto timer = ctx.begin_gpu_timer(cmd, "rings");
    uint32_t f_idx = ctx.frame_index();
    std::vector<rhi::PushDescriptorWrite> writes(scene_writes.begin(), scene_writes.end());
    writes.push_back(rhi::PushDescriptorWrite::storage_buffer(7, host_ssbos_[f_idx], 0, MAX_RING_HOSTS * sizeof(RingHostGPU)));
    writes.push_back(rhi::PushDescriptorWrite::sampled_image(8, gradient_atlas_, ctx.sampler_linear_clamp()));
    writes.push_back(rhi::PushDescriptorWrite::sampled_image(9, shadow_lut_, ctx.sampler_linear_clamp()));
    writes.insert(writes.end(), transparency_writes.begin(), transparency_writes.end());
    auto& pipelines = transparency_writes.empty() ? pipelines_ : transparency_pipelines_;

    uint32_t bound_pipeline = 7;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[bound_pipeline].handle);
    ctx.push_descriptors(cmd, pipelines[bound_pipeline], writes);

    std::vector<uint32_t> order(active_host_count_);
    for (uint32_t i = 0; i < active_host_count_; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        float da = glm::dot(glm::vec3(hosts_gpu_[a].center_min_r), glm::vec3(hosts_gpu_[a].center_min_r));
        float db = glm::dot(glm::vec3(hosts_gpu_[b].center_min_r), glm::vec3(hosts_gpu_[b].center_min_r));
        return da > db;
    });

    for (uint32_t host_idx : order) {
        struct Push { uint32_t host_index, packed_segments, segment_count, filter_profiles; };
        auto draw = [&](uint32_t first, uint32_t count, uint32_t packed, uint32_t segments, uint32_t pipeline_index) {
            if (bound_pipeline != pipeline_index) {
                bound_pipeline = pipeline_index;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[bound_pipeline].handle);
            }
            ctx.push_constants(cmd, pipelines[bound_pipeline], Push{host_idx, packed, segments, settings.ring_reference_base_level ? 0u : 1u});
            vkCmdDrawIndexed(cmd, count, 1, first, 0, 0);
        };
        const auto& host = hosts_gpu_[host_idx];
        bool refracted = refraction_body_ >= 0;
        const auto& mesh = mesh_;
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices.handle, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indices.handle, 0, VK_INDEX_TYPE_UINT32);
        auto classification = classify_ring_draw(glm::vec3(host.center_min_r), glm::vec3(host.pole_max_r), host.pole_max_r.w, stars);
        if (stars.size() != 1 && classification.lighting != RingLightingMode::FiniteSource)
            classification.lighting = RingLightingMode::Point;
        uint32_t pipeline_base = settings.ring_reference_finite_source ? 4u : uint32_t(classification.lighting) * 4u;
        if (refracted || settings.ring_reference_full_annulus || !classification.radial_ranges) {
            uint32_t packed = 0;
            for (uint32_t s = 0; s < host.counts_indices.z; ++s) packed |= s << (s * 3u);
            draw(0, mesh.index_count, packed, host.counts_indices.z, pipeline_base + 3u);
        } else {
            for (const auto& range : host_draw_ranges_[host_idx])
                draw(range.first_index, range.index_count, range.packed_segments, range.segment_count,
                     pipeline_base + std::bit_width(range.segment_count - 1u));
        }
    }
    ctx.end_gpu_timer(cmd, timer);
}


namespace {

struct RingshinePush {
    glm::uvec4 tile;
    glm::vec4 host_params;
    glm::vec4 solar;
    glm::vec4 disk_offset;
    glm::vec4 disk_weight;
};
static_assert(sizeof(RingshinePush) == 80);

double quantize_elevation(double elevation) {
    return std::stod(std::format("{:.5f}", elevation));
}

double quantize_sin_radius(double value) {
    return std::stod(std::format("{:.6g}", value));
}
}

void RingRenderer::bake_ringshine_tile(VkCommandBuffer cmd, rhi::VkContext& ctx, const RingshineJob& job) {
    const RingHostGPU& host = hosts_gpu_[job.host_idx];
    RingshinePush push{};
    push.tile = glm::uvec4(job.host_idx, job.star_col, ringshine_radial_ranges_[job.host_idx]);
    push.host_params = glm::vec4(
        static_cast<float>(job.key.inner),
        static_cast<float>(job.key.outer),
        1.0f,
        static_cast<float>(job.key.flattening));
    push.solar = glm::vec4(
        static_cast<float>(job.key.elevation),
        static_cast<float>(job.key.sin_radius),
        static_cast<float>(host.counts_indices.z),
        static_cast<float>(job.key.bands));
    ringshine_source_samples(job.key.elevation, job.key.sin_radius, push.disk_offset, push.disk_weight);

    uint32_t f_idx = ctx.frame_index();
    auto& atlas = ringshine_atlases_[f_idx];
    ctx.transition_image(cmd, atlas, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    rhi::RenderingAttachment att = rhi::RenderingAttachment::load_color(atlas);
    ctx.begin_rendering(cmd, VkExtent2D{RINGSHINE_TILE_WIDTH * 16, RINGSHINE_TILE_HEIGHT * MAX_RING_HOSTS}, {&att, 1}, nullptr);
    VkViewport vp{};
    vp.width = static_cast<float>(RINGSHINE_TILE_WIDTH);
    vp.height = static_cast<float>(RINGSHINE_TILE_HEIGHT);
    vp.x = static_cast<float>(job.star_col * RINGSHINE_TILE_WIDTH);
    vp.y = static_cast<float>(job.host_idx * RINGSHINE_TILE_HEIGHT);
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{{static_cast<int32_t>(vp.x), static_cast<int32_t>(vp.y)},
                     {RINGSHINE_TILE_WIDTH, RINGSHINE_TILE_HEIGHT}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ringshine_pipeline_.handle);
    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::storage_buffer(7, host_ssbos_[f_idx], 0, MAX_RING_HOSTS * sizeof(RingHostGPU)),
        rhi::PushDescriptorWrite::sampled_image(8, gradient_atlas_, ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(10, ringshine_radial_buffers_[f_idx]),
    };
    ctx.push_descriptors(cmd, ringshine_pipeline_, writes);
    ctx.push_constants(cmd, ringshine_pipeline_, push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    ctx.end_rendering(cmd);

    ctx.transition_image(cmd, atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    ringshine_cache_[f_idx][{job.host_idx, job.star_col}] = job.key;
    ringshine_bakes_++;
}

void RingRenderer::update_ringshine(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                                    const ephemeris::SystemSnapshot& sys, const RenderSettings& settings,
                                    uint32_t body_count, VkExtent2D extent) {
    ringshine_frame_bakes_ = 0;
    if (settings.direct_sunlight_only || !settings.ringshine_enabled || !settings.show_rings || active_host_count_ == 0) return;
    uint32_t f_idx = ctx.frame_index();
    auto& cache = ringshine_cache_[f_idx];
    uint32_t bands = std::clamp(settings.ringshine_band_count, 4u, 1024u);
    if (ringshine_radial_profile_serial_ != ringshine_profile_serial_ || ringshine_radial_bands_ != bands) {
        ringshine_radial_samples_.clear();
        ringshine_radial_ranges_.clear();
        ringshine_sample_counts_.clear();
        for (const auto& host : hosts_gpu_) {
            auto layers = std::span(sys.body(host.counts_indices.x).definition.rings).first(host.counts_indices.z);
            auto samples = build_ringshine_radial_samples(layers, bands);
            ringshine_radial_ranges_.emplace_back(ringshine_radial_samples_.size(), samples.size());
            ringshine_sample_counts_.push_back(static_cast<uint32_t>(samples.size()));
            ringshine_radial_samples_.insert(ringshine_radial_samples_.end(), samples.begin(), samples.end());
        }
        ringshine_radial_profile_serial_ = ringshine_profile_serial_;
        ringshine_radial_bands_ = bands;
        ++ringshine_radial_serial_;
    }
    if (ringshine_radial_upload_serials_[f_idx] != ringshine_radial_serial_) {
        size_t bytes = ringshine_radial_samples_.size() * sizeof(glm::vec2);
        if (bytes > 0) {
            std::memcpy(ringshine_radial_buffers_[f_idx].mapped, ringshine_radial_samples_.data(), bytes);
            ctx.flush_buffer(ringshine_radial_buffers_[f_idx], 0, bytes);
        }
        ringshine_radial_upload_serials_[f_idx] = ringshine_radial_serial_;
    }

    std::vector<glm::dvec3> star_pos;
    std::vector<double> star_radius;
    std::vector<uint32_t> star_body;
    bool has_real_star = false;
    for (uint32_t bi = 0; bi < body_count && star_pos.size() < 16; ++bi) {
        const auto& def = sys.body(bi).definition;
        if (!def.is_star) continue;
        has_real_star = true;
        star_pos.push_back(sys.body(bi).render.position.value);
        star_radius.push_back(def.opaque_radius_au());
        star_body.push_back(bi);
    }
    if (!has_real_star) return;

    double tan_y = std::tan(cam.fov_deg*DEG_TO_RAD*0.5);
    double tan_x = tan_y*double(extent.width)/std::max(1u, extent.height);
    double focal = double(extent.height)/(2.0*tan_y);
    glm::dvec3 eye = cam.eye_pos_render().value;
    std::vector<RingshineJob> jobs;
    for (uint32_t h = 0; h < active_host_count_; ++h) {
        const RingHostGPU& host = hosts_gpu_[h];
        uint32_t bi = host.counts_indices.x;
        if (bi >= body_count) continue;
        glm::dvec3 center = sys.body(bi).render.position.value;
        const auto body_view = sys.body(bi);
        const auto& body = body_view.definition;
        double radius = std::max(body.radius_au(), body_view.opaque_radius_au()) +
            std::max(0.0, body.height_range_km.y)*KM_TO_AU;
        glm::dvec3 to_body = center-eye;
        double depth = glm::dot(to_body, cam.forward_dir());
        if (depth+radius <= 0.0 ||
            std::abs(glm::dot(to_body, cam.right_dir())) > depth*tan_x+radius*std::sqrt(1.0+tan_x*tan_x) ||
            std::abs(glm::dot(to_body, cam.up_dir())) > depth*tan_y+radius*std::sqrt(1.0+tan_y*tan_y)) continue;
        if (depth > radius && radius/(depth-radius)*focal <= 1.5) continue;
        glm::dvec3 pole = glm::length(sys.body(bi).render.pole) > 1e-9
            ? glm::normalize(sys.body(bi).render.pole) : glm::dvec3(0.0, 1.0, 0.0);
        const auto& mapping = host_mappings_[h];
        for (uint32_t s = 0; s < star_pos.size(); ++s) {
            if (!sys.illuminates(star_body[s], bi)) continue;
            glm::dvec3 dir = star_pos[s] - center;
            double dist = glm::length(dir);
            if (dist <= 1e-12) continue;
            double elevation = std::clamp(glm::dot(dir, pole) / dist, -1.0, 1.0);
            RingshineTileKey key{};
            key.profile_serial = ringshine_profile_serial_;
            key.inner = mapping.min_r_ratio;
            key.outer = mapping.max_r_ratio;
            key.flattening = std::clamp(sys.body(bi).definition.oblateness, 0.0, 0.95);
            key.bands = std::clamp(settings.ringshine_band_count, 4u, 1024u);
            key.elevation = quantize_elevation(elevation);
            key.sin_radius = quantize_sin_radius(std::clamp(star_radius[s] / dist, 0.0, 0.999));

            auto it = cache.find({h, s});
            if (it == cache.end() || !(it->second == key)) {
                jobs.push_back({h, s, key});
            }
        }
    }
    for (auto it = cache.begin(); it != cache.end();) {
        if (it->first.first < active_host_count_ && it->first.second < star_pos.size()) ++it;
        else it = cache.erase(it);
    }
    for (const auto& job : jobs) {
        bake_ringshine_tile(cmd, ctx, job);
    }
    ringshine_frame_bakes_ = static_cast<uint32_t>(jobs.size());
}

PlanetshineGPU RingRenderer::atmosphere_sources(const ephemeris::SystemSnapshot& system,
    const RenderSettings& settings, uint32_t body_index) const {
    PlanetshineGPU result{};
    if (!settings.ringshine_enabled || !settings.show_rings || settings.direct_sunlight_only) return result;
    auto found = std::find_if(host_mappings_.begin(),host_mappings_.end(),[&](const auto& h){return h.body_index == body_index;});
    if (found == host_mappings_.end()) return result;
    const auto& host = *found;
    const auto body = system.body(body_index);
    auto pole = glm::normalize(body.render.pole);
    std::array<glm::vec4,16> stars{}, weights{};
    glm::vec3 total(0);
    uint32_t count = 0;
    for (uint32_t i = 0; i < system.size() && count < 16; ++i) {
        auto star = system.body(i);
        if (!star.definition.is_star || !system.illuminates(i, body_index)) continue;
        auto delta = star.render.position.value-body.render.position.value;
        double distance = glm::length(delta);
        if (distance <= 1e-12) continue;
        double elevation = quantize_elevation(glm::clamp(glm::dot(delta,pole)/distance,-1.0,1.0));
        double extent = quantize_sin_radius(std::clamp(star.opaque_radius_au()/distance,0.0,.999));
        stars[count] = {float(elevation),float(extent),0,0};
        float scale = settings.hdr_enabled ? float(std::max(star.definition.lum_eq,0.0)/(distance*distance)) : 1.0f;
        weights[count] = {glm::max(star.definition.color_rgb,glm::vec3(0))*scale,0};
        total += glm::vec3(weights[count++]);
    }
    if (!glm::any(glm::greaterThan(total,glm::vec3(0)))) return result;
    for (uint32_t i = 0; i < count; ++i) for (int c = 0; c < 3; ++c)
        weights[i][c] = total[c] > 0 ? weights[i][c]/total[c] : 0;
    bool dirty = !host.atmosphere_valid || host.atmosphere_stars != stars;
    for (uint32_t i = 0; i < count; ++i)
        dirty |= glm::any(glm::greaterThan(glm::abs(weights[i]-host.atmosphere_weights[i]),glm::vec4(2e-6f)));
    if (dirty) {
        glm::dvec3 faces[2]{};
        double full_area = PI*(host.max_r_ratio*host.max_r_ratio-host.min_r_ratio*host.min_r_ratio);
        for (uint32_t s = 0; s < count; ++s) {
            glm::vec4 offsets, fractions;
            int samples = ringshine_source_samples(stars[s].x,stars[s].y,offsets,fractions);
            if (samples == 1 && stars[s].y >= .01f) {
                samples = 4; offsets = {-.75f,-.25f,.25f,.75f}; fractions = glm::vec4(.25f);
            }
            for (int q = 0; q < samples; ++q) {
                double sun = std::clamp(double(stars[s].x)+double(offsets[q]*stars[s].y),-1.0,1.0);
                if (std::abs(sun) < 1e-8) continue;
                double horizontal = std::sqrt(std::max(1-sun*sun,0.0));
                for (const auto& material : host.atmosphere_materials) {
                    if (material.radius_area_tau.z <= 0) continue;
                    double radius = material.radius_area_tau.x;
                    double eta = 1/std::pow(std::max(1-body.definition.oblateness,.05),2);
                    double shadow = horizontal > 1e-8 ? std::acos(std::clamp(std::sqrt((1-1/(radius*radius))*(horizontal*horizontal+sun*sun*eta))/horizontal,0.0,1.0)) : 0;
                    double area = double(material.radius_area_tau.y)/full_area*(1-shadow/PI)*fractions[q];
                    double mv = .5/std::sqrt(radius*radius+.25);
                    for (int side = 0; side < 2; ++side) for (int angular = 0; angular < 4; ++angular) {
                        double sign = side == 0 ? 1 : -1;
                        double angle = 2*PI*(angular+.5)/4;
                        double phase = (radius*horizontal*std::cos(angle)-.5*sign*sun)/std::sqrt(radius*radius+.25);
                        auto response = ring_radiance_rgb(material.radius_area_tau.z,float(mv),float(sun),float(phase),
                            material.properties,glm::vec3(material.color_textured),material.color_textured.w > .5f,sign*sun >= 0);
                        faces[side] += glm::dvec3(response)*glm::dvec3(weights[s])*(area*.25/PI);
                    }
                }
            }
        }
        glm::dvec3 sum = faces[0]+faces[1];
        uint32_t packed = 0;
        for (int c = 0; c < 3; ++c) {
            double contrast = sum[c] > 0 ? (faces[0][c]-faces[1][c])/sum[c] : 0;
            packed |= uint32_t(std::lround(std::clamp(contrast,-1.0,1.0)*127)+127) << (c*8);
        }
        host.atmosphere_sources = {};
        host.atmosphere_sources.irradiance[0] = {glm::vec3(sum*.5),float(packed)};
        host.atmosphere_stars = stars; host.atmosphere_weights = weights; host.atmosphere_valid = true;
        ++atmosphere_source_bakes_;
    }
    result = host.atmosphere_sources;
    result.direction[0] = {glm::vec3(pole)*float(host.max_r_ratio),-float(host.min_r_ratio)};
    result.irradiance[0] = {glm::vec3(result.irradiance[0])*total,result.irradiance[0].w};
    return result;
}

} // namespace sf::rendering
