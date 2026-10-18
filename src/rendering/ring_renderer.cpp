#include "rendering/ring_renderer.hpp"
#include "core/constants.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace sf::rendering {

void RingRenderer::init(rhi::VkContext& ctx) {
    build_annulus_mesh(ctx, RING_ANGULAR_SEGMENTS, RING_RADIAL_SUBDIVISIONS);

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

    hosts_gpu_.reserve(MAX_RING_HOSTS);
    host_draw_ranges_.reserve(MAX_RING_HOSTS);
    force_rebuild_ = true;
}

void RingRenderer::shutdown(rhi::VkContext& ctx) {
    cached_definition_.reset();
    staging_.shutdown(ctx);
    for (auto& pipeline : pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    for (auto& pipeline : transparency_pipelines_) rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    ctx.destroy_image(shadow_lut_);
    ctx.destroy_image(gradient_atlas_);
    for (auto& buf : host_ssbos_) {
        ctx.destroy_buffer(buf);
    }
    ctx.destroy_buffer(ibo_);
    ctx.destroy_buffer(vbo_);
}

void RingRenderer::build_annulus_mesh(rhi::VkContext& ctx, uint32_t angular_segments, uint32_t radial_subdivisions) {
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

    index_count_ = static_cast<uint32_t>(indices.size());
    vbo_ = ctx.create_device_buffer(verts.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, verts.data());
    ibo_ = ctx.create_device_buffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data());
}

void RingRenderer::rebuild_profiles(VkCommandBuffer cmd, rhi::VkContext& ctx, const ephemeris::SystemSnapshot& sys, uint32_t body_count) {
    host_mappings_.clear();
    uint32_t next_seg_row = MAX_RING_HOSTS;

    std::vector<glm::vec4> atlas_rows;
    std::vector<uint32_t> atlas_row_indices;
    std::vector<std::vector<float>> lut_layers;

    for (uint32_t bi = 0; bi < body_count && host_mappings_.size() < MAX_RING_HOSTS; ++bi) {
        const auto& def = sys.body(bi).definition;
        if (def.rings.empty() || def.is_star || def.is_black_hole || def.is_spacecraft) continue;

        bool wants_texture = std::any_of(def.rings.begin(), def.rings.end(),
                                         [](const ephemeris::RingLayerDefinition& r) { return r.is_textured; });
        std::span<const glm::vec4> raw_tex{};
        if (wants_texture) {
            std::string cache_key = std::string(sys.name()) + "/" + def.name;
            auto it = cached_raw_textures_.find(cache_key);
            if (it == cached_raw_textures_.end()) {
                std::vector<glm::vec4> loaded_tex;
                if (auto path = ephemeris::find_ring_texture_path(sys.name(), def.name)) {
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

    cached_definition_ = sys.share_definition();
    cached_generation_ = sys.generation();
    cached_definition_revision_ = sys.definition_revision();
    force_rebuild_ = false;
}

void RingRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                           const ephemeris::SystemSnapshot& sys, uint32_t body_count) {
    if (force_rebuild_ || cached_definition_.get() != &sys.definition() ||
        cached_generation_ != sys.generation() || cached_definition_revision_ != sys.definition_revision()) {
        rebuild_profiles(cmd, ctx, sys, body_count);
    }

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

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbo_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, ibo_.handle, 0, VK_INDEX_TYPE_UINT32);

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
        auto classification = classify_ring_draw(glm::vec3(host.center_min_r), glm::vec3(host.pole_max_r), host.pole_max_r.w, stars);
        if (stars.size() != 1 && classification.lighting != RingLightingMode::FiniteSource)
            classification.lighting = RingLightingMode::Point;
        uint32_t pipeline_base = settings.ring_reference_finite_source ? 4u : uint32_t(classification.lighting) * 4u;
        if (settings.ring_reference_full_annulus || !classification.radial_ranges) {
            uint32_t packed = 0;
            for (uint32_t s = 0; s < host.counts_indices.z; ++s) packed |= s << (s * 3u);
            draw(0, index_count_, packed, host.counts_indices.z, pipeline_base + 3u);
        } else {
            for (const auto& range : host_draw_ranges_[host_idx])
                draw(range.first_index, range.index_count, range.packed_segments, range.segment_count,
                     pipeline_base + std::bit_width(range.segment_count - 1u));
        }
    }
    ctx.end_gpu_timer(cmd, timer);
}

} // namespace sf::rendering
