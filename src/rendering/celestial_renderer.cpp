#include "rendering/celestial_renderer.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "core/constants.hpp"
#include "core/path_utils.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sf::rendering {

static_assert(sizeof(BodyInstanceGPU) == 96);
static_assert(terrain::MAX_PATCHES >= MAX_CELESTIAL_BODIES*6);

namespace {

struct SphereVertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

} // namespace

void CelestialRenderer::init(rhi::VkContext& ctx, bool load_surface_assets) {
    build_sphere_mesh(ctx, 64, 32);
    textures_.init(ctx);
    normals_.init(ctx, true);
    rings_.init(ctx);
    atmospheres_.init(ctx);
    transparency_.init(ctx);

    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        scene_ubos_[i] = ctx.create_host_buffer(sizeof(SceneCelestialUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        instance_ssbos_[i] = ctx.create_host_buffer(
            MAX_CELESTIAL_BODIES * sizeof(BodyInstanceGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        eclipse_ssbos_[i] = ctx.create_host_buffer(4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1},
    };

    rhi::GraphicsPipelineDesc sphere_desc{};
    sphere_desc.vert_path = "celestial/sphere.vert";
    sphere_desc.frag_path = "celestial/sphere.frag";
    sphere_desc.bindings.assign(std::begin(bindings), std::end(bindings));
    sphere_desc.vertex_bindings = {
        {0, sizeof(SphereVertex), VK_VERTEX_INPUT_RATE_VERTEX},
    };
    sphere_desc.vertex_attributes = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, normal)},
    };
    sphere_desc.color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
    sphere_desc.depth_format = VK_FORMAT_D32_SFLOAT;
    sphere_desc.depth_test = true;
    sphere_desc.depth_write = true;
    sphere_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    sphere_desc.cull_mode = VK_CULL_MODE_BACK_BIT;
    sphere_desc.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    sphere_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, sphere_desc);

    rhi::GraphicsPipelineDesc point_desc{};
    point_desc.vert_path = "celestial/point_celestial.vert";
    point_desc.frag_path = "celestial/point_celestial.frag";
    point_desc.bindings.assign(std::begin(bindings), std::end(bindings));
    point_desc.color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
    point_desc.depth_format = VK_FORMAT_D32_SFLOAT;
    point_desc.depth_test = false;
    point_desc.depth_write = false;
    point_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    point_desc.additive_blend = true;
    point_desc.cull_mode = VK_CULL_MODE_NONE;

    point_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, point_desc);
    terrain_.init(ctx, load_surface_assets);
    for (const auto& [body, source] : terrain_.height_sources()) normals_.set_height_source(body, source);
    if (load_surface_assets) {
        for (const auto& [body, path] : textures_.load_catalog(paths::external_path("data", "surfaces.json")))
            set_height_source(body, terrain::HeightSource::open(path));
    }
}

void CelestialRenderer::shutdown(rhi::VkContext& ctx) {
    transparency_.shutdown(ctx);
    atmospheres_.shutdown(ctx);
    rings_.shutdown(ctx);
    terrain_.shutdown(ctx);
    textures_.shutdown(ctx);
    normals_.shutdown(ctx);
    rhi::ShaderCache::destroy_pipeline(ctx, point_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, sphere_pipeline_);
    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        ctx.destroy_buffer(instance_ssbos_[i]);
        ctx.destroy_buffer(eclipse_ssbos_[i]);
        ctx.destroy_buffer(scene_ubos_[i]);
    }
    ctx.destroy_buffer(sphere_ibo_);
    ctx.destroy_buffer(sphere_vbo_);
}

void CelestialRenderer::build_sphere_mesh(rhi::VkContext& ctx, uint32_t sectors, uint32_t stacks) {
    std::vector<SphereVertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve((stacks + 1) * (sectors + 1));
    indices.reserve(stacks * sectors * 6);

    for (uint32_t i = 0; i <= stacks; ++i) {
        double stack_angle = PI * 0.5 - static_cast<double>(i) * (PI / static_cast<double>(stacks));
        double xy = std::cos(stack_angle);
        double y = std::sin(stack_angle);

        for (uint32_t j = 0; j <= sectors; ++j) {
            double sector_angle = static_cast<double>(j) * (TWO_PI / static_cast<double>(sectors));
            double x = xy * std::cos(sector_angle);
            double z = xy * std::sin(sector_angle);
            glm::vec3 n{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
            vertices.push_back({n, n});
        }
    }

    for (uint32_t i = 0; i < stacks; ++i) {
        uint32_t k1 = i * (sectors + 1);
        uint32_t k2 = k1 + sectors + 1;
        for (uint32_t j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                indices.push_back(k1);
                indices.push_back(k1 + 1);
                indices.push_back(k2);
            }
            if (i != (stacks - 1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2 + 1);
                indices.push_back(k2);
            }
        }
    }

    sphere_index_count_ = static_cast<uint32_t>(indices.size());
    sphere_vbo_ = ctx.create_device_buffer(vertices.size() * sizeof(SphereVertex),
                                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data());
    sphere_ibo_ = ctx.create_device_buffer(indices.size() * sizeof(uint32_t),
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data());
}

void CelestialRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                                const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent) {
    uint32_t count = std::min(static_cast<uint32_t>(sys.size()), MAX_CELESTIAL_BODIES);
    textures_.prepare(cmd, ctx, cam, sys, extent, settings.surface_textures_enabled);
    normals_.prepare(cmd, ctx, cam, sys, extent, settings.terrain_imported_enabled && settings.terrain_normal_maps_enabled);
    rings_.prepare(cmd, ctx, cam, sys, count);
    uint32_t ringshine_timer = ctx.begin_gpu_timer(cmd, "ringshine_bake");
    rings_.update_ringshine(cmd, ctx, cam, sys, settings, count, extent);
    ctx.end_gpu_timer(cmd, ringshine_timer);
    terrain_.prepare(ctx, cam, sys, settings, extent, count);
    terrain_.bake(cmd, ctx);
    atmosphere_minima_.resize(count);
    for (uint32_t i = 0; i < count; ++i)
        atmosphere_minima_[i] = static_cast<float>(terrain_.minimum_height_km(sys.body(i).definition, settings));
    atmospheres_.prepare(cmd, ctx, cam, sys, settings, extent, atmosphere_minima_, rings_.hosts(), &rings_.shadow_lut(), rings_.shadow_revision());
    transparency_.prepare(cmd, ctx, extent, !settings.transparency_reference_legacy && !sys.empty() &&
        ((settings.show_rings && rings_.host_count() > 0) || (settings.show_orbits && sys.size() > 1)));
}

void CelestialRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                               const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent) {
    ring_eclipse_candidate_counts_.clear();
    surface_ring_candidate_masks_.clear();
    uint32_t count = std::min(static_cast<uint32_t>(sys.size()), MAX_CELESTIAL_BODIES);
    if (count == 0) {
        return;
    }

    uint32_t f_idx = ctx.frame_index();
    float fb_w = static_cast<float>(std::max(1u, extent.width));
    float fb_h = static_cast<float>(std::max(1u, extent.height));
    float aspect = fb_w / fb_h;
    float fov_rad = static_cast<float>(cam.fov_deg * DEG_TO_RAD);
    float fov_factor = 1.0f / (2.0f * std::tan(0.5f * fov_rad));
    float z_near = cam.compute_near_plane_au();

    glm::dvec3 eye_pos = cam.eye_pos_render().value;
    glm::mat4 view = cam.view_matrix_eye_relative();
    glm::mat4 proj = cam.proj_matrix(aspect, z_near);

    SceneCelestialUBO ubo{};
    ubo.view = view;
    ubo.view_proj = proj * view;
    ubo.projection = proj;
    ubo.screen_params = {fb_w, fb_h, fov_factor, settings.exposure};
    ubo.point_params.x = std::clamp(settings.star_point_size, 1.5f, 6.0f);
    ubo.num_bodies = count;
    ubo.hdr_enabled = settings.hdr_enabled ? 1u : 0u;
    ubo.num_ring_hosts = (settings.show_rings && settings.ring_shadows_enabled) ? rings_.host_count() : 0u;
    ubo.ringshine_params = {settings.ringshine_enabled && settings.show_rings ? 1u : 0u, rings_.host_count(), 0u, 0u};
    body_instances_.resize(count);
    auto* inst_ptr = body_instances_.data();

    for (uint32_t i = 0; i < count; ++i) {
        const auto& b = sys.body(i);
        glm::vec3 rel_eye = glm::vec3(camera_relative(b.render.position, cam.eye_pos_render()));
        float r_au = static_cast<float>(b.opaque_radius_au());

        BodyInstanceGPU inst{};
        inst.pos_radius = glm::vec4(rel_eye, r_au);
        inst.color_is_star = glm::vec4(b.definition.color_rgb, b.definition.is_star ? 1.0f : 0.0f);
        inst.pole_oblateness = glm::vec4(glm::vec3(b.render.pole), static_cast<float>(b.definition.oblateness));
        inst.pole_color_lum = glm::vec4(b.definition.color_pole_rgb, static_cast<float>(b.definition.lum_eq));
        inst.spin_flags = glm::vec4(static_cast<float>(b.render.spin_angle_rad),
                                    static_cast<float>(b.definition.lum_pole),
                                    b.definition.is_black_hole ? 1.0f : 0.0f,
                                    static_cast<float>(i));
        inst.surface_flags.x = terrain_.covers(i) ? 1u : 0u;
        inst_ptr[i] = inst;

        if (b.definition.is_star && ubo.num_stars < MAX_SCENE_STARS) {
            uint32_t s = ubo.num_stars++;
            ubo.stars_pos_radius[s] = glm::vec4(rel_eye, r_au);
            ubo.stars_color_lum[s] = glm::vec4(b.definition.color_rgb, static_cast<float>(std::max(b.definition.lum_eq, 1e-4)));
        }
    }

    if (ubo.num_stars == 0) {
        ubo.num_stars = 1;
        ubo.stars_pos_radius[0] = glm::vec4(glm::vec3(-eye_pos), static_cast<float>(SOLAR_RADII_TO_AU));
        ubo.stars_color_lum[0] = glm::vec4(1.0f, 0.95f, 0.90f, 1.0f);
    }

    uint32_t ring_count = settings.show_rings ? rings_.host_count() : 0u;
    static_assert(MAX_RING_HOSTS < 32);
    surface_ring_candidate_masks_.assign(ubo.num_ring_hosts > 0u && !settings.ring_shadow_reference_all_hosts
                                       ? count * ubo.num_stars : 0u, 0u);
    ring_eclipse_candidate_counts_.assign(ring_count * ubo.num_stars, 0u);
    eclipse_words_.assign((count + ring_count) * ubo.num_stars * 2, 0u);
    for (uint32_t i = 0; i < count + ring_count; ++i) {
        glm::vec4 receiver_bounds;
        if (i < count) {
            const auto& receiver = inst_ptr[i];
            if (receiver.color_is_star.w > 0.5f || receiver.spin_flags.z > 0.5f) continue;
            float distance = std::max(glm::length(glm::vec3(receiver.pos_radius)), 1e-12f);
            float radius_px = receiver.pos_radius.w / distance * fb_h * fov_factor;
            if (radius_px <= 1.49f && receiver.surface_flags.x == 0u) continue;
            receiver_bounds = receiver.pos_radius;
            if (receiver.surface_flags.x != 0u)
                receiver_bounds.w += terrain_.relief_extent_au(i);
            for (uint32_t s = 0; !surface_ring_candidate_masks_.empty() && s < ubo.num_stars; ++s) {
                uint32_t mask = 0u;
                for (uint32_t h = 0; h < ubo.num_ring_hosts; ++h) {
                    const auto& host = rings_.hosts()[h];
                    if (ring_could_shadow_surface(
                            receiver_bounds, ubo.stars_pos_radius[s],
                            {glm::vec3(host.center_min_r), host.pole_max_r.w},
                            {glm::vec3(host.pole_max_r), host.center_min_r.w}, fb_h * fov_factor))
                        mask |= 1u << h;
                }
                surface_ring_candidate_masks_[i * ubo.num_stars + s] = mask;
            }
        } else {
            const auto& host = rings_.hosts()[i - count];
            receiver_bounds = glm::vec4(glm::vec3(host.center_min_r), host.pole_max_r.w);
        }
        for (uint32_t s = 0; s < ubo.num_stars; ++s) {
            uint32_t range = (i * ubo.num_stars + s) * 2;
            uint32_t start = static_cast<uint32_t>(eclipse_words_.size());
            for (uint32_t j = 0; j < count; ++j) {
                if (i == j || inst_ptr[j].color_is_star.w > 0.5f) continue;
                if (i >= count && inst_ptr[j].pos_radius.w <= 0.0f) continue;
                if (eclipse_could_shadow(receiver_bounds, ubo.stars_pos_radius[s], inst_ptr[j].pos_radius))
                    eclipse_words_.push_back(j);
            }
            eclipse_words_[range] = start;
            eclipse_words_[range + 1] = static_cast<uint32_t>(eclipse_words_.size()) - start;
            if (i >= count)
                ring_eclipse_candidate_counts_[(i - count) * ubo.num_stars + s] = eclipse_words_[range + 1];
        }
    }
    ubo.ring_shadow_params = {static_cast<uint32_t>(eclipse_words_.size()),
                              settings.ring_shadow_reference_all_hosts ? 1u : 0u,
                              settings.ring_shadow_single_box ? 1u : 0u, 0u};
    ubo.eclipse_params.x = settings.eclipse_reference_smoothstep ? 1u : 0u;
    eclipse_words_.insert(eclipse_words_.end(), surface_ring_candidate_masks_.begin(), surface_ring_candidate_masks_.end());
    auto& eclipse_buffer = eclipse_ssbos_[f_idx];
    VkDeviceSize eclipse_bytes = eclipse_words_.size() * sizeof(uint32_t);
    if (eclipse_bytes > eclipse_buffer.size) {
        VkDeviceSize capacity = std::max(eclipse_bytes, eclipse_buffer.size * 2);
        ctx.destroy_buffer(eclipse_buffer);
        eclipse_buffer = ctx.create_host_buffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    std::memcpy(eclipse_buffer.mapped, eclipse_words_.data(), eclipse_bytes);
    std::memcpy(instance_ssbos_[f_idx].mapped, body_instances_.data(), count * sizeof(BodyInstanceGPU));
    std::memcpy(scene_ubos_[f_idx].mapped, &ubo, sizeof(SceneCelestialUBO));
    ctx.flush_buffer(eclipse_buffer, 0, eclipse_bytes);
    ctx.flush_buffer(instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU));
    ctx.flush_buffer(scene_ubos_[f_idx], 0, sizeof(SceneCelestialUBO));

    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
        rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
        rhi::PushDescriptorWrite::storage_buffer(2, eclipse_buffer, 0, eclipse_bytes),
        rhi::PushDescriptorWrite::sampled_image(5, textures_.image(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(6, textures_.pages(f_idx)),
        rhi::PushDescriptorWrite::storage_buffer(7, rings_.host_buffer(f_idx), 0, MAX_RING_HOSTS * sizeof(RingHostGPU)),
        rhi::PushDescriptorWrite::sampled_image(9, rings_.shadow_lut(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::sampled_image(10, normals_.image(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(11, normals_.pages(f_idx)),
        rhi::PushDescriptorWrite::sampled_image(12, atmospheres_.surface_lut(), ctx.sampler_linear_clamp()),
        rhi::PushDescriptorWrite::storage_buffer(13, atmospheres_.surface_records(f_idx)),
        rhi::PushDescriptorWrite::sampled_image(16, rings_.ringshine_atlas(f_idx), ctx.sampler_linear_clamp()),
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sphere_pipeline_.handle);
    ctx.push_descriptors(cmd, sphere_pipeline_, writes);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vbo_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, sphere_ibo_.handle, 0, VK_INDEX_TYPE_UINT32);
    for (uint32_t first = 0; first < count;) {
        if (terrain_.covers(first)) { ++first; continue; }
        uint32_t end = first+1;
        while (end < count && !terrain_.covers(end)) ++end;
        vkCmdDrawIndexed(cmd, sphere_index_count_, end-first, 0, 0, first);
        first = end;
    }

    terrain_.render(cmd, ctx, writes);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipeline_.handle);
    ctx.push_descriptors(cmd, point_pipeline_, writes);
    for (uint32_t first = 0; first < count;) {
        if (terrain_.covers(first)) { ++first; continue; }
        uint32_t end = first+1;
        while (end < count && !terrain_.covers(end)) ++end;
        vkCmdDraw(cmd, 6, end-first, 0, first);
        first = end;
    }

    rhi::PushDescriptorWrite ring_base_writes[] = {
        rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
        rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
        rhi::PushDescriptorWrite::storage_buffer(2, eclipse_buffer, 0, eclipse_bytes),
    };
    auto transparency_writes = transparency_.writes();
    rings_.render(cmd, ctx, settings, ring_base_writes, {ubo.stars_pos_radius, ubo.num_stars},
        transparency_.active() ? std::span{transparency_writes} : std::span<const rhi::PushDescriptorWrite>{});
}

} // namespace sf::rendering
