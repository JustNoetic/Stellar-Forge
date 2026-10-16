#include "rendering/celestial_renderer.hpp"
#include "core/constants.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sf::rendering {

namespace {

struct SphereVertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

} // namespace

void CelestialRenderer::init(rhi::VkContext& ctx) {
    build_sphere_mesh(ctx, 64, 32);

    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        scene_ubos_[i] = ctx.create_host_buffer(sizeof(SceneCelestialUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        instance_ssbos_[i] = ctx.create_host_buffer(
            MAX_CELESTIAL_BODIES * sizeof(BodyInstanceGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1},
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
    point_desc.depth_test = true;
    point_desc.depth_write = false;
    point_desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    point_desc.additive_blend = true;
    point_desc.cull_mode = VK_CULL_MODE_NONE;

    point_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, point_desc);
}

void CelestialRenderer::shutdown(rhi::VkContext& ctx) {
    rhi::ShaderCache::destroy_pipeline(ctx, point_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, sphere_pipeline_);
    for (uint32_t i = 0; i < rhi::FRAMES_IN_FLIGHT; ++i) {
        ctx.destroy_buffer(instance_ssbos_[i]);
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

void CelestialRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                               const ephemeris::SystemState& sys, VkExtent2D extent) {
    uint32_t count = std::min(static_cast<uint32_t>(sys.bodies.size()), MAX_CELESTIAL_BODIES);
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

    glm::dvec3 eye_pos = cam.eye_pos_render();
    glm::mat4 view = cam.view_matrix_eye_relative();
    glm::mat4 proj = cam.proj_matrix(aspect, z_near);

    SceneCelestialUBO ubo{};
    ubo.view = view;
    ubo.view_proj = proj * view;
    ubo.screen_params = {fb_w, fb_h, fov_factor, cam.exposure};
    ubo.num_bodies = count;
    ubo.hdr_enabled = cam.hdr_enabled ? 1u : 0u;

    auto* inst_ptr = static_cast<BodyInstanceGPU*>(instance_ssbos_[f_idx].mapped);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& b = sys.bodies[i];
        glm::vec3 rel_eye = glm::vec3(b.pos_render - eye_pos);
        float r_au = static_cast<float>(b.radius_au);

        BodyInstanceGPU inst{};
        inst.pos_radius = glm::vec4(rel_eye, r_au);
        inst.color_is_star = glm::vec4(b.color_rgb, b.is_star ? 1.0f : 0.0f);
        inst.pole_oblateness = glm::vec4(glm::vec3(b.pole_render), static_cast<float>(b.oblateness));
        inst.pole_color_lum = glm::vec4(b.color_pole_rgb, static_cast<float>(b.lum_eq));
        inst.spin_flags = glm::vec4(static_cast<float>(b.spin_angle_rad),
                                    static_cast<float>(b.lum_pole),
                                    b.is_black_hole ? 1.0f : 0.0f,
                                    static_cast<float>(i));
        inst_ptr[i] = inst;

        if (b.is_star && ubo.num_stars < MAX_SCENE_STARS) {
            uint32_t s = ubo.num_stars++;
            ubo.stars_pos_radius[s] = glm::vec4(rel_eye, r_au);
            ubo.stars_color_lum[s] = glm::vec4(b.color_rgb, static_cast<float>(std::max(b.lum_eq, 1e-4)));
        }
    }

    if (ubo.num_stars == 0) {
        ubo.num_stars = 1;
        ubo.stars_pos_radius[0] = glm::vec4(glm::vec3(-eye_pos), static_cast<float>(SOLAR_RADII_TO_AU));
        ubo.stars_color_lum[0] = glm::vec4(1.0f, 0.95f, 0.90f, 1.0f);
    }

    std::memcpy(scene_ubos_[f_idx].mapped, &ubo, sizeof(SceneCelestialUBO));

    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::uniform_buffer(0, scene_ubos_[f_idx]),
        rhi::PushDescriptorWrite::storage_buffer(1, instance_ssbos_[f_idx], 0, count * sizeof(BodyInstanceGPU)),
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sphere_pipeline_.handle);
    ctx.push_descriptors(cmd, sphere_pipeline_, writes);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vbo_.handle, &offset);
    vkCmdBindIndexBuffer(cmd, sphere_ibo_.handle, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, sphere_index_count_, count, 0, 0, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipeline_.handle);
    ctx.push_descriptors(cmd, point_pipeline_, writes);
    vkCmdDraw(cmd, 6, count, 0, 0);
}

} // namespace sf::rendering
