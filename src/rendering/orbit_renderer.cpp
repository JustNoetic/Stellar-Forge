#include "rendering/orbit_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>

namespace sf::rendering {
namespace {

struct alignas(32) OrbitGpu {
    glm::dvec4 d0, d1, d2, d3, d4;
};

struct alignas(32) OrbitPush {
    glm::mat4 view_projection;
    glm::dvec4 eye;
    glm::vec4 parameters;
};

static_assert(sizeof(OrbitGpu) == 160);
static_assert(sizeof(OrbitPush) == 128);

}

void OrbitRenderer::init(rhi::VkContext& ctx) {
    rhi::GraphicsPipelineDesc desc;
    desc.vert_path = "celestial/orbit.vert";
    desc.frag_path = "celestial/orbit.frag";
    desc.bindings = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1}};
    desc.push_constant_size = sizeof(OrbitPush);
    desc.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    desc.bresenham_lines = true;
    desc.depth_write = false;
    desc.alpha_blend = true;
    desc.cull_mode = VK_CULL_MODE_NONE;
    pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    if (!ctx.bresenham_lines_enabled()) sf::log::warn("Bresenham orbit lines are unavailable on this GPU; using default line rasterization.");
}

void OrbitRenderer::shutdown(rhi::VkContext& ctx) {
    rhi::ShaderCache::destroy_pipeline(ctx, pipeline_);
    for (auto& buffer : buffers_) ctx.destroy_buffer(buffer);
}

void OrbitRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                            const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent) {
    if (!settings.show_orbits || sys.empty()) return;
    if (generation_ != sys.generation() || state_revision_ != sys.state_revision() || definition_revision_ != sys.definition_revision()) {
        curves_ = ephemeris::compute_orbit_curves(sys);
        generation_ = sys.generation();
        state_revision_ = sys.state_revision();
        definition_revision_ = sys.definition_revision();
    }
    if (curves_.empty()) return;
    auto& buffer = buffers_[ctx.frame_index()];
    auto bytes = curves_.size() * sizeof(OrbitGpu);
    if (buffer.size < bytes) {
        ctx.destroy_buffer(buffer);
        buffer = ctx.create_host_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    auto* gpu = static_cast<OrbitGpu*>(buffer.mapped);
    uint32_t count = 0;
    auto eye = cam.eye_pos_render();
    for (const auto& curve : curves_) {
        auto focus = ecliptic_to_render(curve.focus);
        double distance = glm::length(camera_relative(focus, eye));
        if (distance > 1e-6 && curve.semilatus_au / distance < 1e-6) continue;
        gpu[count++] = {
            glm::dvec4(ecliptic_to_render(curve.periapsis), curve.semilatus_au),
            glm::dvec4(ecliptic_to_render(curve.transverse), curve.eccentricity),
            glm::dvec4(focus.value, curve.arc.start),
            glm::dvec4(glm::dvec3(curve.color), curve.arc.span),
            glm::dvec4(curve.arc.phase, curve.patched ? 1.0 : 0.0,
                       curve.arc.span == TWO_PI && curve.eccentricity < 1.0 ? 1.0 : 0.0, 0.0)
        };
    }
    if (count == 0) return;
    OrbitPush push{};
    float aspect = static_cast<float>(extent.width) / static_cast<float>(std::max(1u, extent.height));
    push.view_projection = cam.proj_matrix(aspect, cam.compute_near_plane_au()) * cam.view_matrix_eye_relative();
    push.eye = glm::dvec4(eye.value, 0.0);
    push.parameters = {static_cast<float>(settings.orbit_segments), settings.orbit_opacity, 0.3f, 0.0f};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle);
    rhi::PushDescriptorWrite writes[] = {rhi::PushDescriptorWrite::storage_buffer(0, buffer)};
    ctx.push_descriptors(cmd, pipeline_, writes);
    ctx.push_constants(cmd, pipeline_, push);
    vkCmdDraw(cmd, settings.orbit_segments + 1, count, 0, 0);
}

}
