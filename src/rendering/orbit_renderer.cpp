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
    refraction_fallback_.init(ctx);
    desc.bindings = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
                     {20, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
                     {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_VERTEX_BIT, 1}};
    desc.push_constant_size = sizeof(OrbitPush);
    desc.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    desc.bresenham_lines = true;
    desc.depth_write = false;
    desc.alpha_blend = true;
    desc.cull_mode = VK_CULL_MODE_NONE;
    pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    if (ctx.fragment_stores_enabled()) {
        desc.bindings.push_back({14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT});
        desc.bindings.push_back({15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT});
        desc.macros = {{"TRANSPARENCY_CAPTURE", "1"}};
        transparency_pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    }
    if (!ctx.bresenham_lines_enabled()) sf::log::warn("Bresenham orbit lines are unavailable on this GPU; using default line rasterization.");
}

void OrbitRenderer::shutdown(rhi::VkContext& ctx) {
    rhi::ShaderCache::destroy_pipeline(ctx, pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, transparency_pipeline_);
    for (auto& buffer : buffers_) ctx.destroy_buffer(buffer);
    refraction_fallback_.shutdown(ctx);
}

void OrbitRenderer::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                            const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent,
                            std::span<const rhi::PushDescriptorWrite> transparency_writes,
                            std::span<const rhi::PushDescriptorWrite> refraction_writes) {
    if (!settings.show_orbits || sys.empty()) return;
    bool same_bodies = identities_.size() == sys.size();
    for (size_t i = 0; same_bodies && i < sys.size(); ++i) same_bodies = identities_[i] == sys.body(i).revisions.identity;
    if (!same_bodies || state_revision_ != sys.state_revision() || definition_revision_ != sys.definition_revision() ||
        hierarchy_revision_ != sys.hierarchy_revision()) {
        curves_ = ephemeris::compute_orbit_curves(sys);
        identities_.resize(sys.size());
        for (size_t i = 0; i < sys.size(); ++i) identities_[i] = sys.body(i).revisions.identity;
        state_revision_ = sys.state_revision();
        definition_revision_ = sys.definition_revision();
        hierarchy_revision_ = sys.hierarchy_revision();
    }

    ephemeris::EphemerisOrbitConfig ephem_cfg{
        settings.ephem_frame_mode,
        settings.ephem_rotating_pair,
        settings.ephem_lagrange_days,
        settings.ephem_sample_points,
        settings.ephem_trail_fade,
        settings.ephem_trail_days,
        settings.ephem_lead_days,
        settings.ephem_hide_outside
    };
    trajectory_cache_.evaluate(sys, ephem_cfg, polylines_, polyline_covered_);

    if (curves_.empty() && polylines_.empty()) return;

    auto eye = cam.eye_pos_render();
    size_t total_poly_verts = 0;
    for (const auto& poly : polylines_) {
        total_poly_verts += poly.positions.size();
    }

    auto& buffer = buffers_[ctx.frame_index()];
    auto bytes = (curves_.size() + total_poly_verts) * sizeof(OrbitGpu);
    if (bytes == 0) return;
    if (buffer.size < bytes) {
        ctx.destroy_buffer(buffer);
        buffer = ctx.create_host_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    auto* gpu = static_cast<OrbitGpu*>(buffer.mapped);
    uint32_t count = 0;
    for (const auto& curve : curves_) {
        if (curve.body_index >= 0 && static_cast<size_t>(curve.body_index) < sys.size()) {
            if (!curve.reflex && static_cast<size_t>(curve.body_index) < polyline_covered_.size() &&
                polyline_covered_[curve.body_index]) {
                continue;
            }
            if (settings.ephem_hide_outside && !sys.body(curve.body_index).render.spice_valid) {
                continue;
            }
        }
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

    struct PolyDraw {
        uint32_t first = 0;
        uint32_t vertex_count = 0;
    };
    std::vector<PolyDraw> poly_draws;
    uint32_t v_cursor = count;
    for (const auto& poly : polylines_) {
        if (poly.positions.size() < 2) continue;
        auto center_r = ecliptic_to_render(poly.center);
        double distance = glm::length(camera_relative(center_r, eye));
        if (distance > 1e-6 && poly.extent_au / distance < 1e-6) continue;
        uint32_t first_v = v_cursor;
        for (size_t k = 0; k < poly.positions.size(); ++k) {
            OrbitGpu vert{};
            vert.d0 = glm::dvec4(ecliptic_to_render(poly.positions[k]).value, static_cast<double>(poly.alphas[k]));
            vert.d3 = glm::dvec4(glm::dvec3(poly.color), 1.0);
            gpu[v_cursor++] = vert;
        }
        poly_draws.push_back({first_v, static_cast<uint32_t>(poly.positions.size())});
    }

    if (count == 0 && poly_draws.empty()) return;
    OrbitPush push{};
    float aspect = static_cast<float>(extent.width) / static_cast<float>(std::max(1u, extent.height));
    push.view_projection = cam.proj_matrix(aspect, cam.compute_near_plane_au()) * cam.view_matrix_eye_relative();
    push.eye = glm::dvec4(eye.value, 0.0);
    push.parameters = {static_cast<float>(settings.orbit_segments), settings.orbit_opacity, 0.3f, 0.0f};
    auto& pipeline = transparency_writes.empty() ? pipeline_ : transparency_pipeline_;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
    std::vector<rhi::PushDescriptorWrite> writes{rhi::PushDescriptorWrite::storage_buffer(0, buffer)};
    writes.insert(writes.end(), transparency_writes.begin(), transparency_writes.end());
    auto fallback = refraction_fallback_.writes(ctx);
    if (refraction_writes.empty()) refraction_writes = fallback;
    writes.insert(writes.end(), refraction_writes.begin(), refraction_writes.end());
    ctx.push_descriptors(cmd, pipeline, writes);
    if (count > 0) {
        push.parameters.w = 0.0f;
        ctx.push_constants(cmd, pipeline, push);
        vkCmdDraw(cmd, settings.orbit_segments + 1, count, 0, 0);
    }
    if (!poly_draws.empty()) {
        push.parameters.w = 1.0f;
        ctx.push_constants(cmd, pipeline, push);
        for (const auto& pd : poly_draws) {
            vkCmdDraw(cmd, pd.vertex_count, 1, pd.first, 0);
        }
    }
}

}
