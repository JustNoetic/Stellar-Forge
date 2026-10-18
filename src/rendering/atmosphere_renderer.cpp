#include "rendering/atmosphere_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>

namespace sf::rendering {

static_assert(sizeof(AtmosphereParametersGPU) == 768);

void AtmosphereRenderer::init(rhi::VkContext& ctx) {
    constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
    const rhi::DescriptorLayoutBinding trans[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding multi[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding sky[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    const rhi::DescriptorLayoutBinding composite[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
    transmittance_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/transmittance.comp", trans);
    multiscatter_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/multiscatter.comp", multi);
    sky_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/sky_view.comp", sky);
    composite_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/composite.comp", composite);
}

void AtmosphereRenderer::shutdown(rhi::VkContext& ctx) {
    for (auto& t : tables_) {
        ctx.destroy_image(t->transmittance);
        ctx.destroy_image(t->multiscatter);
        ctx.destroy_image(t->sky);
        ctx.destroy_image(t->throughput);
        for (auto& u : t->uniforms) ctx.destroy_buffer(u);
    }
    tables_.clear();
    active_.clear();
    rhi::ShaderCache::destroy_pipeline(ctx, transmittance_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, multiscatter_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, sky_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, composite_pipeline_);
}

void AtmosphereRenderer::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings, VkExtent2D extent,
                                 std::span<const float> minimum_heights_km) {
    active_.clear();
    stats_ = {};
    if (!settings.show_atmospheres || system.empty()) return;
    struct Candidate { uint32_t index; double distance; };
    std::vector<Candidate> candidates;
    double tan_y = std::tan(camera.fov_deg*DEG_TO_RAD*0.5);
    double tan_x = tan_y*double(extent.width)/std::max(1u, extent.height);
    double focal = double(extent.height)/(2.0*tan_y);
    glm::dvec3 eye = camera.eye_pos_render().value;
    for (uint32_t i = 0; i < system.size(); ++i) {
        const auto b = system.body(i);
        const auto& a = b.definition.atmosphere;
        if (!a.enabled || b.definition.is_star || b.definition.is_black_hole || b.definition.is_spacecraft || !ephemeris::valid_atmosphere(a)) continue;
        double radius = b.definition.radius_au()*AU_TO_KM;
        glm::dvec3 to_body = (b.render.position.value-eye)*AU_TO_KM;
        double distance = glm::length(to_body);
        double outer = radius+a.height_km;
        if (distance > outer) {
            double depth = glm::dot(to_body, camera.forward_dir());
            if (depth+outer <= 0.0 ||
                std::abs(glm::dot(to_body, camera.right_dir())) > depth*tan_x+outer*std::sqrt(1.0+tan_x*tan_x) ||
                std::abs(glm::dot(to_body, camera.up_dir())) > depth*tan_y+outer*std::sqrt(1.0+tan_y*tan_y)) continue;
        }
        if ((radius+a.height_km)/std::max(distance, 1e-12)*focal < 0.5) continue;
        candidates.push_back({i, distance});
    }
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.distance > b.distance; });
    auto image = [&](uint32_t w, uint32_t h) {
        return ctx.create_image_2d(w, h, VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    };
    auto bake = [&](const rhi::Pipeline& pipeline, std::span<const rhi::PushDescriptorWrite> writes, uint32_t x, uint32_t y) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        ctx.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, x, y, 1);
    };
    auto writable = [&](rhi::Image& img) { ctx.transition_image(cmd, img, VK_IMAGE_LAYOUT_GENERAL); };
    auto readable = [&](rhi::Image& img) {
        ctx.transition_image(cmd, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    };
    uint32_t timer = candidates.empty() ? UINT32_MAX : ctx.begin_gpu_timer(cmd, "atmosphere_luts");
    for (auto candidate : candidates) {
        size_t slot = active_.size();
        if (slot == tables_.size()) {
            auto t = std::make_unique<Tables>();
            t->transmittance = image(256, 64);
            t->multiscatter = image(32, 32);
            t->sky = image(192, 108);
            t->throughput = image(192, 108);
            for (auto& u : t->uniforms) u = ctx.create_host_buffer(sizeof(AtmosphereParametersGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            tables_.push_back(std::move(t));
        }
        auto& t = *tables_[slot];
        const auto b = system.body(candidate.index);
        const auto& a = b.definition.atmosphere;
        double polar_ratio = std::clamp(1.0-b.definition.oblateness, 0.05, 1.0);
        glm::dvec3 pole = glm::normalize(b.render.pole);
        auto atmosphere_space = [&](glm::dvec3 v) { return v+pole*glm::dot(v, pole)*(1.0/polar_ratio-1.0); };
        glm::dvec3 relative = atmosphere_space((eye-b.render.position.value)*AU_TO_KM);
        double distance = glm::length(relative);
        glm::dvec3 z = distance > 1e-12 ? relative/distance : glm::dvec3(0.0, 1.0, 0.0);
        glm::dvec3 x = glm::normalize(glm::cross(std::abs(z.y) < 0.99 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), z));
        glm::dvec3 y = glm::cross(z, x);
        float radius = static_cast<float>(b.definition.radius_au()*AU_TO_KM);
        AtmosphereParametersGPU p{};
        double minimum = std::min(0.0, b.definition.height_range_km.x);
        if (candidate.index < minimum_heights_km.size()) minimum = std::min(minimum, double(minimum_heights_km[candidate.index]));
        float bottom = std::max(radius*0.01f, radius+static_cast<float>(minimum/polar_ratio));
        p.radii_heights = {bottom, radius+a.height_km, a.rayleigh_scale_height_km, a.mie_scale_height_km};
        p.reference_shape = {radius, static_cast<float>(polar_ratio), 0.0f, 0.0f};
        p.world_pole = {glm::vec3(pole), static_cast<float>(1.0/polar_ratio-1.0)};
        p.rayleigh = {a.rayleigh_scattering, 0.0f};
        p.mie_scattering = {a.mie_scattering, a.mie_asymmetry};
        p.mie_extinction = {a.mie_extinction, 0.0f};
        p.ground_albedo = {a.ground_albedo, 0.0f};
        p.camera = {0.0f, 0.0f, std::max(static_cast<float>(distance), bottom+0.001f), 0.0f};
        p.axis_x = {glm::vec3(x), 0.0f};
        p.axis_y = {glm::vec3(y), 0.0f};
        p.axis_z = {glm::vec3(z), 0.0f};
        for (uint32_t s = 0; s < system.size() && p.camera.w < 16.0f; ++s) {
            const auto star = system.body(s);
            if (!star.definition.is_star) continue;
            glm::dvec3 delta = star.render.position.value-b.render.position.value;
            double d = glm::length(delta);
            if (d < 1e-12) continue;
            glm::dvec3 direction = glm::normalize(atmosphere_space(delta/d));
            uint32_t index = static_cast<uint32_t>(p.camera.w++);
            p.sun_direction[index] = {glm::vec3(glm::dot(direction, x), glm::dot(direction, y), glm::dot(direction, z)), 0.0f};
            float irradiance = settings.hdr_enabled ? static_cast<float>(std::max(star.definition.lum_eq, 0.0)/(d*d)) : 1.0f;
            p.sun_irradiance[index] = {star.definition.color_rgb*irradiance, 0.0f};
        }
        auto view_projection = camera.proj_matrix(float(extent.width)/std::max(1u, extent.height), camera.compute_near_plane_au()) * camera.view_matrix_eye_relative();
        p.inverse_view_projection = glm::inverse(view_projection);
        p.depth_params.x = settings.hdr_enabled ? settings.exposure : 1.0f;
        auto& uniform = t.uniforms[ctx.frame_index()];
        std::memcpy(uniform.mapped, &p, sizeof(p));
        ctx.flush_buffer(uniform);
        bool identity_dirty = t.definition != system.share_definition() || t.generation != system.generation() ||
            t.body_index != candidate.index || t.geometry_revision != b.revisions.geometry || t.lighting_revision != b.revisions.lighting;
        bool optics_dirty = !t.valid || identity_dirty || p.reference_shape != t.cached.reference_shape || p.radii_heights != t.cached.radii_heights || p.rayleigh != t.cached.rayleigh ||
            p.mie_scattering != t.cached.mie_scattering || p.mie_extinction != t.cached.mie_extinction || p.ground_albedo != t.cached.ground_albedo;
        bool sky_dirty = optics_dirty || p.camera != t.cached.camera || p.sun_direction != t.cached.sun_direction || p.sun_irradiance != t.cached.sun_irradiance;
        auto ubo = rhi::PushDescriptorWrite::uniform_buffer(0, uniform);
        auto sampler = ctx.sampler_linear_clamp();
        if (optics_dirty) {
            writable(t.transmittance);
            rhi::PushDescriptorWrite writes[] = {ubo, rhi::PushDescriptorWrite::storage_image(1, t.transmittance)};
            bake(transmittance_pipeline_, writes, 32, 8);
            readable(t.transmittance);
            writable(t.multiscatter);
            rhi::PushDescriptorWrite multi_writes[] = {ubo, rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler), rhi::PushDescriptorWrite::storage_image(2, t.multiscatter)};
            bake(multiscatter_pipeline_, multi_writes, 32, 32);
            readable(t.multiscatter);
            ++stats_.optical_lut_updates;
        }
        if (sky_dirty) {
            uint32_t sky_timer = ctx.begin_gpu_timer(cmd, "atmosphere_sky");
            writable(t.sky);
            writable(t.throughput);
            rhi::PushDescriptorWrite writes[] = {ubo,
                rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                rhi::PushDescriptorWrite::storage_image(3, t.sky),
                rhi::PushDescriptorWrite::storage_image(4, t.throughput)};
            bake(sky_pipeline_, writes, 24, 14);
            readable(t.sky);
            readable(t.throughput);
            ctx.end_gpu_timer(cmd, sky_timer);
            ++stats_.sky_lut_updates;
        }
        t.cached = p;
        t.definition = system.share_definition();
        t.generation = system.generation();
        t.body_index = candidate.index;
        t.geometry_revision = b.revisions.geometry;
        t.lighting_revision = b.revisions.lighting;
        t.valid = true;
        active_.push_back(&t);
    }
    ctx.end_gpu_timer(cmd, timer);
    stats_.visible = static_cast<uint32_t>(active_.size());
}

void AtmosphereRenderer::composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth) {
    if (active_.empty()) return;
    ctx.transition_image(cmd, color, VK_IMAGE_LAYOUT_GENERAL);
    ctx.transition_image(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint32_t timer = ctx.begin_gpu_timer(cmd, "atmosphere_composite");
    for (auto* t : active_) {
        rhi::PushDescriptorWrite writes[] = {
            rhi::PushDescriptorWrite::uniform_buffer(0, t->uniforms[ctx.frame_index()]),
            rhi::PushDescriptorWrite::sampled_image(1, t->sky, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(2, t->throughput, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(3, depth, ctx.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::storage_image(4, color)};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, composite_pipeline_.handle);
        ctx.push_descriptors(cmd, composite_pipeline_, writes);
        vkCmdDispatch(cmd, (color.extent.width+7)/8, (color.extent.height+7)/8, 1);
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }
    ctx.end_gpu_timer(cmd, timer);
}

}
