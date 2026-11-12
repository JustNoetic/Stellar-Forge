#include "rendering/atmosphere_internal.hpp"
#include "rendering/ring_renderer.hpp"
#include "rendering/planetshine.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace sf::rendering {

void AtmosphereRenderer::bake_luts(const LutRecorder& lut, Tables& t, const AtmosphereFrame& frame, const AtmosphereChanges& changes,
                                   const ephemeris::SystemSnapshot& system, uint32_t index, size_t surface_slot, const RenderSettings& settings) {
    auto cmd = lut.cmd;
    auto& ctx = lut.ctx;
    const auto b = system.body(index);
    const auto& p = frame.p;
    const float bottom = frame.bottom, radius = frame.radius, height = frame.height;
    const double point_radius = frame.point_radius;
    const bool optics_dirty = changes.optics, identity_dirty = changes.identity, lighting_mode_dirty = changes.lighting_mode;
    const bool sky_dirty = changes.sky, aerial_dirty = changes.aerial, small_stars = changes.small_stars;
    const float finite_radius = changes.finite_radius;
    auto& uniform = t.uniforms[ctx.frame_index()];
    auto& records = surface_records_[ctx.frame_index()];
    auto& point_records = point_responses_[ctx.frame_index()];
    auto ubo = rhi::PushDescriptorWrite::uniform_buffer(0, uniform);
    auto sampler = ctx.sampler_linear_clamp();
    if (optics_dirty) {
        for (auto& layer : surface_layers_) if (layer == &t) layer = nullptr;
        lut.writable(t.transmittance);
        rhi::PushDescriptorWrite writes[] = {ubo, rhi::PushDescriptorWrite::storage_image(1, t.transmittance)};
        lut.bake(transmittance_pipeline_, writes, 32, 8);
        lut.readable(t.transmittance);
        lut.writable(t.multiscatter);
        rhi::PushDescriptorWrite multi_writes[] = {ubo, rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler), rhi::PushDescriptorWrite::storage_image(2, t.multiscatter)};
        lut.bake(multiscatter_pipeline_, multi_writes, 32, 32);
        lut.readable(t.multiscatter);
        ++stats_.optical_lut_updates;
    }
    if (p.sun_irradiance[0].w > 0.0f) {
        bool source_dirty = !t.stellar_source || optics_dirty || lighting_mode_dirty ||
            p.sun_irradiance[0].w != t.cached.sun_irradiance[0].w;
        if (!t.stellar_source)
            t.stellar_source = ctx.create_image_2d(128, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 1, 5);
        if (source_dirty) {
            if (!stellar_source_pipeline_)
                stellar_source_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/stellar_source.comp", source_bindings);
            uint32_t source_timer = ctx.begin_gpu_timer(cmd, "atmosphere_stellar_source");
            lut.writable(t.stellar_source);
            rhi::PushDescriptorWrite writes[] = {ubo,
                rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                rhi::PushDescriptorWrite::storage_image(3, t.stellar_source)};
            lut.bake(stellar_source_pipeline_, writes, 16, 8);
            lut.readable(t.stellar_source);
            ctx.end_gpu_timer(cmd, source_timer);
            ++stats_.stellar_lut_updates;
        }
    }
    const auto& source_lut = t.stellar_source ? t.stellar_source : surface_lut_;
    const rhi::Pipeline* angular_pipeline = settings.aerial_perspective_enabled
        ? (small_stars ? &small_aerial_pipeline_ : &aerial_pipeline_)
        : (small_stars ? &small_sky_pipeline_ : &sky_pipeline_);
    if (!small_stars && p.aerosol_count.x < 0.5f &&
        (settings.aerial_perspective_enabled ? aerial_dirty : sky_dirty)) {
        bool molecular = p.aerosol_count.y < 0.5f;
        auto& pipeline = settings.aerial_perspective_enabled
            ? (molecular ? molecular_aerial_pipeline_ : legacy_aerial_pipeline_)
            : (molecular ? molecular_sky_pipeline_ : legacy_sky_pipeline_);
        if (!pipeline) {
            constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
            std::vector<rhi::DescriptorLayoutBinding> bindings{
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
                {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}};
            if (settings.aerial_perspective_enabled) for (uint32_t binding : {5u, 6u, 7u})
                bindings.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage});
            std::vector<std::pair<std::string, std::string>> macros{{"ATMOSPHERE_NO_CUSTOM_AEROSOLS", "1"}};
            if (molecular) macros.emplace_back("ATMOSPHERE_RAYLEIGH_ONLY", "1");
            pipeline = rhi::ShaderCache::create_compute_pipeline(ctx, settings.aerial_perspective_enabled
                ? "atmosphere/aerial_perspective.comp" : "atmosphere/sky_view.comp", bindings, 0, macros);
        }
        angular_pipeline = &pipeline;
    }
    if (surface_slot != SIZE_MAX && index < 1024 && b.render.atmosphere) {
        glm::vec2 surface_key{finite_radius, p.depth_params.z};
        bool surface_dirty = optics_dirty || surface_keys_[surface_slot].y != surface_key.y || surface_layers_[surface_slot] != &t;
        if (surface_dirty || surface_keys_[surface_slot].x != surface_key.x) {
            uint32_t surface_timer = ctx.begin_gpu_timer(cmd, "atmosphere_surface_lut");
            lut.writable(surface_lut_);
            rhi::PushDescriptorWrite writes[] = {ubo,
                rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                rhi::PushDescriptorWrite::storage_image(3, surface_lut_)};
            ctx.push_constants(cmd, surface_pipeline_, glm::uvec2(uint32_t(surface_slot), surface_dirty ? 0u : 1u));
            lut.bake(surface_pipeline_, writes, 16, 4);
            lut.readable(surface_lut_);
            ctx.end_gpu_timer(cmd, surface_timer);
            ++stats_.surface_lut_updates;
            surface_layers_[surface_slot] = &t;
            surface_keys_[surface_slot] = surface_key;
        }
        if (settings.atmosphere_surface_lighting)
            static_cast<glm::vec4*>(records.mapped)[index] = {bottom, radius+height,
                float(b.opaque_radius_au()*AU_TO_KM), float(surface_slot+1)};
    }
    if (point_radius < 2.5) {
        struct PointPush {
            glm::vec4 color;
            physics::SurfaceMaterialGPU material;
            glm::vec4 geometry;
            glm::uvec4 resources;
        } point{glm::vec4(b.definition.color_rgb, 0),
            physics::pack_material(physics::resolve_material(b.definition.material, b.definition.name)),
            glm::vec4(float(b.opaque_radius_au()*AU_TO_KM), 0, 0, 0),
            glm::uvec4(surface_slot == SIZE_MAX ? 0u : uint32_t(surface_slot),
                settings.atmosphere_surface_lighting && surface_slot != SIZE_MAX ? 1u : 0u, 0, 0)};
        if (!point_pipeline_) {
            constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
            const rhi::DescriptorLayoutBinding bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage},
                {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
                {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
                {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}};
            point_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/point_source.comp", bindings, sizeof(PointPush));
            const rhi::DescriptorLayoutBinding reduce[] = {
                {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage}};
            point_reduce_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/point_reduce.comp", reduce, sizeof(uint32_t));
        }
        if (!t.point_response) {
            t.point_partials = ctx.create_device_buffer(16*32*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            t.point_response = ctx.create_device_buffer(16*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
        // The response is a disc integral, so it depends only on the world-space view, pole and star
        // directions and the stars' angular radii, not on the local x/y basis. Orbital motion moves
        // these by float-noise amounts every frame; rebake only once they drift past the tolerance.
        constexpr double POINT_ANGLE_TOLERANCE = 1e-4;  // radians (chord length of unit vectors)
        auto moved = [](glm::dvec3 a, glm::dvec3 b) { return glm::length(a-b) > POINT_ANGLE_TOLERANCE; };
        std::array<glm::dvec4, 16> suns{};
        bool geometry_moved = p.camera.w != t.point_stars || moved(frame.z, t.point_view) || moved(frame.pole, t.point_pole);
        for (uint32_t star = 0; star < uint32_t(p.camera.w); ++star) {
            const glm::vec4& sun = p.sun_direction[star];
            suns[star] = {frame.x*double(sun.x)+frame.y*double(sun.y)+frame.z*double(sun.z), sun.w};
            geometry_moved = geometry_moved || moved(glm::dvec3(suns[star]), glm::dvec3(t.point_suns[star])) ||
                std::abs(suns[star].w-t.point_suns[star].w) > POINT_ANGLE_TOLERANCE*t.point_suns[star].w;
        }
        bool point_dirty = !t.point_valid || identity_dirty || lighting_mode_dirty || geometry_moved || point.color != t.point_color ||
            glm::uvec2(point.resources) != t.point_surface ||
            std::memcmp(&point.material, &t.point_material, sizeof(point.material)) != 0;
        if (point_dirty) {
            uint32_t point_timer = ctx.begin_gpu_timer(cmd, "atmosphere_point_lut");
            ctx.memory_barrier(cmd);
            const rhi::PushDescriptorWrite writes[] = {ubo,
                rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                rhi::PushDescriptorWrite::sampled_image(3, surface_lut_, sampler),
                rhi::PushDescriptorWrite::storage_buffer(4, t.point_partials)};
            ctx.push_constants(cmd, point_pipeline_, point);
            if (p.camera.w > 0.0f) lut.bake(point_pipeline_, writes, 32, uint32_t(p.camera.w));
            ctx.memory_barrier(cmd);
            const rhi::PushDescriptorWrite reduce[] = {
                rhi::PushDescriptorWrite::storage_buffer(0, t.point_partials),
                rhi::PushDescriptorWrite::storage_buffer(1, t.point_response)};
            ctx.push_constants(cmd, point_reduce_pipeline_, uint32_t(p.camera.w));
            lut.bake(point_reduce_pipeline_, reduce, 1, 1);
            ctx.end_gpu_timer(cmd, point_timer);
            t.point_material = point.material;
            t.point_color = point.color;
            t.point_surface = glm::uvec2(point.resources);
            t.point_view = frame.z;
            t.point_pole = frame.pole;
            t.point_suns = suns;
            t.point_stars = p.camera.w;
            t.point_valid = true;
            ++stats_.point_lut_updates;
        }
        ctx.memory_barrier(cmd);
        ctx.copy_buffer(cmd, t.point_response, point_records, 16*sizeof(glm::vec4), 0, index*16*sizeof(glm::vec4));
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }
    if (p.depth_params.w > 0.0f && sky_dirty && !settings.aerial_perspective_enabled) {
        uint32_t sky_timer = ctx.begin_gpu_timer(cmd, "atmosphere_sky");
        lut.writable(t.sky);
        lut.writable(t.throughput);
        rhi::PushDescriptorWrite writes[] = {ubo,
            rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
            rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
            rhi::PushDescriptorWrite::storage_image(3, t.sky),
            rhi::PushDescriptorWrite::storage_image(4, t.throughput),
            rhi::PushDescriptorWrite::sampled_image(10, source_lut, sampler)};
        lut.bake(*angular_pipeline, writes, 24, 14);
        lut.readable(t.sky);
        lut.readable(t.throughput);
        ctx.end_gpu_timer(cmd, sky_timer);
        ++stats_.sky_lut_updates;
    }
    if (p.depth_params.w > 0.0f && settings.aerial_perspective_enabled && aerial_dirty) {
        uint32_t aerial_timer = ctx.begin_gpu_timer(cmd, "atmosphere_aerial");
        lut.writable(t.aerial_radiance);
        lut.writable(t.aerial_optical_depth);
        lut.writable(t.aerial_range);
        lut.writable(t.sky);
        lut.writable(t.throughput);
        rhi::PushDescriptorWrite writes[] = {ubo,
            rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
            rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
            rhi::PushDescriptorWrite::storage_image(3, t.aerial_radiance),
            rhi::PushDescriptorWrite::storage_image(4, t.aerial_optical_depth),
            rhi::PushDescriptorWrite::storage_image(5, t.aerial_range),
            rhi::PushDescriptorWrite::storage_image(6, t.sky),
            rhi::PushDescriptorWrite::storage_image(7, t.throughput),
            rhi::PushDescriptorWrite::sampled_image(10, source_lut, sampler)};
        lut.bake(*angular_pipeline, writes, 24, 27);
        lut.readable(t.aerial_radiance);
        lut.readable(t.aerial_optical_depth);
        lut.readable(t.aerial_range);
        lut.readable(t.sky);
        lut.readable(t.throughput);
        ctx.end_gpu_timer(cmd, aerial_timer);
        ++stats_.sky_lut_updates;
        ++stats_.aerial_lut_updates;
        t.aerial_valid = true;
    }
}

void AtmosphereRenderer::bake_secondary(const LutRecorder& lut, Tables& t, const AtmosphereFrame& frame, const AtmosphereChanges& changes,
                                        const ephemeris::SystemSnapshot& system, uint32_t index, const RenderSettings& settings,
                                        std::span<const PlanetshineGPU> planetshine, const RingRenderer* ring_lighting) {
    auto cmd = lut.cmd;
    auto& ctx = lut.ctx;
    const auto& p = frame.p;
    const glm::dvec3 pole = frame.pole, x = frame.x, y = frame.y, z = frame.z;
    auto atmosphere_space = [&](glm::dvec3 v) { return frame.atmosphere_space(v); };
    const bool optics_dirty = changes.optics, scattering = changes.scattering;
    auto ubo = rhi::PushDescriptorWrite::uniform_buffer(0, t.uniforms[ctx.frame_index()]);
    auto sampler = ctx.sampler_linear_clamp();
    if (optics_dirty) t.secondary_valid = t.secondary_source_valid = false;
    t.secondary_active = false;
    t.secondary_irradiance = {};
    if (p.depth_params.w > 0.0f && !settings.direct_sunlight_only && scattering) {
        PlanetshineGPU lights{};
        if (settings.planetshine_enabled && index < planetshine.size()) lights = planetshine[index];
        if (ring_lighting && !settings.atmosphere_planetshine_reference && settings.ringshine_enabled && settings.show_rings) {
            auto diffuse = ring_lighting->atmosphere_sources(system,settings,index);
            if (glm::any(glm::greaterThan(glm::vec3(diffuse.irradiance[0]),glm::vec3(0)))) {
                size_t weakest = 0;
                for (size_t source = 1; source < PLANETSHINE_LIGHTS; ++source)
                    if (glm::dot(glm::vec3(lights.irradiance[source]),glm::vec3(.2126f,.7152f,.0722f)) <
                        glm::dot(glm::vec3(lights.irradiance[weakest]),glm::vec3(.2126f,.7152f,.0722f))) weakest = source;
                lights.direction[weakest] = diffuse.direction[0]; lights.irradiance[weakest] = diffuse.irradiance[0];
            }
        }
        size_t brightest = 0;
        float maximum = 0.0f;
        for (size_t source = 0; source < PLANETSHINE_LIGHTS; ++source) {
            float flux = glm::dot(glm::vec3(lights.irradiance[source]), glm::vec3(0.2126f, 0.7152f, 0.0722f));
            if (flux > maximum) { maximum = flux; brightest = source; }
        }
        if (maximum > 0.0f) {
            init_secondary(ctx);
            if (!t.secondary_sky) {
                t.secondary_sky = lut.image(96, 54);
                t.secondary_range = lut.image(96, 54, VK_FORMAT_R32_SFLOAT);
                t.secondary_aerial = ctx.create_image_3d(96, 54, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
                lut.readable(t.secondary_aerial);
            }
            if (!settings.atmosphere_planetshine_reference) {
                static_assert(PLANETSHINE_LIGHTS == 4 && sizeof(PlanetshineGPU) == 128);
                PlanetshineGPU sources{};
                glm::vec3 total(0.0f);
                uint32_t count = 0;
                bool diffuse_ring = false;
                uint32_t ring_slot = 0;
                glm::vec3 ring_response(0.0f);
                for (size_t source = 0; source < PLANETSHINE_LIGHTS; ++source) {
                    glm::vec3 flux = glm::max(glm::vec3(lights.irradiance[source]), glm::vec3(0.0f));
                    if (!glm::any(glm::greaterThan(flux, glm::vec3(0.0f)))) continue;
                    bool ring = lights.direction[source].w < 0;
                    if (ring && !ring_lighting) continue;
                    if (ring) {
                        diffuse_ring = true;
                        ring_slot = count;
                        ring_response = glm::max(glm::vec3(lights.direction[source]), glm::vec3(1e-30f));
                        sources.direction[count] = {0.0f, 0.0f, 0.0f, lights.direction[source].w};
                    } else {
                        glm::dvec3 direction = glm::normalize(atmosphere_space(glm::dvec3(lights.direction[source])));
                        sources.direction[count] = {glm::dot(direction, x), glm::dot(direction, y), glm::dot(direction, z), lights.direction[source].w};
                    }
                    sources.irradiance[count++] = {flux, lights.irradiance[source].w};
                    total += flux;
                }
                for (uint32_t source = 0; source < count; ++source) for (int channel = 0; channel < 3; ++channel)
                    sources.irradiance[source][channel] = count == 1 ? 1.0f :
                        (total[channel] > 0.0f ? sources.irradiance[source][channel]/total[channel] : 0.0f);
                std::array<glm::vec4, 17> ring_key{};
                uint64_t ring_revision = 0;
                if (diffuse_ring) {
                    float luminance = glm::dot(ring_response, glm::vec3(0.2126f, 0.7152f, 0.0722f));
                    sources.irradiance[ring_slot] = {glm::vec3(sources.irradiance[ring_slot])*(luminance/ring_response), 1.0f/(float(PI)*luminance)};
                    ring_key[0] = {glm::vec3(glm::dot(pole, x), glm::dot(pole, y), glm::dot(pole, z)), float(settings.ringshine_band_count)};
                    for (size_t star = 0; star < p.sun_direction.size(); ++star) ring_key[star+1] = {glm::vec3(p.sun_direction[star]), 0.0f};
                    ring_revision = ring_lighting->shadow_revision();
                }
                t.secondary_irradiance = {total*float(PI), 0.0f};
                t.secondary_active = true;
                bool weights_changed = false;
                for (uint32_t source = 0; source < PLANETSHINE_LIGHTS; ++source)
                    weights_changed |= glm::any(glm::greaterThan(glm::abs(sources.irradiance[source]-t.secondary_weights[source]), glm::vec4(2e-6f)));
                if (!t.secondary_valid || !t.secondary_approximate || t.secondary_directions != sources.direction ||
                    t.secondary_ring_key != ring_key || t.secondary_ring_revision != ring_revision ||
                    weights_changed || t.secondary_altitude != p.camera.z || (settings.aerial_perspective_enabled && !t.secondary_aerial_valid)) {
                    bool simple = p.aerosol_count.x < 0.5f, molecular = p.aerosol_count.y < 0.5f;
                    auto& pipeline = secondary_approximate_pipelines_[(diffuse_ring ? 12 : 0)+(count-1)*3+(simple ? (molecular ? 1 : 2) : 0)];
                    if (!pipeline) {
                        constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
                        std::vector<rhi::DescriptorLayoutBinding> bindings{
                            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
                            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
                            {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}};
                        std::vector<std::pair<std::string, std::string>> macros{{"ATMOSPHERE_GRID_WIDTH", "96"}, {"ATMOSPHERE_GRID_HEIGHT", "54"},
                            {"ATMOSPHERE_SECONDARY_APPROX", "1"}, {"ATMOSPHERE_SECONDARY_LIGHTS", std::to_string(count)}};
                        if (diffuse_ring) {
                            macros.emplace_back("ATMOSPHERE_SECONDARY_RING", "1");
                            bindings.push_back({16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage});
                        }
                        if (simple) {
                            macros.emplace_back("ATMOSPHERE_NO_CUSTOM_AEROSOLS", "1");
                            if (molecular) macros.emplace_back("ATMOSPHERE_RAYLEIGH_ONLY", "1");
                        }
                        pipeline = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/planetshine.comp", bindings, sizeof(PlanetshineGPU), macros);
                    }
                    uint32_t secondary_timer = ctx.begin_gpu_timer(cmd, "atmosphere_secondary");
                    lut.writable(t.secondary_sky);
                    lut.writable(t.secondary_range);
                    lut.writable(t.secondary_aerial);
                    std::vector<rhi::PushDescriptorWrite> writes{ubo,
                        rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                        rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                        rhi::PushDescriptorWrite::storage_image(3, t.secondary_aerial),
                        rhi::PushDescriptorWrite::storage_image(4, t.secondary_sky),
                        rhi::PushDescriptorWrite::storage_image(5, t.secondary_range)};
                    if (diffuse_ring)
                        writes.push_back(rhi::PushDescriptorWrite::sampled_image(16, ring_lighting->ringshine_atlas(ctx.frame_index()), ctx.sampler_linear_clamp()));
                    ctx.push_constants(cmd, pipeline, sources);
                    lut.bake(pipeline, writes, 12, 14);
                    lut.readable(t.secondary_sky);
                    lut.readable(t.secondary_range);
                    lut.readable(t.secondary_aerial);
                    ctx.end_gpu_timer(cmd, secondary_timer);
                    ++stats_.secondary_lut_updates;
                    t.secondary_directions = sources.direction;
                    t.secondary_ring_key = ring_key;
                    t.secondary_ring_revision = ring_revision;
                    t.secondary_weights = sources.irradiance;
                    t.secondary_altitude = p.camera.z;
                    t.secondary_valid = t.secondary_approximate = true;
                    t.secondary_aerial_valid = settings.aerial_perspective_enabled;
                }
            } else {
                glm::dvec3 direction = glm::normalize(atmosphere_space(glm::dvec3(lights.direction[brightest])));
                glm::vec4 local{glm::vec3(glm::dot(direction, x), glm::dot(direction, y), glm::dot(direction, z)), lights.direction[brightest].w};
                t.secondary_irradiance = {glm::vec3(lights.irradiance[brightest])*float(PI), 0.0f};
                t.secondary_active = true;
                float cache_radius = local.w >= 1e-5f && t.stellar_source_resolved
                    ? std::clamp(std::exp2(std::round(std::log2(local.w)*512.0f)/512.0f), 1e-5f, float(PI*0.5)) : 0.0f;
                if (local.w < 0.01f) cache_radius = std::min(cache_radius, std::nextafter(0.01f, 0.0f));
                else if (cache_radius > 0.0f) cache_radius = std::max(cache_radius, 0.01f);
                if (cache_radius > 0.0f && (!t.secondary_source_valid || t.secondary_source_radius != cache_radius)) {
                    if (!secondary_source_pipeline_) {
                        const std::pair<std::string, std::string> macros[] = {{"ATMOSPHERE_SOURCE_RADIUS_PUSH", "1"}};
                        secondary_source_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx,
                            "atmosphere/stellar_source.comp", source_bindings, sizeof(glm::vec4), macros);
                    }
                    if (!t.secondary_source) t.secondary_source = ctx.create_image_2d(128, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 1, 5);
                    uint32_t source_timer = ctx.begin_gpu_timer(cmd, "atmosphere_secondary_source");
                    lut.writable(t.secondary_source);
                    const rhi::PushDescriptorWrite writes[] = {ubo,
                        rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                        rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                        rhi::PushDescriptorWrite::storage_image(3, t.secondary_source)};
                    ctx.push_constants(cmd, secondary_source_pipeline_, glm::vec4(cache_radius, 0, 0, 0));
                    lut.bake(secondary_source_pipeline_, writes, 16, 8);
                    lut.readable(t.secondary_source);
                    ctx.end_gpu_timer(cmd, source_timer);
                    t.secondary_source_radius = cache_radius;
                    t.secondary_source_valid = true;
                    ++stats_.secondary_source_lut_updates;
                }
                if (!t.secondary_valid || t.secondary_approximate || t.secondary_direction != local || t.secondary_altitude != p.camera.z ||
                    (settings.aerial_perspective_enabled && !t.secondary_aerial_valid)) {
                    bool small = local.w < 0.01f && cache_radius == 0.0f, simple = p.aerosol_count.x < 0.5f, molecular = p.aerosol_count.y < 0.5f;
                    auto& pipeline = secondary_bake_pipelines_[(small ? 0 : 3)+(simple ? (molecular ? 1 : 2) : 0)];
                    if (!pipeline) {
                        constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
                        const rhi::DescriptorLayoutBinding bindings[] = {
                            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage},
                            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}, {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
                            {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage}, {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage},
                            {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage}};
                        std::vector<std::pair<std::string, std::string>> macros{{"ATMOSPHERE_GRID_WIDTH", "96"}, {"ATMOSPHERE_GRID_HEIGHT", "54"}};
                        if (small) macros.emplace_back("ATMOSPHERE_SMALL_STARS", "1");
                        if (simple) {
                            macros.emplace_back("ATMOSPHERE_NO_CUSTOM_AEROSOLS", "1");
                            if (molecular) macros.emplace_back("ATMOSPHERE_RAYLEIGH_ONLY", "1");
                        }
                        pipeline = rhi::ShaderCache::create_compute_pipeline(ctx, "atmosphere/planetshine.comp", bindings, 2*sizeof(glm::vec4), macros);
                    }
                    uint32_t secondary_timer = ctx.begin_gpu_timer(cmd, "atmosphere_secondary");
                    lut.writable(t.secondary_sky);
                    lut.writable(t.secondary_range);
                    lut.writable(t.secondary_aerial);
                    const rhi::PushDescriptorWrite writes[] = {ubo,
                        rhi::PushDescriptorWrite::sampled_image(1, t.transmittance, sampler),
                        rhi::PushDescriptorWrite::sampled_image(2, t.multiscatter, sampler),
                        rhi::PushDescriptorWrite::storage_image(3, t.secondary_aerial),
                        rhi::PushDescriptorWrite::storage_image(4, t.secondary_sky),
                        rhi::PushDescriptorWrite::storage_image(5, t.secondary_range),
                        rhi::PushDescriptorWrite::sampled_image(10, t.secondary_source ? t.secondary_source : surface_lut_, sampler)};
                    const std::array source{local, glm::vec4(cache_radius, 0, 0, 0)};
                    ctx.push_constants(cmd, pipeline, source);
                    lut.bake(pipeline, writes, 12, 14);
                    lut.readable(t.secondary_sky);
                    lut.readable(t.secondary_range);
                    lut.readable(t.secondary_aerial);
                    ctx.end_gpu_timer(cmd, secondary_timer);
                    ++stats_.secondary_lut_updates;
                    t.secondary_direction = local;
                    t.secondary_altitude = p.camera.z;
                    t.secondary_valid = true;
                    t.secondary_approximate = false;
                    t.secondary_aerial_valid = settings.aerial_perspective_enabled;
                }
            }
        }
    }
}

}
