#include "rendering/atmosphere_internal.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "rendering/ring_renderer.hpp"
#include "rendering/refraction_tables.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace sf::rendering {

static_assert(sizeof(AtmosphereParametersGPU) == 14928);
static_assert(sizeof(AtmosphereParametersGPU) <= 16384);
static_assert(sizeof(AtmosphereShadowHeaderGPU) == 560);
static_assert(sizeof(AtmosphereShadowCasterGPU) == 48);

bool stellar_source_resolved(const AtmosphereParametersGPU& p) {
    double bottom = p.radii_heights.x, top = p.radii_heights.y;
    double step = std::sqrt((top-bottom)*(top+bottom))/63.0;
    auto resolved = [&](glm::vec3 extinction, double peak, double width) {
        double optical_depth = std::max({extinction.x, extinction.y, extinction.z})*width*std::sqrt(2.0*PI);
        if (optical_depth <= 0.005 || width <= 0.0) return true;
        double radius = std::clamp(double(p.reference_shape.x)+peak, bottom, top);
        double rho = std::sqrt((radius-bottom)*(radius+bottom));
        double increment = step*(2.0*rho+step);
        double spacing = increment/(std::sqrt(radius*radius+increment)+radius);
        return width >= 2.0*spacing;
    };
    if (p.profile_domain.w > 0.5f && !resolved(glm::vec3(p.ozone_absorption), p.ozone_params.x, p.ozone_params.y)) return false;
    for (int i = 0; i < int(p.aerosol_count.x); ++i)
        if (p.aerosol_shape[i].z > 0.0f && !resolved(glm::vec3(p.aerosol_extinction[i]), p.aerosol_shape[i].y, p.aerosol_shape[i].z)) return false;
    return true;
}

bool prepare_shadows(rhi::VkContext& ctx, rhi::Buffer& buffer, const ephemeris::SystemSnapshot& system,
                     uint32_t receiver, glm::dvec3 eye, float outer, float focal,
                     const RenderSettings& settings, std::span<const RingHostGPU> rings,
                     AtmosphereShadowHeaderGPU& previous, std::vector<AtmosphereShadowCasterGPU>& previous_casters,
                     uint32_t sequence, uint32_t& changes) {
    AtmosphereShadowHeaderGPU header;
    const auto origin = system.body(receiver).render.position.value;
    header.eye_focal = {glm::vec3((eye-origin)*AU_TO_KM), focal};
    header.flags = {settings.atmosphere_shadows_enabled ? 1u : 0u,
                    (settings.ring_shadow_single_box ? 1u : 0u) | (settings.ring_shadow_reference_linear ? 2u : 0u) |
                    (settings.atmosphere_shadow_cached_weights && settings.aerial_perspective_enabled ? 4u : 0u) |
                    (settings.atmosphere_shadow_fast_rings ? 8u : 0u) | (settings.atmosphere_shadow_analytic && !settings.atmosphere_shadow_temporal ? 16u : 0u) |
                    (uint32_t(std::bit_width(std::clamp(settings.atmosphere_shadow_vrs_scale, 2u, 16u))-1) << 8u),
                    settings.eclipse_reference_smoothstep ? 1u : 0u, settings.atmosphere_shadow_reference ? 1u : 0u};
    uint32_t effective_sequence = !settings.atmosphere_shadow_reference &&
        (settings.atmosphere_shadow_temporal || settings.atmosphere_shadow_animate_jitter) ? sequence : 0u;
    header.temporal = {effective_sequence, settings.atmosphere_shadow_noise_type,
                       settings.atmosphere_shadow_samples, settings.atmosphere_shadow_adaptive ? 1u : 0u};
    std::vector<AtmosphereShadowCasterGPU> casters;
    uint32_t star_index = 0;
    for (uint32_t s = 0; s < system.size() && star_index < header.stars.size(); ++s) {
        const auto star = system.body(s);
        if (!star.definition.is_star || glm::length(star.render.position.value-origin) < 1e-12) continue;
        auto& light = header.stars[star_index];
        light = {glm::vec3((star.render.position.value-origin)*AU_TO_KM), float(star.opaque_radius_au()*AU_TO_KM)};
        auto& range = header.ranges[star_index++];
        if (!settings.atmosphere_shadows_enabled || !system.illuminates(s, receiver)) continue;
        range.x = uint32_t(casters.size());
        for (uint32_t j = 0; j < system.size(); ++j) {
            const auto caster = system.body(j);
            if (j == receiver || caster.definition.is_star || caster.opaque_radius_au() <= 0.0) continue;
            glm::vec4 position(glm::vec3((caster.render.position.value-origin)*AU_TO_KM),
                        float(caster.opaque_radius_au()*AU_TO_KM));
            if (!eclipse_could_shadow({0, 0, 0, outer}, light, position)) continue;
            casters.push_back({position, {glm::vec3(caster.render.pole),
                position.w*float(std::clamp(1.0-caster.definition.oblateness, 0.05, 1.0))}, {}});
        }
        range.y = uint32_t(casters.size())-range.x;
        range.z = uint32_t(casters.size());
        if (settings.show_rings && settings.ring_shadows_enabled) {
            for (const auto& ring : rings) {
                uint32_t host = ring.counts_indices.x;
                if (host >= system.size() || ring.pole_max_r.w <= ring.center_min_r.w) continue;
                glm::vec3 center((system.body(host).render.position.value-origin)*AU_TO_KM);
                glm::vec3 pole(ring.pole_max_r);
                float inner = float(ring.center_min_r.w*AU_TO_KM), radius = float(ring.pole_max_r.w*AU_TO_KM);
                glm::vec4 receiver_eye(glm::vec3(origin-eye), outer*float(KM_TO_AU));
                glm::vec4 light_eye(glm::vec3(star.render.position.value-eye), light.w*float(KM_TO_AU));
                if (!ring_could_shadow_surface(receiver_eye, light_eye,
                    {glm::vec3(ring.center_min_r), ring.pole_max_r.w}, {pole, ring.center_min_r.w}, focal, settings.ring_shadow_softness)) continue;
                double light_distance = glm::length(glm::dvec3(light));
                double denominator = std::abs(glm::dot(glm::dvec3(light), glm::dvec3(pole)))-outer-light.w;
                float support = -1.0f;
                if (light_distance > outer && denominator > 1e-6*(light_distance+outer)) {
                    double minimum_mu = denominator/(light_distance+outer+light.w);
                    double hit_distance = (std::abs(glm::dot(glm::dvec3(center), glm::dvec3(pole)))+outer)/minimum_mu;
                    double stretch = 1.0/minimum_mu;
                    double stellar_blur = 2.0*std::max(settings.ring_shadow_softness, 1.0f)*light.w/(light_distance-outer)*hit_distance*stretch;
                    double pixel_blur = 0.75*(glm::length(glm::dvec3(header.eye_focal))+outer)/focal*stretch;
                    double profile_blur = 0.75*(radius-inner)/4096.0;
                    support = float(radius+std::max({stellar_blur, pixel_blur, profile_blur})+0.01);
                }
                casters.push_back({{center, radius}, {pole, inner}, {support, float(ring.counts_indices.y), settings.ring_shadow_softness, 0}});
            }
        }
        range.w = uint32_t(casters.size())-range.z;
    }
    if (casters.empty()) header.flags.x = 0;
    VkDeviceSize bytes = sizeof(header)+casters.size()*sizeof(AtmosphereShadowCasterGPU);
    if (buffer.size < bytes) {
        VkDeviceSize capacity = std::max(bytes, buffer.size*2);
        ctx.destroy_buffer(buffer);
        buffer = ctx.create_host_buffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }
    std::memcpy(buffer.mapped, &header, sizeof(header));
    if (!casters.empty()) std::memcpy(static_cast<std::byte*>(buffer.mapped)+sizeof(header), casters.data(), bytes-sizeof(header));
    ctx.flush_buffer(buffer, 0, bytes);
    changes = 0;
    if (glm::vec3(previous.eye_focal) != glm::vec3(header.eye_focal)) changes |= ShadowHistoryCamera;
    if (previous.eye_focal.w != header.eye_focal.w) changes |= ShadowHistoryProjection;
    if (previous.flags != header.flags || glm::uvec3(previous.temporal.y, previous.temporal.z, previous.temporal.w) !=
        glm::uvec3(header.temporal.y, header.temporal.z, header.temporal.w)) changes |= ShadowHistorySampling;
    if (previous.stars != header.stars) changes |= ShadowHistoryLighting;
    if (previous.ranges != header.ranges || previous_casters.size() != casters.size() || (!casters.empty() &&
        std::memcmp(previous_casters.data(), casters.data(), casters.size()*sizeof(AtmosphereShadowCasterGPU)) != 0))
        changes |= ShadowHistoryCasters;
    auto previous_key = previous;
    previous_key.temporal.x = header.temporal.x;
    if (changes == 0 && std::memcmp(&previous_key, &header, sizeof(header)) != 0) changes |= ShadowHistorySampling;
    previous = header;
    previous_casters = std::move(casters);
    return !previous_casters.empty();
}

AtmosphereParametersGPU make_atmosphere_parameters(const ephemeris::BodyDefinition& body,
    const std::shared_ptr<const physics::PreparedAtmosphere>& prepared, glm::dvec3 pole, double minimum) {
    const auto& a = body.atmosphere;
    float height = prepared ? prepared->height_km : a.height_km;
    double polar_ratio = std::clamp(1.0-body.oblateness, 0.05, 1.0);
    float radius = static_cast<float>(body.radius_au()*AU_TO_KM);
    AtmosphereParametersGPU p{};
    float bottom = std::max(radius*0.01f, radius+static_cast<float>(minimum/polar_ratio));
    if (a.pressure_reference) bottom = radius+float(prepared ? prepared->opaque_altitude_km : a.bottom_altitude_km);
    p.radii_heights = {bottom, radius+height, a.rayleigh_scale_height_km, a.mie_scale_height_km};
    p.reference_shape = {radius, static_cast<float>(polar_ratio), a.pressure_reference ? bottom : radius, 0.0f};
    p.world_pole = {glm::vec3(pole), static_cast<float>(1.0/polar_ratio-1.0)};
    p.rayleigh = {a.rayleigh_scattering, 0.0f};
    p.mie_scattering = {a.mie_scattering, a.mie_asymmetry};
    p.mie_extinction = {a.mie_extinction, 0.0f};
    p.ground_albedo = {a.ground_albedo, 0.0f};
    if (const auto& physical = prepared) {
        p.radii_heights.z = float(physical->surface.scale_height_km);
        p.radii_heights.w = physical->mie_scale_height_km;
        p.mie_scattering = {physical->mie_scattering, physical->mie_asymmetry};
        p.mie_extinction = {physical->mie_extinction, 0.0f};
        p.rayleigh = {physical->rayleigh, 0.0f};
        p.mixed_absorption = {physical->mixed_absorption, 0.0f};
        p.ozone_absorption = {physical->ozone_absorption, 0.0f};
        p.ozone_params = {float(a.ozone_peak_km), float(a.ozone_width_km), 0.0f, 0.0f};
        p.profile_domain = physical->profile_domain;
        p.log_density = physical->log_density;
        p.aerosol_count.x = float(physical->aerosols.size());
        for (size_t i = 0; i < physical->aerosols.size(); ++i) {
            const auto& layer = physical->aerosols[i];
            p.aerosol_shape[i] = layer.shape;
            p.aerosol_scattering[i] = layer.scattering;
            p.aerosol_extinction[i] = layer.extinction;
            std::copy(layer.phase.begin(), layer.phase.end(), p.aerosol_phase.begin()+i*physics::AEROSOL_PHASE_SAMPLES);
        }
        if (a.pressure_pa == 0) p.mie_scattering = p.mie_extinction = glm::vec4(0.0f);
    }
    return p;
}

AtmosphereFrame make_atmosphere_frame(const ephemeris::SystemSnapshot& system, uint32_t index, const Camera& camera,
                                      const RenderSettings& settings, VkExtent2D extent, const glm::dmat3& apparent,
                                      double candidate_distance, double focal, glm::dvec3 eye, std::span<const float> minimum_heights_km) {
    bool limb_active = false;
    const auto b = system.body(index);
    const auto& a = b.definition.atmosphere;
    float height = b.render.atmosphere ? b.render.atmosphere->height_km : a.height_km;
    double polar_ratio = std::clamp(1.0-b.definition.oblateness, 0.05, 1.0);
    glm::dvec3 pole = glm::normalize(b.render.pole);
    auto atmosphere_space = [&](glm::dvec3 v) { return v+pole*glm::dot(v, pole)*(1.0/polar_ratio-1.0); };
    glm::dvec3 relative = atmosphere_space((eye-b.render.position.value)*AU_TO_KM);
    double distance = glm::length(relative);
    glm::dvec3 z = distance > 1e-12 ? relative/distance : glm::dvec3(0.0, 1.0, 0.0);
    glm::dvec3 x = glm::normalize(glm::cross(std::abs(z.y) < 0.99 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0), z));
    glm::dvec3 y = glm::cross(z, x);
    double minimum = std::min(0.0, b.definition.height_range_km.x);
    if (index < minimum_heights_km.size()) minimum = std::min(minimum, double(minimum_heights_km[index]));
    auto p = make_atmosphere_parameters(b.definition, b.render.atmosphere, pole, minimum);
    float bottom = p.radii_heights.x;
    float radius = p.reference_shape.x;
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
        p.sun_direction[index] = {glm::vec3(glm::dot(direction, x), glm::dot(direction, y), glm::dot(direction, z)),
            float(std::asin(std::clamp(star.opaque_radius_au()/d, 1e-7, 1.0)))};
        float irradiance = settings.hdr_enabled ? static_cast<float>(std::max(star.definition.lum_eq, 0.0)/(d*d)) : 1.0f;
        if (!system.illuminates(s, index)) irradiance = 0.0f;
        p.sun_irradiance[index] = {star.definition.color_rgb*(irradiance*static_cast<float>(PI)), 0.0f};
        double star_radius = std::max(star.definition.radius_au(), 1e-12);
        float radiance = !system.illuminates(s, index) ? 0.0f :
            settings.hdr_enabled ? static_cast<float>(std::max(star.definition.lum_eq, 0.0)/(star_radius*star_radius)) : 1.0f;
        p.star_radiance[index] = {star.definition.color_rgb*radiance, 0.0f};
        glm::dvec3 sight = star.render.position.value-eye;
        double range = glm::length(sight);
        glm::dvec3 seen = range > 1e-12 ? sight/range : glm::normalize(delta);
        p.star_disc[index] = {glm::vec3(glm::dot(seen, x), glm::dot(seen, y), glm::dot(seen, z)),
            float(std::asin(std::clamp(star.opaque_radius_au()/std::max(range, 1e-12), 1e-7, 1.0)))};
    }
    auto projection = camera.proj_matrix(float(extent.width)/std::max(1u, extent.height), camera.compute_near_plane_au());
    auto view = camera.view_matrix_eye_relative();
    glm::dmat3 apparent_to_true = glm::transpose(apparent);
    p.inverse_view_projection = glm::mat4(glm::dmat4(apparent_to_true)*glm::dmat4(glm::inverse(projection*view)));
    glm::dmat3 view_to_world = apparent_to_true*glm::inverse(glm::dmat3(glm::mat3(view)));
    glm::dvec3 ray_scale{1.0/projection[0][0], 1.0/projection[1][1], -1.0};
    for (int k = 0; k < 3; ++k) {
        glm::dvec3 axis = k == 0 ? x : k == 1 ? y : z;
        glm::dvec3 row;
        for (int j = 0; j < 3; ++j) row[j] = glm::dot(axis, atmosphere_space(view_to_world[j]))*ray_scale[j];
        p.view_ray[k] = {glm::vec3(row), 1.0f};
    }
    p.depth_params.x = settings.hdr_enabled ? settings.exposure : 1.0f;
    p.depth_params.y = settings.aerial_perspective_enabled ? 1.0f : 0.0f;
    p.depth_params.z = settings.direct_sunlight_only ? 1.0f : 0.0f;
    double point_radius = b.opaque_radius_au()*AU_TO_KM/std::max(candidate_distance, 1e-12)*focal;
    float blend = float(std::clamp(point_radius-1.5, 0.0, 1.0));
    p.depth_params.w = blend*blend*(3.0f-2.0f*blend);
    limb_active = false;
    p.limb_image.z = 2.0f;
    if (distance > p.radii_heights.y && settings.aerial_perspective_enabled) {
        double limb = std::asin(std::clamp(double(p.radii_heights.x)/distance, 0.0, 1.0));
        double edge = std::asin(std::clamp(double(p.radii_heights.y)/distance, 0.0, 1.0));
        bool star_image = false;
        if (settings.refraction_enabled && b.render.atmosphere && b.render.atmosphere->refraction)
            for (uint32_t star = 0; star < uint32_t(p.camera.w); ++star) {
                glm::dvec3 seen{p.star_disc[star]};
                glm::dvec3 sun = atmosphere_space(seen.x*x+seen.y*y+seen.z*z);
                double separation = std::atan2(std::hypot(glm::dot(sun, x), glm::dot(sun, y)), -glm::dot(sun, z));
                star_image = star_image || (glm::any(glm::greaterThan(glm::vec3(p.star_radiance[star]), glm::vec3(0.0f))) &&
                                            separation-p.star_disc[star].w/polar_ratio <= edge);
            }
        if (star_image) {
            const auto& refraction = *b.render.atmosphere->refraction;
            double horizon = refraction.horizon_impact(distance);
            for (size_t row = 0; row < p.limb_bend.size(); ++row) {
                double t_row = double(row)/double(p.limb_bend.size()-1);
                double impact = distance*std::sin(limb+(edge-limb)*t_row*t_row);
                glm::dvec3 bend = refraction.total_deflection(impact);
                for (int channel = 0; channel < 3; ++channel) if (!std::isfinite(bend[channel])) bend[channel] = bend.y;
                bool visible = impact >= horizon && std::isfinite(bend.y);
                p.limb_bend[row] = {glm::vec3(bend), visible ? 1.0f : 0.0f};
                if (visible) p.limb_image.y = std::max(p.limb_image.y, float(std::max({bend.x, bend.y, bend.z})));
            }
            p.limb_image.x = 1.0f;
        }
        limb_active = star_image || (edge-limb)*focal < LIMB_RESOLVED_PIXELS;
        if (limb_active) {
            double margin = 1.5/std::max(focal, 1e-9);
            p.limb_image.z = float(std::sin(std::max(limb-margin, 0.0)));
            p.limb_image.w = float(std::sin(std::min(edge+margin, PI/2)));
        }
    }
    bool particle_scattering = std::any_of(p.aerosol_scattering.begin(), p.aerosol_scattering.end(),
        [](glm::vec4 coefficient) { return glm::any(glm::greaterThan(glm::vec3(coefficient), glm::vec3(0.0f))); });
    p.aerosol_count.z = particle_scattering ? 1.0f : 0.0f;
    p.aerosol_count.y = particle_scattering || glm::any(glm::greaterThan(glm::vec3(p.mie_scattering), glm::vec3(0.0f))) ? 1.0f : 0.0f;
    return {p, pole, x, y, z, polar_ratio, distance, point_radius, height, bottom, radius, projection, view, limb_active};
}

}
