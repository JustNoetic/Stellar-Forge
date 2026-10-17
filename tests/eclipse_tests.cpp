#include "rendering/eclipse_candidates.hpp"
#include "rendering/ring_optics.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

bool fragment_shadowed(glm::vec3 point, glm::vec4 star, glm::vec4 caster) {
    glm::vec3 to_star = glm::vec3(star) - point;
    float distance = glm::length(to_star);
    if (distance <= 1e-9f) return false;
    glm::vec3 direction = to_star / distance;
    glm::vec3 offset = glm::vec3(caster) - point;
    float projection = glm::dot(offset, direction);
    if (projection <= 0.0f || projection >= distance) return false;
    float perpendicular = glm::length(offset - direction * projection);
    return perpendicular < caster.w + star.w * (projection / distance);
}

void ring_eclipse_candidate_contracts() {
    std::mt19937_64 random(0xA66A1u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    auto direction = [&]() {
        double z = 2.0 * unit(random) - 1.0;
        double angle = 6.283185307179586 * unit(random);
        double r = std::sqrt(1.0 - z * z);
        return glm::dvec3(r * std::cos(angle), r * std::sin(angle), z);
    };
    size_t shadowed = 0, rejected = 0;
    for (size_t trial = 0; trial < 6000; ++trial) {
        double outer = std::pow(10.0, -6.0 + 6.0 * unit(random));
        auto center = direction() * outer * (trial % 2 ? 10.0 : 100000.0);
        auto pole = direction();
        auto tangent = glm::normalize(glm::cross(pole, std::abs(pole.z) < 0.9 ? glm::dvec3(0, 0, 1) : glm::dvec3(1, 0, 0)));
        auto bitangent = glm::cross(pole, tangent);
        auto star_center = center + direction() * outer * (10.0 + 1000.0 * unit(random));
        glm::vec4 star{glm::vec3(star_center), float(outer * 0.2)};
        double angle = 6.283185307179586 * unit(random);
        auto target = center + (tangent * std::cos(angle) + bitangent * std::sin(angle)) * outer * (0.5 + 0.5 * unit(random));
        double caster_radius = outer * (0.001 + 0.1 * unit(random));
        auto caster_center = target + (star_center - target) * (0.1 + 0.8 * unit(random)) +
            direction() * (trial % 3 == 0 ? outer * 10.0 : caster_radius * 3.0);
        glm::vec4 caster{glm::vec3(caster_center), float(caster_radius)};
        bool candidate = sf::rendering::eclipse_could_shadow({glm::vec3(center), float(outer)}, star, caster);
        rejected += !candidate;
        for (size_t point = 0; point < 16; ++point) {
            double phi = point == 0 ? angle : 6.283185307179586 * unit(random);
            auto ring_point = point == 0 ? target : center +
                (tangent * std::cos(phi) + bitangent * std::sin(phi)) * outer * (point % 2 ? 1.0 : 0.5 + 0.5 * unit(random));
            if (fragment_shadowed(glm::vec3(ring_point), star, caster)) {
                ++shadowed;
                check(candidate, "Ring receiver bound lost a shadowed annulus sample");
            }
        }
    }
    check(shadowed > 1000 && rejected > 100, "Ring shadow oracle covers retention and rejection across poses and scales");
}

void ring_profile_mip_contracts() {
    using namespace sf::rendering;
    check(build_ring_profile_mips({}).empty(), "Empty ring profiles produce no mip levels");
    std::array<glm::vec4, 8> profile;
    for (size_t i = 0; i < profile.size(); ++i)
        profile[i] = i % 2 ? glm::vec4(0, 0, 1, 0) : glm::vec4(1, 0, 0, 1);
    auto mips = build_ring_profile_mips(profile);
    check(mips.size() == 4 && mips.back().size() == 1, "Ring profile pyramid reaches one texel");
    for (size_t level = 1; level < mips.size(); ++level) {
        check(mips[level].size() == profile.size() >> level, "Ring mip widths follow Vulkan mip extents");
        for (auto texel : mips[level])
            check(glm::length(texel - glm::vec4(0.5f, 0, 0, 0.5f)) < 1e-6f,
                  "Transparent blue does not bleed into opaque red at any mip level");
    }
    profile.fill(glm::vec4(0.5f, 0.25f, 0.75f, 0.4f));
    mips = build_ring_profile_mips(profile);
    auto expected = glm::pow(glm::vec3(profile[0]), glm::vec3(2.2f));
    for (const auto& level : mips)
        for (auto texel : level)
            check(glm::length(glm::vec3(texel) / texel.a - expected) < 1e-6f && std::abs(texel.a - 0.4f) < 1e-6f,
                  "All mip levels preserve uniform linear-light color and opacity");
    profile = {{{0.1f, 0.2f, 0.3f, 0.1f}, {0.2f, 0.7f, 0.4f, 0.4f}, {1, 1, 1, 1}, {0, 0, 0, 0},
                {0.8f, 0.2f, 0.1f, 0.2f}, {0.3f, 0.4f, 0.9f, 0.8f}, {1, 0, 0, 0.6f}, {0, 1, 0, 0.3f}}};
    mips = build_ring_profile_mips(profile);
    glm::vec4 average{0};
    for (auto sample : mips.front()) average += sample / float(profile.size());
    for (const auto& level : mips) {
        glm::vec4 level_average{0};
        for (auto sample : level) level_average += sample / float(level.size());
        check(glm::length(level_average - average) < 1e-6f, "Ring mip reduction conserves premultiplied color and opacity");
    }
}

void ring_draw_contracts() {
    using namespace sf::rendering;
    std::mt19937_64 random(0xBADD1E5u);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    size_t contributing = 0;
    for (uint32_t trial = 0; trial < 300; ++trial) {
        double radius = std::pow(10.0, -8.0 + 6.0 * unit(random));
        std::array<sf::ephemeris::RingLayerDefinition, 8> layers;
        for (auto& layer : layers) {
            layer.inner = 1.0 + 1.6 * unit(random);
            layer.outer = layer.inner + std::pow(10.0, -5.0 + 4.8 * unit(random));
            layer.opacity = unit(random) < 0.2 ? 0.0f : 0.5f;
        }
        double lo = 10.0, hi = 0.0;
        for (const auto& layer : layers) { lo = std::min(lo, layer.inner); hi = std::max(hi, layer.outer); }
        auto ranges = build_ring_draw_ranges(layers, lo, hi, radius);
        std::array<uint32_t, RING_RADIAL_SUBDIVISIONS> masks{};
        uint32_t previous_end = 0;
        constexpr uint32_t strip_indices = RING_ANGULAR_SEGMENTS * 6;
        for (const auto& range : ranges) {
            check(range.first_index >= previous_end && range.index_count > 0 &&
                  range.first_index % strip_indices == 0 && range.index_count % strip_indices == 0,
                  "Radial draws are ordered disjoint mesh strips");
            previous_end = range.first_index + range.index_count;
            check(previous_end <= strip_indices * RING_RADIAL_SUBDIVISIONS, "Radial draw fits the index buffer");
            uint32_t decoded = 0, previous = 0;
            for (uint32_t slot = 0; slot < range.segment_count; ++slot) {
                uint32_t layer = (range.packed_segments >> (slot * 3)) & 7u;
                check(slot == 0 || layer > previous, "Packed layers preserve optical-depth accumulation order");
                previous = layer;
                decoded |= 1u << layer;
            }
            check(decoded == range.segment_mask, "Packed layers match the conservative candidate set");
            for (uint32_t strip = range.first_index / strip_indices; strip < previous_end / strip_indices; ++strip)
                masks[strip] = decoded;
        }
        double angle = 6.283185307179586 / RING_ANGULAR_SEGMENTS;
        glm::dvec2 direction{std::cos(angle), std::sin(angle)};
        for (uint32_t strip = 0; strip < RING_RADIAL_SUBDIVISIONS; ++strip) {
            double inner = std::lerp(lo, hi, double(strip) / RING_RADIAL_SUBDIVISIONS);
            double outer = std::lerp(lo, hi, double(strip + 1) / RING_RADIAL_SUBDIVISIONS);
            for (uint32_t sample = 0; sample < 24; ++sample) {
                double a = unit(random), b = unit(random);
                if (a + b > 1.0) { a = 1.0 - a; b = 1.0 - b; }
                glm::dvec2 p0 = sample % 2 ? direction * inner : glm::dvec2(inner, 0.0);
                glm::dvec2 p1{outer, 0.0};
                glm::dvec2 p2 = direction * (sample % 2 ? outer : inner);
                float r = float((glm::length(p0 * (1.0 - a - b) + p1 * a + p2 * b) +
                                 (unit(random) - 0.5) * hi * 0.0018) * radius);
                for (uint32_t layer = 0; layer < layers.size(); ++layer) {
                    float start = float(layers[layer].inner * radius), end = float(layers[layer].outer * radius);
                    float dr = std::max(end - start, 1e-9f) * 0.05f;
                    if (layers[layer].opacity <= 0.0f || r < start - dr || r > end + dr) continue;
                    ++contributing;
                    check((masks[strip] & (1u << layer)) != 0, "Radial draw omitted a potentially contributing triangle sample");
                }
            }
        }
    }
    check(contributing > 10000, "Radial oracle exercises overlaps, narrow bands, empty gaps and small physical radii");

    glm::vec3 center{0, -5, 0}, pole{0, 1, 0};
    std::array<glm::vec4, 2> stars{{{10, -3, 0, 0.1f}, {10, -7, 0, 0.1f}}};
    check(classify_ring_draw(center, pole, 1, std::span(stars.data(), 1)).lighting == RingLightingMode::Reflection,
          "Star and camera above the plane use reflection");
    check(classify_ring_draw(center, pole, 1, std::span(stars.data() + 1, 1)).lighting == RingLightingMode::Transmission,
          "Opposite sides use transmission");
    check(classify_ring_draw(center, pole, 1, stars).lighting == RingLightingMode::Point, "Mixed star sides retain the dynamic side path");
    stars[0].y = -4.95f;
    check(classify_ring_draw(center, pole, 1, stars).lighting == RingLightingMode::FiniteSource, "A disc crossing the plane retains quadrature");
    check(!classify_ring_draw({5, 0.001f, 0}, pole, 1, stars).radial_ranges, "Grazing camera retains the full annulus");
    size_t specialized = 0;
    for (uint32_t trial = 0; trial < 4000; ++trial) {
        auto direction = [&]() { return glm::normalize(glm::dvec3(unit(random) - 0.5, unit(random) - 0.5, unit(random) - 0.5)); };
        double radius = std::pow(10.0, -7.0 + 6.0 * unit(random));
        glm::vec3 normal = glm::normalize(glm::vec3(direction()));
        glm::vec3 origin(direction() * radius * std::pow(10.0, 1.0 + 5.0 * unit(random)));
        auto tangent = glm::normalize(glm::cross(normal, std::abs(normal.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0)));
        auto bitangent = glm::cross(normal, tangent);
        float star_radius = float(radius * 0.2);
        glm::vec4 light{origin + tangent * float(radius * 100.0) + normal * star_radius * float(trial % 5 - 2.0), star_radius};
        auto classification = classify_ring_draw(origin, normal, float(radius), {&light, 1});
        if (classification.lighting == RingLightingMode::FiniteSource) continue;
        ++specialized;
        for (uint32_t point = 0; point < 32; ++point) {
            float phi = float(6.283185307179586 * unit(random));
            glm::vec3 position = origin + (tangent * std::cos(phi) + bitangent * std::sin(phi)) * float(radius);
            auto to_star = glm::vec3(light) - position;
            float distance = glm::length(to_star);
            if (distance <= 1e-5f) continue;
            float sun_side = glm::dot(normal, to_star / distance);
            float star_angle = std::clamp(light.w / distance, 0.0f, 0.999f);
            check(!(star_angle > 1e-6f && std::abs(sun_side) < star_angle), "Specialization lost a finite-source fragment");
            bool same_side = glm::dot(normal, glm::normalize(-position)) * sun_side >= 0.0f;
            if (classification.lighting != RingLightingMode::Point)
                check(same_side == (classification.lighting == RingLightingMode::Reflection), "Specialization changed the fragment side");
        }
    }
    check(specialized > 100, "Lighting oracle exercises specialized fragments across scales and poses");
}

void ring_radiative_transfer_contracts() {
    using namespace sf::rendering;

    // 1. 3-parameter Double Henyey-Greenstein 4*pi normalization: (1/2) * int_{-1}^{1} P(mu) dmu == 1
    const int N = 20000;
    const double dmu = 2.0 / double(N);
    for (bool is_tex : {false, true}) {
        for (float scatter : {0.0f, 0.25f, 0.5f, 0.85f, 1.0f}) {
            for (float tau : {0.05f, 0.5f, 2.0f}) {
                double integral = 0.0;
                for (int i = 0; i < N; ++i) {
                    float cos_theta = static_cast<float>(-1.0 + (i + 0.5) * dmu);
                    integral += ring_phase_dhg(cos_theta, scatter, 0.75f, -0.45f, tau, is_tex) * dmu;
                }
                double norm = 0.5 * integral;
                check(std::abs(norm - 1.0) < 1e-3, "DHG phase function integrates to unity over [-1, 1]");
            }
        }
    }

    // Verify scatter weight controls forward vs backward dominance even on textured rings
    float p_fwd_low = ring_phase_dhg(0.95f, 0.1f, 0.8f, -0.45f, 0.8f, true);
    float p_fwd_high = ring_phase_dhg(0.95f, 0.9f, 0.8f, -0.45f, 0.8f, true);
    check(p_fwd_high > p_fwd_low * 2.0f, "Scatter weight modulates forward diffraction peak on textured rings");

    float p_back_low = ring_phase_dhg(-0.95f, 0.1f, 0.8f, -0.45f, 0.8f, true);
    float p_back_high = ring_phase_dhg(-0.95f, 0.9f, 0.8f, -0.45f, 0.8f, true);
    check(p_back_low > p_back_high * 2.0f, "Lower scatter weight increases backscatter lobe on textured rings");

    // 2. Single-scattering slab transfer: limits, continuity at mu_s == mu_v, and Helmholtz reciprocity
    check(ring_transfer(0.0f, 0.5f, 0.5f, true) == 0.0f, "Zero optical depth has zero reflection");
    check(ring_transfer(0.0f, 0.5f, 0.5f, false) == 0.0f, "Zero optical depth has zero transmission");

    float t_minus = ring_transfer(1.2f, 0.6f, 0.6f - 1e-4f, false);
    float t_exact = ring_transfer(1.2f, 0.6f, 0.6f, false);
    float t_plus  = ring_transfer(1.2f, 0.6f, 0.6f + 1e-4f, false);
    check(std::abs(t_exact - 0.5f * (t_minus + t_plus)) < 1e-5f, "Transmission slab transfer is continuous at mu_s == mu_v");

    for (bool same_side : {true, false}) {
        float mu_a = 0.35f, mu_b = 0.82f, tau = 1.4f;
        float rab = mu_b * ring_transfer(tau, mu_a, mu_b, same_side);
        float rba = mu_a * ring_transfer(tau, mu_b, mu_a, same_side);
        check(std::abs(rab - rba) < 1e-6f, "Single-scattering slab satisfies Helmholtz reciprocity");
    }

    // 3. Analytic finite-slab Chandrasekhar X/Y multiple scattering
    glm::vec3 albedo{0.8f, 0.7f, 0.5f};
    glm::vec3 ms_zero = ring_multiple_scattering(0.0f, 0.5f, 0.5f, true, albedo);
    check(glm::length(ms_zero) < 1e-6f, "Multiple scattering vanishes at zero optical depth");

    glm::vec3 ms_thin = ring_multiple_scattering(1e-3f, 0.5f, 0.5f, false, albedo);
    check(glm::length(ms_thin) < 1e-3f, "Multiple scattering is negligible in optically very thin rings");

    glm::vec3 ms_mid_trans = ring_multiple_scattering(1.0f, 0.5f, 0.6f, false, albedo);
    check(ms_mid_trans.r > 0.01f && ms_mid_trans.g > 0.01f && ms_mid_trans.b > 0.005f,
          "Intermediate optical depth produces positive diffuse transmission on unlit side");

    glm::vec3 ms_thick_refl = ring_multiple_scattering(25.0f, 0.5f, 0.6f, true, albedo);
    glm::vec3 ms_thick_trans = ring_multiple_scattering(25.0f, 0.5f, 0.6f, false, albedo);
    check(ms_thick_refl.r > 0.03f && ms_thick_refl.r > ms_mid_trans.r,
          "Optically thick slab retains positive multiple-scattered reflection on lit side");
    check(glm::length(ms_thick_trans) < 1e-5f,
          "Optically thick slab attenuates transmitted multiple scattering on unlit side without leakage");

    glm::vec4 sat_props{0.436f, -0.657f, 0.35f, 1.0f};
    glm::vec3 c_ring_rgb{0.25f, 0.20f, 0.17f};
    glm::vec3 b_ring_rgb{0.87f, 0.64f, 0.44f};
    glm::vec3 lit_c = ring_radiance_rgb(0.22f, 0.35f, 0.22f, -0.85f, sat_props, c_ring_rgb, true, true);
    glm::vec3 lit_b = ring_radiance_rgb(1.60f, 0.35f, 0.22f, -0.85f, sat_props, b_ring_rgb, true, true);
    glm::vec3 unlit_c = ring_radiance_rgb(0.22f, 0.35f, 0.22f, 0.35f, sat_props, c_ring_rgb, true, false);
    glm::vec3 unlit_b = ring_radiance_rgb(1.60f, 0.35f, 0.22f, 0.35f, sat_props, b_ring_rgb, true, false);
    check(glm::length(lit_b) > glm::length(lit_c) * 2.5f,
          "Dense B-ring is significantly brighter than optically thin C-ring on the lit side");
    check(glm::length(unlit_c) > glm::length(unlit_b) * 4.0f,
          "Near-opaque B-ring is significantly darker than optically thin C-ring on the unlit side");
    check(unlit_b.r / std::max(unlit_b.b, 1e-6f) < 2.0f,
          "Unlit dense B-ring avoids excessive orange chromatic compounding");

    // 4. Opposition surge (SHOE + CBOE) monotonicity and boundedness
    float opp_90 = ring_opposition_factor(0.0f, 1.0f);
    float opp_10 = ring_opposition_factor(std::cos(10.0f * 0.0174532925f), 1.0f);
    float opp_1  = ring_opposition_factor(std::cos(1.0f * 0.0174532925f), 1.0f);
    float opp_0  = ring_opposition_factor(1.0f, 1.0f);
    check(std::abs(opp_90 - 1.0f) < 0.015f, "Opposition factor is near 1 away from opposition");
    check(opp_0 > opp_1 && opp_1 > opp_10 && opp_10 > opp_90, "Opposition surge increases monotonically toward zero phase angle");

    // 5. Shadow LUT prefix integral monotonicity and oblique angle dependence
    sf::ephemeris::RingLayerDefinition layer{};
    layer.inner = 1.5;
    layer.outer = 2.5;
    layer.opacity = 0.8f;
    layer.gradient = {{0.0f, 0.2f}, {0.5f, 0.9f}, {1.0f, 0.4f}};
    auto seg_prof = generate_ring_segment_profile(layer, {}, 256);
    RingSegmentProfileInput input{layer.inner, layer.outer, layer.opacity, seg_prof};
    auto unified = bake_unified_shadow_profile(std::span{&input, 1}, layer.inner, layer.outer, 256);
    auto lut = bake_ring_shadow_lut_layer(unified, 256, 16);
    check(lut.size() == 16 * 257, "Shadow LUT dimensions match (width + 1) * angles");
    for (uint32_t a = 0; a < 16; ++a) {
        size_t row = a * 257;
        check(lut[row] == 0.0f, "Prefix integral starts at 0");
        for (uint32_t i = 0; i < 256; ++i) {
            check(lut[row + i + 1] >= lut[row + i], "Shadow LUT prefix integral is monotonically non-decreasing");
        }
    }
    check(lut[0 * 257 + 256] > lut[15 * 257 + 256],
          "Grazing solar elevation (row 0) produces stronger total ring shadow extinction than high solar elevation (row 15)");
}

}

int main() {
    try {
        using sf::rendering::eclipse_could_shadow;
        glm::vec4 receiver{0, 0, 0, 1}, star{0, 0, 100, 5};
        check(eclipse_could_shadow(receiver, star, {0, 0, 10, 0.25f}), "Aligned blocker retained");
        check(eclipse_could_shadow(receiver, star, {1.2f, 0, 10, 0.25f}), "Limb penumbra retained");
        check(!eclipse_could_shadow(receiver, star, {30, 0, 10, 0.25f}), "Lateral blocker rejected");
        check(!eclipse_could_shadow(receiver, star, {0, 0, 120, 0.25f}), "Blocker beyond star rejected");
        check(!eclipse_could_shadow(receiver, star, {0, 0, -20, 0.25f}), "Blocker behind receiver rejected");
        check(eclipse_could_shadow(receiver, {0, 0, 0, 5}, {0, 0, 10, 1}), "Overlapping light conservatively retained");
        std::mt19937_64 random(0xEC11F5E);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        auto direction = [&]() {
            double z = 2.0 * unit(random) - 1.0;
            double angle = 6.283185307179586 * unit(random);
            double r = std::sqrt(1.0 - z * z);
            return glm::dvec3(r * std::cos(angle), r * std::sin(angle), z);
        };
        size_t shadowed = 0, rejected = 0;
        for (size_t trial = 0; trial < 30000; ++trial) {
            double scale = std::pow(10.0, -6.0 + 9.0 * unit(random));
            double distance = scale * std::pow(10.0, -0.3 + 5.0 * unit(random));
            auto axis = direction();
            auto center = direction() * distance * (trial % 3 == 0 ? 100.0 : 0.1);
            glm::vec4 target{glm::vec3(center), float(scale)};
            glm::vec4 light{glm::vec3(center + axis * distance), float(distance * unit(random) * 0.1)};
            double along = -0.25 + 1.5 * unit(random);
            double caster_radius = scale * std::pow(10.0, -2.0 + 3.0 * unit(random));
            auto position = center + axis * distance * along + direction() * scale * 6.0 * unit(random);
            glm::vec4 blocker{glm::vec3(position), float(caster_radius)};
            bool candidate = eclipse_could_shadow(target, light, blocker);
            rejected += !candidate;
            for (size_t sample = 0; sample < 64; ++sample) {
                auto point = glm::vec3(center + direction() * scale * (sample % 2 ? 1.0 : unit(random)));
                if (fragment_shadowed(point, light, blocker)) {
                    ++shadowed;
                    check(candidate, "Candidate filter lost a shadowed surface/interior sample");
                }
            }
        }
        check(shadowed > 1000 && rejected > 1000, "Randomized oracle covers shadow retention and rejection");
        ring_eclipse_candidate_contracts();
        ring_draw_contracts();
        ring_profile_mip_contracts();
        ring_radiative_transfer_contracts();
        std::cout << "Eclipse bounds and ring radiative transfer contracts passed: " << shadowed
                  << " shadowed samples retained; " << rejected << " configurations rejected.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
