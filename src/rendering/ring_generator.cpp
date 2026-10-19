#include "rendering/ring_generator.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace sf::rendering {

namespace {

inline float hash_int(int64_t n, uint32_t seed) noexcept {
    uint64_t u = static_cast<uint64_t>(n) * 0x45d9f3bULL + static_cast<uint64_t>(seed);
    uint32_t val = static_cast<uint32_t>(u & 0x7fffffffULL);
    val = ((val >> 16) ^ val) * 0x45d9f3bU;
    val = ((val >> 16) ^ val) * 0x45d9f3bU;
    val = (val >> 16) ^ val;
    val &= 0x7fffffffU;
    return (static_cast<float>(val) / 2147483647.0f) * 2.0f - 1.0f;
}

inline float perlin1d(float x, uint32_t seed) noexcept {
    float x0_f = std::floor(x);
    int64_t x0 = static_cast<int64_t>(x0_f);
    int64_t x1 = x0 + 1;
    float dx0 = x - x0_f;
    float dx1 = dx0 - 1.0f;

    float g0 = hash_int(x0, seed);
    float g1 = hash_int(x1, seed);

    float v0 = g0 * dx0;
    float v1 = g1 * dx1;

    float t = dx0;
    float s = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    return (1.0f - s) * v0 + s * v1;
}

inline float fbm1d(float x, int octaves, float lacunarity, float persistence, uint32_t seed) noexcept {
    float val = 0.0f;
    float amp = 1.0f;
    float freq = 1.0f;
    float max_amp = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        val += perlin1d(x * freq, seed + static_cast<uint32_t>(i) * 1013904223u) * amp;
        max_amp += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    return val / std::max(1e-6f, max_amp);
}

inline float smoothstep(float edge0, float edge1, float x) noexcept {
    float t = std::clamp((x - edge0) / std::max(1e-7f, edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

const std::array<RingPreset, 7> PRESET_TABLE{{
    {
        "Saturnian Ice",
        4, 3, 160.0f, 1.7f, 2.2f, 2.5f,
        1.20f, 2.30f,
        0.436f, -0.657f, 1.0f, 1.0f, 0.45f,
        {0.97f, 0.94f, 0.89f},
        {0.95f, 0.95f, 0.96f},
        {0.75f, 0.86f, 0.98f}
    },
    {
        "Jovian Silicate Dust",
        2, 1, 60.0f, 1.2f, 0.25f, 1.0f,
        1.25f, 1.85f,
        0.75f, -0.15f, 0.85f, 0.5f, 0.8f,
        {0.42f, 0.35f, 0.28f},
        {0.34f, 0.28f, 0.23f},
        {0.22f, 0.18f, 0.15f}
    },
    {
        "Uranian Charcoal",
        6, 5, 220.0f, 2.4f, 2.8f, 2.8f,
        1.45f, 2.05f,
        0.65f, -0.35f, 0.4f, 0.95f, 0.3f,
        {0.16f, 0.16f, 0.17f},
        {0.11f, 0.11f, 0.12f},
        {0.05f, 0.05f, 0.06f}
    },
    {
        "Neptunian Arcs",
        3, 2, 140.0f, 2.0f, 1.4f, 2.2f,
        1.60f, 2.55f,
        0.70f, -0.25f, 0.7f, 0.7f, 0.6f,
        {0.45f, 0.32f, 0.26f},
        {0.32f, 0.24f, 0.20f},
        {0.18f, 0.14f, 0.12f}
    },
    {
        "Chariklo Centaur",
        2, 1, 240.0f, 2.6f, 2.8f, 2.6f,
        1.80f, 2.15f,
        0.50f, -0.55f, 0.35f, 0.98f, 0.35f,
        {0.78f, 0.79f, 0.82f},
        {0.65f, 0.67f, 0.70f},
        {0.40f, 0.42f, 0.46f}
    },
    {
        "Warm Tholin Ice",
        4, 3, 150.0f, 1.8f, 2.2f, 2.4f,
        1.30f, 2.40f,
        0.48f, -0.50f, 0.55f, 1.0f, 0.4f,
        {0.95f, 0.72f, 0.46f},
        {0.82f, 0.54f, 0.32f},
        {0.55f, 0.30f, 0.18f}
    },
    {
        "Custom",
        4, 2, 160.0f, 1.8f, 2.2f, 2.5f,
        1.25f, 2.35f,
        0.55f, -0.50f, 0.5f, 1.0f, 0.5f,
        {0.95f, 0.95f, 0.95f},
        {0.85f, 0.85f, 0.85f},
        {0.60f, 0.70f, 0.80f}
    }
}};

} // namespace

std::span<const RingPreset> get_ring_presets() noexcept {
    return PRESET_TABLE;
}

const RingPreset& get_ring_preset(size_t index) noexcept {
    if (index >= PRESET_TABLE.size()) return PRESET_TABLE.back();
    return PRESET_TABLE[index];
}

const RingPreset& get_ring_preset(std::string_view name) noexcept {
    for (const auto& p : PRESET_TABLE) {
        if (p.name == name) return p;
    }
    return PRESET_TABLE.back();
}

ProceduralRingConfig ProceduralRingConfig::from_preset(const RingPreset& p, uint32_t seed) noexcept {
    ProceduralRingConfig cfg;
    cfg.seed = seed;
    cfg.band_count = p.band_count;
    cfg.gap_count = p.gap_count;
    cfg.striation_freq = p.striation_freq;
    cfg.contrast = p.contrast;
    cfg.base_density = p.base_density;
    cfg.core_boost = p.core_boost;
    cfg.inner_radius_ratio = p.inner_radius_ratio;
    cfg.outer_radius_ratio = p.outer_radius_ratio;
    cfg.opacity = p.opacity;
    cfg.asymmetry = p.asymmetry;
    cfg.backscatter = p.backscatter;
    cfg.scatter = p.scatter;
    cfg.unlit_factor = p.unlit_factor;
    cfg.color_deep = p.color_deep;
    cfg.color_mid = p.color_mid;
    cfg.color_thin = p.color_thin;
    return cfg;
}

ProceduralRingResult generate_procedural_ring_profile(
    const ProceduralRingConfig& cfg,
    uint32_t res) {
    uint32_t effective_res = std::max(64u, res);
    int band_count = std::clamp(cfg.band_count, 1, 8);
    int gap_count = std::clamp(cfg.gap_count, 0, 6);
    float striation_freq = std::clamp(cfg.striation_freq, 10.0f, 400.0f);
    float contrast = std::clamp(cfg.contrast, 0.5f, 4.0f);
    float base_density = std::clamp(cfg.base_density, 0.01f, 10.0f);
    float core_boost = std::clamp(cfg.core_boost, 0.1f, 6.0f);
    float inner_ratio = std::clamp(cfg.inner_radius_ratio, 1.05f, 20.0f);
    float outer_ratio = std::clamp(cfg.outer_radius_ratio, inner_ratio + 0.01f, 50.0f);
    float opacity = std::clamp(cfg.opacity, 0.0f, 10.0f);

    std::mt19937 rng(cfg.seed & 0x7fffffffu);
    auto rand_f = [&](float low, float high) -> float {
        std::uniform_real_distribution<float> dist(low, high);
        return dist(rng);
    };

    std::vector<float> cut_points;
    cut_points.reserve(band_count > 1 ? band_count - 1 : 0);
    for (int i = 0; i < band_count - 1; ++i) {
        cut_points.push_back(rand_f(0.12f, 0.88f));
    }
    std::sort(cut_points.begin(), cut_points.end());

    std::vector<float> band_edges;
    band_edges.reserve(band_count + 1);
    band_edges.push_back(0.0f);
    for (float cp : cut_points) band_edges.push_back(cp);
    band_edges.push_back(1.0f);

    std::vector<float> band_weights(band_count, 1.0f);
    for (int b = 0; b < band_count; ++b) {
        if (band_count == 1) {
            band_weights[b] = 1.0f * core_boost;
        } else if (band_count == 2) {
            band_weights[b] = (b == 0 ? 0.40f : 1.30f * core_boost) * rand_f(0.9f, 1.1f);
        } else if (band_count == 3) {
            float roles[3] = {0.30f, 1.40f * core_boost, 0.85f};
            band_weights[b] = roles[b] * rand_f(0.9f, 1.1f);
        } else {
            if (b == 0) {
                band_weights[b] = 0.28f * rand_f(0.85f, 1.15f);
            } else if (b == 1) {
                band_weights[b] = 1.45f * core_boost * rand_f(0.95f, 1.05f);
            } else if (b == band_count - 1 && band_count >= 5) {
                band_weights[b] = 1.60f * rand_f(0.9f, 1.1f);
            } else {
                band_weights[b] = rand_f(0.70f, 1.15f);
            }
        }
    }

    std::vector<float> macro_density(effective_res, 0.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        for (int b = 0; b < band_count; ++b) {
            float b_start = band_edges[b];
            float b_end = band_edges[b + 1];
            float b_width = b_end - b_start;
            float edge_softness = std::min(0.04f, b_width * 0.15f);
            float band_mask = smoothstep(b_start, b_start + edge_softness, u) *
                              (1.0f - smoothstep(b_end - edge_softness, b_end, u));
            float band_center = 0.5f * (b_start + b_end);
            float band_shape = 1.0f - std::abs(u - band_center) / (0.5f * b_width);
            band_shape = std::clamp(band_shape, 0.0f, 1.0f);
            band_shape = 0.7f + 0.3f * std::sin(band_shape * 1.57079632679f);
            macro_density[i] += band_mask * band_weights[b] * band_shape;
        }
        float boundary_fade = smoothstep(0.0f, 0.025f, u) * (1.0f - smoothstep(0.975f, 1.0f, u));
        macro_density[i] *= boundary_fade;
    }

    struct GapParam {
        float pos = 0.5f;
        float width = 0.01f;
        float sharpness = 0.25f;
        float residual = 0.01f;
    };
    std::vector<GapParam> gaps;
    if (gap_count > 0) {
        std::vector<float> candidate_pos;
        for (float b_cut : cut_points) {
            candidate_pos.push_back(b_cut + rand_f(-0.02f, 0.02f));
        }
        while (candidate_pos.size() < static_cast<size_t>(gap_count)) {
            candidate_pos.push_back(rand_f(0.15f, 0.85f));
        }
        std::sort(candidate_pos.begin(), candidate_pos.begin() + gap_count);
        for (int g = 0; g < gap_count; ++g) {
            bool is_major = (g == 0 && gap_count >= 2);
            GapParam gp;
            gp.pos = candidate_pos[g];
            gp.width = is_major ? rand_f(0.025f, 0.055f) : rand_f(0.005f, 0.015f);
            gp.sharpness = is_major ? rand_f(0.15f, 0.35f) : rand_f(0.25f, 0.5f);
            gp.residual = is_major ? 0.01f : rand_f(0.0f, 0.08f);
            gaps.push_back(gp);
        }
    }

    std::vector<float> gap_transmission(effective_res, 1.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        for (const auto& g : gaps) {
            float dist = std::abs(u - g.pos);
            float notch = smoothstep(g.width * 0.5f * (1.0f - g.sharpness), g.width * 0.5f, dist);
            gap_transmission[i] *= std::max(g.residual, notch);
        }
    }

    std::vector<float> striations(effective_res, 0.0f);
    float s_min = 1e9f;
    float s_max = -1e9f;
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float s_val = fbm1d(u * striation_freq, 7, 2.15f, 0.54f, cfg.seed);
        striations[i] = s_val;
        s_min = std::min(s_min, s_val);
        s_max = std::max(s_max, s_val);
    }
    float s_span = std::max(1e-6f, s_max - s_min);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float s_norm = (striations[i] - s_min) / s_span;
        s_norm = std::pow(s_norm, contrast);
        float micro_raw = perlin1d(u * (striation_freq * 4.2f), cfg.seed + 99991u);
        float micro = 0.5f + 0.5f * micro_raw;
        striations[i] = 0.85f * s_norm + 0.15f * micro;
    }

    std::vector<float> density_waves(effective_res, 0.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        for (const auto& g : gaps) {
            float dist_g = u - g.pos;
            float abs_dist = std::abs(dist_g);
            float wave_mask = smoothstep(0.0f, 0.01f, abs_dist) * std::exp(-abs_dist / 0.05f);
            float sign_g = (dist_g >= 0.0f) ? 1.0f : -1.0f;
            float chirp = std::cos(350.0f * sign_g * std::pow(abs_dist, 1.25f));
            density_waves[i] += wave_mask * chirp * 0.22f;
        }
    }

    std::vector<float> tau_vals(effective_res, 0.0f);
    std::vector<float> alpha_vals(effective_res, 0.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float striation_mod = 0.10f + 1.25f * striations[i];
        float tau = base_density * macro_density[i] * gap_transmission[i] * striation_mod +
                    density_waves[i] * macro_density[i];
        tau = std::max(0.0f, tau);
        tau_vals[i] = tau;
        alpha_vals[i] = std::clamp(1.0f - std::exp(-tau), 0.0f, 1.0f);
    }

    std::vector<glm::vec4> profile(effective_res);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float tau = tau_vals[i];
        float weight_mid = smoothstep(0.05f, 0.45f, tau);
        float weight_deep = smoothstep(0.45f, 1.50f, tau);

        glm::vec3 color_low = glm::mix(cfg.color_thin, cfg.color_mid, weight_mid);
        glm::vec3 color_full = glm::mix(color_low, cfg.color_deep, weight_deep);

        float color_var = 0.5f + 0.5f * perlin1d(u * 12.0f, cfg.seed + 7771u);
        float color_var_3 = 0.95f + 0.10f * color_var;

        glm::vec3 rgb = color_full * color_var_3 * cfg.tint_color;
        rgb = glm::clamp(rgb, 0.0f, 1.0f);

        profile[i] = glm::vec4(rgb, alpha_vals[i]);
    }

    ProceduralRingResult result{};
    result.profile = std::move(profile);
    result.inner_radius_ratio = inner_ratio;
    result.outer_radius_ratio = outer_ratio;
    result.opacity = opacity;
    result.scatter = cfg.scatter;
    result.asymmetry = cfg.asymmetry;
    result.backscatter = cfg.backscatter;
    result.unlit_factor = cfg.unlit_factor;
    result.raw_color = cfg.tint_color;
    return result;
}

void apply_procedural_ring_layer(
    const ProceduralRingResult& result,
    ephemeris::RingLayerDefinition& out_layer) noexcept {
    out_layer.inner = result.inner_radius_ratio;
    out_layer.outer = result.outer_radius_ratio;
    out_layer.color_rgb = result.raw_color;
    out_layer.opacity = result.opacity;
    out_layer.scatter = result.scatter;
    out_layer.asymmetry = result.asymmetry;
    out_layer.backscatter = result.backscatter;
    out_layer.unlit_factor = result.unlit_factor;
    out_layer.is_textured = true;
    out_layer.hue_shift = 0.0f;
    out_layer.saturation = 1.0f;
    out_layer.brightness = 1.0f;
    out_layer.alpha_boost = 1.0f;
    out_layer.gradient = {{0.0f, 1.0f}, {1.0f, 1.0f}};
}

bool bake_and_save_procedural_ring_texture(
    std::string_view system_name,
    std::string_view body_name,
    const ProceduralRingResult& result,
    ephemeris::RingLayerDefinition& out_layer,
    uint32_t texture_height) {
    if (body_name.empty() || result.profile.empty()) return false;
    std::string sys_folder = system_name.empty() ? "Solar System" : std::string(system_name);
    auto out_path = sf::paths::external_path("textures", sys_folder, body_name, "Rings.png");
    if (!save_ring_png_2d(out_path, result.profile, texture_height)) {
        return false;
    }
    apply_procedural_ring_layer(result, out_layer);
    return true;
}

} // namespace sf::rendering
