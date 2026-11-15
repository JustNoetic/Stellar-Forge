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

glm::vec3 rgb_to_hsv(glm::vec3 c) noexcept {
    float maxc = std::max({c.r, c.g, c.b});
    float minc = std::min({c.r, c.g, c.b});
    float d = maxc - minc;
    float h = 0.0f;
    if (d > 1e-6f) {
        if (maxc == c.r) h = std::fmod((c.g - c.b) / d, 6.0f);
        else if (maxc == c.g) h = (c.b - c.r) / d + 2.0f;
        else h = (c.r - c.g) / d + 4.0f;
        h /= 6.0f;
        if (h < 0.0f) h += 1.0f;
    }
    float s = (maxc > 1e-6f) ? (d / maxc) : 0.0f;
    return {h, s, maxc};
}

glm::vec3 hsv_to_rgb(glm::vec3 hsv) noexcept {
    float h = hsv.x - std::floor(hsv.x);
    float s = std::clamp(hsv.y, 0.0f, 1.0f);
    float v = std::clamp(hsv.z, 0.0f, 1.0f);
    float h6 = h * 6.0f;
    int i = static_cast<int>(std::floor(h6)) % 6;
    float f = h6 - std::floor(h6);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
    }
}

glm::vec3 jitter_color(glm::vec3 rgb, float dh, float ds, float dv) noexcept {
    glm::vec3 hsv = rgb_to_hsv(glm::clamp(rgb, 0.0f, 1.0f));
    hsv.x = std::fmod(hsv.x + dh + 1.0f, 1.0f);
    hsv.y = std::clamp(hsv.y + ds, 0.0f, 1.0f);
    hsv.z = std::clamp(hsv.z + dv, 0.02f, 1.0f);
    return hsv_to_rgb(hsv);
}

const std::array<RingPreset, 7> PRESET_TABLE{{
    {
        "Saturnian Ice",
        4, 3, 160.0f, 1.7f, 1.15f, 1.65f,
        1.20f, 2.30f,
        0.436f, -0.657f, 0.68f, 1.0f, 0.75f,
        {0.97f, 0.93f, 0.86f},
        {0.92f, 0.92f, 0.94f},
        {0.72f, 0.84f, 0.98f},
        {
            0.25f, 0.82f, 0.18f, 0.68f, -0.82f, -0.36f, 0.45f, 1.15f,
            1.14f, 1.40f, 2.00f, 2.80f, 3, 7, 2, 6,
            100.0f, 290.0f, 1.2f, 2.8f, 0.65f, 2.10f, 1.1f, 2.4f, 0.85f, 1.15f,
            0.06f, 0.18f, 0.12f, 0.70f, 0.014f
        }
    },
    {
        "Jovian Silicate Dust",
        2, 1, 60.0f, 1.2f, 0.22f, 1.0f,
        1.25f, 1.85f,
        0.78f, -0.15f, 0.85f, 0.45f, 0.90f,
        {0.42f, 0.35f, 0.28f},
        {0.34f, 0.28f, 0.23f},
        {0.22f, 0.18f, 0.15f},
        {
            0.65f, 0.96f, 0.62f, 0.94f, -0.32f, -0.04f, 0.65f, 1.35f,
            1.15f, 1.55f, 1.65f, 2.45f, 1, 4, 0, 2,
            25.0f, 120.0f, 0.7f, 1.7f, 0.08f, 0.60f, 0.8f, 1.6f, 0.15f, 0.75f,
            0.05f, 0.18f, 0.14f, 0.85f, 0.025f
        }
    },
    {
        "Uranian Charcoal",
        6, 5, 220.0f, 2.4f, 1.45f, 1.85f,
        1.45f, 2.05f,
        0.58f, -0.35f, 0.40f, 0.95f, 0.40f,
        {0.16f, 0.16f, 0.17f},
        {0.11f, 0.11f, 0.12f},
        {0.05f, 0.05f, 0.06f},
        {
            0.18f, 0.62f, 0.28f, 0.75f, -0.58f, -0.18f, 0.22f, 0.65f,
            1.30f, 1.70f, 1.85f, 2.65f, 4, 8, 3, 6,
            150.0f, 340.0f, 1.8f, 3.4f, 0.80f, 2.60f, 1.2f, 2.8f, 0.70f, 1.10f,
            0.05f, 0.10f, 0.07f, 0.45f, 0.006f
        }
    },
    {
        "Neptunian Arcs",
        3, 2, 140.0f, 2.0f, 0.85f, 1.55f,
        1.60f, 2.55f,
        0.70f, -0.25f, 0.68f, 0.65f, 0.65f,
        {0.45f, 0.32f, 0.26f},
        {0.32f, 0.24f, 0.20f},
        {0.18f, 0.14f, 0.12f},
        {
            0.45f, 0.85f, 0.48f, 0.88f, -0.45f, -0.10f, 0.40f, 0.95f,
            1.35f, 1.85f, 2.10f, 3.10f, 2, 6, 1, 5,
            80.0f, 240.0f, 1.4f, 3.0f, 0.35f, 1.60f, 1.0f, 2.3f, 0.35f, 0.90f,
            0.06f, 0.20f, 0.12f, 0.65f, 0.014f
        }
    },
    {
        "Chariklo Centaur",
        2, 1, 240.0f, 2.6f, 1.40f, 1.80f,
        1.80f, 2.15f,
        0.48f, -0.55f, 0.36f, 0.98f, 0.42f,
        {0.78f, 0.79f, 0.82f},
        {0.65f, 0.67f, 0.70f},
        {0.40f, 0.42f, 0.46f},
        {
            0.18f, 0.58f, 0.25f, 0.68f, -0.72f, -0.32f, 0.25f, 0.72f,
            1.55f, 2.35f, 1.95f, 3.10f, 1, 4, 0, 3,
            140.0f, 340.0f, 1.8f, 3.4f, 0.85f, 2.50f, 1.2f, 2.7f, 0.75f, 1.15f,
            0.06f, 0.16f, 0.12f, 0.35f, 0.005f
        }
    },
    {
        "Warm Tholin Ice",
        4, 3, 150.0f, 1.8f, 1.15f, 1.65f,
        1.30f, 2.40f,
        0.48f, -0.50f, 0.56f, 1.0f, 0.55f,
        {0.95f, 0.72f, 0.46f},
        {0.82f, 0.54f, 0.32f},
        {0.55f, 0.30f, 0.18f},
        {
            0.28f, 0.76f, 0.30f, 0.74f, -0.70f, -0.28f, 0.35f, 0.95f,
            1.15f, 1.50f, 1.95f, 2.85f, 2, 7, 1, 6,
            90.0f, 260.0f, 1.2f, 2.8f, 0.60f, 2.10f, 1.1f, 2.5f, 0.75f, 1.15f,
            0.07f, 0.22f, 0.14f, 0.65f, 0.015f
        }
    },
    {
        "Custom",
        4, 2, 160.0f, 1.8f, 1.15f, 1.65f,
        1.25f, 2.35f,
        0.55f, -0.50f, 0.55f, 1.0f, 0.60f,
        {0.95f, 0.95f, 0.95f},
        {0.85f, 0.85f, 0.85f},
        {0.60f, 0.70f, 0.80f},
        {
            0.15f, 0.90f, 0.20f, 0.90f, -0.80f, -0.10f, 0.25f, 1.20f,
            1.12f, 1.80f, 1.80f, 3.20f, 1, 8, 0, 6,
            40.0f, 320.0f, 0.8f, 3.2f, 0.25f, 2.50f, 0.9f, 2.6f, 0.35f, 1.25f,
            0.10f, 0.25f, 0.16f, 0.60f, 0.018f,
            0.35f, 1.65f, 0.30f, 1.65f, 0.30f, 1.65f
        }
    }
}};

enum class RingMorphology {
    ClassicMultiZone = 0,
    SparseRinglets = 1,
    DiffuseDust = 2,
    CoreWithFilaments = 3
};

struct SubBand {
    float start = 0.0f;
    float end = 1.0f;
    float density_mult = 1.0f;
    float slope = 0.0f;
    float edge_soft = 0.005f;
    glm::vec3 tint{1.0f};
};

struct BandSpec {
    float start = 0.0f;
    float end = 1.0f;
    float weight = 1.0f;
    float inner_soft = 0.02f;
    float outer_soft = 0.02f;
    float skew = 0.0f;
    float flat_top = 0.5f;
    float freq_mult = 1.0f;
    float contrast_mult = 1.0f;
    float smoothness = 0.25f;
    glm::vec3 tint{1.0f};
    std::vector<SubBand> sub_bands;
};

struct GapParam {
    float pos = 0.5f;
    float width = 0.01f;
    float sharpness = 0.25f;
    float residual = 0.01f;
    int wave_side = 0;
    float wave_freq = 350.0f;
    float wave_decay = 0.045f;
    float wave_pow = 1.25f;
    float wave_amp = 0.20f;
};

RingMorphology choose_morphology(const ProceduralRingConfig& cfg, int band_count, std::mt19937& rng) noexcept {
    if (cfg.morphology >= 0 && cfg.morphology <= 3) {
        return static_cast<RingMorphology>(cfg.morphology);
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    const auto& lim = cfg.limits;
    if (band_count <= 1) {
        if (lim.density_max <= 1.0f || lim.scatter_min >= 0.60f) {
            return RingMorphology::DiffuseDust;
        }
        return (r < 0.78f) ? RingMorphology::ClassicMultiZone : RingMorphology::DiffuseDust;
    }
    if (lim.density_max <= 1.0f || lim.scatter_min >= 0.60f) {
        return (r < 0.68f) ? RingMorphology::DiffuseDust : RingMorphology::CoreWithFilaments;
    }
    if (lim.contrast_min >= 1.7f && lim.striation_min >= 130.0f) {
        if (r < 0.58f) return RingMorphology::SparseRinglets;
        if (r < 0.86f) return RingMorphology::CoreWithFilaments;
        return RingMorphology::ClassicMultiZone;
    }
    if (r < 0.44f) return RingMorphology::ClassicMultiZone;
    if (r < 0.74f) return RingMorphology::CoreWithFilaments;
    if (r < 0.90f) return RingMorphology::SparseRinglets;
    return RingMorphology::DiffuseDust;
}

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
    cfg.opacity_variety = p.opacity_variety;
    cfg.single_band_color_variety = p.single_band_color_variety;
    cfg.per_band_color_variety = p.per_band_color_variety;
    cfg.asymmetry = p.asymmetry;
    cfg.backscatter = p.backscatter;
    cfg.scatter = p.scatter;
    cfg.unlit_factor = p.unlit_factor;
    cfg.color_deep = p.color_deep;
    cfg.color_mid = p.color_mid;
    cfg.color_thin = p.color_thin;
    cfg.limits = p.limits;
    cfg.randomize_optics = true;
    cfg.morphology = -1;
    return cfg;
}

ProceduralRingConfig ProceduralRingConfig::randomize_from_preset(const RingPreset& p, uint32_t seed) noexcept {
    ProceduralRingConfig cfg = from_preset(p, seed);
    std::mt19937 rng((seed ^ 0x6c8e9cf5u) & 0x7fffffffu);
    auto rand_f = [&](float lo, float hi) {
        if (hi <= lo) return lo;
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    auto rand_i = [&](int lo, int hi) {
        if (hi <= lo) return lo;
        return std::uniform_int_distribution<int>(lo, hi)(rng);
    };

    const auto& lim = p.limits;
    cfg.band_count = rand_i(lim.band_min, lim.band_max);
    cfg.gap_count = rand_i(lim.gap_min, lim.gap_max);
    cfg.striation_freq = std::round(rand_f(lim.striation_min, lim.striation_max));
    cfg.contrast = std::round(rand_f(lim.contrast_min, lim.contrast_max) * 100.0f) / 100.0f;
    cfg.base_density = std::round(rand_f(lim.density_min, lim.density_max) * 100.0f) / 100.0f;
    cfg.core_boost = std::round(rand_f(lim.core_boost_min, lim.core_boost_max) * 100.0f) / 100.0f;
    cfg.inner_radius_ratio = std::round(rand_f(lim.inner_min, lim.inner_max) * 100.0f) / 100.0f;
    float min_outer = std::max(lim.outer_min, cfg.inner_radius_ratio + 0.18f);
    float max_outer = std::max(min_outer + 0.05f, lim.outer_max);
    cfg.outer_radius_ratio = std::round(rand_f(min_outer, max_outer) * 100.0f) / 100.0f;
    auto sample_variety = [&](float lo, float hi) {
        float cl_lo = std::clamp(lo, 0.0f, 2.0f);
        float cl_hi = std::clamp(hi, cl_lo, 2.0f);
        float p_roll = rand_f(0.0f, 1.0f);
        float val = 1.0f;
        if (p_roll < 0.015f) {
            bool hit_low = (rand_f(0.0f, 1.0f) < 0.5f);
            bool exact_edge = (rand_f(0.0f, 1.0f) < 0.5f);
            if (hit_low) {
                val = exact_edge ? 0.0f : rand_f(0.0f, 0.12f);
            } else {
                val = exact_edge ? 2.0f : rand_f(1.88f, 2.0f);
            }
        } else if (p_roll < 0.14f) {
            bool go_low = (cl_lo > 0.12f) && (cl_hi >= 1.88f || rand_f(0.0f, 1.0f) < 0.5f);
            val = go_low ? rand_f(0.12f, cl_lo) : rand_f(cl_hi, std::max(cl_hi, 1.88f));
        } else {
            val = rand_f(cl_lo, cl_hi);
        }
        return std::round(std::clamp(val, 0.0f, 2.0f) * 100.0f) / 100.0f;
    };

    cfg.opacity = std::round(rand_f(lim.opacity_min, lim.opacity_max) * 100.0f) / 100.0f;
    cfg.opacity_variety = sample_variety(lim.opacity_variety_min, lim.opacity_variety_max);
    cfg.single_band_color_variety = sample_variety(lim.single_band_color_variety_min, lim.single_band_color_variety_max);
    cfg.per_band_color_variety = sample_variety(lim.per_band_color_variety_min, lim.per_band_color_variety_max);
    cfg.scatter = rand_f(lim.scatter_min, lim.scatter_max);
    cfg.asymmetry = rand_f(lim.asymmetry_min, lim.asymmetry_max);
    cfg.backscatter = rand_f(lim.backscatter_min, lim.backscatter_max);
    cfg.unlit_factor = rand_f(lim.unlit_min, lim.unlit_max);
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
    float op_var = std::clamp(cfg.opacity_variety, 0.0f, 2.0f);
    float sb_col_var = std::clamp(cfg.single_band_color_variety, 0.0f, 2.0f);
    float pb_col_var = std::clamp(cfg.per_band_color_variety, 0.0f, 2.0f);

    std::mt19937 rng(cfg.seed & 0x7fffffffu);
    auto rand_f = [&](float low, float high) -> float {
        if (high <= low) return low;
        return std::uniform_real_distribution<float>(low, high)(rng);
    };

    RingMorphology morph = choose_morphology(cfg, band_count, rng);

    std::mt19937 opt_rng((cfg.seed ^ 0x9e3779b9u) & 0x7fffffffu);
    auto opt_f = [&](float low, float high) -> float {
        if (high <= low) return low;
        return std::uniform_real_distribution<float>(low, high)(opt_rng);
    };

    float out_scatter = cfg.scatter;
    float out_asymmetry = cfg.asymmetry;
    float out_backscatter = cfg.backscatter;
    float out_unlit = cfg.unlit_factor;
    glm::vec3 c_deep = cfg.color_deep;
    glm::vec3 c_mid = cfg.color_mid;
    glm::vec3 c_thin = cfg.color_thin;

    if (cfg.randomize_optics) {
        const auto& lim = cfg.limits;
        out_scatter = std::clamp(opt_f(lim.scatter_min, lim.scatter_max), 0.0f, 1.0f);
        out_asymmetry = std::clamp(opt_f(lim.asymmetry_min, lim.asymmetry_max), -0.99f, 0.99f);
        out_backscatter = std::clamp(opt_f(lim.backscatter_min, lim.backscatter_max), -0.99f, 0.0f);
        out_unlit = std::clamp(opt_f(lim.unlit_min, lim.unlit_max), 0.0f, 2.0f);

        float dh = opt_f(-lim.hue_jitter, lim.hue_jitter);
        float ds = opt_f(-lim.sat_jitter, lim.sat_jitter);
        float dv = opt_f(-lim.val_jitter, lim.val_jitter);
        c_deep = jitter_color(c_deep, dh, ds, dv);
        c_mid = jitter_color(c_mid, dh, ds * 0.7f, dv * 0.7f);
        c_thin = jitter_color(c_thin, dh, ds * 0.7f, dv * 0.6f);
    }

    auto make_band_tint = [&]() {
        float t_h = rand_f(-0.015f, 0.015f) * pb_col_var;
        float t_s = 1.0f + (rand_f(0.95f, 1.05f) - 1.0f) * pb_col_var;
        return glm::vec3(
            std::clamp(t_s + t_h, 0.80f, 1.18f),
            std::clamp(t_s, 0.80f, 1.18f),
            std::clamp(t_s - t_h, 0.80f, 1.18f));
    };

    std::vector<BandSpec> bands;
    std::vector<float> transition_cuts;

    if (morph == RingMorphology::SparseRinglets) {
        int ringlet_n = std::max(2, band_count);
        float cursor = rand_f(0.03f, 0.12f);
        float avail = 0.94f - cursor;
        bool has_wide = (rand_f(0.0f, 1.0f) < 0.65f);
        int wide_idx = has_wide ? static_cast<int>(rng() % static_cast<uint32_t>(ringlet_n)) : -1;
        std::vector<float> widths(ringlet_n);
        float w_sum = 0.0f;
        for (int b = 0; b < ringlet_n; ++b) {
            widths[b] = (b == wide_idx) ? rand_f(0.07f, 0.16f) : rand_f(0.015f, 0.055f);
            w_sum += widths[b];
        }
        float max_fill = avail * 0.58f;
        if (w_sum > max_fill) {
            float scale = max_fill / w_sum;
            for (float& w : widths) w *= scale;
            w_sum = max_fill;
        }
        float empty_budget = std::max(0.06f, avail - w_sum);
        std::vector<float> gaps_between(std::max(1, ringlet_n - 1), 0.0f);
        float g_sum = 0.0f;
        for (float& g : gaps_between) {
            g = rand_f(0.3f, 1.6f);
            g_sum += g;
        }
        for (float& g : gaps_between) g = (g / std::max(1e-5f, g_sum)) * empty_budget;

        for (int b = 0; b < ringlet_n; ++b) {
            BandSpec bs;
            bs.start = cursor;
            bs.end = std::min(0.985f, cursor + widths[b]);
            float bw = std::max(0.006f, bs.end - bs.start);
            bs.weight = (b == wide_idx ? 1.25f * core_boost : rand_f(0.45f, 1.35f * core_boost));
            bs.inner_soft = std::min(bw * 0.25f, rand_f(0.002f, 0.010f));
            bs.outer_soft = std::min(bw * 0.25f, rand_f(0.002f, 0.010f));
            bs.skew = rand_f(-0.45f, 0.45f);
            bs.flat_top = rand_f(0.35f, 0.90f);
            bs.freq_mult = rand_f(0.9f, 1.7f);
            bs.contrast_mult = rand_f(1.0f, 1.35f);
            bs.smoothness = rand_f(0.05f, 0.30f);
            bs.tint = make_band_tint();
            bands.push_back(bs);
            if (b + 1 < ringlet_n) {
                transition_cuts.push_back(bs.end + 0.5f * gaps_between[b]);
                cursor = bs.end + gaps_between[b];
            }
        }
    } else if (morph == RingMorphology::DiffuseDust) {
        int env_n = std::clamp(band_count, 1, 4);
        for (int b = 0; b < env_n; ++b) {
            float c = (env_n == 1) ? rand_f(0.46f, 0.54f) : rand_f(0.18f, 0.82f);
            float half_w = (env_n == 1) ? rand_f(0.46f, 0.50f) : rand_f(0.16f, 0.38f);
            BandSpec bs;
            bs.start = std::max(0.01f, c - half_w);
            bs.end = std::min(0.99f, c + half_w);
            float bw = std::max(0.05f, bs.end - bs.start);
            bs.weight = rand_f(0.55f, 1.15f * std::max(1.0f, core_boost * 0.6f));
            bs.inner_soft = bw * ((env_n == 1) ? rand_f(0.08f, 0.16f) : rand_f(0.18f, 0.40f));
            bs.outer_soft = bw * ((env_n == 1) ? rand_f(0.08f, 0.16f) : rand_f(0.18f, 0.40f));
            bs.skew = rand_f(-0.55f, 0.55f);
            bs.flat_top = rand_f(0.05f, 0.35f);
            bs.freq_mult = rand_f(0.35f, 0.80f);
            bs.contrast_mult = rand_f(0.55f, 0.90f);
            bs.smoothness = rand_f(0.55f, 0.88f);
            bs.tint = make_band_tint();
            bands.push_back(bs);
        }
        if (rand_f(0.0f, 1.0f) < 0.6f) {
            BandSpec core;
            float c = rand_f(0.30f, 0.75f);
            float hw = rand_f(0.025f, 0.08f);
            core.start = std::max(0.05f, c - hw);
            core.end = std::min(0.95f, c + hw);
            core.weight = rand_f(0.8f, 1.4f * core_boost);
            core.inner_soft = hw * 0.35f;
            core.outer_soft = hw * 0.35f;
            core.skew = rand_f(-0.3f, 0.3f);
            core.flat_top = rand_f(0.3f, 0.7f);
            core.freq_mult = rand_f(0.8f, 1.4f);
            core.contrast_mult = rand_f(0.85f, 1.2f);
            core.smoothness = rand_f(0.25f, 0.55f);
            core.tint = make_band_tint();
            bands.push_back(core);
        }
    } else if (morph == RingMorphology::CoreWithFilaments) {
        int core_n = (band_count >= 4 && rand_f(0.0f, 1.0f) < 0.5f) ? 2 : 1;
        float core_start = rand_f(0.08f, 0.35f);
        float core_span = rand_f(0.34f, 0.56f);
        float core_end = std::min(0.88f, core_start + core_span);
        float split = std::lerp(core_start, core_end, rand_f(0.35f, 0.65f));
        for (int c = 0; c < core_n; ++c) {
            BandSpec bs;
            bs.start = (core_n == 1) ? core_start : (c == 0 ? core_start : split + rand_f(0.015f, 0.055f));
            bs.end = (core_n == 1) ? core_end : (c == 0 ? split : core_end);
            float bw = std::max(0.04f, bs.end - bs.start);
            bs.weight = rand_f(0.95f, 1.45f) * core_boost;
            bs.inner_soft = std::min(bw * 0.25f, rand_f(0.008f, 0.045f));
            bs.outer_soft = std::min(bw * 0.25f, rand_f(0.008f, 0.045f));
            bs.skew = rand_f(-0.5f, 0.5f);
            bs.flat_top = rand_f(0.35f, 0.85f);
            bs.freq_mult = rand_f(0.75f, 1.35f);
            bs.contrast_mult = rand_f(0.85f, 1.25f);
            bs.smoothness = rand_f(0.15f, 0.45f);
            bs.tint = make_band_tint();
            bands.push_back(bs);
            if (core_n == 2 && c == 0) transition_cuts.push_back(split);
        }
        int fil_n = std::clamp(band_count - core_n, 1, 5);
        for (int f = 0; f < fil_n; ++f) {
            bool place_inner = (core_start > 0.14f) && ((f == 0 && rand_f(0.0f, 1.0f) < 0.65f) || core_end > 0.90f);
            float pos = place_inner
                ? rand_f(0.04f, std::max(0.06f, core_start - 0.04f))
                : rand_f(std::min(0.92f, core_end + 0.03f), 0.96f);
            float hw = rand_f(0.010f, 0.040f);
            BandSpec bs;
            bs.start = std::max(0.015f, pos - hw);
            bs.end = std::min(0.985f, pos + hw);
            float bw = std::max(0.008f, bs.end - bs.start);
            bs.weight = rand_f(0.25f, 1.15f * std::max(1.0f, core_boost * 0.7f));
            bs.inner_soft = bw * rand_f(0.15f, 0.40f);
            bs.outer_soft = bw * rand_f(0.15f, 0.40f);
            bs.skew = rand_f(-0.4f, 0.4f);
            bs.flat_top = rand_f(0.25f, 0.75f);
            bs.freq_mult = rand_f(0.9f, 1.6f);
            bs.contrast_mult = rand_f(0.9f, 1.3f);
            bs.smoothness = rand_f(0.20f, 0.60f);
            bs.tint = make_band_tint();
            bands.push_back(bs);
            transition_cuts.push_back(pos);
        }
    } else {
        float min_band_w = std::min(0.08f, 0.65f / static_cast<float>(band_count));
        std::vector<float> raw_weights(band_count);
        float raw_sum = 0.0f;
        for (int b = 0; b < band_count; ++b) {
            raw_weights[b] = rand_f(0.4f, 1.8f);
            raw_sum += raw_weights[b];
        }
        float total_gap = 0.0f;
        std::vector<float> inter_gaps(std::max(0, band_count - 1), 0.0f);
        for (float& ig : inter_gaps) {
            if (rand_f(0.0f, 1.0f) < 0.45f) {
                ig = rand_f(0.015f, 0.075f);
                total_gap += ig;
            }
        }
        float usable = std::max(0.35f, 0.96f - total_gap - min_band_w * static_cast<float>(band_count));
        float cursor = 0.02f;
        int dominant_idx = static_cast<int>(rng() % static_cast<uint32_t>(band_count));
        int tenuous_idx = (band_count >= 3) ? static_cast<int>(rng() % static_cast<uint32_t>(band_count)) : -1;
        if (tenuous_idx == dominant_idx) tenuous_idx = (dominant_idx + band_count - 1) % band_count;

        for (int b = 0; b < band_count; ++b) {
            float bw = min_band_w + usable * (raw_weights[b] / std::max(1e-5f, raw_sum));
            BandSpec bs;
            bs.start = cursor;
            bs.end = std::min(0.985f, cursor + bw);
            float actual_w = std::max(0.02f, bs.end - bs.start);
            if (b == dominant_idx) {
                bs.weight = rand_f(1.15f, 1.55f) * core_boost;
            } else if (b == tenuous_idx) {
                bs.weight = rand_f(0.18f, 0.50f);
            } else {
                bs.weight = rand_f(0.45f, 1.25f * std::max(1.0f, core_boost * 0.6f));
            }
            bs.inner_soft = std::min(actual_w * 0.30f, rand_f(0.005f, 0.048f));
            bs.outer_soft = std::min(actual_w * 0.30f, rand_f(0.005f, 0.048f));
            bs.skew = rand_f(-0.6f, 0.6f);
            bs.flat_top = rand_f(0.25f, 0.85f);
            bs.freq_mult = rand_f(0.60f, 1.55f);
            bs.contrast_mult = rand_f(0.75f, 1.35f);
            bs.smoothness = rand_f(0.10f, 0.55f);
            bs.tint = make_band_tint();
            bands.push_back(bs);
            if (b + 1 < band_count) {
                transition_cuts.push_back(bs.end + 0.5f * inter_gaps[b]);
                cursor = bs.end + inter_gaps[b];
            }
        }
    }

    for (auto& bs : bands) {
        float bw = std::max(1e-5f, bs.end - bs.start);
        int sub_n = std::clamp(static_cast<int>(std::round(bw * rand_f(8.0f, 18.0f))), 2, 12);
        std::vector<float> sw(static_cast<size_t>(sub_n));
        float sw_sum = 0.0f;
        for (float& w : sw) {
            w = rand_f(0.50f, 1.65f);
            sw_sum += w;
        }
        float cur = bs.start;
        bs.sub_bands.reserve(static_cast<size_t>(sub_n));
        for (int s = 0; s < sub_n; ++s) {
            float sub_w = bw * (sw[static_cast<size_t>(s)] / std::max(1e-5f, sw_sum));
            SubBand sb;
            sb.start = cur;
            sb.end = (s + 1 == sub_n) ? bs.end : std::min(bs.end, cur + sub_w);
            float roll = rand_f(0.0f, 1.0f);
            float raw_mult = (roll < 0.15f) ? rand_f(0.45f, 0.68f) : rand_f(0.72f, 1.24f);
            sb.density_mult = std::clamp(1.0f + (raw_mult - 1.0f) * op_var, 0.15f, 1.85f);
            sb.slope = std::clamp(rand_f(-0.22f, 0.22f) * op_var, -0.45f, 0.45f);
            sb.edge_soft = sub_w * rand_f(0.18f, 0.42f);
            float al_var = 1.0f + (rand_f(0.95f, 1.03f) - 1.0f) * sb_col_var;
            float warm = rand_f(-0.010f, 0.010f) * sb_col_var;
            sb.tint = glm::clamp(glm::vec3(al_var + warm, al_var, al_var - warm), 0.82f, 1.16f);
            bs.sub_bands.push_back(sb);
            cur = sb.end;
        }
    }

    std::vector<float> macro_density(effective_res, 0.0f);
    std::vector<glm::vec3> band_tint(effective_res, glm::vec3(1.0f));
    std::vector<float> local_freq_mult(effective_res, 1.0f);
    std::vector<float> local_contrast_mult(effective_res, 1.0f);
    std::vector<float> local_smoothness(effective_res, 0.25f);

    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float d_sum = 0.0f;
        glm::vec3 tint_acc(0.0f);
        float freq_acc = 0.0f;
        float cont_acc = 0.0f;
        float smth_acc = 0.0f;

        for (const auto& bs : bands) {
            if (u < bs.start || u > bs.end) continue;
            float bw = std::max(1e-5f, bs.end - bs.start);
            float mask = smoothstep(bs.start, bs.start + bs.inner_soft, u) *
                         (1.0f - smoothstep(bs.end - bs.outer_soft, bs.end, u));
            float t = std::clamp((u - bs.start) / bw, 0.0f, 1.0f);
            float dome = std::sin(t * 3.14159265f);
            float shaped = std::lerp(dome, 1.0f, bs.flat_top * 0.72f);
            float ramp = std::clamp(1.0f + bs.skew * (2.0f * t - 1.0f), 0.45f, 1.50f);

            float sub_mod = 1.0f;
            glm::vec3 sub_tint(1.0f);
            if (!bs.sub_bands.empty()) {
                float sub_w_acc = 0.0f;
                float sub_d_acc = 0.0f;
                glm::vec3 sub_t_acc(0.0f);
                for (const auto& sb : bs.sub_bands) {
                    float pad = sb.edge_soft;
                    if (u < sb.start - pad || u > sb.end + pad) continue;
                    float sbw = std::max(1e-5f, sb.end - sb.start);
                    float sm = smoothstep(sb.start - pad, sb.start + pad, u) *
                               (1.0f - smoothstep(sb.end - pad, sb.end + pad, u));
                    float st = std::clamp((u - sb.start) / sbw, 0.0f, 1.0f);
                    float sramp = std::clamp(1.0f + sb.slope * (2.0f * st - 1.0f), 0.35f, 1.65f);
                    sub_w_acc += sm;
                    sub_d_acc += sm * sb.density_mult * sramp;
                    sub_t_acc += sm * sb.tint;
                }
                if (sub_w_acc > 1e-5f) {
                    sub_mod = sub_d_acc / sub_w_acc;
                    sub_tint = sub_t_acc / sub_w_acc;
                }
            }

            float contribution = mask * bs.weight * shaped * ramp * sub_mod;
            d_sum += contribution;
            tint_acc += (bs.tint * sub_tint) * contribution;
            freq_acc += bs.freq_mult * contribution;
            cont_acc += bs.contrast_mult * contribution;
            smth_acc += bs.smoothness * contribution;
        }

        float boundary_fade = smoothstep(0.0f, 0.022f, u) * (1.0f - smoothstep(0.978f, 1.0f, u));
        macro_density[i] = d_sum * boundary_fade;
        if (d_sum > 1e-6f) {
            band_tint[i] = tint_acc / d_sum;
            local_freq_mult[i] = freq_acc / d_sum;
            local_contrast_mult[i] = cont_acc / d_sum;
            local_smoothness[i] = smth_acc / d_sum;
        }
    }

    std::vector<GapParam> gaps;
    if (gap_count > 0) {
        std::vector<float> candidate_pos;
        for (float tc : transition_cuts) {
            candidate_pos.push_back(std::clamp(tc + rand_f(-0.018f, 0.018f), 0.06f, 0.94f));
        }
        for (const auto& bs : bands) {
            if (bs.end - bs.start > 0.14f) {
                candidate_pos.push_back(std::lerp(bs.start, bs.end, rand_f(0.25f, 0.75f)));
            }
        }
        while (candidate_pos.size() < static_cast<size_t>(gap_count)) {
            candidate_pos.push_back(rand_f(0.10f, 0.90f));
        }
        std::shuffle(candidate_pos.begin(), candidate_pos.end(), rng);
        candidate_pos.resize(static_cast<size_t>(gap_count));
        std::sort(candidate_pos.begin(), candidate_pos.end());

        int major_idx = (gap_count >= 2) ? static_cast<int>(rng() % static_cast<uint32_t>(gap_count)) : -1;
        for (int g = 0; g < gap_count; ++g) {
            bool is_major = (g == major_idx);
            GapParam gp;
            gp.pos = candidate_pos[g];
            gp.width = is_major ? rand_f(0.022f, 0.055f) : rand_f(0.003f, 0.016f);
            gp.sharpness = is_major ? rand_f(0.12f, 0.35f) : rand_f(0.22f, 0.65f);
            gp.residual = is_major ? rand_f(0.0f, 0.025f) : rand_f(0.0f, 0.12f);
            float wave_roll = rand_f(0.0f, 1.0f);
            gp.wave_side = (wave_roll < 0.32f) ? 0 : ((wave_roll < 0.66f) ? 1 : -1);
            gp.wave_freq = rand_f(220.0f, 580.0f);
            gp.wave_decay = rand_f(0.020f, 0.065f);
            gp.wave_pow = rand_f(1.12f, 1.48f);
            gp.wave_amp = rand_f(0.10f, 0.32f);
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

    std::vector<float> striations_raw(effective_res, 0.0f);
    std::vector<float> ringlets_raw(effective_res, 0.0f);
    std::vector<float> meso_raw(effective_res, 0.0f);
    float s_min = 1e9f;
    float s_max = -1e9f;
    float m_min = 1e9f;
    float m_max = -1e9f;
    float phase_offset = rand_f(0.0f, 6.283185f);
    float meso_freq = rand_f(14.0f, 30.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float freq = striation_freq * local_freq_mult[i];
        float s_val = fbm1d(u * freq, 7, 2.15f, 0.54f, cfg.seed);
        striations_raw[i] = s_val;
        s_min = std::min(s_min, s_val);
        s_max = std::max(s_max, s_val);

        float m_val = fbm1d(u * meso_freq, 4, 2.1f, 0.58f, cfg.seed + 51511u);
        meso_raw[i] = m_val;
        m_min = std::min(m_min, m_val);
        m_max = std::max(m_max, m_val);

        float warp = 2.6f * fbm1d(u * (freq * 0.16f), 4, 2.0f, 0.55f, cfg.seed + 31337u);
        float osc = 0.5f + 0.5f * std::sin(u * freq * 1.65f + warp + phase_offset);
        float osc_hi = 0.5f + 0.5f * std::sin(u * freq * 4.3f - warp * 1.4f);
        ringlets_raw[i] = 0.68f * osc + 0.32f * osc_hi;
    }

    float s_span = std::max(1e-6f, s_max - s_min);
    float m_span = std::max(1e-6f, m_max - m_min);
    std::vector<float> striations(effective_res, 0.0f);
    std::vector<float> meso_mod(effective_res, 1.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float s_norm = std::clamp((striations_raw[i] - s_min) / s_span, 0.0f, 1.0f);
        float c_eff = std::clamp(contrast * local_contrast_mult[i], 0.45f, 4.2f);
        s_norm = std::pow(s_norm, c_eff);
        float r_shaped = std::pow(std::clamp(ringlets_raw[i], 0.0f, 1.0f), std::max(0.75f, c_eff * 0.7f));
        float detail = std::lerp(s_norm, 0.52f * s_norm + 0.48f * r_shaped, 0.50f);
        float micro1 = 0.5f + 0.5f * perlin1d(u * (striation_freq * local_freq_mult[i] * 4.2f), cfg.seed + 99991u);
        float micro2 = 0.5f + 0.5f * perlin1d(u * (striation_freq * local_freq_mult[i] * 10.5f), cfg.seed + 199933u);
        float textured = 0.76f * detail + 0.15f * micro1 + 0.09f * micro2;
        striations[i] = std::lerp(textured, 0.48f + 0.40f * s_norm, local_smoothness[i]);

        float m_norm = std::clamp((meso_raw[i] - m_min) / m_span, 0.0f, 1.0f);
        float m_ridge = smoothstep(0.15f, 0.85f, m_norm);
        float m_base = std::lerp(0.72f + 0.42f * m_ridge, 0.85f + 0.25f * m_ridge, local_smoothness[i]);
        meso_mod[i] = std::clamp(1.0f + (m_base - 1.0f) * op_var, 0.25f, 1.65f);
    }

    std::vector<float> density_waves(effective_res, 0.0f);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        for (const auto& g : gaps) {
            if (g.wave_side == 0) continue;
            float dist_g = u - g.pos;
            if ((g.wave_side > 0 && dist_g < 0.0f) || (g.wave_side < 0 && dist_g > 0.0f)) continue;
            float abs_dist = std::abs(dist_g);
            float wave_mask = smoothstep(g.width * 0.35f, g.width * 0.5f + 0.012f, abs_dist) *
                              std::exp(-abs_dist / g.wave_decay);
            float chirp = std::cos(g.wave_freq * std::pow(abs_dist, g.wave_pow));
            density_waves[i] += wave_mask * chirp * g.wave_amp;
        }
    }

    constexpr float kTauCalibration = 0.78f;
    std::vector<float> tau_vals(effective_res, 0.0f);
    std::vector<float> alpha_vals(effective_res, 0.0f);
    float alpha_sum = 0.0f;
    for (uint32_t i = 0; i < effective_res; ++i) {
        float striation_raw = 0.42f + 0.88f * striations[i];
        float striation_mod = std::clamp(std::lerp(0.86f, striation_raw, op_var), 0.06f, 1.75f);
        float tau = kTauCalibration * base_density * macro_density[i] * meso_mod[i] *
                    gap_transmission[i] * striation_mod +
                    kTauCalibration * density_waves[i] * op_var * macro_density[i];
        tau = std::max(0.0f, tau);
        tau_vals[i] = tau;
        alpha_vals[i] = std::clamp(1.0f - std::exp(-tau), 0.0f, 1.0f);
        alpha_sum += alpha_vals[i];
    }

    if (alpha_sum < 15.0f && effective_res >= 128) {
        float boost = (18.0f / std::max(0.01f, alpha_sum));
        boost = std::min(boost, 12.0f);
        for (uint32_t i = 0; i < effective_res; ++i) {
            tau_vals[i] *= boost;
            alpha_vals[i] = std::clamp(1.0f - std::exp(-tau_vals[i]), 0.0f, 1.0f);
        }
    }

    float comp_freq = rand_f(6.0f, 22.0f);
    std::vector<glm::vec4> profile(effective_res);
    for (uint32_t i = 0; i < effective_res; ++i) {
        float u = (effective_res > 1) ? static_cast<float>(i) / static_cast<float>(effective_res - 1) : 0.0f;
        float tau = tau_vals[i];
        float weight_mid = smoothstep(0.05f, 0.45f, tau);
        float weight_deep = smoothstep(0.45f, 1.50f, tau);

        glm::vec3 color_low = glm::mix(c_thin, c_mid, weight_mid);
        glm::vec3 color_full = glm::mix(color_low, c_deep, weight_deep);

        float color_var = 0.5f + 0.5f * fbm1d(u * comp_freq, 3, 2.0f, 0.5f, cfg.seed + 7771u);
        float color_var_3 = std::clamp(1.0f + (color_var - 0.5f) * 0.09f * sb_col_var, 0.84f, 1.16f);

        glm::vec3 rgb = color_full * band_tint[i] * color_var_3 * cfg.tint_color;
        rgb = glm::clamp(rgb, 0.0f, 1.0f);

        profile[i] = glm::vec4(rgb, alpha_vals[i]);
    }

    std::vector<ephemeris::RingLayerDefinition> extra_layers;
    if (cfg.randomize_optics && cfg.limits.halo_prob > 0.0f && opacity > 1e-4f) {
        float op_scale = std::min(1.0f, opacity / std::max(0.1f, cfg.limits.opacity_min));
        if (opt_f(0.0f, 1.0f) < cfg.limits.halo_prob) {
            ephemeris::RingLayerDefinition halo{};
            bool narrow_arc_ringlet = (morph == RingMorphology::SparseRinglets || opt_f(0.0f, 1.0f) < 0.35f);
            if (narrow_arc_ringlet) {
                float gap_off = opt_f(0.015f, 0.075f);
                float width = opt_f(0.008f, 0.028f);
                halo.inner = outer_ratio + gap_off;
                halo.outer = halo.inner + width;
                halo.opacity = opt_f(0.002f, cfg.limits.halo_opacity_max * 1.5f) * op_scale;
                halo.gradient = {{0.0f, 0.0f}, {0.25f, 0.85f}, {0.5f, 1.0f}, {0.75f, 0.85f}, {1.0f, 0.0f}};
            } else {
                halo.inner = std::max(1.04f, outer_ratio * opt_f(0.95f, 1.03f));
                halo.outer = outer_ratio * opt_f(1.12f, 1.52f);
                halo.opacity = opt_f(0.0008f, cfg.limits.halo_opacity_max) * op_scale;
                halo.gradient = {{0.0f, 0.0f}, {0.22f, 0.45f}, {0.48f, 1.0f}, {0.75f, 0.35f}, {1.0f, 0.0f}};
            }
            halo.color_rgb = glm::clamp(glm::mix(c_thin, c_mid, 0.4f) * cfg.tint_color, 0.05f, 1.0f);
            halo.scatter = opt_f(0.72f, 0.94f);
            halo.asymmetry = opt_f(0.82f, 0.96f);
            halo.backscatter = opt_f(-0.35f, -0.04f);
            halo.unlit_factor = opt_f(0.70f, 1.15f);
            halo.is_textured = false;
            extra_layers.push_back(std::move(halo));
        }
    }

    ProceduralRingResult result{};
    result.profile = std::move(profile);
    result.inner_radius_ratio = inner_ratio;
    result.outer_radius_ratio = outer_ratio;
    result.opacity = opacity;
    result.scatter = out_scatter;
    result.asymmetry = out_asymmetry;
    result.backscatter = out_backscatter;
    result.unlit_factor = out_unlit;
    result.raw_color = cfg.tint_color;
    result.extra_layers = std::move(extra_layers);
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

std::vector<ephemeris::RingLayerDefinition> build_procedural_ring_layers(
    const ProceduralRingResult& result) {
    std::vector<ephemeris::RingLayerDefinition> layers;
    layers.reserve(1 + result.extra_layers.size());
    ephemeris::RingLayerDefinition main_layer{};
    apply_procedural_ring_layer(result, main_layer);
    layers.push_back(std::move(main_layer));
    for (const auto& extra : result.extra_layers) {
        layers.push_back(extra);
    }
    return layers;
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
