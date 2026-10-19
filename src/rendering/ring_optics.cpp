#include "rendering/ring_optics.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace sf::rendering {

namespace {

float smoothstep_f(float edge0, float edge1, float x) noexcept {
    float t = std::clamp((x - edge0) / std::max(edge1 - edge0, 1e-12f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

double disk_cdf(double y) {
    return 0.5 + (std::asin(y) + y * std::sqrt(std::max(0.0, 1.0 - y * y))) / 3.14159265358979323846;
}

double disk_moment(double y) {
    return -2.0 / (3.0 * 3.14159265358979323846) * std::pow(std::max(0.0, 1.0 - y * y), 1.5);
}

double inverse_disk_cdf(double value) {
    double lo = -1.0, hi = 1.0;
    for (int i = 0; i < 40; ++i) {
        double mid = 0.5 * (lo + hi);
        if (disk_cdf(mid) < value) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

uint32_t read_be32(const uint8_t* p) noexcept {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

void append_be32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(uint8_t((v >> 24) & 0xFFu));
    out.push_back(uint8_t((v >> 16) & 0xFFu));
    out.push_back(uint8_t((v >> 8) & 0xFFu));
    out.push_back(uint8_t(v & 0xFFu));
}

uint32_t crc32_png(std::span<const uint8_t> bytes) noexcept {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = 0xFFFFFFFFu;
    for (uint8_t b : bytes) c = table[(c ^ b) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

uint32_t adler32_zlib(std::span<const uint8_t> bytes) noexcept {
    uint32_t s1 = 1u;
    uint32_t s2 = 0u;
    for (uint8_t b : bytes) {
        s1 = (s1 + b) % 65521u;
        s2 = (s2 + s1) % 65521u;
    }
    return (s2 << 16) | s1;
}

struct HuffmanTable {
    static constexpr int MAX_BITS = 15;
    std::array<int, MAX_BITS + 1> count{};
    std::vector<uint16_t> symbols;

    bool build(std::span<const uint8_t> lengths) {
        count.fill(0);
        for (uint8_t len : lengths) {
            if (len > MAX_BITS) return false;
            if (len > 0) ++count[len];
        }
        std::array<int, MAX_BITS + 1> offs{};
        int total = 0;
        for (int b = 1; b <= MAX_BITS; ++b) {
            offs[b] = total;
            total += count[b];
        }
        symbols.assign(static_cast<size_t>(total), 0);
        for (size_t sym = 0; sym < lengths.size(); ++sym) {
            uint8_t len = lengths[sym];
            if (len > 0) {
                symbols[static_cast<size_t>(offs[len]++)] = static_cast<uint16_t>(sym);
            }
        }
        return true;
    }
};

struct BitStream {
    std::span<const uint8_t> data;
    size_t pos = 0;
    uint32_t bitbuf = 0;
    int bitcnt = 0;

    bool ensure(int n) noexcept {
        while (bitcnt < n) {
            if (pos >= data.size()) return false;
            bitbuf |= uint32_t(data[pos++]) << bitcnt;
            bitcnt += 8;
        }
        return true;
    }

    uint32_t read_bits(int n) noexcept {
        if (n == 0) return 0;
        if (!ensure(n)) return 0;
        uint32_t val = bitbuf & ((1u << n) - 1u);
        bitbuf >>= n;
        bitcnt -= n;
        return val;
    }

    void align_byte() noexcept {
        bitbuf = 0;
        bitcnt = 0;
    }

    int decode(const HuffmanTable& h) noexcept {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= HuffmanTable::MAX_BITS; ++len) {
            if (!ensure(1)) return -1;
            code |= int(bitbuf & 1u);
            bitbuf >>= 1;
            --bitcnt;
            int cnt = h.count[len];
            if (code - cnt < first) {
                return h.symbols[static_cast<size_t>(index + (code - first))];
            }
            index += cnt;
            first += cnt;
            first <<= 1;
            code <<= 1;
        }
        return -1;
    }
};

bool inflate_zlib(std::span<const uint8_t> src, std::vector<uint8_t>& out, size_t expected_size) {
    if (src.size() < 6) return false;
    uint8_t cmf = src[0];
    uint8_t flg = src[1];
    if ((cmf & 0x0Fu) != 8u || (((uint32_t(cmf) << 8) | flg) % 31u) != 0u) return false;
    size_t offset = (flg & 0x20u) ? 6u : 2u;
    if (offset >= src.size()) return false;

    BitStream bs{src.subspan(offset)};
    out.clear();
    out.reserve(expected_size);

    static constexpr std::array<uint16_t, 29> len_base = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
        35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258
    };
    static constexpr std::array<uint8_t, 29> len_extra = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
    };
    static constexpr std::array<uint16_t, 30> dist_base = {
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
        257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577
    };
    static constexpr std::array<uint8_t, 30> dist_extra = {
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
        7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
    };
    static constexpr std::array<uint8_t, 19> cl_order = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
    };

    bool bfinal = false;
    while (!bfinal) {
        if (!bs.ensure(3)) return false;
        bfinal = bs.read_bits(1) != 0;
        uint32_t btype = bs.read_bits(2);
        if (btype == 0u) {
            bs.align_byte();
            if (bs.pos + 4 > bs.data.size()) return false;
            uint16_t len = uint16_t(bs.data[bs.pos]) | (uint16_t(bs.data[bs.pos + 1]) << 8);
            uint16_t nlen = uint16_t(bs.data[bs.pos + 2]) | (uint16_t(bs.data[bs.pos + 3]) << 8);
            bs.pos += 4;
            if ((len ^ nlen) != 0xFFFFu || bs.pos + len > bs.data.size()) return false;
            out.insert(out.end(), bs.data.begin() + static_cast<std::ptrdiff_t>(bs.pos),
                       bs.data.begin() + static_cast<std::ptrdiff_t>(bs.pos + len));
            bs.pos += len;
        } else if (btype == 1u || btype == 2u) {
            HuffmanTable litlen_table{};
            HuffmanTable dist_table{};
            if (btype == 1u) {
                std::array<uint8_t, 288> ll_lens{};
                for (size_t i = 0; i <= 143; ++i) ll_lens[i] = 8;
                for (size_t i = 144; i <= 255; ++i) ll_lens[i] = 9;
                for (size_t i = 256; i <= 279; ++i) ll_lens[i] = 7;
                for (size_t i = 280; i <= 287; ++i) ll_lens[i] = 8;
                std::array<uint8_t, 32> d_lens{};
                d_lens.fill(5);
                litlen_table.build(ll_lens);
                dist_table.build(d_lens);
            } else {
                if (!bs.ensure(14)) return false;
                uint32_t hlit = bs.read_bits(5) + 257u;
                uint32_t hdist = bs.read_bits(5) + 1u;
                uint32_t hclen = bs.read_bits(4) + 4u;
                std::array<uint8_t, 19> cl_lens{};
                for (uint32_t i = 0; i < hclen; ++i) {
                    if (!bs.ensure(3)) return false;
                    cl_lens[cl_order[i]] = static_cast<uint8_t>(bs.read_bits(3));
                }
                HuffmanTable cl_table{};
                if (!cl_table.build(cl_lens)) return false;

                std::vector<uint8_t> unpacked(hlit + hdist, 0);
                size_t idx = 0;
                while (idx < unpacked.size()) {
                    int sym = bs.decode(cl_table);
                    if (sym < 0) return false;
                    if (sym < 16) {
                        unpacked[idx++] = static_cast<uint8_t>(sym);
                    } else if (sym == 16) {
                        if (idx == 0 || !bs.ensure(2)) return false;
                        uint32_t rep = bs.read_bits(2) + 3u;
                        uint8_t prev = unpacked[idx - 1];
                        while (rep-- > 0 && idx < unpacked.size()) unpacked[idx++] = prev;
                    } else if (sym == 17) {
                        if (!bs.ensure(3)) return false;
                        uint32_t rep = bs.read_bits(3) + 3u;
                        while (rep-- > 0 && idx < unpacked.size()) unpacked[idx++] = 0;
                    } else if (sym == 18) {
                        if (!bs.ensure(7)) return false;
                        uint32_t rep = bs.read_bits(7) + 11u;
                        while (rep-- > 0 && idx < unpacked.size()) unpacked[idx++] = 0;
                    } else {
                        return false;
                    }
                }
                std::span<const uint8_t> sp(unpacked);
                if (!litlen_table.build(sp.first(hlit)) || !dist_table.build(sp.subspan(hlit, hdist))) {
                    return false;
                }
            }

            for (;;) {
                int sym = bs.decode(litlen_table);
                if (sym < 0) return false;
                if (sym < 256) {
                    out.push_back(static_cast<uint8_t>(sym));
                } else if (sym == 256) {
                    break;
                } else {
                    int l_idx = sym - 257;
                    if (l_idx < 0 || l_idx >= static_cast<int>(len_base.size())) return false;
                    uint32_t length = len_base[static_cast<size_t>(l_idx)] + bs.read_bits(len_extra[static_cast<size_t>(l_idx)]);
                    int d_sym = bs.decode(dist_table);
                    if (d_sym < 0 || d_sym >= static_cast<int>(dist_base.size())) return false;
                    uint32_t dist = dist_base[static_cast<size_t>(d_sym)] + bs.read_bits(dist_extra[static_cast<size_t>(d_sym)]);
                    if (dist == 0 || dist > out.size()) return false;
                    for (uint32_t k = 0; k < length; ++k) {
                        out.push_back(out[out.size() - dist]);
                    }
                }
            }
        } else {
            return false;
        }
    }
    return true;
}

uint8_t paeth_predictor(int a, int b, int c) noexcept {
    int p = a + b - c;
    int pa = std::abs(p - a);
    int pb = std::abs(p - b);
    int pc = std::abs(p - c);
    if (pa <= pb && pa <= pc) return static_cast<uint8_t>(a);
    if (pb <= pc) return static_cast<uint8_t>(b);
    return static_cast<uint8_t>(c);
}

void write_png_chunk(std::vector<uint8_t>& out, const char type[4], std::span<const uint8_t> payload) {
    append_be32(out, static_cast<uint32_t>(payload.size()));
    size_t crc_start = out.size();
    out.push_back(static_cast<uint8_t>(type[0]));
    out.push_back(static_cast<uint8_t>(type[1]));
    out.push_back(static_cast<uint8_t>(type[2]));
    out.push_back(static_cast<uint8_t>(type[3]));
    out.insert(out.end(), payload.begin(), payload.end());
    uint32_t crc = crc32_png(std::span<const uint8_t>(out).subspan(crc_start));
    append_be32(out, crc);
}

} // namespace

std::vector<RingDrawRange> build_ring_draw_ranges(std::span<const ephemeris::RingLayerDefinition> layers,
                                                double min_radius, double max_radius, double planet_radius_au,
                                                uint32_t angular_segments, uint32_t radial_subdivisions) {
    std::vector<RingDrawRange> ranges;
    if (angular_segments < 3 || radial_subdivisions == 0 || max_radius <= min_radius || planet_radius_au <= 0.0) return ranges;
    double chord = std::cos(3.14159265358979323846 / angular_segments);
    double padding = max_radius * 0.001;
    for (uint32_t strip = 0; strip < radial_subdivisions; ++strip) {
        double inner = std::lerp(min_radius, max_radius, double(strip) / radial_subdivisions) * chord - padding;
        double outer = std::lerp(min_radius, max_radius, double(strip + 1) / radial_subdivisions) + padding;
        uint32_t mask = 0;
        uint32_t packed = 0;
        uint32_t count = 0;
        for (uint32_t s = 0; s < std::min<uint32_t>(static_cast<uint32_t>(layers.size()), MAX_RING_SEGMENTS); ++s) {
            const auto& layer = layers[s];
            double fade = std::max(layer.outer - layer.inner, 1e-9 / planet_radius_au) * 0.05;
            if (outer < layer.inner - fade || inner > layer.outer + fade || layer.opacity <= 0.0f) continue;
            mask |= 1u << s;
            packed |= s << (count++ * 3u);
        }
        if (mask == 0) continue;
        uint32_t first = strip * angular_segments * 6;
        uint32_t indices = angular_segments * 6;
        if (!ranges.empty() && ranges.back().segment_mask == mask &&
            ranges.back().first_index + ranges.back().index_count == first) {
            ranges.back().index_count += indices;
        } else {
            ranges.push_back({first, indices, mask, packed, count});
        }
    }
    return ranges;
}

RingDrawClassification classify_ring_draw(glm::vec3 center, glm::vec3 pole, float outer_radius,
                                          std::span<const glm::vec4> stars) noexcept {
    float minimum_cosine = std::abs(glm::dot(center, pole)) /
        std::max(glm::length(center) + outer_radius, 1e-20f);
    if (minimum_cosine < 0.01f) return {};
    bool finite_source = false;
    uint32_t sides = 0;
    glm::dvec3 normal = glm::normalize(glm::dvec3(pole));
    double camera_side = -glm::dot(glm::dvec3(center), normal);
    for (const auto& star : stars) {
        double height = glm::dot(glm::dvec3(star) - glm::dvec3(center), normal);
        double magnitude = std::max(glm::length(glm::dvec3(star)), glm::length(glm::dvec3(center))) + outer_radius;
        double padding = 64.0 * std::numeric_limits<float>::epsilon() * std::max(magnitude, 1e-12);
        if (std::abs(height) <= star.w + padding) finite_source = true;
        sides |= camera_side * height >= 0.0 ? 1u : 2u;
    }
    auto lighting = finite_source ? RingLightingMode::FiniteSource : sides == 1u ? RingLightingMode::Reflection :
        sides == 2u ? RingLightingMode::Transmission : RingLightingMode::Point;
    return {lighting, true};
}

float ring_tau(float alpha) noexcept {
    float a = std::clamp(alpha, 0.0f, RING_MAX_ALPHA);
    return -std::log(1.0f - a);
}

float ring_absorbed(float tau_val) noexcept {
    return 1.0f - std::exp(-std::max(tau_val, 0.0f));
}

float ring_transfer(float tau_val, float mu_s, float mu_v, bool same_side) noexcept {
    float tau = std::max(tau_val, 0.0f);
    float ms = std::max(mu_s, 1e-5f);
    float mv = std::max(mu_v, 1e-5f);
    float xs = tau / ms;
    float xv = tau / mv;

    if (same_side) {
        float a = std::min(xs + xv, 80.0f);
        float one_minus_exp = (a < 1e-3f)
            ? a * (1.0f - 0.5f * a + (1.0f / 6.0f) * a * a)
            : (1.0f - std::exp(-a));
        return (ms / (ms + mv)) * one_minus_exp;
    }

    float d = xs - xv;
    if (std::abs(d) < 1e-3f) {
        float xm = 0.5f * (xs + xv);
        return xv * std::exp(-xm) * (1.0f + (d * d) / 24.0f);
    }
    float ad = std::abs(d);
    float one_minus_exp = ad < 1e-2f
        ? ad * (1.0f - 0.5f * ad + ad * ad / 6.0f)
        : -std::expm1(-ad);
    return (xv / ad) * std::exp(-std::min(xs, xv)) * one_minus_exp;
}

float ring_opposition_factor(float cos_alpha, float tau_val) noexcept {
    float ca = std::clamp(cos_alpha, -1.0f, 1.0f);
    float alpha_rad = std::acos(std::clamp(ca, -1.0f, 1.0f - 1e-7f));

    float shoe_width = 0.030f;
    float shoe_amp = 0.55f * smoothstep_f(0.02f, 0.45f, tau_val);
    float shoe = shoe_amp / (1.0f + alpha_rad / shoe_width);

    float cboe_width = 0.0025f;
    float cboe_amp = 0.35f * smoothstep_f(0.005f, 0.18f, tau_val);
    float x_cb = alpha_rad / cboe_width;
    float cboe = cboe_amp / (1.0f + x_cb * x_cb);

    return 1.0f + shoe + cboe;
}

float ring_phase_hg(float cos_theta, float g) noexcept {
    float gg = std::clamp(g, -0.99f, 0.99f);
    float mu = std::clamp(cos_theta, -1.0f, 1.0f);
    float one_m_g = (gg >= 0.0f) ? (1.0f - gg) : (1.0f + gg);
    float delta_mu = (gg >= 0.0f) ? (1.0f - mu) : (1.0f + mu);
    float denom = std::max(one_m_g * one_m_g + 2.0f * std::abs(gg) * delta_mu, 1e-6f);
    return (1.0f - gg * gg) / (denom * std::sqrt(denom));
}

float ring_phase_dhg(float cos_theta, float scatter, float g_fwd, float g_back, float tau_val, bool is_textured) noexcept {
    float dust_factor = 1.0f - smoothstep_f(0.05f, 0.50f, tau_val);
    float core_factor = smoothstep_f(0.30f, 1.80f, tau_val);

    float g1 = std::clamp(g_fwd + (is_textured ? 0.14f * dust_factor : 0.0f), -0.92f, 0.92f);
    float g2_mag = std::clamp(std::abs(g_back) + (is_textured ? 0.08f * core_factor : 0.0f), 0.0f, 0.92f);
    float g2 = -g2_mag;

    float base_w = (is_textured && scatter > 1.0f) ? 0.35f : std::clamp(scatter, 0.0f, 1.0f);
    float w_f = std::clamp(base_w + (is_textured ? (0.12f * dust_factor - 0.08f * core_factor) : 0.0f), 0.0f, 1.0f);

    float p_back = ring_phase_hg(cos_theta, g2);
    float p_fwd = ring_phase_hg(cos_theta, g1);
    return p_back * (1.0f - w_f) + p_fwd * w_f;
}

glm::vec3 ring_chandrasekhar_h(float mu, glm::vec3 w0) noexcept {
    float m = std::max(mu, 0.0f);
    glm::vec3 one_minus = glm::max(glm::vec3(1.0f) - w0, glm::vec3(1e-6f));
    glm::vec3 s = glm::sqrt(one_minus);
    return glm::vec3(1.0f + m * 1.7320508f) / (glm::vec3(1.0f) + m * 1.7320508f * s);
}

glm::vec3 ring_slab_x(float mu, float tau_val, glm::vec3 w0) noexcept {
    glm::vec3 h = ring_chandrasekhar_h(mu, w0);
    float m = std::max(mu, 1e-4f);
    glm::vec3 k = glm::sqrt(3.0f * glm::max(glm::vec3(1.0f) - w0, glm::vec3(1e-5f))) + glm::vec3(0.5f);
    glm::vec3 arg = -std::max(tau_val, 0.0f) * (glm::vec3(1.0f / m) + k);
    glm::vec3 atten = glm::vec3(1.0f) - glm::vec3(std::exp(arg.x), std::exp(arg.y), std::exp(arg.z));
    return glm::vec3(1.0f) + (h - glm::vec3(1.0f)) * atten;
}

glm::vec3 ring_multiple_scattering(float tau_val, float mu_s, float mu_v, bool same_side, glm::vec3 linear_rgb) noexcept {
    float tau = std::max(tau_val, 0.0f);
    if (tau < 1e-5f) return glm::vec3(0.0f);

    float luma = glm::dot(linear_rgb, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    glm::vec3 w_transport = glm::clamp((glm::vec3(luma) * 0.80f + linear_rgb * 0.20f) * 0.78f, glm::vec3(0.0f), glm::vec3(0.92f));
    glm::vec3 w_color = glm::clamp(linear_rgb * 0.78f, glm::vec3(0.0f), glm::vec3(0.96f));
    glm::vec3 x_s = ring_slab_x(mu_s, tau, w_transport);
    glm::vec3 x_v = ring_slab_x(mu_v, tau, w_transport);
    glm::vec3 multi_bounce = glm::max(x_s * x_v - glm::vec3(1.0f), glm::vec3(0.0f));

    if (same_side) {
        float slab_r = ring_transfer(tau * 1.5f, mu_s, mu_v, true);
        return 0.35f * w_color * multi_bounce * slab_r;
    }

    float ms = std::max(mu_s, 1e-4f);
    float mv = std::max(mu_v, 1e-4f);
    float slab_t = ring_transfer(tau, ms, mv, false);
    float slant_tau = std::min(tau * (0.5f / ms + 0.5f / mv), 80.0f);
    glm::vec3 k_diff = glm::sqrt(3.0f * glm::max(glm::vec3(1.0f) - w_transport, glm::vec3(1e-4f)));
    glm::vec3 exp_arg = -slant_tau * (glm::vec3(0.55f) + 0.45f * k_diff);
    glm::vec3 exp_k(std::exp(exp_arg.x), std::exp(exp_arg.y), std::exp(exp_arg.z));
    glm::vec3 diff_core = (ms / (ms + mv)) * slant_tau * exp_k;
    float buildup = 1.0f - std::exp(-slant_tau);
    glm::vec3 y_env = (glm::vec3(slab_t) + 0.35f * diff_core) * buildup;
    return 0.25f * w_color * multi_bounce * y_env;
}

glm::vec3 ring_radiance_rgb(float total_tau, float mu_v, float sun_elev, float cos_theta,
                            glm::vec4 props, glm::vec3 linear_rgb, bool is_textured, bool same_side) noexcept {
    float mu_s = std::max(std::abs(sun_elev), 1e-5f);
    float g_fwd = props.x;
    float g_back = props.y;
    float scatter = props.z;
    float unlit_factor = props.w;

    glm::vec3 eff_rgb = glm::clamp(linear_rgb, glm::vec3(0.0f), glm::vec3(1.0f));
    float phase = ring_phase_dhg(cos_theta, scatter, g_fwd, g_back, total_tau, is_textured);
    float opp = same_side ? ring_opposition_factor(-cos_theta, total_tau) : 1.0f;
    float transfer = ring_transfer(total_tau, mu_s, mu_v, same_side);

    glm::vec3 ss = eff_rgb * (0.25f * phase * opp * transfer);
    glm::vec3 ms = ring_multiple_scattering(total_tau, mu_s, mu_v, same_side, eff_rgb);
    float side_scale = same_side ? 1.0f : unlit_factor;
    return (ss + ms) * side_scale;
}

int ringshine_source_samples(double elevation, double sin_radius, glm::vec4& offsets, glm::vec4& weights) {
    offsets = glm::vec4(0.0f);
    weights = glm::vec4(0.0f);
    if (sin_radius <= 1e-6 || std::abs(elevation) >= 2.0 * sin_radius) {
        weights.x = 1.0f;
        return 1;
    }
    double boundary = std::clamp(-elevation / sin_radius, -1.0, 1.0);
    double faces[2][2] = {{-1.0, boundary}, {boundary, 1.0}};
    for (int face = 0; face < 2; ++face) {
        double lo = faces[face][0], hi = faces[face][1];
        double area = disk_cdf(hi) - disk_cdf(lo);
        if (area < 1e-14) continue;
        double middle = inverse_disk_cdf(disk_cdf(lo) + 0.5 * area);
        double parts[2][2] = {{lo, middle}, {middle, hi}};
        for (int part = 0; part < 2; ++part) {
            double left = parts[part][0], right = parts[part][1];
            double weight = disk_cdf(right) - disk_cdf(left);
            if (weight <= 0.0) continue;
            int slot = 2 * face + part;
            offsets[slot] = static_cast<float>((disk_moment(right) - disk_moment(left)) / weight);
            weights[slot] = static_cast<float>(weight);
        }
    }
    return 4;
}

std::vector<glm::vec2> build_ringshine_radial_samples(std::span<const ephemeris::RingLayerDefinition> layers,
                                                    uint32_t band_count) {
    layers = layers.first(std::min<size_t>(layers.size(), MAX_RING_SEGMENTS));
    std::vector<double> boundaries;
    for (const auto& layer : layers) {
        double inner = std::max(layer.inner, 1.000001);
        if (layer.opacity <= 0.0f || layer.outer <= inner) continue;
        boundaries.push_back(inner);
        boundaries.push_back(layer.outer);
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    struct Interval {
        double inner;
        double outer;
        double span;
        double remainder = 0.0;
        uint32_t bands = 1;
    };
    std::vector<Interval> intervals;
    double total_span = 0.0;
    for (size_t i = 1; i < boundaries.size(); ++i) {
        double inner = boundaries[i - 1], outer = boundaries[i], middle = 0.5 * (inner + outer);
        if (std::none_of(layers.begin(), layers.end(), [&](const auto& layer) {
                return layer.opacity > 0.0f && middle > layer.inner && middle < layer.outer;
            })) continue;
        double span = std::log((outer - 1.0) / (inner - 1.0));
        intervals.push_back({inner, outer, span});
        total_span += span;
    }
    if (intervals.empty()) return {};
    uint32_t budget = std::max(std::clamp(band_count, 4u, 1024u), static_cast<uint32_t>(intervals.size()));
    uint32_t remaining = budget - static_cast<uint32_t>(intervals.size()), assigned = 0;
    for (auto& interval : intervals) {
        double extra = remaining * interval.span / total_span;
        uint32_t whole = static_cast<uint32_t>(extra);
        interval.bands += whole;
        interval.remainder = extra - whole;
        assigned += whole;
    }
    for (; assigned < remaining; ++assigned) {
        auto it = std::max_element(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
            return a.remainder < b.remainder;
        });
        ++it->bands;
        it->remainder = -1.0;
    }
    constexpr double nodes[] = {0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970262};
    constexpr double weights[] = {0.17392742256872693, 0.32607257743127307, 0.32607257743127307, 0.17392742256872693};
    std::vector<glm::vec2> samples;
    samples.reserve(4 * budget);
    for (const auto& interval : intervals) {
        for (uint32_t band = 0; band < interval.bands; ++band) {
            for (uint32_t q = 0; q < 4; ++q) {
                double t = (band + nodes[q]) / interval.bands;
                double gap = (interval.inner - 1.0) * std::exp(t * interval.span);
                samples.emplace_back(1.0 + gap, gap * interval.span * weights[q] / interval.bands);
            }
        }
    }
    return samples;
}

float sample_ring_gradient(std::span<const ephemeris::RingGradientStop> stops, float p) noexcept {
    if (stops.empty()) return 1.0f;
    if (p <= stops.front().p) return stops.front().a;
    if (p >= stops.back().p) return stops.back().a;
    for (size_t i = 0; i + 1 < stops.size(); ++i) {
        const auto& g0 = stops[i];
        const auto& g1 = stops[i + 1];
        if (p >= g0.p && p <= g1.p) {
            float span = g1.p - g0.p;
            if (span <= 1e-9f) return g1.a;
            float t = (p - g0.p) / span;
            return g0.a * (1.0f - t) + g1.a * t;
        }
    }
    return 1.0f;
}

glm::vec4 apply_hsba_pixel(glm::vec4 rgba, float hue_shift, float saturation, float brightness,
                           float opacity_mult, float alpha_boost) noexcept {
    float r = rgba.r, g = rgba.g, b = rgba.b;
    float maxc = std::max({r, g, b});
    float minc = std::min({r, g, b});
    float rangec = maxc - minc;

    float h = 0.0f;
    float s = 0.0f;
    float v = std::clamp(maxc * brightness, 0.0f, 1.0f);

    if (rangec > 1e-6f) {
        s = std::clamp((rangec / (maxc + 1e-6f)) * saturation, 0.0f, 1.0f);
        float rc = (maxc - r) / (rangec + 1e-6f);
        float gc = (maxc - g) / (rangec + 1e-6f);
        float bc = (maxc - b) / (rangec + 1e-6f);
        if (r == maxc) h = bc - gc;
        else if (g == maxc) h = 2.0f + rc - bc;
        else h = 4.0f + gc - rc;
        h = std::fmod(h / 6.0f + hue_shift, 1.0f);
        if (h < 0.0f) h += 1.0f;
    }

    float h6 = h * 6.0f;
    int i = static_cast<int>(std::floor(h6)) % 6;
    if (i < 0) i += 6;
    float f = h6 - std::floor(h6);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    glm::vec3 rgb_out{v, t, p};
    switch (i) {
        case 0: rgb_out = {v, t, p}; break;
        case 1: rgb_out = {q, v, p}; break;
        case 2: rgb_out = {p, v, t}; break;
        case 3: rgb_out = {p, q, v}; break;
        case 4: rgb_out = {t, p, v}; break;
        default: rgb_out = {v, p, q}; break;
    }

    float a_out = std::clamp(rgba.a * opacity_mult, 0.0f, 1.0f);
    if (std::abs(alpha_boost - 1.0f) > 1e-4f && a_out > 1e-5f) {
        a_out = std::clamp(std::pow(a_out, 1.0f / std::max(0.01f, alpha_boost)), 0.0f, 1.0f);
    }
    return glm::vec4(glm::clamp(rgb_out, glm::vec3(0.0f), glm::vec3(1.0f)), a_out);
}

std::vector<glm::vec4> generate_ring_segment_profile(
    const ephemeris::RingLayerDefinition& layer,
    std::span<const glm::vec4> raw_texture_samples,
    uint32_t res) {
    std::vector<glm::vec4> out(res, glm::vec4(1.0f));
    bool use_tex = layer.is_textured && !raw_texture_samples.empty();
    bool has_hsba = std::abs(layer.hue_shift) > 1e-4f ||
                    std::abs(layer.saturation - 1.0f) > 1e-4f ||
                    std::abs(layer.brightness - 1.0f) > 1e-4f ||
                    std::abs(layer.alpha_boost - 1.0f) > 1e-4f;

    for (uint32_t i = 0; i < res; ++i) {
        float u = (res > 1) ? float(i) / float(res - 1) : 0.0f;
        float grad_a = sample_ring_gradient(layer.gradient, u);
        if (use_tex) {
            float sample_pos = u * float(raw_texture_samples.size() - 1);
            size_t lo = static_cast<size_t>(std::floor(sample_pos));
            size_t hi = std::min(lo + 1, raw_texture_samples.size() - 1);
            float frac = sample_pos - float(lo);
            glm::vec4 tex = glm::mix(raw_texture_samples[lo], raw_texture_samples[hi], frac);
            if (has_hsba) {
                tex = apply_hsba_pixel(tex, layer.hue_shift, layer.saturation, layer.brightness, 1.0f, layer.alpha_boost);
            }
            out[i] = glm::vec4(glm::vec3(tex) * layer.color_rgb, tex.a * grad_a);
        } else {
            out[i] = glm::vec4(layer.color_rgb, grad_a);
        }
    }
    return out;
}

std::vector<std::vector<glm::vec4>> build_ring_profile_mips(std::span<const glm::vec4> profile) {
    if (profile.empty() || !std::has_single_bit(profile.size())) return {};
    std::vector<std::vector<glm::vec4>> mips(1);
    mips.front().reserve(profile.size());
    for (auto sample : profile) {
        float alpha = std::clamp(sample.a, 0.0f, 1.0f);
        auto color = glm::pow(glm::max(glm::vec3(sample), glm::vec3(0.0f)), glm::vec3(2.2f));
        mips.front().emplace_back(color * alpha, alpha);
    }
    while (mips.back().size() > 1) {
        const auto& previous = mips.back();
        std::vector<glm::vec4> next(previous.size() / 2);
        for (size_t i = 0; i < next.size(); ++i) next[i] = (previous[2 * i] + previous[2 * i + 1]) * 0.5f;
        mips.push_back(std::move(next));
    }
    return mips;
}

std::vector<glm::vec4> bake_unified_shadow_profile(
    std::span<const RingSegmentProfileInput> segments,
    double min_r, double max_r,
    uint32_t res) {
    std::vector<glm::vec4> combined(res, glm::vec4(0.0f));
    if (segments.empty() || max_r <= min_r) return combined;

    for (uint32_t i = 0; i < res; ++i) {
        double u = (res > 1) ? double(i) / double(res - 1) : 0.0;
        double radius = min_r + (max_r - min_r) * u;
        double tau = 0.0;
        glm::dvec3 weighted_color{0.0};

        for (const auto& seg : segments) {
            if (seg.profile.empty() || radius < seg.inner_r || radius > seg.outer_r) continue;
            double t = std::clamp((radius - seg.inner_r) / std::max(1e-12, seg.outer_r - seg.inner_r), 0.0, 1.0);
            double sample = t * double(seg.profile.size() - 1);
            size_t lo = static_cast<size_t>(std::floor(sample));
            size_t hi = std::min(lo + 1, seg.profile.size() - 1);
            float frac = static_cast<float>(sample - double(lo));
            glm::vec4 val = glm::mix(seg.profile[lo], seg.profile[hi], frac);
            float alpha = std::clamp(val.a * seg.opacity, 0.0f, RING_MAX_ALPHA);
            double layer_tau = -std::log1p(-double(alpha));
            tau += layer_tau;
            weighted_color += glm::dvec3(val.r, val.g, val.b) * layer_tau;
        }

        if (tau > 0.0) {
            combined[i] = glm::vec4(
                glm::vec3(weighted_color / tau),
                static_cast<float>(-std::expm1(-tau))
            );
        }
    }
    return combined;
}

std::vector<float> bake_ring_shadow_lut_layer(
    std::span<const glm::vec4> unified_profile,
    uint32_t width,
    uint32_t angles) {
    uint32_t lut_w = width + 1;
    std::vector<float> lut(static_cast<size_t>(angles) * lut_w, 0.0f);
    if (unified_profile.empty()) return lut;

    std::vector<double> tau(width, 0.0);
    for (uint32_t i = 0; i < width; ++i) {
        size_t idx = std::min<size_t>(i, unified_profile.size() - 1);
        double a = std::clamp(double(unified_profile[idx].a), 0.0, double(RING_MAX_ALPHA));
        tau[i] = -std::log1p(-a);
    }

    double inv_w = 1.0 / double(std::max(1u, width));
    for (uint32_t a = 0; a < angles; ++a) {
        double mu = (double(a) + 0.5) / double(std::max(1u, angles));
        size_t row_off = static_cast<size_t>(a) * lut_w;
        double acc = 0.0;
        lut[row_off] = 0.0f;
        for (uint32_t i = 0; i < width; ++i) {
            double ext = 1.0 - std::exp(-std::min(tau[i] / mu, 80.0));
            acc += ext * inv_w;
            lut[row_off + i + 1] = static_cast<float>(acc);
        }
    }
    return lut;
}

std::optional<std::vector<glm::vec4>> load_ring_png_1d(
    const std::filesystem::path& path,
    uint32_t target_samples) {
    auto bytes_opt = sf::paths::read_binary_file(path);
    if (!bytes_opt || bytes_opt->size() < 33) return std::nullopt;
    std::span<const uint8_t> data(*bytes_opt);

    static constexpr uint8_t png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (std::memcmp(data.data(), png_sig, 8) != 0) return std::nullopt;

    uint32_t width = 0, height = 0;
    uint8_t bit_depth = 0, color_type = 0, interlace = 0;
    std::vector<uint8_t> plte;
    std::vector<uint8_t> trns;
    std::vector<uint8_t> idat;

    size_t pos = 8;
    while (pos + 12 <= data.size()) {
        uint32_t len = read_be32(data.data() + pos);
        const uint8_t* type = data.data() + pos + 4;
        if (pos + 12 + len > data.size()) return std::nullopt;
        std::span<const uint8_t> chunk(data.data() + pos + 8, len);

        if (std::memcmp(type, "IHDR", 4) == 0) {
            if (len < 13) return std::nullopt;
            width = read_be32(chunk.data());
            height = read_be32(chunk.data() + 4);
            bit_depth = chunk[8];
            color_type = chunk[9];
            interlace = chunk[12];
        } else if (std::memcmp(type, "PLTE", 4) == 0) {
            plte.assign(chunk.begin(), chunk.end());
        } else if (std::memcmp(type, "tRNS", 4) == 0) {
            trns.assign(chunk.begin(), chunk.end());
        } else if (std::memcmp(type, "IDAT", 4) == 0) {
            idat.insert(idat.end(), chunk.begin(), chunk.end());
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + static_cast<size_t>(len);
    }

    if (width == 0 || height == 0 || interlace != 0 || idat.empty()) return std::nullopt;
    int channels = 0;
    switch (color_type) {
        case 0: channels = 1; break;
        case 2: channels = 3; break;
        case 3: channels = 1; break;
        case 4: channels = 2; break;
        case 6: channels = 4; break;
        default: return std::nullopt;
    }
    if (bit_depth != 8 && bit_depth != 16) return std::nullopt;
    if (color_type == 3 && bit_depth != 8) return std::nullopt;

    size_t bpp = static_cast<size_t>(channels) * (bit_depth / 8u);
    size_t stride = static_cast<size_t>(width) * bpp;
    size_t raw_size = static_cast<size_t>(height) * (1 + stride);

    std::vector<uint8_t> raw;
    if (!inflate_zlib(idat, raw, raw_size) || raw.size() < raw_size) return std::nullopt;

    std::vector<uint8_t> recon(static_cast<size_t>(height) * stride, 0);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t filter = raw[static_cast<size_t>(y) * (1 + stride)];
        const uint8_t* src_row = raw.data() + static_cast<size_t>(y) * (1 + stride) + 1;
        uint8_t* dst_row = recon.data() + static_cast<size_t>(y) * stride;
        const uint8_t* prev_row = (y > 0) ? (dst_row - stride) : nullptr;

        for (size_t x = 0; x < stride; ++x) {
            int a = (x >= bpp) ? dst_row[x - bpp] : 0;
            int b = prev_row ? prev_row[x] : 0;
            int c = (prev_row && x >= bpp) ? prev_row[x - bpp] : 0;
            uint8_t val = src_row[x];
            switch (filter) {
                case 0: dst_row[x] = val; break;
                case 1: dst_row[x] = static_cast<uint8_t>((val + a) & 0xFF); break;
                case 2: dst_row[x] = static_cast<uint8_t>((val + b) & 0xFF); break;
                case 3: dst_row[x] = static_cast<uint8_t>((val + ((a + b) >> 1)) & 0xFF); break;
                case 4: dst_row[x] = static_cast<uint8_t>((val + paeth_predictor(a, b, c)) & 0xFF); break;
                default: return std::nullopt;
            }
        }
    }

    std::vector<glm::vec4> strip(width, glm::vec4(0.0f));
    float inv_h = 1.0f / float(height);
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = recon.data() + static_cast<size_t>(y) * stride;
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * bpp;
            auto ch_val = [&](int c) -> float {
                if (bit_depth == 8) return float(px[c]) / 255.0f;
                uint16_t v16 = (uint16_t(px[c * 2]) << 8) | uint16_t(px[c * 2 + 1]);
                return float(v16) / 65535.0f;
            };
            glm::vec4 rgba{1.0f};
            if (color_type == 0) {
                float g = ch_val(0);
                rgba = {g, g, g, 1.0f};
            } else if (color_type == 2) {
                rgba = {ch_val(0), ch_val(1), ch_val(2), 1.0f};
            } else if (color_type == 3) {
                uint8_t idx = px[0];
                if (static_cast<size_t>(idx) * 3 + 2 < plte.size()) {
                    rgba.r = float(plte[static_cast<size_t>(idx) * 3 + 0]) / 255.0f;
                    rgba.g = float(plte[static_cast<size_t>(idx) * 3 + 1]) / 255.0f;
                    rgba.b = float(plte[static_cast<size_t>(idx) * 3 + 2]) / 255.0f;
                }
                rgba.a = (static_cast<size_t>(idx) < trns.size()) ? (float(trns[idx]) / 255.0f) : 1.0f;
            } else if (color_type == 4) {
                float g = ch_val(0);
                rgba = {g, g, g, ch_val(1)};
            } else if (color_type == 6) {
                rgba = {ch_val(0), ch_val(1), ch_val(2), ch_val(3)};
            }
            strip[x] += rgba * inv_h;
        }
    }

    if (target_samples == 0 || strip.size() == target_samples) {
        return strip;
    }

    std::vector<glm::vec4> resampled(target_samples);
    for (uint32_t i = 0; i < target_samples; ++i) {
        float u = (target_samples > 1) ? float(i) / float(target_samples - 1) : 0.0f;
        float pos_f = u * float(strip.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(pos_f));
        size_t hi = std::min(lo + 1, strip.size() - 1);
        float frac = pos_f - float(lo);
        resampled[i] = glm::mix(strip[lo], strip[hi], frac);
    }
    return resampled;
}

bool save_ring_png_2d(
    const std::filesystem::path& path,
    std::span<const glm::vec4> radial_rgba,
    uint32_t height) {
    if (radial_rgba.empty() || height == 0) return false;
    uint32_t width = static_cast<uint32_t>(radial_rgba.size());
    size_t row_bytes = 1 + static_cast<size_t>(width) * 4;
    std::vector<uint8_t> single_row(row_bytes, 0);
    for (uint32_t x = 0; x < width; ++x) {
        glm::vec4 c = glm::clamp(radial_rgba[x], glm::vec4(0.0f), glm::vec4(1.0f));
        single_row[1 + static_cast<size_t>(x) * 4 + 0] = static_cast<uint8_t>(std::lround(c.r * 255.0f));
        single_row[1 + static_cast<size_t>(x) * 4 + 1] = static_cast<uint8_t>(std::lround(c.g * 255.0f));
        single_row[1 + static_cast<size_t>(x) * 4 + 2] = static_cast<uint8_t>(std::lround(c.b * 255.0f));
        single_row[1 + static_cast<size_t>(x) * 4 + 3] = static_cast<uint8_t>(std::lround(c.a * 255.0f));
    }

    std::vector<uint8_t> raw_scanlines;
    raw_scanlines.reserve(row_bytes * height);
    for (uint32_t y = 0; y < height; ++y) {
        raw_scanlines.insert(raw_scanlines.end(), single_row.begin(), single_row.end());
    }

    std::vector<uint8_t> zlib_stream;
    zlib_stream.push_back(0x78u);
    zlib_stream.push_back(0x01u);
    size_t pos = 0;
    while (pos < raw_scanlines.size()) {
        size_t chunk_len = std::min<size_t>(65535u, raw_scanlines.size() - pos);
        bool final_block = (pos + chunk_len == raw_scanlines.size());
        zlib_stream.push_back(final_block ? 0x01u : 0x00u);
        uint16_t len16 = static_cast<uint16_t>(chunk_len);
        uint16_t nlen16 = static_cast<uint16_t>(~len16);
        zlib_stream.push_back(uint8_t(len16 & 0xFFu));
        zlib_stream.push_back(uint8_t((len16 >> 8) & 0xFFu));
        zlib_stream.push_back(uint8_t(nlen16 & 0xFFu));
        zlib_stream.push_back(uint8_t((nlen16 >> 8) & 0xFFu));
        zlib_stream.insert(zlib_stream.end(),
                           raw_scanlines.begin() + static_cast<std::ptrdiff_t>(pos),
                           raw_scanlines.begin() + static_cast<std::ptrdiff_t>(pos + chunk_len));
        pos += chunk_len;
    }
    append_be32(zlib_stream, adler32_zlib(raw_scanlines));

    std::vector<uint8_t> png;
    static constexpr uint8_t png_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), std::begin(png_sig), std::end(png_sig));

    std::vector<uint8_t> ihdr;
    append_be32(ihdr, width);
    append_be32(ihdr, height);
    ihdr.push_back(8u);
    ihdr.push_back(6u);
    ihdr.push_back(0u);
    ihdr.push_back(0u);
    ihdr.push_back(0u);
    write_png_chunk(png, "IHDR", ihdr);
    write_png_chunk(png, "IDAT", zlib_stream);
    write_png_chunk(png, "IEND", {});

    return sf::paths::write_binary_file(path, png.data(), png.size());
}

bool bake_and_save_ring_texture(
    std::string_view system_name,
    std::string_view body_name,
    ephemeris::RingLayerDefinition& layer) {
    if (body_name.empty()) return false;
    std::vector<glm::vec4> raw_tex;
    std::error_code ec;
    std::filesystem::path existing;
    if (!system_name.empty()) {
        auto p1 = sf::paths::external_path("textures", system_name, body_name, "Rings.png");
        if (std::filesystem::is_regular_file(p1, ec)) existing = p1;
    }
    if (existing.empty()) {
        auto p2 = sf::paths::external_path("textures", "Solar System", body_name, "Rings.png");
        if (std::filesystem::is_regular_file(p2, ec)) existing = p2;
    }
    if (!existing.empty()) {
        if (auto loaded = load_ring_png_1d(existing, RING_ATLAS_WIDTH)) {
            raw_tex = std::move(*loaded);
        }
    }
    auto baked = generate_ring_segment_profile(layer, raw_tex, RING_ATLAS_WIDTH);
    std::string sys_folder = system_name.empty() ? "Solar System" : std::string(system_name);
    auto out_path = sf::paths::external_path("textures", sys_folder, body_name, "Rings.png");
    if (!save_ring_png_2d(out_path, baked, 64)) return false;

    layer.is_textured = true;
    layer.color_rgb = {1.0f, 1.0f, 1.0f};
    layer.hue_shift = 0.0f;
    layer.saturation = 1.0f;
    layer.brightness = 1.0f;
    layer.alpha_boost = 1.0f;
    layer.gradient = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    return true;
}

} // namespace sf::rendering
