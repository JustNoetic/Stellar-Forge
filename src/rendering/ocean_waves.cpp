#include "rendering/ocean_waves.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <random>

namespace sf::rendering {

namespace {

struct EvolvePush {
    double seconds;
    float gravity;
    float choppiness;
    glm::vec4 lengths;
    float last_length;
    float padding[3];
};

static_assert(sizeof(EvolvePush) == 48);

double donelan_banner_beta(double ratio) {
    if (ratio < 0.95) return 2.61 * std::pow(ratio, 1.3);
    if (ratio < 1.6) return 2.28 * std::pow(ratio, -1.3);
    return std::pow(10.0, -0.4 + 0.8393 * std::exp(-0.567 * std::log(ratio * ratio)));
}

double wavenumber_spectrum(glm::dvec2 k, const OceanSpectrumParameters& p) {
    double length = glm::length(k);
    if (length <= 0.0) return 0.0;
    double g = p.gravity, u = std::max(p.wind_ms, 0.5);
    double omega = std::sqrt(g * length);
    double peak = std::max(22.0 * std::cbrt(g * g / (u * OCEAN_FETCH_M)), 0.855 * g / u);
    double alpha = std::max(0.076 * std::pow(u * u / (OCEAN_FETCH_M * g), 0.22), 0.0081);
    double sigma = omega <= peak ? 0.07 : 0.09;
    double r = std::exp(-(omega - peak) * (omega - peak) / (2.0 * sigma * sigma * peak * peak));
    double s = alpha * g * g / std::pow(omega, 5.0) * std::exp(-1.25 * std::pow(peak / omega, 4.0)) * std::pow(3.3, r);
    double theta = std::remainder(std::atan2(k.y, k.x) - p.wind_direction_rad, TWO_PI);
    double beta = donelan_banner_beta(omega / peak);
    double sech = 1.0 / std::cosh(beta * theta);
    double spreading = beta / (2.0 * std::tanh(beta * PI)) * sech * sech;
    spreading = 0.1 / TWO_PI + 0.9 * spreading;
    double suppression = std::exp(-length * length * 1e-4);
    return s * spreading * (g / (2.0 * omega)) / length * suppression;
}

}

OceanSpectrum build_ocean_spectrum(const OceanSpectrumParameters& p, uint32_t seed) {
    constexpr uint32_t n = OCEAN_FFT_SIZE;
    constexpr int half = int(n / 2);
    OceanSpectrum result;
    result.amplitudes.assign(size_t(OCEAN_CASCADE_COUNT) * n * n, glm::vec4(0.0f));
    std::mt19937 random(seed);
    std::normal_distribution<double> gaussian;
    std::vector<glm::dvec2> h0(size_t(n) * n);
    for (uint32_t c = 0; c < OCEAN_CASCADE_COUNT; ++c) {
        double dk = TWO_PI / OCEAN_CASCADE_SIZES_M[c];
        double band = c + 1 < OCEAN_CASCADE_COUNT ? PI * n / OCEAN_CASCADE_SIZES_M[c + 1] : 0.0;
        for (uint32_t y = 0; y < n; ++y) {
            for (uint32_t x = 0; x < n; ++x) {
                glm::dvec2 noise{gaussian(random), gaussian(random)};
                int nx = int(x) - half, ny = int(y) - half;
                glm::dvec2 k = glm::dvec2(nx, ny) * dk;
                bool excluded = nx == -half || ny == -half || (std::abs(k.x) < band && std::abs(k.y) < band);
                double amplitude = excluded ? 0.0 : std::sqrt(wavenumber_spectrum(k, p) * dk * dk * 0.5);
                h0[size_t(y) * n + x] = noise * (amplitude * std::sqrt(0.5));
            }
        }
        double slope = 0.0;
        for (uint32_t y = 0; y < n; ++y) {
            for (uint32_t x = 0; x < n; ++x) {
                auto a = h0[size_t(y) * n + x];
                auto b = h0[size_t((n - y) % n) * n + (n - x) % n];
                result.amplitudes[(size_t(c) * n + y) * n + x] = glm::vec4(float(a.x), float(a.y), float(b.x), float(-b.y));
                double k2 = (std::pow(int(x) - half, 2.0) + std::pow(int(y) - half, 2.0)) * dk * dk;
                double energy = glm::dot(a, a) + glm::dot(b, b);
                slope += k2 * energy;
                result.height_variance += energy;
            }
        }
        result.mean_square_slope[c] = slope;
        result.total_mean_square_slope += slope;
    }
    return result;
}

void OceanWaves::init(rhi::VkContext& ctx) {
    constexpr uint32_t n = OCEAN_FFT_SIZE;
    field_ = ctx.create_image_2d(n, n, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT, 1, OCEAN_CASCADE_COUNT, true);
    base_ = ctx.create_image_2d(n, n, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        1, 2 * OCEAN_CASCADE_COUNT, true);
    moments_ = ctx.create_image_2d(n, n, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        uint32_t(std::bit_width(n)), 2 * OCEAN_CASCADE_COUNT, true);
    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        ctx.transition_image(cmd, field_, VK_IMAGE_LAYOUT_GENERAL);
        ctx.transition_image(cmd, base_, VK_IMAGE_LAYOUT_GENERAL);
        ctx.transition_image(cmd, moments_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    for (auto& buffer : amplitudes_)
        buffer = ctx.create_host_buffer(size_t(OCEAN_CASCADE_COUNT) * n * n * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    rhi::DescriptorLayoutBinding evolve_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
    };
    evolve_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "ocean/spectrum_evolve.comp", evolve_bindings, sizeof(EvolvePush));
    rhi::DescriptorLayoutBinding fft_bindings[] = {
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
    };
    fft_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "ocean/fft.comp", fft_bindings, sizeof(uint32_t));
    rhi::DescriptorLayoutBinding moment_bindings[] = {
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
    };
    moments_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "ocean/moments.comp", moment_bindings, sizeof(float));
}

void OceanWaves::shutdown(rhi::VkContext& ctx) {
    rhi::ShaderCache::destroy_pipeline(ctx, evolve_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, fft_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, moments_pipeline_);
    for (auto& buffer : amplitudes_) ctx.destroy_buffer(buffer);
    ctx.destroy_image(field_);
    ctx.destroy_image(base_);
    ctx.destroy_image(moments_);
}

void OceanWaves::track(uint64_t body_identity, glm::dvec3 camera_fixed_m) {
    double distance = glm::length(camera_fixed_m);
    glm::dvec3 up = distance > 1.0 ? camera_fixed_m / distance : glm::dvec3(1.0, 0.0, 0.0);
    if (!tracking_ || body_identity != body_identity_) {
        glm::dvec3 east = glm::cross(glm::dvec3(0.0, 1.0, 0.0), up);
        tangent_ = glm::length(east) > 1e-6 ? glm::normalize(east) : glm::dvec3(0.0, 0.0, 1.0);
        bitangent_ = glm::cross(up, tangent_);
        camera_ = camera_fixed_m;
        body_identity_ = body_identity;
        tracking_ = true;
        return;
    }
    glm::dvec3 delta = camera_fixed_m - camera_;
    planar_ += glm::dvec2(glm::dot(delta, tangent_), glm::dot(delta, bitangent_));
    planar_ -= glm::floor(planar_ / OCEAN_CASCADE_SIZES_M.back()) * OCEAN_CASCADE_SIZES_M.back();
    camera_ = camera_fixed_m;
    tangent_ = glm::normalize(tangent_ - up * glm::dot(tangent_, up));
    bitangent_ = glm::cross(up, tangent_);
}

OceanCascadesGPU OceanWaves::cascades() const {
    OceanCascadesGPU result;
    result.tangent = glm::vec4(glm::vec3(tangent_), 0.0f);
    result.bitangent = glm::vec4(glm::vec3(bitangent_), 0.0f);
    for (uint32_t c = 0; c < OCEAN_CASCADE_COUNT; ++c) {
        glm::dvec2 offset = glm::fract(planar_ / OCEAN_CASCADE_SIZES_M[c]);
        result.cascades[c] = glm::vec4(float(offset.x), float(offset.y), float(1.0 / OCEAN_CASCADE_SIZES_M[c]),
                                       float(spectrum_.mean_square_slope[c]));
    }
    return result;
}

double OceanWaves::residual_mean_square_slope(double target) const noexcept {
    return std::max(target - spectrum_.total_mean_square_slope, 0.0);
}

void OceanWaves::update(VkCommandBuffer cmd, rhi::VkContext& ctx, const OceanSpectrumParameters& parameters, double seconds) {
    constexpr uint32_t n = OCEAN_FFT_SIZE;
    if (spectrum_builds_ == 0 || !(parameters == parameters_)) {
        spectrum_ = build_ocean_spectrum(parameters);
        parameters_ = parameters;
        ++spectrum_builds_;
        log::info("Ocean spectrum: wind {:.1f} m/s, gravity {:.2f} m/s^2, Hs {:.2f} m, mean square slope {:.4f}",
                  parameters.wind_ms, parameters.gravity, 4.0 * std::sqrt(spectrum_.height_variance), spectrum_.total_mean_square_slope);
    }
    uint32_t frame = ctx.frame_index();
    if (uploaded_[frame] != spectrum_builds_) {
        size_t bytes = spectrum_.amplitudes.size() * sizeof(glm::vec4);
        std::memcpy(amplitudes_[frame].mapped, spectrum_.amplitudes.data(), bytes);
        ctx.flush_buffer(amplitudes_[frame], 0, bytes);
        uploaded_[frame] = spectrum_builds_;
    }
    auto compute_barrier = [&] {
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    };
    compute_barrier();
    EvolvePush push{};
    push.seconds = seconds;
    push.gravity = float(parameters.gravity);
    push.choppiness = float(OCEAN_CHOPPINESS);
    push.lengths = glm::vec4(float(OCEAN_CASCADE_SIZES_M[0]), float(OCEAN_CASCADE_SIZES_M[1]),
                             float(OCEAN_CASCADE_SIZES_M[2]), float(OCEAN_CASCADE_SIZES_M[3]));
    push.last_length = float(OCEAN_CASCADE_SIZES_M[4]);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, evolve_pipeline_.handle);
    ctx.push_descriptors(cmd, evolve_pipeline_, {
        rhi::PushDescriptorWrite::storage_buffer(0, amplitudes_[frame]),
        rhi::PushDescriptorWrite::storage_image(1, field_),
    });
    ctx.push_constants(cmd, evolve_pipeline_, push);
    vkCmdDispatch(cmd, n / 8, n / 8, OCEAN_CASCADE_COUNT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fft_pipeline_.handle);
    ctx.push_descriptors(cmd, fft_pipeline_, {rhi::PushDescriptorWrite::storage_image(1, field_)});
    for (uint32_t vertical = 0; vertical < 2; ++vertical) {
        compute_barrier();
        ctx.push_constants(cmd, fft_pipeline_, vertical);
        vkCmdDispatch(cmd, n, 1, OCEAN_CASCADE_COUNT);
    }
    compute_barrier();
    ctx.transition_image(cmd, base_, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_BLIT_BIT, 0,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, moments_pipeline_.handle);
    ctx.push_descriptors(cmd, moments_pipeline_, {
        rhi::PushDescriptorWrite::storage_image(1, field_),
        rhi::PushDescriptorWrite::storage_image(2, base_),
    });
    ctx.push_constants(cmd, moments_pipeline_, float(OCEAN_CHOPPINESS));
    vkCmdDispatch(cmd, n / 8, n / 8, OCEAN_CASCADE_COUNT);
    ctx.build_mip_chain(cmd, base_, moments_, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
}

}
