#pragma once

#include "rhi/vk_context.hpp"
#include <array>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

namespace sf::rendering {

inline constexpr uint32_t OCEAN_FFT_SIZE = 256;
inline constexpr uint32_t OCEAN_CASCADE_COUNT = 5;
inline constexpr std::array<double, OCEAN_CASCADE_COUNT> OCEAN_CASCADE_SIZES_M{2.0, 28.0, 392.0, 5488.0, 76832.0};
inline constexpr double OCEAN_FETCH_M = 1.0e6;
inline constexpr double OCEAN_CHOPPINESS = 1.0;
inline constexpr double OCEAN_ACTIVE_ALTITUDE_M = 1.0e6;

struct OceanSpectrumParameters {
    double wind_ms = 7.0;
    double wind_direction_rad = 0.0;
    double gravity = 9.81;
    bool operator==(const OceanSpectrumParameters&) const = default;
};

struct OceanSpectrum {
    std::vector<glm::vec4> amplitudes;
    std::array<double, OCEAN_CASCADE_COUNT> mean_square_slope{};
    double total_mean_square_slope = 0.0;
    double height_variance = 0.0;
};

OceanSpectrum build_ocean_spectrum(const OceanSpectrumParameters& parameters, uint32_t seed = 0x5eed0ceau);

struct alignas(16) OceanCascadesGPU {
    glm::vec4 tangent{0.0f};
    glm::vec4 bitangent{0.0f};
    glm::vec4 cascades[OCEAN_CASCADE_COUNT]{};
};

class OceanWaves {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void track(uint64_t body_identity, glm::dvec3 camera_fixed_m);
    void reset() noexcept { tracking_ = false; }
    void update(VkCommandBuffer cmd, rhi::VkContext& ctx, const OceanSpectrumParameters& parameters, double seconds);
    OceanCascadesGPU cascades() const;
    double residual_mean_square_slope(double target) const noexcept;
    double mean_square_slope() const noexcept { return spectrum_.total_mean_square_slope; }
    const rhi::Image& moments() const noexcept { return moments_; }
    uint64_t spectrum_builds() const noexcept { return spectrum_builds_; }

private:
    rhi::Image field_{};
    rhi::Image base_{};
    rhi::Image moments_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> amplitudes_{};
    std::array<uint64_t, rhi::FRAMES_IN_FLIGHT> uploaded_{};
    rhi::Pipeline evolve_pipeline_{}, fft_pipeline_{}, moments_pipeline_{};
    OceanSpectrumParameters parameters_{};
    OceanSpectrum spectrum_{};
    uint64_t spectrum_builds_ = 0;
    bool tracking_ = false;
    uint64_t body_identity_ = 0;
    glm::dvec3 camera_{0.0}, tangent_{1.0, 0.0, 0.0}, bitangent_{0.0, 0.0, 1.0};
    glm::dvec2 planar_{0.0};
};

}
