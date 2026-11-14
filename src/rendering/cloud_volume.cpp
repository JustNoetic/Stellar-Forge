#include "rendering/cloud_volume.hpp"
#include "rendering/cloud_maps.hpp"
#include "rendering/render_settings.hpp"
#include "core/camera.hpp"
#include "core/constants.hpp"
#include "core/coordinates.hpp"
#include "core/log.hpp"
#include "ephemeris/system_manager.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <glm/gtc/packing.hpp>

namespace sf::rendering {

namespace {

double smooth(double a, double b, double x) {
    double t = std::clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

glm::dmat3 body_fixed_rotation(glm::dvec3 pole, double spin) {
    glm::dvec3 reference = std::abs(pole.x) < 0.9 ? glm::dvec3(1.0, 0.0, 0.0) : glm::dvec3(0.0, 0.0, 1.0);
    glm::dvec3 x = glm::normalize(reference - pole * glm::dot(reference, pole));
    glm::dvec3 z = glm::cross(x, pole);
    double c = std::cos(spin), s = std::sin(spin);
    return glm::dmat3(x * c - z * s, pole, x * s + z * c);
}

glm::dvec3 fract(glm::dvec3 v) { return v - glm::floor(v); }

glm::dvec3 wrap(glm::dvec3 v, double period) { return v - glm::floor(v / period) * period; }

struct NoisePush {
    uint32_t detail = 0;
    uint32_t normalize = 0;
    uint32_t size = 0;
    uint32_t range_offset = 0;
};

}

uint32_t cloud_type_index(std::string_view type) noexcept {
    if (type == "stratus") return uint32_t(CloudType::Stratus);
    if (type == "mixed") return uint32_t(CloudType::Mixed);
    if (type == "cirrus") return uint32_t(CloudType::Cirrus);
    return uint32_t(CloudType::Cumulus);
}

CloudProfileSample cloud_profile(uint32_t type, double h) noexcept {
    if (!(h > 0.0 && h < 1.0)) return {};
    switch (CloudType(type)) {
    case CloudType::Stratus:
        return {smooth(0.0, 0.08, h) * (1.0 - smooth(0.18, 0.4, h)), 1.0};
    case CloudType::Mixed:
        return {smooth(0.0, 0.07, h) * (1.0 - smooth(0.3, 0.7, h)), smooth(0.0, 0.1, h) * (1.0 - 0.3 * h)};
    case CloudType::Cirrus:
        return {smooth(0.75, 0.85, h) * (1.0 - smooth(0.9, 1.0, h)), 0.5};
    case CloudType::Cumulus:
    default:
        // Flat bases, and tops that rise with coverage: density needs shape > 1 - coverage.
        return {smooth(0.0, 0.06, h) * (1.0 - smooth(0.15, 1.0, h)), smooth(0.0, 0.12, h) * (1.0 - 0.55 * h)};
    }
}

CloudProfileTable build_cloud_profile_table() {
    CloudProfileTable table;
    constexpr uint32_t N = CLOUD_PROFILE_HEIGHTS;
    table.texels.resize(size_t(CLOUD_PROFILE_TYPES) * N * 2);
    for (uint32_t type = 0; type < CLOUD_PROFILE_TYPES; ++type) {
        std::array<float, N> shape{};
        for (uint32_t i = 0; i < N; ++i) {
            auto sample = cloud_profile(type, double(i) / double(N - 1));
            uint16_t s = glm::packHalf1x16(float(sample.shape)), d = glm::packHalf1x16(float(sample.density));
            table.texels[(size_t(type) * N + i) * 2] = s;
            table.texels[(size_t(type) * N + i) * 2 + 1] = d;
            shape[i] = glm::unpackHalf1x16(s);
        }
        // Linear filtering keeps a value inside its segment's texel range, so the interval of segments whose
        // larger end exceeds t contains every height the filtered shape can exceed t. Half a texel of margin
        // absorbs the sampler's sub-texel quantization.
        const float margin = 0.5f / float(N - 1);
        for (uint32_t k = 0; k < CLOUD_PROFILE_THRESHOLDS; ++k) {
            float t = float(k) / float(CLOUD_PROFILE_THRESHOLDS - 1);
            int first = -1, last = -1;
            for (uint32_t i = 0; i + 1 < N; ++i)
                if (std::max(shape[i], shape[i + 1]) > t) { if (first < 0) first = int(i); last = int(i); }
            glm::vec2 interval(1.0f, 0.0f);
            if (first >= 0)
                interval = {std::max(float(first) / float(N - 1) - margin, 0.0f),
                            std::min(float(last + 1) / float(N - 1) + margin, 1.0f)};
            table.bounds[type * CLOUD_PROFILE_THRESHOLDS + k] = interval;
        }
    }
    return table;
}

void CloudWindOffset::advance(uint64_t identity, glm::dvec3 camera_fixed_km, double seconds, double wind_ms,
                              double shape_tile_km, double detail_tile_km) {
    if (!tracking_ || identity != identity_ || shape_tile_km != shape_tile_km_ || detail_tile_km != detail_tile_km_) {
        tracking_ = true;
        identity_ = identity;
        seconds_ = seconds;
        shape_tile_km_ = shape_tile_km;
        detail_tile_km_ = detail_tile_km;
        wind_shape_ = wind_detail_ = glm::dvec3(0.0);
        return;
    }
    double distance = glm::length(camera_fixed_km);
    glm::dvec3 up = distance > 1e-9 ? camera_fixed_km / distance : glm::dvec3(1.0, 0.0, 0.0);
    // The body-fixed pole is +y; the coverage drift in cloud_weather moves features toward +longitude (east).
    glm::dvec3 east = glm::cross(glm::dvec3(0.0, 1.0, 0.0), up);
    east = glm::length(east) > 1e-9 ? glm::normalize(east) : glm::dvec3(0.0, 0.0, 1.0);
    glm::dvec3 step = east * (wind_ms * 1e-3 * (seconds - seconds_));
    seconds_ = seconds;
    wind_shape_ = wrap(wind_shape_ + step, shape_tile_km_);
    wind_detail_ = wrap(wind_detail_ + step, detail_tile_km_);
}

glm::dvec3 CloudWindOffset::shape_anchor(glm::dvec3 camera_fixed_km) const noexcept {
    return fract(wrap(camera_fixed_km, shape_tile_km_) / shape_tile_km_ - wind_shape_ / shape_tile_km_);
}

glm::dvec3 CloudWindOffset::detail_anchor(glm::dvec3 camera_fixed_km) const noexcept {
    return fract(wrap(camera_fixed_km, detail_tile_km_) / detail_tile_km_ - wind_detail_ / detail_tile_km_);
}

void CloudVolumeRenderer::init(rhi::VkContext& ctx) {
    profile_ = build_cloud_profile_table();
    profile_lut_ = ctx.create_image_2d(CLOUD_PROFILE_HEIGHTS, CLOUD_PROFILE_TYPES, VK_FORMAT_R16G16_SFLOAT,
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    ctx.upload_image_2d(profile_lut_, profile_.texels.data(), profile_.texels.size() * sizeof(uint16_t));
    for (auto& buffer : param_buffers_)
        buffer = ctx.create_host_buffer(sizeof(CloudVolumeGPU), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    generate_noise(ctx);
}

void CloudVolumeRenderer::generate_noise(rhi::VkContext& ctx) {
    auto start = std::chrono::steady_clock::now();
    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    shape_noise_ = ctx.create_image_3d(CLOUD_SHAPE_NOISE_SIZE, CLOUD_SHAPE_NOISE_SIZE, CLOUD_SHAPE_NOISE_SIZE,
                                       VK_FORMAT_R8G8B8A8_UNORM, usage);
    detail_noise_ = ctx.create_image_3d(CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE, CLOUD_DETAIL_NOISE_SIZE,
                                        VK_FORMAT_R8G8B8A8_UNORM, usage);
    // Per-channel float-bit ranges: [offset + c] minimum, [offset + 4 + c] maximum. Values are non-negative,
    // so unsigned comparison of their bits orders them.
    std::array<uint32_t, 16> range{};
    for (uint32_t i = 0; i < 16; ++i) range[i] = (i % 8) < 4 ? 0x7f800000u : 0u;
    auto range_buffer = ctx.create_device_buffer(sizeof(range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, range.data());
    constexpr auto compute = VK_SHADER_STAGE_COMPUTE_BIT;
    const rhi::DescriptorLayoutBinding bindings[] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, compute},
                                                     {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, compute}};
    auto pipeline = rhi::ShaderCache::create_compute_pipeline(ctx, "clouds/noise_generate.comp", bindings, sizeof(NoisePush));
    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        for (uint32_t detail = 0; detail < 2; ++detail) {
            rhi::Image& image = detail ? detail_noise_ : shape_noise_;
            uint32_t size = detail ? CLOUD_DETAIL_NOISE_SIZE : CLOUD_SHAPE_NOISE_SIZE;
            ctx.transition_image(cmd, image, VK_IMAGE_LAYOUT_GENERAL);
            ctx.push_descriptors(cmd, pipeline, {rhi::PushDescriptorWrite::storage_image(0, image),
                                                 rhi::PushDescriptorWrite::storage_buffer(1, range_buffer)});
            for (uint32_t pass = 0; pass < 2; ++pass) {
                NoisePush push{detail, pass, size, detail * 8u};
                vkCmdPushConstants(cmd, pipeline.layout, compute, 0, sizeof(push), &push);
                vkCmdDispatch(cmd, size / 4, size / 4, size / 4);
                ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }
            ctx.transition_image(cmd, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });
    rhi::ShaderCache::destroy_pipeline(ctx, pipeline);
    ctx.destroy_buffer(range_buffer);
    noise_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    log::info("Cloud volume noise: {}^3 shape, {}^3 detail, {:.0f} ms", CLOUD_SHAPE_NOISE_SIZE,
              CLOUD_DETAIL_NOISE_SIZE, noise_ms_);
}

void CloudVolumeRenderer::shutdown(rhi::VkContext& ctx) {
    ctx.destroy_image(shape_noise_);
    ctx.destroy_image(detail_noise_);
    ctx.destroy_image(profile_lut_);
    for (auto& buffer : param_buffers_) ctx.destroy_buffer(buffer);
    wind_.reset();
}

CloudVolumeGPU CloudVolumeRenderer::base_parameters(const CloudMaps& maps) const {
    CloudVolumeGPU p{};
    const auto& bounds = maps.bound_image();
    p.flags = {CLOUD_VOLUME_NONE, bounds.mip_levels, bounds.extent.width, bounds.extent.height};
    p.erosion = {0.5f, 0.3f, float(CLOUD_VOLUME_MAX_COVERAGE_LOD), 0.0f};
    for (size_t i = 0; i < profile_.bounds.size(); ++i) p.bounds[i] = glm::vec4(profile_.bounds[i], 0.0f, 0.0f);
    return p;
}

void CloudVolumeRenderer::prepare(rhi::VkContext& ctx, const Camera& cam, const ephemeris::SystemSnapshot& sys,
                                  const RenderSettings& settings, const CloudMaps& maps, uint32_t count) {
    params_ = base_parameters(maps);
    int focus = -1;
    double best = std::numeric_limits<double>::infinity();
    glm::dvec3 focus_offset{0.0};
    for (uint32_t i = 0; settings.clouds_enabled && i < count; ++i) {
        const auto& b = sys.body(i);
        const auto& d = b.definition;
        const auto& c = d.clouds;
        if (d.is_star || d.is_black_hole || !c.enabled || c.coverage <= 0 || c.density <= 0 || c.extinction_per_km <= 0)
            continue;
        glm::dvec3 offset = -camera_relative(b.render.position, cam.eye_pos_render()) * AU_TO_KM;
        glm::dvec3 pole = glm::length(b.render.pole) > 1e-4 ? glm::normalize(b.render.pole) : glm::dvec3(0.0, 1.0, 0.0);
        double flattening = 1.0 / std::max(1.0 - d.oblateness, 0.05) - 1.0;
        double radius = glm::length(offset + pole * glm::dot(offset, pole) * flattening);
        double top = d.radius_au() * AU_TO_KM + b.render.cloud_base_altitude_km + c.thickness_km;
        double altitude = radius - top;
        if (altitude >= CLOUD_VOLUME_FADE_END_KM || altitude >= best) continue;
        best = altitude;
        focus = int(i);
        focus_offset = offset;
    }
    if (focus < 0) wind_.reset();
    else {
        const auto& b = sys.body(uint32_t(focus));
        const auto& c = b.definition.clouds;
        glm::dvec3 pole = glm::length(b.render.pole) > 1e-4 ? glm::normalize(b.render.pole) : glm::dvec3(0.0, 1.0, 0.0);
        glm::dvec3 camera_fixed = glm::transpose(body_fixed_rotation(pole, b.render.spin_angle_rad)) * focus_offset;
        double shape_tile = CLOUD_NOISE_TILE_CELLS * std::max(c.shape_size_km, 1e-3);
        double detail_tile = CLOUD_NOISE_TILE_CELLS * std::max(c.detail_size_km, 1e-3);
        wind_.advance(b.revisions.identity, camera_fixed, sys.time().value * SECONDS_PER_YEAR, c.wind_speed_ms,
                      shape_tile, detail_tile);
        params_.flags.x = uint32_t(focus);
        params_.camera_local = glm::vec4(glm::vec3(focus_offset), 0.0f);
        params_.shape_anchor = glm::vec4(glm::vec3(wind_.shape_anchor(camera_fixed)), float(1.0 / shape_tile));
        params_.detail_anchor = glm::vec4(glm::vec3(wind_.detail_anchor(camera_fixed)), float(1.0 / detail_tile));
        params_.erosion.x = float(std::clamp(0.35 + 0.5 * c.erosion, 0.0, 0.9));
    }
    std::memcpy(param_buffers_[ctx.frame_index()].mapped, &params_, sizeof(params_));
}

std::array<rhi::PushDescriptorWrite, 4> CloudVolumeRenderer::resource_writes(rhi::VkContext& ctx, const CloudMaps& maps) const {
    return {rhi::PushDescriptorWrite::sampled_image(41, shape_noise_, ctx.sampler_linear_repeat()),
            rhi::PushDescriptorWrite::sampled_image(42, detail_noise_, ctx.sampler_linear_repeat()),
            rhi::PushDescriptorWrite::sampled_image(43, profile_lut_, ctx.sampler_linear_clamp()),
            rhi::PushDescriptorWrite::sampled_image(44, maps.bound_image(), ctx.sampler_nearest_clamp())};
}

std::array<rhi::PushDescriptorWrite, 5> CloudVolumeRenderer::writes(rhi::VkContext& ctx, const CloudMaps& maps) const {
    auto resources = resource_writes(ctx, maps);
    return {rhi::PushDescriptorWrite::uniform_buffer(40, param_buffers_[ctx.frame_index()]),
            resources[0], resources[1], resources[2], resources[3]};
}

}
