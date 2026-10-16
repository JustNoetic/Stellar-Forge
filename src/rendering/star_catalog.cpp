#include "rendering/star_catalog.hpp"
#include "core/constants.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"
#include "ephemeris/system_manager.hpp"
#include "rhi/shader_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace sf::rendering {

namespace {

#pragma pack(push, 1)
struct GaiaBinHeader {
    char magic[5];
    uint32_t version;
    uint32_t count;
};

struct GaiaBinRecord {
    float ra_deg;
    float dec_deg;
    float plx_mas;
    float pmra_mas_yr;
    float pmdec_mas_yr;
    float rv_km_s;
    float g_mag;
    float bp_rp;
};
#pragma pack(pop)

inline constexpr double PC_TO_AU = 206264.80624709636;
inline constexpr double MAS_TO_RAD = PI / (180.0 * 3600.0 * 1000.0);
inline constexpr double EPOCH_OFFSET_YEARS = 16.0;
inline constexpr double FLUX_SCALE = 1.0e13;

inline constexpr double TEFF_A = 8842.0;
inline constexpr double TEFF_B = -4578.875;
inline constexpr double TEFF_C = 1031.17;
inline constexpr double TEFF_D = -96.233;

struct alignas(16) StarfieldPushConstants {
    glm::mat4 view_proj{1.0f};
    glm::dvec4 eye_world_au{0.0};
    double sim_time_yr = 0.0;
    float flux_calib = 1.0f;
    float screen_height = 1080.0f;
    float point_base_px = 2.0f;
    float min_point_px = 2.0f;
    float max_point_px = 10.0f;
    float intensity_scale = 1.0f;
    float exposure = 1.0f;
    uint32_t hdr_enabled = 1u;
};

} // namespace

bool StarCatalog::init(rhi::VkContext& ctx, const std::filesystem::path& bin_path) {
    auto bytes = sf::paths::read_binary_file(bin_path);
    if (!bytes || bytes->size() < sizeof(GaiaBinHeader)) {
        sf::log::warn("Gaia star catalog not found at {}; starfield disabled.", bin_path.string());
        return false;
    }

    GaiaBinHeader hdr{};
    std::memcpy(&hdr, bytes->data(), sizeof(GaiaBinHeader));
    if (std::memcmp(hdr.magic, "GAIA1", 5) != 0 || hdr.version != 1) {
        sf::log::warn("Invalid Gaia catalog header in {}", bin_path.string());
        return false;
    }

    size_t expected_bytes = sizeof(GaiaBinHeader) + static_cast<size_t>(hdr.count) * sizeof(GaiaBinRecord);
    if (bytes->size() < expected_bytes) {
        sf::log::warn("Truncated Gaia catalog in {}", bin_path.string());
        return false;
    }

    const auto* raw_records = reinterpret_cast<const GaiaBinRecord*>(bytes->data() + sizeof(GaiaBinHeader));
    count_ = hdr.count;

    std::vector<StarGpuRecord> gpu_records(count_);
    const double ce = std::cos(OBLIQUITY);
    const double se = std::sin(OBLIQUITY);

    for (uint32_t i = 0; i < count_; ++i) {
        const GaiaBinRecord& r = raw_records[i];
        double ra = static_cast<double>(r.ra_deg) * DEG_TO_RAD;
        double dec = static_cast<double>(r.dec_deg) * DEG_TO_RAD;
        double plx = std::max(static_cast<double>(r.plx_mas), 1e-4);
        double pmra = static_cast<double>(r.pmra_mas_yr);
        double pmdec = static_cast<double>(r.pmdec_mas_yr);
        double g = static_cast<double>(r.g_mag);
        double bmrp = std::clamp(static_cast<double>(r.bp_rp), -0.5, 5.0);

        double cos_d = std::cos(dec);
        double sin_d = std::sin(dec);
        double cos_a = std::cos(ra);
        double sin_a = std::sin(ra);

        glm::dvec3 eq_dir{cos_d * cos_a, cos_d * sin_a, sin_d};
        glm::dvec3 east{-sin_a, cos_a, 0.0};
        glm::dvec3 north{-sin_d * cos_a, -sin_d * sin_a, cos_d};
        glm::dvec3 pm_eq = (pmra * east + pmdec * north) * MAS_TO_RAD;

        auto to_ecl = [&](const glm::dvec3& v) -> glm::dvec3 {
            return {v.x, ce * v.y + se * v.z, -se * v.y + ce * v.z};
        };

        glm::dvec3 dir_r = sf::ecliptic_to_render(to_ecl(eq_dir));
        glm::dvec3 pm_rate = sf::ecliptic_to_render(to_ecl(pm_eq));

        double dist_pc = 1000.0 / plx;
        double dist_au = dist_pc * PC_TO_AU;

        glm::dvec3 pm_vec = pm_rate * dist_au;
        glm::dvec3 pos0 = dir_r * dist_au - pm_vec * EPOCH_OFFSET_YEARS;

        double abs_mag = g - 5.0 * (std::log10(dist_pc) - 1.0);
        double teff = std::clamp(TEFF_A + TEFF_B * bmrp + TEFF_C * bmrp * bmrp + TEFF_D * bmrp * bmrp * bmrp,
                                 2800.0, 40000.0);
        glm::vec3 rgb = sf::ephemeris::temperature_to_rgb(teff);
        float flux_scaled = static_cast<float>(std::pow(10.0, -0.4 * abs_mag) * FLUX_SCALE);

        StarGpuRecord out{};
        out.pos0 = glm::dvec4(pos0, 0.0);
        out.pm_flux = glm::vec4(static_cast<float>(pm_vec.x), static_cast<float>(pm_vec.y),
                                static_cast<float>(pm_vec.z), flux_scaled);
        out.color = glm::vec4(rgb, 1.0f);
        gpu_records[i] = out;
    }

    star_ssbo_ = ctx.create_device_buffer(
        gpu_records.size() * sizeof(StarGpuRecord),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        gpu_records.data());

    rhi::GraphicsPipelineDesc desc{};
    desc.vert_path = "celestial/starfield.vert";
    desc.frag_path = "celestial/starfield.frag";
    desc.bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1},
    };
    desc.push_constant_size = sizeof(StarfieldPushConstants);
    desc.color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
    desc.depth_format = VK_FORMAT_D32_SFLOAT;
    desc.depth_test = true;
    desc.depth_write = false;
    desc.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    desc.additive_blend = true;
    desc.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    desc.cull_mode = VK_CULL_MODE_NONE;

    pipeline_ = rhi::ShaderCache::create_graphics_pipeline(ctx, desc);
    loaded_ = true;

    sf::log::info("Loaded {} Gaia DR3 stars into GPU SSBO ({:.1f} MB)", count_,
                  (gpu_records.size() * sizeof(StarGpuRecord)) / (1024.0 * 1024.0));
    return true;
}

void StarCatalog::shutdown(rhi::VkContext& ctx) {
    rhi::ShaderCache::destroy_pipeline(ctx, pipeline_);
    ctx.destroy_buffer(star_ssbo_);
    loaded_ = false;
    count_ = 0;
}

void StarCatalog::render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                         units::Years sim_time, const RenderSettings& settings, VkExtent2D extent) const {
    if (!loaded_ || !settings.show_gaia_stars || count_ == 0) {
        return;
    }

    float fb_w = static_cast<float>( std::max(1u, extent.width));
    float fb_h = static_cast<float>(std::max(1u, extent.height));
    float aspect = fb_w / fb_h;
    float fov_rad = static_cast<float>(cam.fov_deg * DEG_TO_RAD);
    float fov_factor = 1.0f / (2.0f * std::tan(0.5f * fov_rad));

    float star_size_px = std::clamp(settings.star_point_size * (fb_h / 1080.0f), 2.0f, 10.0f);
    float flux_calib = static_cast<float>(3.879 * std::pow(static_cast<double>(fb_h * fov_factor), 2.0) *
                                          star_size_px * star_size_px / 1.0e13);

    StarfieldPushConstants pc{};
    pc.view_proj = cam.proj_matrix(aspect, 0.1f) * cam.view_matrix_eye_relative();
    pc.eye_world_au = glm::dvec4(cam.eye_pos_render().value, 0.0);
    pc.sim_time_yr = sim_time.value;
    pc.flux_calib = flux_calib;
    pc.screen_height = fb_h;
    pc.point_base_px = settings.star_point_size;
    pc.min_point_px = 2.0f;
    pc.max_point_px = 10.0f;
    pc.intensity_scale = settings.star_intensity;
    pc.exposure = settings.exposure;
    pc.hdr_enabled = settings.hdr_enabled ? 1u : 0u;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle);
    rhi::PushDescriptorWrite writes[] = {
        rhi::PushDescriptorWrite::storage_buffer(0, star_ssbo_),
    };
    ctx.push_descriptors(cmd, pipeline_, writes);
    ctx.push_constants(cmd, pipeline_, pc);
    vkCmdDraw(cmd, count_, 1, 0, 0);
}

} // namespace sf::rendering
