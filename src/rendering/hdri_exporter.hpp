#pragma once

#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "rendering/celestial_renderer.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/star_catalog.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sf::rendering {

enum class HdriQuality {
    P720,
    P1080,
    K2,
    K4,
    K8,
    K16,
    K32,
    Count
};

enum class HdriOrientation {
    CameraHorizon,
    EclipticNorth
};

enum class HdriExposureMode {
    NormalizedCurrent,
    RawPhysical
};

struct HdriPresetInfo {
    const char* label;
    uint32_t width;
    uint32_t height;
    uint32_t face_size;
};

struct HdriExportOptions {
    std::filesystem::path output_path;
    HdriQuality quality = HdriQuality::K8;
    HdriOrientation orientation = HdriOrientation::CameraHorizon;
    HdriExposureMode exposure_mode = HdriExposureMode::NormalizedCurrent;
};

class HdriExporter {
public:
    static HdriPresetInfo preset_info(HdriQuality quality, uint32_t max_image_dim = 32768) noexcept;
    static std::optional<std::filesystem::path> save_file_dialog(const std::filesystem::path& initial_path, void* owner_hwnd = nullptr);

    void begin(const HdriExportOptions& options, const Camera& camera, const RenderSettings& settings,
               uint32_t max_image_dim, bool fp32_supported, bool was_paused);
    bool step(rhi::VkContext& ctx, CelestialRenderer& celestial, StarCatalog& stars,
              const ephemeris::SystemSnapshot& system);
    void cancel(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);

    bool active() const noexcept { return phase_ != Phase::Idle; }
    bool finished() const noexcept { return phase_ == Phase::Done || phase_ == Phase::Failed; }
    bool succeeded() const noexcept { return phase_ == Phase::Done; }
    bool was_paused() const noexcept { return was_paused_; }
    uint32_t current_face() const noexcept { return face_index_; }
    float progress() const noexcept;
    const std::string& status_text() const noexcept { return status_text_; }
    const std::string& error() const noexcept { return error_; }
    const std::filesystem::path& output_path() const noexcept { return options_.output_path; }
    void acknowledge() noexcept { phase_ = Phase::Idle; error_.clear(); status_text_.clear(); }

private:
    enum class Phase { Idle, RenderFace, WriteExr, Done, Failed };

    struct MappedFaceFile {
        std::filesystem::path path;
        void* file_handle = nullptr;
        void* mapping_handle = nullptr;
        const float* data = nullptr;
    };

    void release_gpu_targets(rhi::VkContext& ctx);
    void release_face_storage();
    void render_single_face(rhi::VkContext& ctx, CelestialRenderer& celestial, StarCatalog& stars,
                            const ephemeris::SystemSnapshot& system, uint32_t face);
    void write_equirectangular_exr();

    Phase phase_ = Phase::Idle;
    HdriExportOptions options_{};
    HdriPresetInfo preset_{};
    uint32_t padded_face_ = 0;
    VkFormat target_format_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    bool was_paused_ = false;
    uint32_t face_index_ = 0;
    std::string status_text_;
    std::string error_;

    Camera base_camera_{};
    RenderSettings export_settings_{};
    glm::dvec3 basis_x_{1.0, 0.0, 0.0};
    glm::dvec3 basis_y_{0.0, 0.0, -1.0};
    glm::dvec3 basis_z_{0.0, 1.0, 0.0};

    rhi::Image export_color_{};
    rhi::Image export_depth_{};
    rhi::Buffer readback_{};

    bool use_file_backing_ = false;
    std::array<std::vector<float>, 6> ram_faces_{};
    std::array<MappedFaceFile, 6> mapped_faces_{};
};

} // namespace sf::rendering
