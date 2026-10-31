#pragma once

#include "terrain/surface_textures.hpp"
#include "terrain/terrain_source.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace sf::terrain {

inline constexpr uint32_t SURFACE_PREVIEW_W = 512;
inline constexpr uint32_t SURFACE_PREVIEW_H = 256;

struct StagedImageInfo {
    std::filesystem::path path;
    uint32_t width = 0;
    uint32_t height = 0;
    int recommended_lod = 3;
    std::vector<uint8_t> preview_rgba;
};

struct SurfaceBakeRequest {
    std::string system_name;
    std::string body_name;
    bool is_height = false;
    std::filesystem::path source_file;
    int level = 3;
    double polar_ratio = 1.0;
    double radius_m = 6371000.0;
    double height_min_km = -10.0;
    double height_max_km = 10.0;
    std::string backend = "auto";
    bool copy_to_system_textures = true;
};

struct SurfaceBakeResult {
    bool success = false;
    std::string body_name;
    bool is_height = false;
    int level = 0;
    double height_min_km = 0.0;
    double height_max_km = 0.0;
    std::filesystem::path pack_path;
    std::string backend_used;
    double elapsed_sec = 0.0;
    std::string error;
};

class SurfaceBaker {
public:
    SurfaceBaker() = default;
    ~SurfaceBaker();

    SurfaceBaker(const SurfaceBaker&) = delete;
    SurfaceBaker& operator=(const SurfaceBaker&) = delete;

    bool start_bake(SurfaceBakeRequest request);
    std::optional<SurfaceBakeResult> poll();

    bool is_baking() const noexcept { return active_.load(std::memory_order_relaxed); }
    float progress() const noexcept { return progress_.load(std::memory_order_relaxed); }
    double elapsed_seconds() const noexcept;
    std::string status_text() const;
    std::string backend_name() const;

    static std::optional<std::filesystem::path> open_image_file_dialog(void* owner_hwnd = nullptr);
    static std::optional<std::filesystem::path> find_existing_source_image(
        std::string_view system_name, std::string_view body_name, bool is_height);
    static std::optional<StagedImageInfo> inspect_and_load_image(const std::filesystem::path& path);
    static std::vector<uint8_t> build_color_pack_preview(const SurfaceSource& source);
    static std::vector<uint8_t> build_height_pack_preview(const HeightSource& source);
    static std::vector<uint8_t> build_solid_preview(float r, float g, float b);
    static bool remove_from_catalog(std::string_view body_name, bool is_height);
    static std::filesystem::path resolve_python_executable();

private:
    void run_worker(SurfaceBakeRequest req);

    std::jthread worker_;
    std::atomic<bool> active_{false};
    std::atomic<bool> completed_{false};
    std::atomic<float> progress_{0.0f};
    std::chrono::steady_clock::time_point start_time_{};
    mutable std::mutex mutex_;
    std::string status_text_;
    std::string backend_name_;
    SurfaceBakeResult result_{};
};

} // namespace sf::terrain
