#pragma once

#include "terrain/geometry_residency.hpp"
#include "rhi/staging_uploads.hpp"
#include "core/camera.hpp"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>

namespace sf::terrain {

inline constexpr uint32_t COLOR_CORE = 128;
inline constexpr uint32_t COLOR_GUTTER = 2;
inline constexpr uint32_t COLOR_SIDE = COLOR_CORE + 2*COLOR_GUTTER;
inline constexpr uint32_t COLOR_BYTES = COLOR_SIDE*COLOR_SIDE*4;
inline constexpr uint32_t COLOR_SLOTS = 512;
inline constexpr uint32_t NORMAL_SLOTS = 512;
inline constexpr uint32_t COLOR_UPLOADS = 64;
inline constexpr uint32_t HEIGHT_UPLOADS = 128;
inline constexpr uint32_t NORMAL_BAKES_PER_FRAME = 64;
inline constexpr uint32_t COLOR_BODIES = 1024;
inline constexpr uint32_t COLOR_SOURCES = 32;
inline constexpr uint32_t HORIZON_DIRECTIONS = 16;
inline constexpr uint32_t HORIZON_LAYERS_PER_SLOT = 4;
inline constexpr uint32_t HORIZON_APRON_PAD = 30;
inline constexpr uint32_t HORIZON_APRON_SIDE = COLOR_SIDE + 2*HORIZON_APRON_PAD;
inline constexpr uint32_t HORIZON_BAKES_PER_FRAME = 32;
inline constexpr uint32_t HEIGHT_TILE_FLOATS = DENSE_HEIGHT_SIDE * DENSE_HEIGHT_SIDE;
inline constexpr uint32_t HEIGHT_TILE_BYTES = HEIGHT_TILE_FLOATS * sizeof(float);
using ColorTile = std::array<uint8_t, COLOR_BYTES>;

struct alignas(16) HorizonBakeJobGPU {
    glm::uvec4 address{0};
    glm::uvec4 target{0};
    glm::vec4 metrics{0};
    glm::vec4 bounds{0};
};
static_assert(sizeof(HorizonBakeJobGPU) == 64);

struct alignas(16) NormalBakeJobGPU {
    glm::uvec4 address{0};
    glm::uvec4 target{0};
    glm::vec4 metrics{0};
};
static_assert(sizeof(NormalBakeJobGPU) == 48);

struct SurfaceSource {
    virtual ~SurfaceSource() = default;
    SourceDomain domain;
    uint32_t level = 0;
    double polar_ratio = 1;
    virtual uint32_t count() const noexcept = 0;
    virtual std::vector<std::byte> load(TileId tile) const = 0;
    virtual std::optional<glm::vec3> mean_albedo() const noexcept { return std::nullopt; }
    virtual bool has_specular_mask() const noexcept { return false; }
    uint32_t index(TileId tile) const;
    virtual bool compatible(double radius_m, double polar) const noexcept;
};

struct ColorSource final : SurfaceSource {
    std::filesystem::path payload;
    std::vector<uint32_t> checksums;
    std::optional<glm::vec3> mean_albedo_;
    bool has_specular_ = false;
    static std::shared_ptr<const ColorSource> open(const std::filesystem::path& directory);
    std::shared_ptr<const ColorTile> read(TileId tile) const;
    uint32_t count() const noexcept override { return uint32_t(checksums.size()); }
    std::vector<std::byte> load(TileId tile) const override;
    std::optional<glm::vec3> mean_albedo() const noexcept override { return mean_albedo_; }
    bool has_specular_mask() const noexcept override { return has_specular_; }
};

struct NormalSource final : SurfaceSource {
    struct TileWithApron {
        std::vector<std::byte> normals;
        std::vector<float> apron_m;
    };
    explicit NormalSource(std::shared_ptr<const HeightSource> height);
    uint32_t count() const noexcept override { return 2*((1u << (2*(level+1)))-1); }
    std::vector<std::byte> load(TileId tile) const override;
    TileWithApron load_with_apron(TileId tile) const;
    std::vector<float> load_apron(TileId tile) const;
    uint32_t global_level() const noexcept { return std::min(level, 2u); }
    uint32_t global_face_size() const noexcept { return COLOR_CORE << global_level(); }
    std::vector<float> load_global_face(uint32_t face) const;
    bool compatible(double radius_m, double polar) const noexcept override;
    std::shared_ptr<const HeightSource> height;
    double height_scale_m = 1.0;

private:
    TileWithApron evaluate_tile(TileId tile, bool need_normals, bool need_apron) const;
};

struct ColorBinding {
    std::string body;
    std::shared_ptr<const SurfaceSource> source;
};

struct TextureStats {
    uint32_t sources = 0;
    uint32_t resident = 0;
    uint32_t uploads = 0;
    uint32_t pending = 0;
    uint32_t ready = 0;
    uint32_t failed = 0;
    uint32_t capacity = 0;
    uint32_t evictions = 0;
    uint32_t selection_rebuilds = 0;
    uint32_t selection_nodes = 0;
    uint32_t horizon_bakes = 0;
    uint32_t global_height_uploads = 0;
};

struct SurfaceTextureView {
    const ephemeris::BodyDefinition* definition = nullptr;
    SelectionView view;
    double radius_pixels = 0;
    uint32_t index = 0;
    float ocean_floor_m = -1e30f;
};

class SurfaceTextures {
public:
    void init(rhi::VkContext& ctx, bool normals = false, bool horizons = false);
    void shutdown(rhi::VkContext& ctx);
    void set_sources(std::vector<ColorBinding> bindings);
    void set_color_source(std::string body, std::shared_ptr<const SurfaceSource> source);
    void set_height_source(std::string body, std::shared_ptr<const HeightSource> source);
    std::shared_ptr<const SurfaceSource> color_source(std::string_view body) const noexcept;
    std::vector<std::pair<std::string, std::filesystem::path>> load_catalog(const std::filesystem::path& path);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                 const ephemeris::SystemSnapshot& system, VkExtent2D extent, bool enabled,
                 bool horizons_enabled = false, bool oceans_enabled = false,
                 double target_error_pixels = 1.0);
    void prepare_views(VkCommandBuffer cmd, rhi::VkContext& ctx, std::span<const SurfaceTextureView> views,
                       bool enabled, bool horizons_enabled = false, double target_error_pixels = 1.0);
    int source_index(const ephemeris::BodyDefinition& body) const;
    std::optional<glm::vec3> mean_albedo(const ephemeris::BodyDefinition& body) const noexcept;
    const rhi::Image& image() const noexcept { return atlas_; }
    const rhi::Image& horizon_image() const noexcept { return horizon_atlas_; }
    const rhi::Buffer& horizon_bounds() const noexcept { return horizon_bounds_; }
    const rhi::Buffer& pages(uint32_t frame) const noexcept { return pages_[frame]; }
    const TextureStats& stats() const noexcept { return stats_; }
    bool body_active(uint32_t body) const noexcept {
        return body < COLOR_BODIES && words_.size() >= COLOR_BODIES * 4 && words_[body * 4] != 0u;
    }
    uint32_t body_flags(uint32_t body) const noexcept {
        return body_active(body) ? words_[body * 4 + 2] : 0u;
    }

private:
    struct SelectionEntry { SelectionView view; Selection selection; bool valid = false; };
    struct Completed { GeometryKey key; std::vector<std::byte> payload; };
    struct CompletedHeight { GeometryKey key; std::shared_ptr<const HeightTile> tile; };
    struct GlobalHeightState {
        rhi::Image raw;
        rhi::Image pyramid;
        std::vector<VkImageView> mip_views;
        bool raw_baked = false;
        bool pyramid_initialized = false;
        bool pyramid_built = false;
        float built_ocean_floor_m = -1e30f;
    };
    struct RetiredGlobal {
        rhi::Image raw;
        rhi::Image pyramid;
        std::vector<VkImageView> mip_views;
        uint64_t serial = 0;
    };
    GeometryKey key(uint32_t source, TileId tile) const;
    void stop_workers();
    void start_workers();
    void run(std::stop_token stop);
    rhi::Image atlas_;
    rhi::Image horizon_atlas_;
    rhi::Buffer horizon_bounds_;
    rhi::Image apron_scratch_;
    rhi::Buffer height_pool_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> height_pages_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> normal_staging_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> normal_jobs_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> apron_jobs_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> horizon_jobs_;
    rhi::Pipeline normal_pipeline_;
    rhi::Pipeline apron_pipeline_;
    rhi::Pipeline global_bake_pipeline_;
    rhi::Pipeline horizon_pipeline_;
    rhi::Pipeline height_mip_pipeline_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> pages_;
    rhi::StagingUploads staging_;
    GeometryResidency slots_{COLOR_SLOTS};
    GeometryResidency height_slots_{0};
    uint32_t capacity_ = COLOR_SLOTS;
    uint32_t height_capacity_ = 0;
    std::array<std::optional<GeometryKey>, COLOR_SLOTS> owners_;
    std::vector<std::optional<GeometryKey>> height_owners_;
    std::array<uint8_t, COLOR_SLOTS> horizon_valid_{};
    std::vector<ColorBinding> sources_;
    std::vector<SelectionEntry> selections_;
    std::vector<GlobalHeightState> globals_;
    std::vector<RetiredGlobal> retired_globals_;
    SelectionScratch selection_scratch_;
    std::vector<uint32_t> offsets_;
    std::vector<uint32_t> words_;
    std::vector<uint32_t> height_words_;
    TextureStats stats_;
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::deque<GeometryKey> queue_;
    std::unordered_set<GeometryKey, GeometryKeyHash> in_flight_;
    std::unordered_set<GeometryKey, GeometryKeyHash> failed_;
    std::vector<Completed> ready_;
    std::deque<GeometryKey> height_queue_;
    std::unordered_set<GeometryKey, GeometryKeyHash> height_in_flight_;
    std::vector<CompletedHeight> height_ready_;
    std::string error_;
    bool reported_error_ = false;
    bool normals_ = false;
    bool horizons_ = false;
    std::vector<std::jthread> workers_;
};

}
