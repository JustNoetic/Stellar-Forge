#pragma once

#include "terrain/geometry_residency.hpp"
#include "rhi/staging_uploads.hpp"
#include "core/camera.hpp"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace sf::terrain {

inline constexpr uint32_t COLOR_CORE = 128;
inline constexpr uint32_t COLOR_GUTTER = 2;
inline constexpr uint32_t COLOR_SIDE = COLOR_CORE + 2*COLOR_GUTTER;
inline constexpr uint32_t COLOR_BYTES = COLOR_SIDE*COLOR_SIDE*4;
inline constexpr uint32_t COLOR_SLOTS = 512;
inline constexpr uint32_t COLOR_UPLOADS = 16;
inline constexpr uint32_t COLOR_BODIES = 1024;
inline constexpr uint32_t COLOR_SOURCES = 32;
using ColorTile = std::array<uint8_t, COLOR_BYTES>;

struct SurfaceSource {
    virtual ~SurfaceSource() = default;
    SourceDomain domain;
    uint32_t level = 0;
    double polar_ratio = 1;
    virtual uint32_t count() const noexcept = 0;
    virtual std::vector<std::byte> load(TileId tile) const = 0;
    uint32_t index(TileId tile) const;
    virtual bool compatible(double radius_m, double polar) const noexcept;
};

struct ColorSource final : SurfaceSource {
    std::filesystem::path payload;
    std::vector<uint32_t> checksums;
    static std::shared_ptr<const ColorSource> open(const std::filesystem::path& directory);
    std::shared_ptr<const ColorTile> read(TileId tile) const;
    uint32_t count() const noexcept override { return uint32_t(checksums.size()); }
    std::vector<std::byte> load(TileId tile) const override;
};

struct NormalSource final : SurfaceSource {
    explicit NormalSource(std::shared_ptr<const HeightSource> height);
    uint32_t count() const noexcept override { return 2*((1u << (2*(level+1)))-1); }
    std::vector<std::byte> load(TileId tile) const override;
    bool compatible(double radius_m, double polar) const noexcept override;
    std::shared_ptr<const HeightSource> height;
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
};

struct SurfaceTextureView {
    const ephemeris::BodyDefinition* definition = nullptr;
    SelectionView view;
    double radius_pixels = 0;
    uint32_t index = 0;
};

class SurfaceTextures {
public:
    void init(rhi::VkContext& ctx, bool normals = false);
    void shutdown(rhi::VkContext& ctx);
    void set_sources(std::vector<ColorBinding> bindings);
    void set_height_source(std::string body, std::shared_ptr<const HeightSource> source);
    std::vector<std::pair<std::string, std::filesystem::path>> load_catalog(const std::filesystem::path& path);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                 const ephemeris::SystemSnapshot& system, VkExtent2D extent, bool enabled);
    void prepare_views(VkCommandBuffer cmd, rhi::VkContext& ctx, std::span<const SurfaceTextureView> views, bool enabled);
    int source_index(const ephemeris::BodyDefinition& body) const;
    const rhi::Image& image() const noexcept { return atlas_; }
    const rhi::Buffer& pages(uint32_t frame) const noexcept { return pages_[frame]; }
    const TextureStats& stats() const noexcept { return stats_; }

private:
    struct SelectionEntry { SelectionView view; Selection selection; bool valid = false; };
    struct Completed { GeometryKey key; std::vector<std::byte> payload; };
    GeometryKey key(uint32_t source, TileId tile) const;
    void request(const GeometryKey& key);
    void run(std::stop_token stop);
    rhi::Image atlas_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> pages_;
    rhi::StagingUploads staging_;
    GeometryResidency slots_{COLOR_SLOTS};
    uint32_t capacity_ = COLOR_SLOTS;
    std::array<std::optional<GeometryKey>, COLOR_SLOTS> owners_;
    std::vector<ColorBinding> sources_;
    std::vector<SelectionEntry> selections_;
    std::vector<uint32_t> offsets_;
    std::vector<uint32_t> words_;
    TextureStats stats_;
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::deque<GeometryKey> queue_;
    std::unordered_set<GeometryKey, GeometryKeyHash> pending_;
    std::unordered_set<GeometryKey, GeometryKeyHash> failed_;
    std::vector<Completed> ready_;
    std::string error_;
    bool reported_error_ = false;
    bool planning_ = false;
    bool normals_ = false;
    std::jthread worker_;
};

}
