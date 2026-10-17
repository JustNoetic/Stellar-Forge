#pragma once

#include "terrain/tile_metadata.hpp"
#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <mutex>
#include <atomic>

namespace sf::terrain {

struct SourceDomain {
    uint64_t identity = 0;
    uint64_t revision = 0;
    bool operator==(const SourceDomain&) const = default;
};

struct SurfaceTile {
    TileBounds bounds;
    double geometric_error = 0.0;
};

class TerrainSource {
public:
    virtual ~TerrainSource() = default;
    virtual SourceDomain domain() const noexcept = 0;
    virtual SurfaceTile describe(TileId tile, double polar_ratio, double amplitude) const = 0;
};

class ProceduralSource final : public TerrainSource {
public:
    explicit ProceduralSource(TileMetadataCache* cache = nullptr) : cache_(cache) {}
    SourceDomain domain() const noexcept override { return {}; }
    SurfaceTile describe(TileId tile, double polar_ratio, double amplitude) const override;

private:
    TileMetadataCache* cache_;
};

inline constexpr uint32_t HEIGHT_SIDE = GRID_CELLS + 3;
inline constexpr size_t MAX_SOURCE_RECORDS = 16384;

struct HeightTile {
    std::array<float, HEIGHT_SIDE * HEIGHT_SIDE> heights_m{};
};

struct HeightRecord {
    TileId tile;
    uint64_t offset = 0;
    uint32_t checksum = 0;
    double minimum_m = 0.0;
    double maximum_m = 0.0;
    double height_error_m = 0.0;
    double gradient_m = 0.0;
};

struct HeightSourceInfo {
    SourceDomain domain;
    std::string name;
    double equatorial_radius_m = 0.0;
    double polar_ratio = 1.0;
    double prime_meridian_rad = 0.0;
    double minimum_m = 0.0;
    double maximum_m = 0.0;
    double encoding_error_m = 0.0;
    double maximum_gradient_m = 0.0;
    uint32_t data_level = 0;
    std::array<double, MAX_LEVEL + 1> shared_height_error_m{};
};

class HeightSource final : public TerrainSource {
public:
    static std::shared_ptr<HeightSource> open(const std::filesystem::path& directory);
    SourceDomain domain() const noexcept override { return info_.domain; }
    SurfaceTile describe(TileId tile, double polar_ratio, double amplitude) const override;
    const HeightSourceInfo& info() const noexcept { return info_; }
    const HeightRecord& record(TileId tile) const;
    HeightTile read(TileId tile) const;
    void retain_envelope(TileId tile, const HeightTile& payload) const;
    double morph_error(uint32_t level) const noexcept;
    double approximation_error(TileId tile, uint32_t stride = 1) const;
    uint64_t bounds_revision() const noexcept { return bounds_revision_.load(std::memory_order_relaxed); }
    bool compatible(double radius_m, double polar_ratio) const noexcept;
    std::span<const HeightRecord> records() const noexcept { return records_; }

private:
    HeightSourceInfo info_;
    std::filesystem::path payload_;
    std::vector<HeightRecord> records_;
    std::array<size_t, MAX_LEVEL + 1> level_offsets_{};
    mutable std::mutex envelope_mutex_;
    mutable std::atomic<uint64_t> bounds_revision_{0};
    mutable std::list<std::pair<uint64_t, std::shared_ptr<const HeightTile>>> envelopes_;
};

double height_sample(const HeightTile& payload, double x, double y) noexcept;
uint32_t height_checksum(std::span<const std::byte> bytes) noexcept;

}
