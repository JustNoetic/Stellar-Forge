#pragma once

#include "terrain/quadtree.hpp"
#include <list>
#include <unordered_map>

namespace sf::terrain {

inline constexpr double RELIEF_ROUNDING = 2e-5;
inline constexpr size_t TILE_METADATA_CAPACITY = 16384;

struct TileMetadata {
    double minimum_height = -1.0;
    double maximum_height = 1.0;
    double gradient_bound = 0.0;
    double curvature_bound = 0.0;
};

TileMetadata tile_metadata(TileId tile) noexcept;
TileBounds surface_bounds(TileId tile, double polar_ratio, double height_amplitude, const TileMetadata& metadata) noexcept;
double surface_error_coefficient(double polar_ratio, double height_amplitude, const TileMetadata& metadata) noexcept;

class TileMetadataCache {
public:
    explicit TileMetadataCache(size_t capacity = TILE_METADATA_CAPACITY) : capacity_(capacity) {}
    TileMetadataCache(const TileMetadataCache&) = delete;
    TileMetadataCache& operator=(const TileMetadataCache&) = delete;
    TileMetadata get(TileId tile);
    size_t size() const noexcept { return entries_.size(); }
    uint64_t builds() const noexcept { return builds_; }
    void clear();

private:
    struct Entry { TileMetadata metadata; std::list<uint64_t>::iterator age; };
    size_t capacity_;
    uint64_t builds_ = 0;
    std::list<uint64_t> ages_;
    std::unordered_map<uint64_t, Entry> entries_;
};

}
