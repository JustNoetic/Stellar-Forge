#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <vector>

namespace sf::terrain {

inline constexpr uint32_t GRID_CELLS = 16;
inline constexpr uint32_t MAX_LEVEL = 24;
inline constexpr uint32_t MAX_PATCHES = 4096;

struct TileId {
    uint32_t face = 0;
    uint32_t level = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t key() const noexcept;
    TileId parent() const noexcept;
    bool operator==(const TileId&) const = default;
};

struct TileBounds {
    glm::dvec3 center{0.0};
    double radius = 0.0;
};

struct SelectionView {
    glm::dvec3 eye{0.0, 0.0, 3.0};
    glm::dmat3 view_from_local{1.0};
    double tan_half_fov_x = 1.0;
    double tan_half_fov_y = 1.0;
    double focal_pixels = 800.0;
    double target_error_pixels = 2.0;
    double polar_ratio = 1.0;
    double height_amplitude = 0.0;
    uint32_t max_level = MAX_LEVEL;
    uint32_t max_patches = 1024;
    bool cull = true;
    bool operator==(const SelectionView&) const = default;
};

struct Patch {
    TileId tile;
    uint32_t coarse_edges = 0;
};

struct Selection {
    std::vector<Patch> patches;
    uint32_t deepest_level = 0;
    uint32_t visited = 0;
    bool budget_limited = false;
};

glm::dvec3 cube_vector(uint32_t face, double u, double v) noexcept;
TileId tile_at(glm::dvec3 direction, uint32_t level) noexcept;
glm::dvec3 ellipsoid_point(uint32_t face, double u, double v, double polar_ratio) noexcept;
glm::dmat3 body_rotation(glm::dvec3 pole, double spin) noexcept;
TileBounds bounds(TileId tile, double polar_ratio, double height_amplitude) noexcept;
bool visible(const TileBounds& tile, const SelectionView& view) noexcept;
double error_coefficient(double polar_ratio, double height_amplitude) noexcept;
class TileMetadataCache;
Selection select(const SelectionView& view, TileMetadataCache* metadata = nullptr);
std::vector<uint32_t> grid_indices(uint32_t coarse_edges);

}
