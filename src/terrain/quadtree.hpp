#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace sf::terrain {

inline constexpr uint32_t GRID_CELLS = 16;
inline constexpr uint32_t MAX_LEVEL = 24;
inline constexpr uint32_t MAX_PATCHES = 8192;

struct TileId {
    uint32_t face = 0;
    uint32_t level = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t key() const noexcept;
    static TileId from_key(uint64_t key) noexcept;
    TileId parent() const noexcept;
    bool operator==(const TileId&) const = default;
};

struct TileBounds {
    glm::dvec3 center{0.0};
    double radius = 0.0;
    glm::dvec3 surface_center{0.0};
    glm::dvec3 direction{0.0};
    double angular_radius = 0.0;
    double polar_ratio = 1.0;
    double padding = 0.0;
    double maximum_scaled_radius = 0.0;
    double support(glm::dvec3 axis) const noexcept;
    double distance(glm::dvec3 eye) const noexcept;
};

struct SelectionView {
    glm::dvec3 eye{0.0, 0.0, 3.0};
    glm::dmat3 view_from_local{1.0};
    double tan_half_fov_x = 1.0;
    double tan_half_fov_y = 1.0;
    double focal_pixels = 800.0;
    double orthographic_pixels = 0.0;
    glm::dvec4 orthographic_bounds{-1, -1, 1, 1};
    double target_error_pixels = 2.0;
    double polar_ratio = 1.0;
    double height_amplitude = 0.0;
    double occluder_scale = 1.0;
    uint32_t max_level = MAX_LEVEL;
    uint32_t max_patches = 1024;
    bool cull = true;
    bool operator==(const SelectionView&) const = default;
};

struct Patch {
    TileId tile;
    uint32_t coarse_edges = 0;
    bool operator==(const Patch&) const = default;
};

struct Selection {
    std::vector<Patch> patches;
    std::vector<TileId> required_tiles;
    double max_error_pixels = 0.0;
    uint32_t deepest_level = 0;
    uint32_t visited = 0;
    bool budget_limited = false;
    bool loading_limited = false;
};

glm::dvec3 cube_vector(uint32_t face, double u, double v) noexcept;
TileId tile_at(glm::dvec3 direction, uint32_t level) noexcept;
glm::dvec3 ellipsoid_point(uint32_t face, double u, double v, double polar_ratio) noexcept;
glm::dmat3 body_rotation(glm::dvec3 pole, double spin) noexcept;
TileBounds bounds(TileId tile, double polar_ratio, double height_amplitude) noexcept;
bool visible(const TileBounds& tile, const SelectionView& view) noexcept;
double error_coefficient(double polar_ratio, double height_amplitude) noexcept;
class TileMetadataCache;
class TerrainSource;
struct TileAvailability {
    std::function<bool(TileId)> ready;
    std::function<void(TileId)> request;
};
class SelectionPlan {
public:
    SelectionPlan();
    ~SelectionPlan();
    SelectionPlan(SelectionPlan&&) noexcept;
    SelectionPlan& operator=(SelectionPlan&&) noexcept;
    void invalidate() noexcept;
    uint32_t checks() const noexcept;

private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
    friend Selection select(const SelectionView&, const TerrainSource&, const TileAvailability&, class SelectionScratch*, SelectionPlan*);
    friend bool update_selection(Selection&, SelectionPlan&, const SelectionView&, const TerrainSource&, const TileAvailability&, class SelectionScratch*);
};

class SelectionScratch {
public:
    SelectionScratch();
    ~SelectionScratch();
    SelectionScratch(SelectionScratch&&) noexcept;
    SelectionScratch& operator=(SelectionScratch&&) noexcept;

private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
    friend Selection select(const SelectionView&, const TerrainSource&, const TileAvailability&, SelectionScratch*, SelectionPlan*);
};

Selection select(const SelectionView& view, TileMetadataCache* metadata = nullptr, SelectionScratch* scratch = nullptr);
Selection select(const SelectionView& view, const TerrainSource& source, const TileAvailability& availability = {}, SelectionScratch* scratch = nullptr, SelectionPlan* plan = nullptr);
bool update_selection(Selection& selection, SelectionPlan& plan, const SelectionView& view, const TerrainSource& source,
                      const TileAvailability& availability = {}, SelectionScratch* scratch = nullptr);
std::vector<uint32_t> grid_indices(uint32_t coarse_edges);

}
