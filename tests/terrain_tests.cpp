#include "terrain/quadtree.hpp"
#include "terrain/geometry_residency.hpp"
#include "terrain/tile_metadata.hpp"
#include "rendering/render_settings.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

using namespace sf::terrain;

void mapping() {
    std::mt19937 random(1979);
    std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
    for (uint32_t face = 0; face < 6; ++face) {
        for (int i = 0; i < 1000; ++i) {
            double u = coordinate(random), v = coordinate(random);
            auto p = ellipsoid_point(face, u, v, 0.8);
            check(std::abs(p.x * p.x + p.y * p.y / 0.64 + p.z * p.z - 1.0) < 1e-12, "Points lie on the oblate ellipsoid");
            for (uint32_t level : {0u, 8u, 24u}) {
                auto tile = tile_at(cube_vector(face, u, v), level);
                double width = std::ldexp(2.0, -int(level));
                check(tile.face == face && u >= -1.0 + tile.x * width && u <= -1.0 + (tile.x + 1) * width &&
                      v >= -1.0 + tile.y * width && v <= -1.0 + (tile.y + 1) * width, "Cube-face mapping round trip");
            }
            auto tile = tile_at(cube_vector(face, u, v), 5);
            auto bound = bounds(tile, 0.8, 0.01);
            double w = std::ldexp(2.0, -int(tile.level));
            for (int y = 0; y <= 4; ++y) {
                for (int x = 0; x <= 4; ++x) {
                    auto sample = ellipsoid_point(face, -1.0 + (tile.x + x / 4.0) * w,
                                                  -1.0 + (tile.y + y / 4.0) * w, 0.8);
                    check(glm::length(sample - bound.center) + 0.01 <= bound.radius + 1e-12,
                          "Bounds contain interior samples, corners and full relief range");
                }
            }
        }
        auto p = ellipsoid_point(face, 0.0, 0.0, 0.8);
        auto du = ellipsoid_point(face, 0.001, 0.0, 0.8) - p;
        auto dv = ellipsoid_point(face, 0.0, 0.001, 0.8) - p;
        check(glm::dot(glm::cross(du, dv), p) > 0.0, "Every cube face has outward winding");
    }
    for (auto pole : {glm::dvec3(0, 1, 0), glm::dvec3(1, 0, 0), glm::normalize(glm::dvec3(1, 2, 3))}) {
        for (double spin : {0.0, 0.7, 3.5}) {
            auto r = body_rotation(pole, spin);
            check(glm::length(r[1] - pole) < 1e-12, "Body rotation aligns its north pole");
            check(std::abs(glm::determinant(r) - 1.0) < 1e-12, "Body frame is orthonormal and right handed");
        }
    }
}

void topology(const Selection& selection) {
    std::unordered_map<uint64_t, TileId> leaves;
    for (const auto& patch : selection.patches) {
        check(leaves.emplace(patch.tile.key(), patch.tile).second, "Unique tile keys");
        check(patch.tile.level <= MAX_LEVEL, "Tile depth stays bounded");
    }
    for (const auto& patch : selection.patches) {
        auto ancestor = patch.tile;
        while (ancestor.level) {
            ancestor = ancestor.parent();
            check(!leaves.contains(ancestor.key()), "No parent overlaps a rendered child");
        }
        auto tile = patch.tile;
        double w = std::ldexp(2.0, -int(tile.level));
        for (uint32_t edge = 0; edge < 4; ++edge) {
            for (double fraction : {0.125, 0.375, 0.625, 0.875}) {
                double u = -1.0 + (tile.x + ((edge == 0 || edge == 2) ? fraction : 0.5)) * w;
                double v = -1.0 + (tile.y + ((edge == 1 || edge == 3) ? fraction : 0.5)) * w;
                if (edge == 0) v = -1.0 + tile.y * w - w * 1e-5;
                if (edge == 1) u = -1.0 + (tile.x + 1) * w + w * 1e-5;
                if (edge == 2) v = -1.0 + (tile.y + 1) * w + w * 1e-5;
                if (edge == 3) u = -1.0 + tile.x * w - w * 1e-5;
                auto other = tile_at(cube_vector(tile.face, u, v), MAX_LEVEL);
                for (;;) {
                    if (leaves.contains(other.key())) {
                        check(std::abs(int(other.level) - int(tile.level)) <= 1, "Neighbor LODs differ by at most one across all faces");
                        check(((patch.coarse_edges >> edge) & 1u) == uint32_t(other.level < tile.level), "Stitch masks identify every coarser neighbor");
                        break;
                    }
                    if (!other.level) break;
                    other = other.parent();
                }
            }
        }
    }
}

void selections() {
    SelectionView view;
    view.cull = false;
    view.max_patches = 512;
    view.height_amplitude = 0.001;
    for (auto direction : {glm::dvec3(0, 0, 1), glm::dvec3(0, 1, 0), glm::normalize(glm::dvec3(1, 0, 1)), glm::normalize(glm::dvec3(1, 1, 1))}) {
        for (double distance : {100.0, 3.0, 1.1, 1.00000015}) {
            view.eye = direction * distance;
            auto selection = select(view);
            check(selection.patches.size() <= view.max_patches, "Hard patch budget at every altitude");
            topology(selection);
            double area = 0.0;
            for (const auto& patch : selection.patches) area += std::ldexp(1.0, -2 * int(patch.tile.level));
            check(std::abs(area - 6.0) < 1e-12, "Full six-face coverage under budget pressure");
        }
    }
    double ground = view.height_amplitude * (.30 * std::sin(-11.0) + .15 * std::sin(61.0));
    view.eye = {0, 0, 1.0 + ground + .00000015};
    auto near = select(view);
    check(near.deepest_level >= 12, "Earth-scale meter approach reaches fine LODs");
    check(near.budget_limited, "Budget pressure is reported");
    view.eye = {0, 0, 100};
    auto far = select(view);
    check(far.patches.size() < near.patches.size(), "Far views use fewer patches");
    view.max_level = 5;
    view.eye = {0, 0, 1.000001};
    check(select(view).deepest_level <= 5, "Configured maximum level respected");
    view = {};
    view.eye = {0, 0, 3};
    auto back = bounds({5, 5, 16, 16}, 1.0, 0.0);
    auto front = bounds({4, 5, 16, 16}, 1.0, 0.0);
    check(!visible(back, view) && visible(front, view), "Horizon culling rejects the far side and retains the front");
    check(!visible({{100, 0, 0}, 0.1}, view), "Frustum culling rejects lateral tiles");
    view.eye = {0, 0, 0};
    check(visible({{0, 0, -1}, 0.1}, view), "Camera inside body disables horizon occlusion");
}

float relief_at(glm::dvec3 direction) {
    glm::vec3 d{direction};
    return .55f * std::sin(18.f * d.x + 5.f * d.y) * std::cos(14.f * d.z)
         + .30f * std::sin(37.f * d.y - 11.f * d.z) * std::cos(23.f * d.x)
         + .15f * std::sin(61.f * d.z + 19.f * d.x) * std::cos(47.f * d.y);
}

double analytic_relief(glm::dvec3 d) {
    return .55 * std::sin(18.0 * d.x + 5.0 * d.y) * std::cos(14.0 * d.z)
         + .30 * std::sin(37.0 * d.y - 11.0 * d.z) * std::cos(23.0 * d.x)
         + .15 * std::sin(61.0 * d.z + 19.0 * d.x) * std::cos(47.0 * d.y);
}

glm::dvec3 displaced_point(uint32_t face, double u, double v, double polar, double amplitude) {
    auto direction = glm::normalize(cube_vector(face, u, v));
    glm::dvec3 scale{1, polar, 1};
    return direction * scale + glm::normalize(direction / scale) * (amplitude * analytic_relief(direction));
}

void metadata_bounds() {
    std::mt19937 random(1703);
    for (uint32_t face = 0; face < 6; ++face) {
        for (uint32_t level : {0u, 2u, 6u, 10u, 17u, 24u}) {
            uint32_t n = 1u << level;
            for (uint32_t trial = 0; trial < 8; ++trial) {
                TileId tile{face, level, uint32_t(random()) % n, uint32_t(random()) % n};
                auto metadata = tile_metadata(tile);
                double width = std::ldexp(2.0, -int(level));
                double u0 = -1.0 + tile.x * width, v0 = -1.0 + tile.y * width;
                for (double polar : {1.0, .8, .05}) {
                    for (double amplitude : {.0032, .01}) {
                        auto bound = surface_bounds(tile, polar, amplitude, metadata);
                        if (level >= 17) {
                            check(metadata.maximum_height - metadata.minimum_height < .003, "Fine tile height ranges narrow with extent");
                            check(bound.radius < bounds(tile, polar, amplitude).radius * .1, "Local relief bounds shrink below the global envelope");
                        }
                        for (uint32_t y = 0; y <= 8; ++y) {
                            for (uint32_t x = 0; x <= 8; ++x) {
                                double u = u0 + x * width / 8.0, v = v0 + y * width / 8.0;
                                auto d = glm::normalize(cube_vector(face, u, v));
                                for (double height : {analytic_relief(d), double(relief_at(d))}) {
                                    check(height >= metadata.minimum_height && height <= metadata.maximum_height,
                                          "Analytic and float heights lie inside tile metadata");
                                    auto scale = glm::dvec3(1, polar, 1);
                                    auto point = d * scale + glm::normalize(d / scale) * (amplitude * height);
                                    check(glm::length(point - bound.center) <= bound.radius + 1e-13,
                                          "Local bounds contain displaced interiors and edges even on flattened bodies");
                                }
                            }
                        }
                        double step = width / GRID_CELLS;
                        double error_bound = surface_error_coefficient(polar, amplitude, metadata) * step * step;
                        for (uint32_t cell = 0; cell < GRID_CELLS; ++cell) {
                            double u = u0 + cell * step, v = v0 + cell * step;
                            auto a = displaced_point(face, u, v, polar, amplitude);
                            auto b = displaced_point(face, u + step, v, polar, amplitude);
                            auto c = displaced_point(face, u, v + step, polar, amplitude);
                            auto midpoint = displaced_point(face, u + step / 3.0, v + step / 3.0, polar, amplitude);
                            check(glm::length(midpoint - (a + b + c) / 3.0) <= error_bound + 1e-13,
                                  "Local curvature bounds contain triangle approximation errors");
                        }
                    }
                }
            }
        }
    }
    TileMetadataCache cache(2);
    TileId a{4, 17, 40000, 50000}, b{4, 17, 40001, 50000}, c{4, 17, 40002, 50000};
    cache.get(a); cache.get(b); cache.get(a);
    check(cache.builds() == 2, "Repeated tile metadata is reused");
    cache.get(c); cache.get(a);
    check(cache.size() == 2 && cache.builds() == 3, "Metadata LRU remains bounded and protects recent tiles");
    cache.get(b);
    check(cache.builds() == 4, "Evicted metadata is recomputed when needed");
    cache.clear();
    check(cache.size() == 0 && cache.builds() == 0, "Metadata lifetime can be reset");
    TileMetadataCache uncached(0);
    uncached.get(a); uncached.get(a);
    check(uncached.size() == 0 && uncached.builds() == 2, "Zero metadata capacity computes safely without retaining data");
    SelectionView view;
    auto direction = glm::dvec3(std::sin(.15), 0, std::cos(.15));
    view.eye = direction * (1.0 + 2.0 / 6378.137);
    view.view_from_local = glm::dmat3(glm::lookAt(glm::dvec3(0), -direction, glm::dvec3(0, 1, 0)));
    view.polar_ratio = 1.0 - 1.0 / 298.257223563;
    view.height_amplitude = 20.0 / 6378.137;
    view.tan_half_fov_y = std::tan(glm::radians(30.0));
    view.tan_half_fov_x = view.tan_half_fov_y * 1920.0 / 1080.0;
    view.focal_pixels = 1080.0 / (2.0 * view.tan_half_fov_y);
    TileMetadataCache shared;
    auto selection = select(view, &shared);
    check(!selection.budget_limited && selection.patches.size() < 512, "Local high-relief metadata avoids unnecessary budget exhaustion at two kilometers");
    topology(selection);
    uint64_t builds = shared.builds();
    auto repeated = select(view, &shared);
    check(shared.builds() == builds && repeated.patches.size() == selection.patches.size(), "Repeated selection reuses bounded source metadata");
    for (double x : {-1000.0, 0.0, 1000.0}) {
        auto sample = glm::normalize(direction + glm::dvec3(std::cos(.15), 0, -std::sin(.15)) * (x / 6378137.0));
        uint32_t level = 0;
        for (const auto& patch : selection.patches) {
            if (tile_at(sample, patch.tile.level) == patch.tile) level = patch.tile.level;
        }
        check(level >= 9, "Tight local bounds preserve visible high-relief ground coverage");
    }
}

void ground_refinement() {
    for (auto direction : {glm::normalize(glm::dvec3(std::sin(.15), 0, std::cos(.15))),
                           glm::dvec3(0, 0, 1), glm::dvec3(0, 1, 0),
                           glm::normalize(glm::dvec3(1, 0, 1)), glm::normalize(glm::dvec3(1, 1, 1)),
                           glm::normalize(glm::dvec3(-2, .4, -1))}) {
        for (double amplitude : {0.0, 2000.0 / 6378137.0}) {
            for (bool horizon : {false, true}) {
                SelectionView view;
                view.polar_ratio = 1.0 - 1.0 / 298.257223563;
                view.height_amplitude = amplitude;
                glm::dvec3 scale{1, view.polar_ratio, 1};
                auto normal = glm::normalize(direction / scale);
                view.eye = direction * scale + normal * (amplitude * relief_at(direction) + 1.0 / 6378137.0);
                auto tangent = glm::normalize(glm::cross(std::abs(normal.y) < .9 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0), normal));
                auto forward = horizon ? tangent : -normal;
                auto up = horizon ? normal : glm::cross(normal, tangent);
                view.view_from_local = glm::dmat3(glm::lookAt(glm::dvec3(0), forward, up));
                view.tan_half_fov_y = std::tan(glm::radians(30.0));
                view.tan_half_fov_x = view.tan_half_fov_y * 1920.0 / 1080.0;
                view.focal_pixels = 1080.0 / (2.0 * view.tan_half_fov_y);
                auto selection = select(view);
                topology(selection);
                check(selection.patches.size() <= view.max_patches, "Ground refinement respects the patch budget");
                check(selection.deepest_level >= 15, "Ground detail survives balancing at interiors, poles and cube seams");
                if (!horizon) {
                    for (double x : {-.25, 0.0, .25}) {
                        for (double y : {-.25, 0.0, .25}) {
                            auto sample = glm::normalize(direction + (tangent * x + up * y) / 6378137.0);
                            uint32_t covering_level = 0;
                            for (const auto& patch : selection.patches) {
                                if (tile_at(sample, patch.tile.level) == patch.tile) covering_level = patch.tile.level;
                            }
                            check(covering_level >= 15, "Visible ground below the camera retains local detail without holes");
                        }
                    }
                }
                for (uint32_t budget : {64u, 128u, 512u}) {
                    view.max_patches = budget;
                    auto constrained = select(view);
                    check(constrained.patches.size() <= budget, "Transition geometry fits small budgets atomically");
                    topology(constrained);
                    bool covered = horizon;
                    for (const auto& patch : constrained.patches) covered |= tile_at(direction, patch.tile.level) == patch.tile;
                    check(covered, "Budget rejection retains visible ground coverage");
                }
            }
        }
    }
}

void grids() {
    for (uint32_t mask = 0; mask < 16; ++mask) {
        auto indices = grid_indices(mask);
        check(indices.size() == GRID_CELLS * GRID_CELLS * 6, "Fixed grid index count");
        double area = 0.0;
        for (size_t i = 0; i < indices.size(); i += 3) {
            glm::dvec2 p[3];
            for (size_t j = 0; j < 3; ++j) {
                uint32_t index = indices[i + j];
                check(index < (GRID_CELLS + 1) * (GRID_CELLS + 1), "Stitched grid indices remain in range");
                uint32_t x = index % (GRID_CELLS + 1), y = index / (GRID_CELLS + 1);
                if ((y == 0 && (mask & 1)) || (y == GRID_CELLS && (mask & 4))) check(x % 2 == 0, "Horizontal seams reference only coarse vertices");
                if ((x == 0 && (mask & 8)) || (x == GRID_CELLS && (mask & 2))) check(y % 2 == 0, "Vertical seams reference only coarse vertices");
                p[j] = {x, y};
            }
            auto a = p[1] - p[0], b = p[2] - p[0];
            double triangle_area = (a.x * b.y - a.y * b.x) * 0.5;
            check(triangle_area >= 0.0, "Stitching never reverses triangle winding");
            area += triangle_area;
        }
        check(area == GRID_CELLS * GRID_CELLS, "Every seam combination covers the whole patch without holes");
    }
    sf::rendering::RenderSettingsStore store;
    auto draft = store.values();
    draft.terrain_error_pixels = std::numeric_limits<float>::quiet_NaN();
    check(!store.apply(draft).accepted, "Nonfinite terrain settings rejected");
    draft = store.values();
    draft.terrain_patch_limit = 0;
    check(!store.apply(draft).accepted, "Invalid patch budgets rejected");
    draft = store.values();
    draft.terrain_enabled = true;
    check(store.apply(draft).accepted, "Terrain settings use the validated store");
}

void geometry_residency() {
    GeometryKey a{1, 1, TileId{0, 3, 2, 2}.key(), 4, 0.9, 0.001};
    auto b = a, c = a, d = a, e = a;
    b.tile = TileId{1, 3, 2, 2}.key();
    c.tile = TileId{2, 3, 2, 2}.key();
    d.tile = TileId{3, 3, 2, 2}.key();
    e.tile = TileId{4, 3, 2, 2}.key();
    GeometryResidency cache(3);
    auto cold = cache.request(std::vector{a, b, c}, 3);
    check(cold.bakes.size() == 3 && cold.hits == 0, "Cold geometry entries require bakes");
    auto warm = cache.request(std::vector{a, b, c}, 3);
    check(warm.bakes.empty() && warm.hits == 3 && warm.slots == cold.slots, "Unchanged geometry reuses resident slots");
    auto moved = cache.request(std::vector{d, a}, 3);
    check(moved.slots[1] == cold.slots[0] && moved.slots[0] != moved.slots[1], "All current hits are protected before allocating misses");
    auto pinned = cache.request(std::vector{e, a, c, d}, 3);
    check(pinned.slots[0] == -1 && pinned.hits == 3 && pinned.bakes.empty(), "A full protected cache preserves draw references and falls back");
    check(cache.size() == 3, "Geometry residency remains within its fixed budget");
    GeometryResidency limited(4);
    auto partial = limited.request(std::vector{a, a, b}, 1);
    check(partial.bakes.size() == 1 && partial.slots[0] == partial.slots[1] && partial.slots[2] == -1,
          "Duplicate jobs coalesce and excess misses use the reference path");
    auto next = limited.request(std::vector{a, b}, 1);
    check(next.hits == 1 && next.bakes.size() == 1 && next.slots[1] >= 0, "Deferred geometry converges without invalidating existing entries");
    for (uint32_t field = 0; field < 6; ++field) {
        auto changed = a;
        if (field == 0) ++changed.generation;
        if (field == 1) ++changed.geometry_revision;
        if (field == 2) ++changed.tile;
        if (field == 3) ++changed.body;
        if (field == 4) changed.polar_ratio = 0.8;
        if (field == 5) changed.height_amplitude = 0.002;
        check(limited.request(std::vector{changed}, 1).bakes.size() == 1, "Geometry keys include source domain, shape and relief");
    }
    GeometryResidency zero(0);
    check(zero.request(std::vector{a}, 1).slots[0] == -1, "Zero-capacity cache falls back safely");
}

}

int main() {
    try {
        mapping();
        grids();
        geometry_residency();
        selections();
        metadata_bounds();
        ground_refinement();
        std::cout << "Terrain mapping, bounds, seams, coverage, precision-level and budget contracts passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
