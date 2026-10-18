#include "terrain/quadtree.hpp"
#include "terrain/geometry_residency.hpp"
#include "terrain/tile_metadata.hpp"
#include "terrain/height_stream.hpp"
#include "core/path_utils.hpp"
#include "rhi/upload_ring.hpp"
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
#include <chrono>
#include <thread>
#include <fstream>

namespace {

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

using namespace sf::terrain;

void enclosed(const TileBounds& bound, glm::dvec3 point) {
    for (auto axis : {glm::dvec3(1, 0, 0), glm::dvec3(0, 1, 0), glm::dvec3(0, 0, 1),
                      bound.direction, glm::normalize(glm::dvec3(.7, -.3, .2))}) {
        check(std::abs(glm::dot(axis, point-bound.surface_center)) <= bound.support(axis)+2e-14,
              "Surface enclosure contains displaced samples and parent morph vertices");
    }
    check(bound.distance(point) <= 1.1e-12, "Distance lower bound never excludes a surface sample");
    auto eye = point+glm::dvec3(.001, -.002, .003);
    check(bound.distance(eye) <= glm::length(eye-point)+3e-14, "Refinement distance stays below sampled surface distance");
}

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
                    enclosed(bound, sample);
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
                                    enclosed(bound, point);
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
    check(selection.max_error_pixels <= view.target_error_pixels, "Ready relief leaves meet the screen-space error target");
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

void visibility_bounds() {
    std::mt19937 random(20261018);
    std::uniform_real_distribution<double> coordinate(-1, 1);
    uint32_t retained = 0;
    for (double polar : {.05, .5, .99665, 1.0}) {
        for (uint32_t trial = 0; trial < 400; ++trial) {
            uint32_t face = random()%6;
            double u = coordinate(random), v = coordinate(random);
            auto direction = glm::normalize(cube_vector(face, u, v));
            auto scale = glm::dvec3(1, polar, 1);
            auto normal = glm::normalize(direction/scale);
            double height = trial%3 == 0 ? -.001 : trial%3 == 1 ? .001 : 0;
            auto point = direction*scale+normal*height;
            auto tangent = glm::normalize(glm::cross(normal, std::abs(normal.y) < .9 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0)));
            SelectionView view;
            view.polar_ratio = polar;
            view.height_amplitude = std::abs(height);
            view.eye = point+(normal-tangent*(trial%2 ? 4.0 : 0.0))*(trial%2 ? 1e-7 : .02);
            auto forward = trial%2 ? tangent-normal*.1 : -normal;
            auto up = trial%2 ? normal : tangent;
            view.view_from_local = glm::dmat3(glm::lookAt(glm::dvec3(0), forward, up));
            view.tan_half_fov_x = 1;
            view.tan_half_fov_y = .6;
            for (uint32_t level : {0u, 2u, 8u, 16u, 24u}) {
                auto tile = tile_at(direction, level);
                auto bound = surface_bounds(tile, polar, 1, {height, height, 0, 0});
                enclosed(bound, point);
                check(visible(bound, view), "Frustum and ellipsoid horizon culling retain visible signed-height ground through LOD24");
                ++retained;
            }
        }
    }
    SelectionView view;
    view.polar_ratio = .5;
    view.eye = {0, 0, 1.0001};
    view.view_from_local = glm::dmat3(glm::lookAt(glm::dvec3(0), glm::dvec3(1, 0, 0), glm::dvec3(0, 0, 1)));
    auto hidden = bounds(tile_at({std::sin(.2), 0, std::cos(.2)}, 12), .5, 0);
    check(!visible(hidden, view), "Ellipsoid horizon rejects terrain retained by the old polar-radius sphere");
    auto root = bounds({4}, 1, 0);
    check(root.distance({0, 0, 1.001}) > .000999999,
          "Thin enclosure retains a useful ground distance when the eye lies inside the old sphere");
    view.polar_ratio = 1;
    auto tile = bounds({4, 8, 128, 128}, 1, 0);
    auto sphere = tile;
    sphere.angular_radius = 0;
    view.eye = tile.center+tile.direction*.001;
    view.view_from_local = glm::dmat3(glm::lookAt(glm::dvec3(0), tile.direction, glm::dvec3(0, 1, 0)));
    check(visible(sphere, view) && !visible(tile, view), "Thin surface bounds reject an offscreen patch whose sphere surrounds the eye");
    check(retained == 8000, "Visibility oracle exercises every configured shape, height and depth");
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
    for (uint32_t field = 0; field < 8; ++field) {
        auto changed = a;
        if (field == 0) ++changed.generation;
        if (field == 1) ++changed.geometry_revision;
        if (field == 2) ++changed.tile;
        if (field == 3) ++changed.body;
        if (field == 4) changed.polar_ratio = 0.8;
        if (field == 5) changed.height_amplitude = 0.002;
        if (field == 6) changed.source.identity = 123;
        if (field == 7) changed.source.revision = 123;
        check(limited.request(std::vector{changed}, 1).bakes.size() == 1, "Geometry keys include source domain, shape and relief");
    }
    GeometryResidency zero(0);
    check(zero.request(std::vector{a}, 1).slots[0] == -1, "Zero-capacity cache falls back safely");
}

void staging_residency() {
    sf::rhi::UploadRing ring(64);
    check(ring.allocate(32, 1, 0) == 0, "Staging starts at the beginning");
    check(ring.allocate(32, 2, 0) == 32, "Independent submissions own disjoint staging ranges");
    check(!ring.allocate(16, 3, 0), "Uncompleted staging is never overwritten");
    check(ring.allocate(16, 3, 1) == 0, "Completed ranges can wrap and be reused");
    check(!ring.allocate(48, 4, 1), "Wrap padding respects live allocations");
    check(ring.allocate(48, 4, 3) == 16, "Draining completion frees the whole ring");
    check(!ring.allocate(65, 5, 4), "Oversized staging requests are rejected");
}

void resident_refinement() {
    SelectionView view;
    view.cull = false;
    view.max_patches = 128;
    view.target_error_pixels = .2;
    view.max_level = 4;
    std::unordered_set<uint64_t> resident;
    std::vector<TileId> pending;
    auto choose = [&] {
        pending.clear();
        return select(view, ProceduralSource{}, {
            [&](TileId tile) { return resident.contains(tile.key()); },
            [&](TileId tile) { pending.push_back(tile); }});
    };
    check(choose().patches.empty(), "No terrain cover is advertised before roots are ready");
    for (uint32_t face = 0; face < 6; ++face) resident.insert(TileId{face}.key());
    auto roots = choose();
    check(roots.patches.size() == 6 && roots.loading_limited, "Resident parents cover delayed children");
    resident.insert(pending.front().key());
    check(choose().patches.size() == 6, "A partially ready sibling group cannot remove its parent");
    for (int frame = 0; frame < 80; ++frame) {
        auto selection = choose();
        topology(selection);
        check(selection.patches.size() <= view.max_patches, "Loading respects the same patch budget");
        double area = 0;
        for (const auto& patch : selection.patches) area += std::ldexp(1.0, -int(2*patch.tile.level));
        check(std::abs(area-6) < 1e-12, "Loading retains complete nonoverlapping coverage");
        if (pending.empty()) break;
        for (size_t i = 0; i < pending.size(); i += 3) resident.insert(pending[i].key());
    }
    check(choose().deepest_level > 0, "Out-of-order payload readiness eventually refines terrain");
    GeometryResidency cache(64);
    auto key_for = [](TileId tile) { GeometryKey key; key.tile = tile.key(); return key; };
    view.max_patches = 32;
    uint32_t admissions = 0;
    for (uint32_t frame = 0; frame < 100; ++frame) {
        view.eye = glm::normalize(glm::dvec3(std::cos(frame*.17), .2, std::sin(frame*.17)))*1.01;
        std::vector<GeometryKey> keys;
        for (uint32_t face = 0; face < 6; ++face) keys.push_back(key_for({face}));
        std::vector<TileId> requests;
        auto selection = select(view, ProceduralSource{}, {
            [&](TileId tile) { return cache.find(key_for(tile)) >= 0; },
            [&](TileId tile) { requests.push_back(tile); }});
        if (!selection.patches.empty()) topology(selection);
        for (const auto& patch : selection.patches) {
            auto tile = patch.tile;
            for (;;) {
                keys.push_back(key_for(tile));
                if (!tile.level) break;
                tile = tile.parent();
            }
        }
        for (auto tile : requests) keys.push_back(key_for(tile));
        auto plan = cache.request(keys, 8);
        admissions += uint32_t(plan.bakes.size());
        check(cache.size() <= 64, "Resident-parent planning stays bounded through eviction");
        for (uint32_t face = 0; face < 6; ++face) check(cache.find(key_for({face})) >= 0, "Root coverage survives geometry eviction");
        for (const auto& patch : selection.patches) check(cache.find(key_for(patch.tile)) >= 0, "Cache admission preserves every current draw");
    }
    check(admissions > 64, "Moving residency test exercises slot eviction and reuse");
}

void resident_detail_retention() {
    SelectionView view;
    view.eye = {0, 0, 1.001};
    view.max_patches = 128;
    view.max_level = 12;
    std::vector<GeometryKey> warm_keys;
    auto key_for = [](TileId tile) { GeometryKey key; key.tile = tile.key(); return key; };
    for (uint32_t face = 0; face < 6; ++face) warm_keys.push_back(key_for({face}));
    auto desired = select(view, ProceduralSource{}, {{}, [&](TileId tile) { warm_keys.push_back(key_for(tile)); }});
    GeometryResidency cache(uint32_t(warm_keys.size()) + 32);
    cache.request(warm_keys, uint32_t(warm_keys.size()));
    auto choose = [&] {
        return select(view, ProceduralSource{}, {[&](TileId tile) { return cache.find(key_for(tile)) >= 0; }, {}});
    };
    auto ready = choose();
    check(!ready.loading_limited && ready.deepest_level >= 5, "Eviction probe starts with fully loaded ground detail");
    std::vector<GeometryKey> retained;
    for (uint32_t face = 0; face < 6; ++face) retained.push_back(key_for({face}));
    for (auto tile : ready.required_tiles) retained.push_back(key_for(tile));
    auto protected_count = retained.size();
    uint32_t admissions = 0;
    for (uint32_t frame = 0; frame < 8; ++frame) {
        retained.resize(protected_count);
        for (uint32_t tile = 0; tile < 64; ++tile) {
            auto key = key_for({5, 12, frame*64+tile, 0});
            key.body = 99;
            retained.push_back(key);
        }
        admissions += uint32_t(cache.request(retained, 64).bakes.size());
    }
    check(admissions > 64 && cache.size() <= warm_keys.size()+32, "Detail retention probe forces bounded cache eviction");
    view.eye.z += 1e-10;
    desired = select(view);
    auto after = choose();
    check(!after.loading_limited && after.patches.size() == desired.patches.size(),
          "Evicting unrelated geometry cannot discard loaded terrain refinement prerequisites");
    for (size_t index = 0; index < after.patches.size(); ++index)
        check(after.patches[index].tile == desired.patches[index].tile, "Loaded terrain detail survives cache churn");
}

void imported_error_bounds(const HeightSource& source) {
    std::unordered_map<uint64_t, HeightTile> payloads;
    auto point = [&](uint32_t face, double u, double v) {
        auto direction = glm::normalize(cube_vector(face, u, v));
        auto address = tile_at(direction, source.info().data_level);
        auto a = glm::abs(direction);
        glm::dvec2 uv;
        switch (address.face) {
        case 0: uv = {-direction.z/a.x, direction.y/a.x}; break;
        case 1: uv = {direction.z/a.x, direction.y/a.x}; break;
        case 2: uv = {direction.x/a.y, -direction.z/a.y}; break;
        case 3: uv = {direction.x/a.y, direction.z/a.y}; break;
        case 4: uv = {direction.x/a.z, direction.y/a.z}; break;
        default: uv = {-direction.x/a.z, direction.y/a.z}; break;
        }
        if (!payloads.contains(address.key())) payloads.emplace(address.key(), source.read(address));
        double n = double(1u << address.level);
        double h = height_sample(payloads.at(address.key()), ((uv.x+1)*.5*n-address.x)*16, ((uv.y+1)*.5*n-address.y)*16);
        auto scale = glm::dvec3(1, source.info().polar_ratio, 1);
        return direction*scale+glm::normalize(direction/scale)*(h/source.info().equatorial_radius_m);
    };
    for (uint32_t face = 0; face < 6; ++face) for (uint32_t level : {0u, 1u, 2u, 3u, 6u, 12u, 24u}) {
        uint32_t n = 1u << level;
        for (uint32_t tx : {0u, n/2, n-1}) for (uint32_t ty : {0u, n/2, n-1}) {
            TileId tile{face, level, tx, ty};
            double width = std::ldexp(2.0, -int(level));
            auto description = source.describe(tile, source.info().polar_ratio, 0);
            for (uint32_t stride : {1u, 2u, 4u}) {
                if (stride > (1u << std::min(level, 2u))) continue;
                double error = source.approximation_error(tile, stride);
                for (uint32_t x : {0u, 8u, 16-stride}) for (uint32_t y : {0u, 8u, 16-stride}) {
                    double u = -1+(tx+double(x)/16)*width, v = -1+(ty+double(y)/16)*width;
                    double step = width*stride/16;
                    auto a = point(face, u, v), b = point(face, u+step, v);
                    auto c = point(face, u, v+step), d = point(face, u+step, v+step);
                    for (auto weights : {glm::dvec3(1./3), glm::dvec3(.1,.2,.7), glm::dvec3(.8,.15,.05)}) {
                        auto sample = point(face, u+step*weights.y, v+step*weights.z);
                        auto upper = point(face, u+step*(weights.x+weights.y), v+step*(weights.y+weights.z));
                        check(glm::length(sample-(a*weights.x+b*weights.y+c*weights.z)) <= error+2e-15 &&
                              glm::length(upper-(b*weights.x+d*weights.y+c*weights.z)) <= error+2e-15,
                              "Imported error bounds contain fine, parent and grandparent triangles on every face and oblate shape");
                        check(glm::length(sample-description.bounds.center) <= description.bounds.radius+2e-15,
                              "Imported bounds contain sampled interiors at edges, poles and LOD24");
                        enclosed(description.bounds, sample);
                        enclosed(description.bounds, a*weights.x+b*weights.y+c*weights.z);
                        enclosed(description.bounds, b*weights.x+d*weights.y+c*weights.z);
                    }
                }
            }
        }
    }
    for (uint32_t level = 0; level <= MAX_LEVEL; ++level) {
        double shared = source.morph_error(level);
        for (uint32_t face = 0; face < 6; ++face)
            check(shared >= source.approximation_error({face, level, 0, 0}), "Shared morph errors contain local fine-grid errors");
        if (level) check(shared <= source.morph_error(level-1), "Shared morph errors decrease with level");
    }
}

void imported_source(const std::filesystem::path& pointer) {
    auto path = sf::paths::read_text_file(pointer);
    check(path.has_value(), "Imported fixture path exists");
    auto source = HeightSource::open(*path);
    check(source->info().minimum_m < 0 && source->info().maximum_m > 0, "Imported signed heights retain valleys and peaks");
    check(source->compatible(6378137, 1-1/298.257223563), "Dataset shape matches its reference ellipsoid");
    check(!source->compatible(6000000, 1), "Incompatible body edits cannot reuse geographic metadata");
    imported_error_bounds(*source);
    auto flat = HeightSource::open(std::filesystem::path(*path).parent_path()/"flat");
    imported_error_bounds(*flat);
    check(flat->approximation_error({4}) < 1.01*std::pow(2.0/GRID_CELLS,2), "Flat imported terrain avoids the old inflated curvature allowance");
    auto dense_path = std::filesystem::path(*path).parent_path()/"dense";
    if (std::filesystem::exists(dense_path)) {
        auto dense = HeightSource::open(dense_path);
        check(dense->info().grid_cells == DENSE_HEIGHT_CELLS && dense->info().gutter == DENSE_HEIGHT_GUTTER &&
              dense->info().cubic && dense->info().finest_level == dense->info().data_level + 3,
              "Dense height source exposes 128x128 cubic-bspline metadata");
        imported_error_bounds(*dense);
        auto tile_payload = dense->read({4, dense->info().data_level, 0, 0});
        double c0 = height_sample_cubic(tile_payload, 4.25, 7.75);
        check(c0 >= dense->info().minimum_m && c0 <= dense->info().maximum_m,
              "Uniform cubic B-spline height reconstruction remains within convex hull bounds");
    }
    auto revisions = HeightSource::open(*path);
    check(revisions->bounds_revision() == 0, "Metadata-only source starts without resident payload envelopes");
    revisions->read({4, revisions->info().data_level, 0, 0});
    auto first_bounds_revision = revisions->bounds_revision();
    check(first_bounds_revision > 0, "New payload envelopes invalidate derived selection bounds");
    revisions->read({4, revisions->info().data_level, 0, 0});
    check(revisions->bounds_revision() == first_bounds_revision, "Resident envelope hits retain the selection domain");
    for (const auto& record : source->records()) {
        auto payload = source->read(record.tile);
        for (uint32_t y = 0; y <= GRID_CELLS; ++y)
            for (uint32_t x = 0; x <= GRID_CELLS; ++x) {
                double h = payload.heights_m[(y+1)*HEIGHT_SIDE+x+1];
                check(h >= record.minimum_m-1e-5 && h <= record.maximum_m+1e-5, "Imported metadata contains its payload");
            }
    }
    {
        double angle = .15;
        glm::dvec3 direction{std::sin(angle), 0, std::cos(angle)};
        auto tile = tile_at(direction, source->info().data_level);
        auto payload = source->read(tile);
        double n = double(1u << tile.level);
        double height = height_sample(payload, ((direction.x/direction.z+1)*.5*n-tile.x)*16, (.5*n-tile.y)*16);
        SelectionView view;
        view.eye = direction*(1+(height+1)/6378137);
        view.view_from_local = glm::dmat3(glm::lookAt(view.eye, glm::dvec3(0), glm::dvec3(0, 1, 0)));
        view.focal_pixels = 935;
        view.tan_half_fov_y = std::tan(30.0*std::acos(-1)/180);
        view.tan_half_fov_x = view.tan_half_fov_y*1920/1080;
        view.polar_ratio = source->info().polar_ratio;
        view.height_amplitude = .001;
        auto selection = select(view, *source);
        std::cout << "Imported ground selector: " << selection.patches.size() << " patches, LOD " << selection.deepest_level << ".\n";
        check(selection.deepest_level >= 21, "Metadata-only selection retains meter-scale imported detail");
        check(!selection.budget_limited && selection.patches.size() < 128, "Resident virtual bounds avoid spurious near-ground refinement");
        check(selection.max_error_pixels <= view.target_error_pixels, "Ready imported leaves meet the screen-space error target");
        std::unordered_map<uint64_t, HeightTile> payloads;
        auto point = [&](double u, double v) {
            auto d = glm::normalize(cube_vector(4, u, v));
            auto address = tile_at(d, source->info().data_level);
            if (!payloads.contains(address.key())) payloads.emplace(address.key(), source->read(address));
            double h = height_sample(payloads.at(address.key()), ((u+1)*.5*n-address.x)*16, ((v+1)*.5*n-address.y)*16);
            auto scale = glm::dvec3(1, view.polar_ratio, 1);
            return d*scale+glm::normalize(d/scale)*(h/6378137);
        };
        for (uint32_t level : {0u, 2u, 6u, 18u, 24u}) {
            auto address = tile_at(direction, level);
            auto description = source->describe(address, view.polar_ratio, 0);
            double width = std::ldexp(2.0, -int(level));
            for (uint32_t y = 0; y < 16; ++y) {
                for (uint32_t x = 0; x < 16; ++x) {
                    double u = -1+(address.x+double(x)/16)*width;
                    double v = -1+(address.y+double(y)/16)*width;
                    double step = width/16;
                    auto a = point(u, v), b = point(u+step, v), c = point(u, v+step);
                    auto sample = point(u+step/3, v+step/3);
                    check(glm::length(sample-description.bounds.center) <= description.bounds.radius+1e-12,
                          "Imported displaced bounds contain reconstructed interiors through LOD24");
                    check(glm::length(sample-(a+b+c)/3.0) <= description.geometric_error+1e-12,
                          "Imported geometric error contains oblate triangle approximation");
                }
            }
        }
    }
    HeightStream stream(source, 16, 8);
    for (uint32_t face = 0; face < 6; ++face) stream.request({face});
    auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (stream.stats().ready < 6 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto ready = stream.completed();
    check(ready.size() == 6 && stream.stats().failed == 0, "Bounded worker loads and bakes root coverage");
    struct Sample { glm::dvec3 position, normal; };
    std::unordered_map<std::string, Sample> edges;
    for (const auto& [tile, geometry] : ready) {
        auto anchor = ellipsoid_point(tile.face, 0, 0, source->info().polar_ratio);
        for (uint32_t y = 0; y <= 16; ++y) {
            for (uint32_t x = 0; x <= 16; ++x) {
                const auto& vertex = (*geometry)[y*17+x];
                check(std::abs(glm::length(glm::vec3(vertex.normal))-1) < 1e-5, "Resident heights yield finite unit normals");
                if (x != 0 && y != 0 && x != 16 && y != 16) continue;
                auto p = cube_vector(tile.face, -1+double(x)/8, -1+double(y)/8);
                std::string key = std::to_string(int(p.x*8))+","+std::to_string(int(p.y*8))+","+std::to_string(int(p.z*8));
                Sample sample{anchor+glm::dvec3(vertex.position), glm::dvec3(vertex.normal)};
                if (auto prior = edges.find(key); prior != edges.end()) {
                    check(glm::length(prior->second.position-sample.position)*6378137 < .6, "FP32 patch-local geometry meets across cube faces");
                    check(glm::length(prior->second.normal-sample.normal) < 1e-6, "Canonical normal evaluation matches at poles and cube corners");
                } else edges.emplace(key, sample);
            }
        }
    }
    for (uint32_t x = 0; x < 100; ++x) stream.request({0, 8, x, 0});
    check(stream.stats().pending <= 8 && stream.stats().ready <= 16, "Producer and completed queues stay bounded");
    HeightStream small(source, 2, 4);
    for (uint32_t face = 0; face < 3; ++face) small.request({face});
    deadline = std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (small.stats().builds < 3 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(small.stats().builds == 3 && small.stats().ready == 2, "Completed geometry evicts within its explicit CPU budget");
    auto copy = pointer.parent_path()/("validation-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::copy(*path, copy, std::filesystem::copy_options::recursive);
    auto manifest = *sf::paths::read_json(copy/"manifest.json");
    for (double polar : {.05, .5, 1.0}) {
        auto altered = manifest;
        altered["polar_ratio"] = polar;
        altered["revision"] = source->domain().revision+2;
        check(sf::paths::write_json(copy/"manifest.json", altered), "Write oblate error fixture");
        imported_error_bounds(*HeightSource::open(copy));
    }
    manifest["revision"] = source->domain().revision+1;
    check(sf::paths::write_json(copy/"manifest.json", manifest), "Write source replacement fixture");
    auto changed = HeightSource::open(copy);
    check(changed->domain() != source->domain(), "Source revisions form independent cache domains");
    {
        std::fstream payload(copy/"heights.bin", std::ios::in|std::ios::out|std::ios::binary);
        char byte;
        payload.read(&byte, 1);
        byte ^= 1;
        payload.seekp(0);
        payload.write(&byte, 1);
    }
    bool rejected = false;
    try { changed->read({0}); } catch (const std::exception&) { rejected = true; }
    check(rejected, "Corrupt height payloads are rejected before becoming resident");
    manifest["tiles"][0][6] = 1000000000;
    check(sf::paths::write_json(copy/"manifest.json", manifest), "Write invalid metadata fixture");
    rejected = false;
    try { HeightSource::open(copy); } catch (const std::exception&) { rejected = true; }
    check(rejected, "Invalid extrema cannot enter the metadata domain");
}

}

int main(int argc, char** argv) {
    try {
        mapping();
        grids();
        geometry_residency();
        selections();
        metadata_bounds();
        visibility_bounds();
        ground_refinement();
        staging_residency();
        resident_refinement();
        resident_detail_retention();
        if (argc == 2) imported_source(argv[1]);
        std::cout << "Terrain mapping, bounds, seams, coverage, precision-level and budget contracts passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
