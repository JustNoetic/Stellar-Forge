#include "terrain/quadtree.hpp"
#include "terrain/tile_metadata.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <unordered_map>

namespace sf::terrain {

uint64_t TileId::key() const noexcept {
    return (uint64_t(face) << 61) | (uint64_t(level) << 56) | (uint64_t(x) << 28) | y;
}

TileId TileId::parent() const noexcept {
    return level ? TileId{face, level - 1, x / 2, y / 2} : *this;
}

glm::dvec3 cube_vector(uint32_t face, double u, double v) noexcept {
    switch (face) {
    case 0: return {1, v, -u};
    case 1: return {-1, v, u};
    case 2: return {u, 1, -v};
    case 3: return {u, -1, v};
    case 4: return {u, v, 1};
    default: return {-u, v, -1};
    }
}

TileId tile_at(glm::dvec3 d, uint32_t level) noexcept {
    auto a = glm::abs(d);
    uint32_t face;
    glm::dvec2 uv;
    if (a.x >= a.y && a.x >= a.z) {
        face = d.x >= 0 ? 0 : 1;
        uv = {d.x >= 0 ? -d.z / a.x : d.z / a.x, d.y / a.x};
    } else if (a.y >= a.z) {
        face = d.y >= 0 ? 2 : 3;
        uv = {d.x / a.y, d.y >= 0 ? -d.z / a.y : d.z / a.y};
    } else {
        face = d.z >= 0 ? 4 : 5;
        uv = {d.z >= 0 ? d.x / a.z : -d.x / a.z, d.y / a.z};
    }
    uint32_t n = 1u << std::min(level, MAX_LEVEL);
    auto index = [n](double value) {
        return static_cast<uint32_t>(std::clamp(std::floor((value + 1.0) * 0.5 * n), 0.0, double(n - 1)));
    };
    return {face, std::min(level, MAX_LEVEL), index(uv.x), index(uv.y)};
}

glm::dvec3 ellipsoid_point(uint32_t face, double u, double v, double polar_ratio) noexcept {
    auto p = glm::normalize(cube_vector(face, u, v));
    p.y *= polar_ratio;
    return p;
}

glm::dmat3 body_rotation(glm::dvec3 pole, double spin) noexcept {
    pole = glm::length(pole) > 1e-12 ? glm::normalize(pole) : glm::dvec3(0, 1, 0);
    auto reference = std::abs(pole.x) < 0.9 ? glm::dvec3(1, 0, 0) : glm::dvec3(0, 0, 1);
    auto x = glm::normalize(reference - pole * glm::dot(reference, pole));
    auto z = glm::cross(x, pole);
    double c = std::cos(spin), s = std::sin(spin);
    return {x * c - z * s, pole, x * s + z * c};
}

TileBounds bounds(TileId tile, double polar_ratio, double height_amplitude) noexcept {
    double width = std::ldexp(2.0, -int(tile.level));
    auto center = ellipsoid_point(tile.face, -1.0 + (tile.x + 0.5) * width,
                                 -1.0 + (tile.y + 0.5) * width, polar_ratio);
    return {center, width / std::sqrt(2.0) + height_amplitude};
}

bool visible(const TileBounds& tile, const SelectionView& view) noexcept {
    if (!view.cull) return true;
    auto v = view.view_from_local * (tile.center - view.eye);
    double depth = -v.z;
    if (depth + tile.radius <= 0.0) return false;
    if (std::abs(v.x) - depth * view.tan_half_fov_x > tile.radius * std::sqrt(1.0 + view.tan_half_fov_x * view.tan_half_fov_x)) return false;
    if (std::abs(v.y) - depth * view.tan_half_fov_y > tile.radius * std::sqrt(1.0 + view.tan_half_fov_y * view.tan_half_fov_y)) return false;
    double occluder = view.polar_ratio - view.height_amplitude * (1.0 + RELIEF_ROUNDING);
    double eye_distance = glm::length(view.eye);
    if (occluder <= 0.0 || eye_distance <= occluder) return true;
    auto to_tile = tile.center - view.eye;
    double along = glm::dot(to_tile, -view.eye / eye_distance);
    double perpendicular = glm::length(to_tile + view.eye / eye_distance * along);
    double sine = occluder / eye_distance;
    double cosine = std::sqrt(1.0 - sine * sine);
    bool inside_cone = along * sine - perpendicular * cosine > tile.radius;
    bool beyond_tangent = glm::length(to_tile) - tile.radius > eye_distance * cosine;
    return !(inside_cone && beyond_tangent);
}

double error_coefficient(double polar_ratio, double height_amplitude) noexcept {
    return 4.0 + 4000.0 * height_amplitude / (polar_ratio * polar_ratio * polar_ratio);
}

namespace {

using LeafMap = std::unordered_map<uint64_t, size_t>;

int neighbor(const LeafMap& leaves, TileId tile, uint32_t edge) {
    double width = std::ldexp(2.0, -int(tile.level));
    double u = -1.0 + (tile.x + 0.5) * width;
    double v = -1.0 + (tile.y + 0.5) * width;
    double step = width * 0.50001;
    if (edge == 0) v -= step;
    if (edge == 1) u += step;
    if (edge == 2) v += step;
    if (edge == 3) u -= step;
    auto candidate = tile_at(cube_vector(tile.face, u, v), tile.level);
    for (;;) {
        if (auto found = leaves.find(candidate.key()); found != leaves.end()) return int(found->second);
        if (candidate.level == 0) return -1;
        candidate = candidate.parent();
    }
}

}

Selection select(const SelectionView& input, TileMetadataCache* metadata) {
    SelectionView view = input;
    view.max_level = std::min(view.max_level, MAX_LEVEL);
    view.max_patches = std::clamp(view.max_patches, 6u, MAX_PATCHES);
    view.polar_ratio = std::clamp(view.polar_ratio, 0.05, 1.0);
    view.height_amplitude = std::clamp(view.height_amplitude, 0.0, 0.01);
    view.target_error_pixels = std::max(view.target_error_pixels, 0.1);
    struct Candidate { TileId tile; double error; };
    struct Node { Candidate candidate; bool active = true; bool scheduled = false; };
    struct Request {
        double error;
        size_t index;
        bool operator<(const Request& other) const noexcept {
            return error != other.error ? error < other.error : index > other.index;
        }
    };
    Selection result;
    std::vector<Node> nodes;
    nodes.reserve(view.max_patches * 4);
    LeafMap leaves;
    leaves.reserve(view.max_patches * 2);
    std::priority_queue<Request> queue;
    std::vector<size_t> splits;
    std::vector<Candidate> children;
    splits.reserve(64);
    children.reserve(256);
    auto evaluate = [&](TileId tile, Candidate& candidate) {
        ++result.visited;
        TileMetadata local;
        if (view.height_amplitude > 0.0) local = metadata ? metadata->get(tile) : tile_metadata(tile);
        auto bound = view.height_amplitude > 0.0 ? surface_bounds(tile, view.polar_ratio, view.height_amplitude, local)
                                               : bounds(tile, view.polar_ratio, 0.0);
        if (!visible(bound, view)) return false;
        double coefficient = surface_error_coefficient(view.polar_ratio, view.height_amplitude, local);
        double width = std::ldexp(2.0, -int(tile.level));
        double distance = std::max(glm::length(bound.center - view.eye) - bound.radius, 1e-12);
        double error = coefficient * width * width / (GRID_CELLS * GRID_CELLS) * view.focal_pixels / distance;
        candidate = {tile, error};
        return true;
    };
    auto add = [&](Candidate candidate) {
        leaves.emplace(candidate.tile.key(), nodes.size());
        queue.push({candidate.error, nodes.size()});
        nodes.push_back({candidate});
    };
    for (uint32_t face = 0; face < 6; ++face) {
        Candidate candidate;
        if (evaluate({face}, candidate)) add(candidate);
    }
    auto schedule = [&](auto&& self, size_t index) -> void {
        if (nodes[index].scheduled) return;
        nodes[index].scheduled = true;
        auto tile = nodes[index].candidate.tile;
        for (uint32_t edge = 0; edge < 4; ++edge) {
            int other = neighbor(leaves, tile, edge);
            if (other >= 0 && nodes[other].candidate.tile.level < tile.level) self(self, size_t(other));
        }
        splits.push_back(index);
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                Candidate candidate;
                if (evaluate({tile.face, tile.level + 1, tile.x * 2 + x, tile.y * 2 + y}, candidate)) children.push_back(candidate);
            }
        }
    };
    while (!queue.empty()) {
        auto request = queue.top();
        queue.pop();
        if (!nodes[request.index].active) continue;
        auto tile = nodes[request.index].candidate.tile;
        if (request.error <= view.target_error_pixels || tile.level >= view.max_level) continue;
        splits.clear();
        children.clear();
        schedule(schedule, request.index);
        if (leaves.size() - splits.size() + children.size() > view.max_patches) {
            result.budget_limited = true;
            break;
        }
        for (size_t index : splits) {
            nodes[index].active = false;
            leaves.erase(nodes[index].candidate.tile.key());
        }
        for (auto candidate : children) add(candidate);
    }
    result.patches.reserve(leaves.size());
    for (const auto& node : nodes) {
        if (!node.active) continue;
        Patch patch{node.candidate.tile};
        for (uint32_t edge = 0; edge < 4; ++edge) {
            int other = neighbor(leaves, patch.tile, edge);
            if (other >= 0 && nodes[other].candidate.tile.level < patch.tile.level) patch.coarse_edges |= 1u << edge;
        }
        result.patches.push_back(patch);
    }
    std::sort(result.patches.begin(), result.patches.end(), [](const Patch& a, const Patch& b) {
        return a.tile.key() < b.tile.key();
    });
    for (const auto& patch : result.patches) result.deepest_level = std::max(result.deepest_level, patch.tile.level);
    return result;
}

std::vector<uint32_t> grid_indices(uint32_t coarse_edges) {
    std::vector<uint32_t> indices;
    indices.reserve(GRID_CELLS * GRID_CELLS * 6);
    auto vertex = [coarse_edges](uint32_t x, uint32_t y) {
        if ((y == 0 && (coarse_edges & 1)) || (y == GRID_CELLS && (coarse_edges & 4))) x &= ~1u;
        if ((x == GRID_CELLS && (coarse_edges & 2)) || (x == 0 && (coarse_edges & 8))) y &= ~1u;
        return y * (GRID_CELLS + 1) + x;
    };
    for (uint32_t y = 0; y < GRID_CELLS; ++y) {
        for (uint32_t x = 0; x < GRID_CELLS; ++x) {
            auto a = vertex(x, y), b = vertex(x + 1, y), c = vertex(x, y + 1), d = vertex(x + 1, y + 1);
            indices.insert(indices.end(), {a, b, c, b, d, c});
        }
    }
    return indices;
}

}
