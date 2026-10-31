#include "terrain/quadtree.hpp"
#include "terrain/tile_metadata.hpp"
#include "terrain/terrain_source.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory_resource>
#include <limits>
#include <unordered_map>

namespace sf::terrain {

uint64_t TileId::key() const noexcept {
    return (uint64_t(face) << 61) | (uint64_t(level) << 56) | (uint64_t(x) << 28) | y;
}

TileId TileId::from_key(uint64_t key) noexcept {
    return {uint32_t(key >> 61), uint32_t((key >> 56) & 31), uint32_t((key >> 28) & 0xfffffff), uint32_t(key & 0xfffffff)};
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
    return surface_bounds(tile, polar_ratio, height_amplitude, TileMetadata{-1.0, 1.0, 0.0, 0.0});
}

double TileBounds::support(glm::dvec3 axis) const noexcept {
    if (angular_radius == 0.0) return radius*glm::length(axis);
    auto scaled = axis*glm::dvec3(1, polar_ratio, 1);
    double radial = glm::dot(scaled, direction);
    double tangent = glm::length(scaled-direction*radial);
    return std::abs(radial)*angular_radius*angular_radius*.25
        + tangent*std::min(angular_radius, 1.0) + padding*glm::length(axis);
}

double TileBounds::distance(glm::dvec3 eye) const noexcept {
    double result = glm::length(center-eye)-radius;
    if (angular_radius > 0.0) {
        auto offset = eye-surface_center;
        double length = glm::length(offset);
        if (length > 0.0) result = std::max(result, length-support(offset/length));
    }
    return std::max(result, 1e-12);
}

namespace {

class Visibility {
public:
    explicit Visibility(const SelectionView& view) : view_(view) {
        auto local_from_view = glm::transpose(view.view_from_local);
        planes_ = {glm::dvec3(0, 0, -1), glm::dvec3(1, 0, -view.tan_half_fov_x),
            glm::dvec3(-1, 0, -view.tan_half_fov_x), glm::dvec3(0, 1, -view.tan_half_fov_y),
            glm::dvec3(0, -1, -view.tan_half_fov_y)};
        for (auto& plane : planes_) plane = glm::normalize(local_from_view*plane);
        scale_ = {1, 1/view.polar_ratio, 1};
        eye_ = view.eye*scale_;
        double distance = glm::length(eye_);
        double occluder = (1.0-view.height_amplitude*(1.0+RELIEF_ROUNDING)/view.polar_ratio)*view.occluder_scale;
        if (occluder > 0.0 && distance > occluder) {
            axis_ = -eye_/distance;
            sine_ = occluder/distance;
            cosine_ = std::sqrt(std::max(0.0, 1.0-sine_*sine_));
            tangent_distance_ = distance*cosine_;
            horizon_ = occluder*sine_;
            occluder_squared_ = occluder*occluder;
            horizon_normal_ = -axis_*scale_;
        }
        tolerance_ = 64*std::numeric_limits<double>::epsilon()*(1+glm::length(view.eye));
    }

    bool visible(const TileBounds& tile) const noexcept {
        if (!view_.cull) return true;
        if (view_.orthographic_pixels > 0.0) {
            for (int axis = 0; axis < 2; ++axis) {
                auto local_axis = glm::transpose(view_.view_from_local)[axis];
                double center = glm::dot(local_axis,tile.angular_radius > 0.0 ? tile.surface_center : tile.center);
                double support = tile.support(local_axis);
                if (center+support < view_.orthographic_bounds[axis]-tolerance_ ||
                    center-support > view_.orthographic_bounds[axis+2]+tolerance_) return false;
            }
            auto front = glm::transpose(view_.view_from_local)[2];
            auto normal_axis = front*glm::dvec3(1,1/view_.polar_ratio,1);
            return glm::dot(normal_axis,tile.direction)+glm::length(normal_axis)*tile.angular_radius >= -tolerance_;
        }
        for (auto plane : planes_) {
            double distance = glm::dot(plane, tile.center-view_.eye);
            if (distance+tile.radius < -tolerance_) return false;
            if (tile.angular_radius > 0.0 && distance < tile.radius &&
                glm::dot(plane, tile.surface_center-view_.eye)+tile.support(plane) < -tolerance_) return false;
        }
        if (sine_ == 0.0) return true;
        if (tile.maximum_scaled_radius > 0.0) {
            double extent = tile.angular_radius > 0.0
                ? glm::dot(horizon_normal_, tile.surface_center)+tile.support(horizon_normal_)
                : glm::dot(horizon_normal_, tile.center)+tile.radius*glm::length(horizon_normal_);
            double height = std::sqrt(std::max(0.0,
                tile.maximum_scaled_radius*tile.maximum_scaled_radius-occluder_squared_));
            if (extent < horizon_-cosine_*height-tolerance_) return false;
        }
        auto to_tile = tile.center*scale_-eye_;
        double radius = tile.radius/view_.polar_ratio;
        double along = glm::dot(to_tile, axis_);
        double perpendicular = glm::length(to_tile-axis_*along);
        bool inside_cone = along*sine_-perpendicular*cosine_ > radius+tolerance_;
        bool beyond_tangent = glm::length(to_tile)-radius > tangent_distance_+tolerance_;
        return !(inside_cone && beyond_tangent);
    }

private:
    const SelectionView& view_;
    std::array<glm::dvec3, 5> planes_;
    glm::dvec3 scale_, eye_, axis_{0}, horizon_normal_{0};
    double sine_ = 0, cosine_ = 0, tangent_distance_ = 0, horizon_ = 0, occluder_squared_ = 0, tolerance_ = 0;
};

}

bool visible(const TileBounds& tile, const SelectionView& view) noexcept {
    return Visibility(view).visible(tile);
}

double error_coefficient(double polar_ratio, double height_amplitude) noexcept {
    return 4.0 + 4000.0 * height_amplitude / (polar_ratio * polar_ratio * polar_ratio);
}

namespace {

using LeafMap = std::pmr::unordered_map<uint64_t, size_t>;

int coarser_neighbor(const LeafMap& leaves, TileId tile, uint32_t edge) {
    if (tile.level == 0) return -1;
    if (edge == 0 && (tile.y & 1)) return -1;
    if (edge == 1 && !(tile.x & 1)) return -1;
    if (edge == 2 && !(tile.y & 1)) return -1;
    if (edge == 3 && (tile.x & 1)) return -1;
    tile = tile.parent();
    TileId candidate = tile;
    uint32_t n = 1u << tile.level;
    if (edge == 0 && tile.y > 0) --candidate.y;
    else if (edge == 1 && tile.x + 1 < n) ++candidate.x;
    else if (edge == 2 && tile.y + 1 < n) ++candidate.y;
    else if (edge == 3 && tile.x > 0) --candidate.x;
    else {
        double width = std::ldexp(2.0, -int(tile.level));
        double u = -1.0 + (tile.x + 0.5) * width;
        double v = -1.0 + (tile.y + 0.5) * width;
        double step = width * 0.50001;
        if (edge == 0) v -= step;
        if (edge == 1) u += step;
        if (edge == 2) v += step;
        if (edge == 3) u -= step;
        candidate = tile_at(cube_vector(tile.face, u, v), tile.level);
    }
    if (auto found = leaves.find(candidate.key()); found != leaves.end()) return int(found->second);
    return -1;
}

}

struct SelectionScratch::Storage {
    struct Candidate { TileId tile; double error; };
    struct Node { Candidate candidate; bool active = true; bool scheduled = false; uint32_t decision = UINT32_MAX; };
    struct Request {
        double error;
        size_t index;
        bool operator<(const Request& other) const noexcept {
            return error != other.error ? error < other.error : index > other.index;
        }
    };
    std::pmr::unsynchronized_pool_resource pool;
    std::vector<Node> nodes;
    LeafMap leaves{&pool};
    std::vector<Request> queue;
    std::vector<size_t> splits;
    std::vector<Candidate> children;
    std::vector<uint32_t> child_decisions;
    std::vector<TileId> prerequisites;
    uint32_t capacity = 0;
};

SelectionScratch::SelectionScratch() = default;
SelectionScratch::~SelectionScratch() = default;
SelectionScratch::SelectionScratch(SelectionScratch&&) noexcept = default;
SelectionScratch& SelectionScratch::operator=(SelectionScratch&&) noexcept = default;

struct SelectionPlan::Storage {
    struct Decision {
        SurfaceTile surface;
        bool visible = false;
        bool refine = false;
        bool selected = false;
        uint8_t children = 0;
    };
    SelectionView view;
    SourceDomain source;
    uint64_t bounds_revision = 0;
    std::vector<Decision> decisions;
    uint32_t checks = 0;
    uint32_t budget_bound = 0;
    bool valid = false;
    bool limited = false;
    uint32_t misses = 0;
    uint32_t cooldown = 0;
    void missed() noexcept {
        if (++misses == 2) { cooldown = 32; misses = 0; }
    }
};

SelectionPlan::SelectionPlan() = default;
SelectionPlan::~SelectionPlan() = default;
SelectionPlan::SelectionPlan(SelectionPlan&&) noexcept = default;
SelectionPlan& SelectionPlan::operator=(SelectionPlan&&) noexcept = default;
void SelectionPlan::invalidate() noexcept {
    if (storage_) {
        if (storage_->valid) storage_->missed();
        storage_->valid = false;
        storage_->limited = false;
    }
}
uint32_t SelectionPlan::checks() const noexcept { return storage_ ? storage_->checks : 0; }

static SelectionView normalized_view(SelectionView view) {
    view.max_level = std::min(view.max_level, MAX_LEVEL);
    view.max_patches = std::clamp(view.max_patches, 6u, MAX_PATCHES);
    view.polar_ratio = std::clamp(view.polar_ratio, 0.05, 1.0);
    view.height_amplitude = std::clamp(view.height_amplitude, 0.0, 0.01);
    view.target_error_pixels = std::max(view.target_error_pixels, 0.1);
    return view;
}

bool update_selection(Selection& selection, SelectionPlan& plan, const SelectionView& input, const TerrainSource& source,
                      const TileAvailability& availability, SelectionScratch* scratch) {
    auto view = normalized_view(input);
    if (plan.storage_) {
        auto& previous = *plan.storage_;
        previous.checks = 0;
        auto comparable = previous.view;
        comparable.eye = view.eye;
        comparable.view_from_local = view.view_from_local;
        comparable.tan_half_fov_x = view.tan_half_fov_x;
        comparable.tan_half_fov_y = view.tan_half_fov_y;
        comparable.focal_pixels = view.focal_pixels;
        comparable.orthographic_pixels = view.orthographic_pixels;
        comparable.orthographic_bounds = view.orthographic_bounds;
        comparable.target_error_pixels = view.target_error_pixels;
        comparable.occluder_scale = view.occluder_scale;
        if (previous.valid && !availability.ready && comparable == view && previous.source == source.domain() &&
            previous.bounds_revision == source.bounds_revision()) {
            Visibility visibility(view);
            double maximum_error = 0;
            bool valid = true;
            for (const auto& decision : previous.decisions) {
                ++previous.checks;
                bool seen = visibility.visible(decision.surface.bounds);
                if (seen != decision.visible) { valid = false; break; }
                if (!seen) continue;
                double error = decision.surface.geometric_error * (view.orthographic_pixels > 0 ?
                    view.orthographic_pixels : view.focal_pixels/decision.surface.bounds.distance(view.eye));
                if ((error > view.target_error_pixels) != decision.refine) { valid = false; break; }
                if (decision.selected) maximum_error = std::max(maximum_error, error);
            }
            if (valid) {
                previous.misses = 0;
                selection.max_error_pixels = maximum_error;
                return false;
            }
        }
        if (previous.valid) previous.missed();
    }
    selection = select(view, source, availability, scratch, &plan);
    return true;
}

Selection select(const SelectionView& input, TileMetadataCache* metadata, SelectionScratch* scratch) {
    return select(input, ProceduralSource{metadata}, {}, scratch);
}

Selection select(const SelectionView& input, const TerrainSource& source, const TileAvailability& availability, SelectionScratch* scratch, SelectionPlan* plan) {
    SelectionView view = normalized_view(input);
    SelectionPlan::Storage* retained = nullptr;
    if (plan) {
        if (!plan->storage_) plan->storage_ = std::make_unique<SelectionPlan::Storage>();
        retained = plan->storage_.get();
        retained->valid = false;
        retained->decisions.clear();
        retained->budget_bound = 0;
        retained->view = view;
        retained->source = source.domain();
        retained->bounds_revision = source.bounds_revision();
    }
    bool capture = retained && !retained->limited && !retained->cooldown && !availability.ready;
    if (retained && retained->cooldown) --retained->cooldown;
    uint32_t decision_limit = std::min(MAX_PATCHES*4, view.max_patches*4+MAX_LEVEL*4+6);
    Visibility visibility(view);
    SelectionScratch local;
    auto& storage = scratch ? scratch->storage_ : local.storage_;
    if (!storage) storage = std::make_unique<SelectionScratch::Storage>();
    auto& [pool, nodes, leaves, queue, splits, children, child_decisions, prerequisites, capacity] = *storage;
    using Candidate = SelectionScratch::Storage::Candidate;
    nodes.clear();
    leaves.clear();
    queue.clear();
    if (capacity < view.max_patches) {
        nodes.reserve(view.max_patches * 4);
        leaves.reserve(view.max_patches * 2);
        queue.reserve(view.max_patches * 4);
        capacity = view.max_patches;
    }
    splits.reserve(64);
    children.reserve(256);
    prerequisites.reserve(256);
    Selection result;
    auto evaluate = [&](TileId tile, Candidate& candidate) {
        ++result.visited;
        auto description = source.describe(tile, view.polar_ratio, view.height_amplitude);
        auto bound = description.bounds;
        bool seen = visibility.visible(bound);
        double error = seen ? description.geometric_error * (view.orthographic_pixels > 0.0 ?
            view.orthographic_pixels : view.focal_pixels / bound.distance(view.eye)) : 0;
        if (capture) {
            if (retained->decisions.size() == decision_limit) capture = false;
            else retained->decisions.push_back({description, seen, error > view.target_error_pixels});
        }
        if (!seen) return false;
        candidate = {tile, error};
        return true;
    };
    auto add = [&](Candidate candidate, uint32_t decision) {
        leaves.emplace(candidate.tile.key(), nodes.size());
        queue.push_back({candidate.error, nodes.size()});
        std::push_heap(queue.begin(), queue.end());
        nodes.push_back({candidate, true, false, decision});
    };
    for (uint32_t face = 0; face < 6; ++face) {
        Candidate candidate;
        if (evaluate({face}, candidate)) {
            if (availability.ready && !availability.ready({face})) {
                if (availability.request) availability.request({face});
                result.loading_limited = true;
            }
            add(candidate, capture ? uint32_t(retained->decisions.size()-1) : UINT32_MAX);
            if (capture) ++retained->budget_bound;
            if (availability.ready || availability.request) result.required_tiles.push_back({face});
        }
    }
    if (result.loading_limited) return result;
    auto schedule = [&](auto&& self, size_t index) -> void {
        if (nodes[index].scheduled) return;
        nodes[index].scheduled = true;
        auto tile = nodes[index].candidate.tile;
        for (uint32_t edge = 0; edge < 4; ++edge) {
            int other = coarser_neighbor(leaves, tile, edge);
            if (other >= 0) self(self, size_t(other));
        }
        splits.push_back(index);
        uint8_t visible_children = 0;
        for (uint32_t y = 0; y < 2; ++y) {
            for (uint32_t x = 0; x < 2; ++x) {
                Candidate candidate;
                TileId child{tile.face, tile.level + 1, tile.x * 2 + x, tile.y * 2 + y};
                if (availability.ready || availability.request) prerequisites.push_back(child);
                if (evaluate(child, candidate)) {
                    ++visible_children;
                    children.push_back(candidate);
                    child_decisions.push_back(capture ? uint32_t(retained->decisions.size()-1) : UINT32_MAX);
                }
            }
        }
        if (capture) retained->decisions[nodes[index].decision].children = visible_children;
    };
    while (!queue.empty()) {
        std::pop_heap(queue.begin(), queue.end());
        auto request = queue.back();
        queue.pop_back();
        if (!nodes[request.index].active) continue;
        auto tile = nodes[request.index].candidate.tile;
        if (request.error <= view.target_error_pixels || tile.level >= view.max_level) continue;
        splits.clear();
        children.clear();
        child_decisions.clear();
        prerequisites.clear();
        schedule(schedule, request.index);
        if (leaves.size() - splits.size() + children.size() > view.max_patches) {
            result.budget_limited = true;
            break;
        }
        bool ready = true;
        if (availability.ready) {
            for (auto child : prerequisites) {
                if (!availability.ready(child)) {
                    ready = false;
                    if (availability.request) availability.request(child);
                }
            }
        }
        if (!ready) {
            result.loading_limited = true;
            for (size_t index : splits) nodes[index].scheduled = false;
            continue;
        }
        if (!availability.ready && availability.request) for (auto child : prerequisites) availability.request(child);
        result.required_tiles.insert(result.required_tiles.end(), prerequisites.begin(), prerequisites.end());
        for (size_t index : splits) {
            if (capture) retained->budget_bound += std::max(1u, uint32_t(retained->decisions[nodes[index].decision].children))-1;
            nodes[index].active = false;
            leaves.erase(nodes[index].candidate.tile.key());
        }
        for (size_t index = 0; index < children.size(); ++index) add(children[index], child_decisions[index]);
    }
    result.patches.reserve(leaves.size());
    for (const auto& node : nodes) {
        if (!node.active) continue;
        if (capture) retained->decisions[node.decision].selected = true;
        result.max_error_pixels = std::max(result.max_error_pixels, node.candidate.error);
        Patch patch{node.candidate.tile};
        for (uint32_t edge = 0; edge < 4; ++edge) {
            if (coarser_neighbor(leaves, patch.tile, edge) >= 0) patch.coarse_edges |= 1u << edge;
        }
        result.patches.push_back(patch);
    }
    std::sort(result.patches.begin(), result.patches.end(), [](const Patch& a, const Patch& b) {
        return a.tile.key() < b.tile.key();
    });
    for (const auto& patch : result.patches) result.deepest_level = std::max(result.deepest_level, patch.tile.level);
    if (retained) {
        retained->valid = capture && !result.budget_limited && !result.loading_limited && retained->budget_bound <= view.max_patches;
        retained->limited = result.budget_limited || result.visited > decision_limit;
    }
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
