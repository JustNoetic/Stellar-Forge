#include "ephemeris/reference_hierarchy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sf::ephemeris {
namespace {

constexpr double enter_scale = 0.9;
constexpr double exit_scale = 1.1;

bool can_parent(const SystemDefinition& definition, int parent, int body) {
    const auto& p = definition.bodies[parent];
    const auto& b = definition.bodies[body];
    return parent != body && !p.is_spacecraft && p.mass_msun > 0.0 &&
           (b.is_spacecraft || p.mass_msun > b.mass_msun || (p.mass_msun == b.mass_msun && parent < body));
}

}

int ReferenceHierarchyTracker::build(std::span<const BodyMotion> motions, size_t first, size_t last, int parent) {
    int index = static_cast<int>(nodes_.size());
    nodes_.push_back({});
    nodes_[index].parent = parent;
    auto lower = motions[entries_[first]].position.value, upper = lower;
    for (size_t i = first + 1; i < last; ++i) {
        lower = glm::min(lower, motions[entries_[i]].position.value);
        upper = glm::max(upper, motions[entries_[i]].position.value);
    }
    nodes_[index].lower = lower;
    nodes_[index].upper = upper;
    if (last - first == 1) {
        nodes_[index].body = entries_[first];
        leaves_[entries_[first]] = index;
    } else {
        auto extent = upper - lower;
        int axis = extent.y > extent.x ? 1 : 0;
        if (extent.z > extent[axis]) axis = 2;
        size_t middle = first + (last - first) / 2;
        std::nth_element(entries_.begin() + first, entries_.begin() + middle, entries_.begin() + last,
                         [&](int a, int b) { return motions[a].position.value[axis] < motions[b].position.value[axis]; });
        int left = build(motions, first, middle, index);
        int right = build(motions, middle, last, index);
        nodes_[index].left = left;
        nodes_[index].right = right;
    }
    return index;
}

void ReferenceHierarchyTracker::refit(int body, glm::dvec3 position, double radius) {
    int index = leaves_[body];
    if (index < 0) return;
    nodes_[index].lower = position - glm::dvec3(radius * exit_scale);
    nodes_[index].upper = position + glm::dvec3(radius * exit_scale);
    for (int parent = nodes_[index].parent; parent >= 0; parent = nodes_[parent].parent) {
        const auto& left = nodes_[nodes_[parent].left];
        const auto& right = nodes_[nodes_[parent].right];
        nodes_[parent].lower = glm::min(left.lower, right.lower);
        nodes_[parent].upper = glm::max(left.upper, right.upper);
    }
}

void ReferenceHierarchyTracker::query(int node, int body, int current, const SystemDefinition& definition,
                                     std::span<const BodyMotion> motions, int& best, double& best_radius, double& best_distance) {
    const auto& entry = nodes_[node];
    auto position = motions[body].position.value;
    if (glm::any(glm::lessThan(position, entry.lower)) || glm::any(glm::greaterThan(position, entry.upper))) return;
    if (entry.body < 0) {
        query(entry.left, body, current, definition, motions, best, best_radius, best_distance);
        query(entry.right, body, current, definition, motions, best, best_radius, best_distance);
        return;
    }
    int candidate = entry.body;
    double radius = radii_[candidate];
    if (!(radius > 0.0) || !can_parent(definition, candidate, body)) return;
    ++candidate_checks_;
    auto delta = position - motions[candidate].position.value;
    double distance = glm::dot(delta, delta) / (radius * radius);
    double scale = candidate == current ? exit_scale : enter_scale;
    if (distance > scale * scale) return;
    if (radius < best_radius || (radius == best_radius && (distance < best_distance ||
        (distance == best_distance && candidate < best)))) {
        best = candidate;
        best_radius = radius;
        best_distance = distance;
    }
}

ReferenceHierarchy ReferenceHierarchyTracker::evaluate(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                                       const ReferenceHierarchy& previous) {
    size_t count = motions.size();
    ReferenceHierarchy result;
    result.parents.resize(count);
    result.children.resize(count);
    candidate_checks_ = 0;
    bool dynamic = definition.motion_model == MotionModel::NBody;
    if (dynamic) {
        ranked_.resize(count);
        std::iota(ranked_.begin(), ranked_.end(), 0);
        std::sort(ranked_.begin(), ranked_.end(), [&](int a, int b) {
            const auto& x = definition.bodies[a];
            const auto& y = definition.bodies[b];
            if (x.is_spacecraft != y.is_spacecraft) return !x.is_spacecraft;
            return x.mass_msun != y.mass_msun ? x.mass_msun > y.mass_msun : a < b;
        });
        anchors_.resize(count);
        entries_.clear();
        leaves_.assign(count, -1);
        radii_.assign(count, 0.0);
        nodes_.clear();
        nodes_.reserve(count * 2);
        for (size_t i = 0; i < count; ++i) {
            const auto& body = definition.bodies[i];
            anchors_[i] = body.parent_idx < 0 ? static_cast<int>(i) : anchors_[body.parent_idx];
            if (body.parent_idx >= 0 && body.mass_msun > 0.0 && !body.is_spacecraft) entries_.push_back(static_cast<int>(i));
        }
        if (!entries_.empty()) build(motions, 0, entries_.size(), -1);
        for (int i : ranked_) {
            const auto& body = definition.bodies[i];
            int parent = -1;
            if (body.parent_idx >= 0) {
                int anchor = anchors_[i];
                if (can_parent(definition, anchor, i)) parent = anchor;
                else if (can_parent(definition, ranked_.front(), i)) parent = ranked_.front();
                int current = previous.parents.size() == count ? previous.parents[i] : body.parent_idx;
                double best_radius = std::numeric_limits<double>::infinity();
                double best_distance = std::numeric_limits<double>::infinity();
                if (!nodes_.empty()) query(0, i, current, definition, motions, parent, best_radius, best_distance);
            }
            result.parents[i] = parent;
            if (parent >= 0 && leaves_[i] >= 0) {
                double distance = glm::length(motions[i].position.value - motions[parent].position.value);
                radii_[i] = std::max(body.radius_au(), distance * std::cbrt((body.mass_msun / definition.bodies[parent].mass_msun) / 3.0));
                refit(i, motions[i].position.value, radii_[i]);
            }
        }
    } else {
        for (size_t i = 0; i < count; ++i) result.parents[i] = definition.bodies[i].parent_idx;
    }
    if (result.parents == previous.parents) result.children = previous.children;
    else for (size_t i = 0; i < count; ++i) if (result.parents[i] >= 0) result.children[result.parents[i]].push_back(static_cast<int>(i));
    for (size_t i = 0; i < count; ++i) {
        if (result.parents[i] < 0) result.roots.push_back(static_cast<int>(i));
        auto less = [&](int a, int b) {
            auto da = motions[a].position.value - motions[i].position.value;
            auto db = motions[b].position.value - motions[i].position.value;
            double ra = glm::dot(da, da), rb = glm::dot(db, db);
            return ra != rb ? ra < rb : a < b;
        };
        auto& children = result.children[i];
        if (!std::is_sorted(children.begin(), children.end(), less)) std::sort(children.begin(), children.end(), less);
    }
    return result;
}

}
