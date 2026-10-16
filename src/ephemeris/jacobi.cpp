#include "ephemeris/jacobi.hpp"
#include "core/math_utils.hpp"
#include <algorithm>

namespace sf::ephemeris {
namespace {

void merge(JacobiSubsystem& inner, const JacobiSubsystem& outer) {
    double mass = inner.mass + outer.mass;
    if (mass > 0.0) {
        double fraction = outer.mass / mass;
        inner.position += fraction * (outer.position - inner.position);
        inner.velocity += fraction * (outer.velocity - inner.velocity);
    }
    inner.mass = mass;
}

}

JacobiHierarchy build_jacobi_hierarchy(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                      const JacobiHierarchy* previous) {
    size_t count = definition.bodies.size();
    JacobiHierarchy hierarchy;
    hierarchy.children.resize(count);
    hierarchy.subsystem_mass.resize(count);
    hierarchy.inner_mass.resize(count);
    if (previous) hierarchy.children = previous->children;
    for (size_t i = 0; i < count; ++i) {
        hierarchy.subsystem_mass[i] = definition.bodies[i].mass_msun;
        int parent = definition.bodies[i].parent_idx;
        if (!previous && parent >= 0) hierarchy.children[parent].push_back(static_cast<int>(i));
    }
    for (size_t i = count; i-- > 0;) {
        int parent = definition.bodies[i].parent_idx;
        if (parent >= 0) hierarchy.subsystem_mass[parent] += hierarchy.subsystem_mass[i];
    }
    for (size_t parent = 0; parent < count; ++parent) {
        auto& children = hierarchy.children[parent];
        if (!previous) std::stable_sort(children.begin(), children.end(), [&](int a, int b) {
            auto ra = motions[a].position.value - motions[parent].position.value;
            auto rb = motions[b].position.value - motions[parent].position.value;
            return glm::dot(ra, ra) < glm::dot(rb, rb);
        });
        double mass = definition.bodies[parent].mass_msun;
        for (int child : children) {
            hierarchy.inner_mass[child] = mass;
            mass += hierarchy.subsystem_mass[child];
        }
    }
    return hierarchy;
}

JacobiCoordinates extract_jacobi(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                 const JacobiHierarchy& hierarchy) {
    JacobiCoordinates coordinates;
    coordinates.subsystems.resize(motions.size());
    coordinates.frames.resize(motions.size());
    for (size_t i = 0; i < motions.size(); ++i) {
        coordinates.subsystems[i] = {definition.bodies[i].mass_msun, motions[i].position.value, motions[i].velocity.value};
    }
    for (size_t i = motions.size(); i-- > 0;) {
        int parent = definition.bodies[i].parent_idx;
        if (parent >= 0) merge(coordinates.subsystems[parent], coordinates.subsystems[i]);
    }
    for (size_t parent = 0; parent < motions.size(); ++parent) {
        JacobiSubsystem inner{definition.bodies[parent].mass_msun, motions[parent].position.value, motions[parent].velocity.value};
        for (int child : hierarchy.children[parent]) {
            auto& frame = coordinates.frames[child];
            frame.inner = inner;
            frame.relative_position = coordinates.subsystems[child].position - inner.position;
            frame.relative_velocity = coordinates.subsystems[child].velocity - inner.velocity;
            merge(inner, coordinates.subsystems[child]);
            frame.barycenter = inner.position;
        }
    }
    return coordinates;
}

void JacobiPropagator::reset(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                            const JacobiHierarchy& hierarchy, units::Years time) {
    auto coordinates = extract_jacobi(definition, motions, hierarchy);
    epoch_ = time;
    seeds_.resize(motions.size());
    initial_centers_.resize(motions.size());
    relative_.resize(motions.size());
    centers_.resize(motions.size());
    for (size_t i = 0; i < motions.size(); ++i) {
        const auto& frame = coordinates.frames[i];
        seeds_[i] = {EclipticPosition{frame.relative_position}, EclipticVelocity{frame.relative_velocity}};
        initial_centers_[i] = {EclipticPosition{coordinates.subsystems[i].position}, EclipticVelocity{coordinates.subsystems[i].velocity}};
    }
}

void JacobiPropagator::evaluate(const SystemDefinition& definition, const JacobiHierarchy& hierarchy,
                               units::Years time, std::vector<BodyMotion>& output) {
    double dt = time.value - epoch_.value;
    for (size_t i = 0; i < seeds_.size(); ++i) {
        if (definition.bodies[i].parent_idx < 0) {
            centers_[i] = initial_centers_[i];
            centers_[i].position.value += dt * centers_[i].velocity.value;
        } else {
            double mu = G_CONST * (hierarchy.inner_mass[i] + hierarchy.subsystem_mass[i]);
            auto state = math::propagate_conic_state(seeds_[i].position.value, seeds_[i].velocity.value, mu, dt);
            relative_[i] = {EclipticPosition{state.pos}, EclipticVelocity{state.vel}};
        }
    }
    output.resize(seeds_.size());
    for (size_t parent = 0; parent < seeds_.size(); ++parent) {
        auto inner = centers_[parent];
        const auto& children = hierarchy.children[parent];
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            int child = *it;
            double total = hierarchy.inner_mass[child] + hierarchy.subsystem_mass[child];
            double fraction = total > 0.0 ? hierarchy.subsystem_mass[child] / total : 0.0;
            inner.position.value -= fraction * relative_[child].position.value;
            inner.velocity.value -= fraction * relative_[child].velocity.value;
            centers_[child] = {EclipticPosition{inner.position.value + relative_[child].position.value},
                               EclipticVelocity{inner.velocity.value + relative_[child].velocity.value}};
        }
        output[parent] = inner;
    }
}

}
