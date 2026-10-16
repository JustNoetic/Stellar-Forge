#include "ephemeris/jacobi.hpp"
#include "core/math_utils.hpp"
#include <algorithm>
#include <numeric>

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
                                      const JacobiHierarchy* previous, std::span<const int> parents) {
    size_t count = definition.bodies.size();
    JacobiHierarchy hierarchy;
    hierarchy.children.resize(count);
    hierarchy.subsystem_mass.resize(count);
    hierarchy.inner_mass.resize(count);
    hierarchy.parents.resize(count);
    for (size_t i = 0; i < count; ++i) hierarchy.parents[i] = parents.empty() ? definition.bodies[i].parent_idx : parents[i];
    bool preserve = previous && previous->parents == hierarchy.parents;
    if (preserve) hierarchy.children = previous->children;
    for (size_t i = 0; i < count; ++i) {
        hierarchy.subsystem_mass[i] = definition.bodies[i].mass_msun;
        int parent = hierarchy.parents[i];
        if (!preserve && parent >= 0) hierarchy.children[parent].push_back(static_cast<int>(i));
    }
    if (parents.empty()) {
        hierarchy.traversal.resize(count);
        std::iota(hierarchy.traversal.begin(), hierarchy.traversal.end(), 0);
    } else {
        for (size_t i = 0; i < count; ++i) if (hierarchy.parents[i] < 0) hierarchy.traversal.push_back(static_cast<int>(i));
        for (size_t i = 0; i < hierarchy.traversal.size(); ++i) {
            const auto& children = hierarchy.children[hierarchy.traversal[i]];
            hierarchy.traversal.insert(hierarchy.traversal.end(), children.begin(), children.end());
        }
    }
    for (auto it = hierarchy.traversal.rbegin(); it != hierarchy.traversal.rend(); ++it) {
        int i = *it;
        int parent = hierarchy.parents[i];
        if (parent >= 0) hierarchy.subsystem_mass[parent] += hierarchy.subsystem_mass[i];
    }
    for (size_t parent = 0; parent < count; ++parent) {
        auto& children = hierarchy.children[parent];
        if (!preserve) std::stable_sort(children.begin(), children.end(), [&](int a, int b) {
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
    for (auto it = hierarchy.traversal.rbegin(); it != hierarchy.traversal.rend(); ++it) {
        int i = *it;
        int parent = hierarchy.parents[i];
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
    size_t count = motions.size();
    seeds_.resize(count);
    initial_centers_.resize(count);
    relative_.resize(count);
    centers_.resize(count);
    orbits_.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& frame = coordinates.frames[i];
        seeds_[i] = {EclipticPosition{frame.relative_position}, EclipticVelocity{frame.relative_velocity}};
        initial_centers_[i] = {EclipticPosition{coordinates.subsystems[i].position}, EclipticVelocity{coordinates.subsystems[i].velocity}};
        orbits_[i] = KeplerOrbit{};
    }

    std::vector<bool> computed(count, false);
    auto compute_body_orbit = [&](auto& self, size_t i) -> void {
        if (computed[i]) return;
        computed[i] = true;

        int parent_idx = definition.bodies[i].parent_idx;
        if (parent_idx < 0 || parent_idx >= static_cast<int>(count)) return;

        self(self, static_cast<size_t>(parent_idx));

        double mu = G_CONST * (hierarchy.inner_mass[i] + hierarchy.subsystem_mass[i]);
        const auto& rel_pos = seeds_[i].position.value;
        const auto& rel_vel = seeds_[i].velocity.value;

        auto kv = math::cartesian_to_orbital_vectors(rel_pos, rel_vel, mu);
        auto& orb = orbits_[i];
        orb.a = kv.a;
        orb.e = kv.e;
        orb.M0 = kv.M0;
        orb.mu = mu;
        orb.n = kv.n;
        orb.u_e = kv.u_e;
        orb.prec_vec = glm::dvec3{0.0};
        orb.domega_peri = 0.0;
        orb.is_elliptic = (kv.a > 1e-12 && mu > 1e-30 && kv.e < 0.999);

        if (orb.a > 0.0 && orb.e < 1.0) {
            double n_mean = std::sqrt(mu / (orb.a * orb.a * orb.a));
            double p_sl = orb.a * (1.0 - orb.e * orb.e);

            const auto& parent_def = definition.bodies[parent_idx];
            double parent_req = parent_def.radius_au();
            double parent_j2 = parent_def.j2;
            if (parent_j2 != 0.0 && parent_req > 0.0 && p_sl > 0.0) {
                glm::dvec3 parent_pole = parent_def.pole_ecl;
                double pole_len = glm::length(parent_pole);
                if (pole_len > 1e-10) parent_pole /= pole_len;
                else parent_pole = glm::dvec3{0.0, 0.0, 1.0};

                double cos_i_j2 = glm::dot(orb.n, parent_pole);
                double j2_ratio = parent_req / p_sl;
                double j2_factor = 1.5 * parent_j2 * (j2_ratio * j2_ratio) * n_mean;

                double dOmega_j2 = -j2_factor * cos_i_j2;
                orb.prec_vec += dOmega_j2 * parent_pole;
                orb.domega_peri += 0.5 * j2_factor * (5.0 * cos_i_j2 * cos_i_j2 - 1.0);
            }

            int gp_idx = parent_def.parent_idx;
            if (gp_idx >= 0 && gp_idx < static_cast<int>(count)) {
                const auto& gp_orb = orbits_[parent_idx];
                if (gp_orb.a > 0.0 && gp_orb.mu > 0.0) {
                    double n_3rd = std::sqrt(gp_orb.mu / (gp_orb.a * gp_orb.a * gp_orb.a));
                    glm::dvec3 gp_pole = gp_orb.n;

                    double cos_i_3rd = glm::dot(orb.n, gp_pole);
                    double e_fac = std::sqrt(std::max(0.0001, 1.0 - orb.e * orb.e));
                    double tb_factor = 0.75 * (n_3rd * n_3rd / n_mean) / e_fac;

                    double dOmega_3rd = -tb_factor * cos_i_3rd;
                    orb.prec_vec += dOmega_3rd * gp_pole;
                    orb.domega_peri += tb_factor * (5.0 * cos_i_3rd * cos_i_3rd - 1.0);
                }
            }

            if (p_sl > 0.0) {
                double gr_factor = 3.0 * mu * n_mean / (C_AU_YR * C_AU_YR * p_sl);
                orb.domega_peri += gr_factor;
            }
        }
    };

    for (size_t i = 0; i < count; ++i) {
        compute_body_orbit(compute_body_orbit, i);
    }
}

void JacobiPropagator::evaluate(const SystemDefinition& definition, const JacobiHierarchy& hierarchy,
                               units::Years time, std::vector<BodyMotion>& output) {
    double dt = time.value - epoch_.value;
    for (size_t i = 0; i < seeds_.size(); ++i) {
        if (definition.bodies[i].parent_idx < 0) {
            centers_[i] = initial_centers_[i];
            centers_[i].position.value += dt * centers_[i].velocity.value;
        } else if (dt == 0.0) {
            relative_[i] = seeds_[i];
        } else {
            const auto& orb = orbits_[i];
            if (!orb.is_elliptic) {
                auto state = math::propagate_conic_state(seeds_[i].position.value, seeds_[i].velocity.value, orb.mu, dt);
                if (orb.e < 1.0 && orb.a > 1e-12 && orb.mu > 1e-30) {
                    double prec_mag = glm::length(orb.prec_vec);
                    if (prec_mag > 1e-15) {
                        glm::dvec3 axis = orb.prec_vec / prec_mag;
                        state.pos = math::axis_angle_rotation(state.pos, axis, prec_mag * dt);
                        state.vel = math::axis_angle_rotation(state.vel, axis, prec_mag * dt);
                    }
                    if (std::abs(orb.domega_peri) > 1e-15) {
                        glm::dvec3 n_curr = orb.n;
                        if (prec_mag > 1e-15) {
                            glm::dvec3 axis = orb.prec_vec / prec_mag;
                            n_curr = math::axis_angle_rotation(n_curr, axis, prec_mag * dt);
                        }
                        state.pos = math::axis_angle_rotation(state.pos, n_curr, orb.domega_peri * dt);
                        state.vel = math::axis_angle_rotation(state.vel, n_curr, orb.domega_peri * dt);
                    }
                }
                relative_[i] = {EclipticPosition{state.pos}, EclipticVelocity{state.vel}};
            } else {
                double mean_motion = std::sqrt(orb.mu / (orb.a * orb.a * orb.a));
                double M_t = orb.M0 + mean_motion * dt;
                M_t = std::fmod(M_t, TWO_PI);
                if (M_t < 0.0) M_t += TWO_PI;

                glm::dvec3 curr_n = orb.n;
                glm::dvec3 curr_e = orb.u_e;

                double prec_mag = glm::length(orb.prec_vec);
                if (prec_mag > 1e-15) {
                    glm::dvec3 axis = orb.prec_vec / prec_mag;
                    double theta = prec_mag * dt;
                    curr_n = math::axis_angle_rotation(curr_n, axis, theta);
                    curr_e = math::axis_angle_rotation(curr_e, axis, theta);
                }

                if (std::abs(orb.domega_peri) > 1e-15) {
                    double omega_theta = orb.domega_peri * dt;
                    curr_e = math::axis_angle_rotation(curr_e, curr_n, omega_theta);
                }

                auto state = math::vector_orbital_to_cartesian(orb.a, orb.e, curr_n, curr_e, M_t, orb.mu);
                relative_[i] = {EclipticPosition{state.pos}, EclipticVelocity{state.vel}};
            }
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
