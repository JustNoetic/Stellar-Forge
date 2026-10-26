#include "comparison/body_metrics.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <unordered_map>

namespace sf::comparison {

using namespace ephemeris;

BodyMetrics measure_body(const SystemSnapshot& snapshot, size_t index) {
    const auto body = snapshot.body(index);
    const auto& def = body.definition;
    BodyMetrics metrics;
    metrics.name = def.name;
    metrics.type = def.type;
    metrics.star = def.is_star;
    metrics.mass_msun = def.mass_msun;
    metrics.radius_km = def.radius_au() * AU_TO_KM;
    metrics.parent = snapshot.reference_parent(index);
    if (metrics.parent >= 0 && size_t(metrics.parent) != index && size_t(metrics.parent) < snapshot.size()) {
        const auto host = snapshot.body(size_t(metrics.parent));
        double mu = G_CONST * (std::max(host.definition.mass_msun, 0.0) + std::max(def.mass_msun, 0.0));
        glm::dvec3 r = body.motion.position.value - host.motion.position.value;
        glm::dvec3 v = body.motion.velocity.value - host.motion.velocity.value;
        double distance = glm::length(r), speed2 = glm::dot(v, v);
        if (mu > 0.0 && distance > 0.0 && std::isfinite(distance) && std::isfinite(speed2)) {
            glm::dvec3 e = ((speed2 - mu / distance) * r - glm::dot(r, v) * v) / mu;
            metrics.eccentricity = glm::length(e);
            double energy = 0.5 * speed2 - mu / distance;
            if (energy < 0.0) {
                double a = -mu / (2.0 * energy);
                metrics.semi_major_au = a;
                metrics.period_yr = TWO_PI * std::sqrt(a * a * a / mu);
            }
        }
    }
    if (!def.is_star) {
        double flux = 0.0;
        for (size_t s = 0; s < snapshot.size(); ++s) {
            const auto star = snapshot.body(s);
            if (s == index || !star.definition.is_star) continue;
            double distance = glm::length(star.motion.position.value - body.motion.position.value);
            if (distance > 0.0) flux += std::max(star.definition.lum_eq, 0.0) / (distance * distance);
        }
        if (flux > 0.0 && std::isfinite(flux)) metrics.equilibrium_k = EQUILIBRIUM_TEMPERATURE_1AU_K * std::pow(flux, 0.25);
    }
    return metrics;
}

namespace {

std::vector<int> planets_by_distance(const SystemSnapshot& snapshot) {
    std::vector<std::pair<double, int>> planets;
    for (size_t i = 0; i < snapshot.size(); ++i) {
        const auto body = snapshot.body(i);
        int parent = snapshot.reference_parent(i);
        if (body.definition.is_star || parent < 0 || size_t(parent) >= snapshot.size() ||
            !snapshot.body(size_t(parent)).definition.is_star) continue;
        auto metrics = measure_body(snapshot, i);
        double distance = metrics.semi_major_au.value_or(
            glm::length(body.motion.position.value - snapshot.body(size_t(parent)).motion.position.value));
        planets.emplace_back(distance, int(i));
    }
    std::ranges::sort(planets);
    std::vector<int> result;
    for (const auto& [distance, index] : planets) result.push_back(index);
    return result;
}

}

std::vector<MatchRow> match_bodies(std::span<const SystemSnapshot* const> systems, MatchMode mode) {
    std::vector<MatchRow> rows;
    if (mode == MatchMode::Name) {
        std::unordered_map<std::string, size_t> seen;
        for (size_t slot = 0; slot < systems.size(); ++slot) {
            const auto& snapshot = *systems[slot];
            for (size_t i = 0; i < snapshot.size(); ++i) {
                const auto& name = snapshot.body(i).definition.name;
                auto [it, inserted] = seen.emplace(name, rows.size());
                if (inserted) rows.push_back({name, std::vector<int>(systems.size(), -1)});
                auto& cell = rows[it->second].bodies[slot];
                if (cell < 0) cell = int(i);
            }
        }
        return rows;
    }
    MatchRow stars{"Star", std::vector<int>(systems.size(), -1)};
    std::vector<std::vector<int>> planets(systems.size());
    size_t count = 0;
    for (size_t slot = 0; slot < systems.size(); ++slot) {
        const auto& snapshot = *systems[slot];
        int star = snapshot.star_index();
        if (star >= 0 && size_t(star) < snapshot.size() && snapshot.body(size_t(star)).definition.is_star) stars.bodies[slot] = star;
        planets[slot] = planets_by_distance(snapshot);
        count = std::max(count, planets[slot].size());
    }
    rows.push_back(std::move(stars));
    for (size_t rank = 0; rank < count; ++rank) {
        MatchRow row{std::format("Planet {}", rank + 1), std::vector<int>(systems.size(), -1)};
        for (size_t slot = 0; slot < systems.size(); ++slot)
            if (rank < planets[slot].size()) row.bodies[slot] = planets[slot][rank];
        rows.push_back(std::move(row));
    }
    return rows;
}

}
