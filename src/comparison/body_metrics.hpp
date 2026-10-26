#pragma once

#include "ephemeris/system_manager.hpp"
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sf::comparison {

inline constexpr double EQUILIBRIUM_TEMPERATURE_1AU_K = 278.6;

struct BodyMetrics {
    std::string name;
    std::string type;
    bool star = false;
    int parent = -1;
    double mass_msun = 0.0;
    double radius_km = 0.0;
    std::optional<double> semi_major_au;
    std::optional<double> eccentricity;
    std::optional<double> period_yr;
    std::optional<double> equilibrium_k;
};

BodyMetrics measure_body(const ephemeris::SystemSnapshot& snapshot, size_t index);

enum class MatchMode { Name, OrbitOrder };

struct MatchRow {
    std::string label;
    std::vector<int> bodies;
};

std::vector<MatchRow> match_bodies(std::span<const ephemeris::SystemSnapshot* const> systems, MatchMode mode);

}
