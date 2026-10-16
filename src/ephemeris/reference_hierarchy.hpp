#pragma once

#include "ephemeris/system_manager.hpp"

namespace sf::ephemeris {

class ReferenceHierarchyTracker {
public:
    ReferenceHierarchy evaluate(const SystemDefinition& definition, std::span<const BodyMotion> motions,
                                const ReferenceHierarchy& previous);
    size_t candidate_checks() const noexcept { return candidate_checks_; }

private:
    struct Node {
        glm::dvec3 lower{}, upper{};
        int left = -1, right = -1, parent = -1, body = -1;
    };

    int build(std::span<const BodyMotion> motions, size_t first, size_t last, int parent);
    void refit(int body, glm::dvec3 position, double radius);
    void query(int node, int body, int current, const SystemDefinition& definition,
               std::span<const BodyMotion> motions, int& best, double& best_radius, double& best_distance);

    std::vector<Node> nodes_;
    std::vector<int> entries_, leaves_, ranked_, anchors_;
    std::vector<double> radii_;
    size_t candidate_checks_ = 0;
};

}
