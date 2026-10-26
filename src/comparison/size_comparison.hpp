#pragma once

#include "ephemeris/system_manager.hpp"
#include <map>
#include <span>

namespace sf::comparison {

inline constexpr double SIZE_SCALE = 1e-5;
inline const glm::dvec3 DISPLAY_POLE{-0.48989795, 0.84852814, 0.2};

struct SizeBody {
    std::shared_ptr<const ephemeris::SystemDefinition> system;
    size_t index = 0;
    int parent = -1;
    glm::dvec2 position{};
    double radius = 0, outer = 0, spin = 0;
    std::shared_ptr<const physics::PreparedAtmosphere> atmosphere;
    const ephemeris::BodyDefinition& definition() const { return system->bodies[index]; }
};

glm::dvec4 arrange_bodies(std::span<SizeBody> bodies);

struct SizeView {
    glm::dvec2 center{};
    double height = 20;
    bool fit = true;
    glm::dvec2 screen_to_world(glm::dvec2 point, glm::dvec2 viewport) const;
    glm::dvec2 world_to_screen(glm::dvec2 point, glm::dvec2 viewport) const;
    void pan(glm::dvec2 delta, double viewport_height);
    void zoom(double steps, glm::dvec2 cursor, glm::dvec2 viewport);
    void frame(glm::dvec4 bounds, glm::dvec4 available, glm::dvec2 viewport);
    void focus(const SizeBody& body, glm::dvec4 available, glm::dvec2 viewport);
};

class SizeComparison {
public:
    std::vector<std::string> selected_systems;
    std::vector<SizeBody> bodies;
    std::map<std::string, std::string> errors;
    glm::dvec4 bounds{-1, -1, 1, 1};
    std::string selected_system, selected_body;
    uint64_t revision = 0;

    bool synchronize(const ephemeris::SystemManager& manager, const ephemeris::SystemSnapshot& live);
    void refresh();
    int selection() const;
    int pick(glm::dvec2 point, double minimum_radius) const;
    void select(int index);

private:
    struct CachedSystem {
        std::filesystem::file_time_type stamp{};
        std::shared_ptr<const ephemeris::SystemDefinition> definition;
        std::vector<std::shared_ptr<const physics::PreparedAtmosphere>> atmospheres;
    };
    std::map<std::string, CachedSystem> cache_;
    std::vector<std::shared_ptr<const ephemeris::SystemDefinition>> definitions_;
};

}
