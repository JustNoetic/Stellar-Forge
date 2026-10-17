#include "rendering/eclipse_candidates.hpp"
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

bool fragment_shadowed(glm::vec3 point, glm::vec4 star, glm::vec4 caster) {
    glm::vec3 to_star = glm::vec3(star) - point;
    float distance = glm::length(to_star);
    if (distance <= 1e-9f) return false;
    glm::vec3 direction = to_star / distance;
    glm::vec3 offset = glm::vec3(caster) - point;
    float projection = glm::dot(offset, direction);
    if (projection <= 0.0f || projection >= distance) return false;
    float perpendicular = glm::length(offset - direction * projection);
    return perpendicular < caster.w + star.w * (projection / distance);
}

}

int main() {
    try {
        using sf::rendering::eclipse_could_shadow;
        glm::vec4 receiver{0, 0, 0, 1}, star{0, 0, 100, 5};
        check(eclipse_could_shadow(receiver, star, {0, 0, 10, 0.25f}), "Aligned blocker retained");
        check(eclipse_could_shadow(receiver, star, {1.2f, 0, 10, 0.25f}), "Limb penumbra retained");
        check(!eclipse_could_shadow(receiver, star, {30, 0, 10, 0.25f}), "Lateral blocker rejected");
        check(!eclipse_could_shadow(receiver, star, {0, 0, 120, 0.25f}), "Blocker beyond star rejected");
        check(!eclipse_could_shadow(receiver, star, {0, 0, -20, 0.25f}), "Blocker behind receiver rejected");
        check(eclipse_could_shadow(receiver, {0, 0, 0, 5}, {0, 0, 10, 1}), "Overlapping light conservatively retained");
        std::mt19937_64 random(0xEC11F5E);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        auto direction = [&]() {
            double z = 2.0 * unit(random) - 1.0;
            double angle = 6.283185307179586 * unit(random);
            double r = std::sqrt(1.0 - z * z);
            return glm::dvec3(r * std::cos(angle), r * std::sin(angle), z);
        };
        size_t shadowed = 0, rejected = 0;
        for (size_t trial = 0; trial < 30000; ++trial) {
            double scale = std::pow(10.0, -6.0 + 9.0 * unit(random));
            double distance = scale * std::pow(10.0, -0.3 + 5.0 * unit(random));
            auto axis = direction();
            auto center = direction() * distance * (trial % 3 == 0 ? 100.0 : 0.1);
            glm::vec4 target{glm::vec3(center), float(scale)};
            glm::vec4 light{glm::vec3(center + axis * distance), float(distance * unit(random) * 0.1)};
            double along = -0.25 + 1.5 * unit(random);
            double caster_radius = scale * std::pow(10.0, -2.0 + 3.0 * unit(random));
            auto position = center + axis * distance * along + direction() * scale * 6.0 * unit(random);
            glm::vec4 blocker{glm::vec3(position), float(caster_radius)};
            bool candidate = eclipse_could_shadow(target, light, blocker);
            rejected += !candidate;
            for (size_t sample = 0; sample < 64; ++sample) {
                auto point = glm::vec3(center + direction() * scale * (sample % 2 ? 1.0 : unit(random)));
                if (fragment_shadowed(point, light, blocker)) {
                    ++shadowed;
                    check(candidate, "Candidate filter lost a shadowed surface/interior sample");
                }
            }
        }
        check(shadowed > 1000 && rejected > 1000, "Randomized oracle covers shadow retention and rejection");
        std::cout << "Eclipse bounds passed: " << shadowed << " shadowed samples retained; " << rejected << " configurations rejected.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
