#include "comparison/size_comparison.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sf::comparison {

glm::dvec4 arrange_bodies(std::span<SizeBody> bodies) {
    const int count = int(bodies.size());
    std::vector<std::vector<int>> children(count);
    std::vector<int> roots;
    auto star = [&](int i) { return bodies[i].definition().is_star || bodies[i].definition().is_black_hole; };
    for (int i = 0; i < count; ++i) {
        auto& b = bodies[i];
        b.radius = std::max(0.0, b.definition().radius_au()*AU_TO_KM)*SIZE_SCALE;
        b.outer = b.radius;
        for (const auto& ring : b.definition().rings) b.outer = std::max(b.outer, ring.outer*b.radius);
        if (star(i)) roots.push_back(i);
        if (b.parent >= 0 && b.parent < count && b.parent != i && !star(i)) children[b.parent].push_back(i);
    }
    for (int i = 0; i < count; ++i)
        if (!star(i) && (bodies[i].parent < 0 || bodies[i].parent >= count || bodies[i].parent == i)) roots.push_back(i);
    for (auto& items : children) std::stable_sort(items.begin(), items.end(), [&](int a, int b) {
        return bodies[a].definition().semi_major_au < bodies[b].definition().semi_major_au;
    });
    for (int i = 0; i < count; ++i) roots.push_back(i);
    std::vector<bool> visited(count);
    double bottom = 0;
    bool first = true;
    auto extent = [&](int i) { return std::max(bodies[i].radius, bodies[i].outer*.52); };
    for (int root : roots) {
        if (visited[root]) continue;
        std::vector<int> row{root};
        row.insert(row.end(), children[root].begin(), children[root].end());
        double top = 0;
        for (int i : row) top = std::max(top, extent(i));
        double y = first ? 0 : bottom-2-top, x = 0, row_bottom = y;
        first = false;
        for (int i : row) {
            if (visited[i]) continue;
            auto& b = bodies[i];
            double spacing = b.radius+(b.outer-b.radius)*.25;
            x += .25+spacing;
            b.position = {x, y};
            visited[i] = true;
            double moon_y = y-extent(i)-.15;
            std::vector<int> pending;
            if (i != root) pending.assign(children[i].rbegin(), children[i].rend());
            while (!pending.empty()) {
                int moon = pending.back();
                pending.pop_back();
                if (visited[moon]) continue;
                moon_y -= extent(moon);
                bodies[moon].position = {x, moon_y};
                visited[moon] = true;
                moon_y -= extent(moon)+.125;
                pending.insert(pending.end(), children[moon].rbegin(), children[moon].rend());
            }
            row_bottom = std::min({row_bottom, y-extent(i), moon_y});
            x += spacing+.25;
        }
        bottom = row_bottom;
    }
    if (bodies.empty()) return {-1, -1, 1, 1};
    glm::dvec2 low(std::numeric_limits<double>::max()), high(-std::numeric_limits<double>::max());
    for (int i = 0; i < count; ++i) {
        glm::dvec2 e{bodies[i].outer, extent(i)};
        low = glm::min(low, bodies[i].position-e);
        high = glm::max(high, bodies[i].position+e);
    }
    return {low.x, low.y, high.x, high.y};
}

glm::dvec2 SizeView::screen_to_world(glm::dvec2 point, glm::dvec2 viewport) const {
    return center+(point-viewport*.5)*glm::dvec2(1, -1)*(height/std::max(viewport.y, 1.0));
}

glm::dvec2 SizeView::world_to_screen(glm::dvec2 point, glm::dvec2 viewport) const {
    return (point-center)*glm::dvec2(1, -1)*(std::max(viewport.y, 1.0)/height)+viewport*.5;
}

void SizeView::pan(glm::dvec2 delta, double viewport_height) {
    center -= delta*glm::dvec2(1, -1)*(height/std::max(viewport_height, 1.0));
}

void SizeView::zoom(double steps, glm::dvec2 cursor, glm::dvec2 viewport) {
    auto before = screen_to_world(cursor, viewport);
    height = std::clamp(height*std::exp(-std::clamp(steps, -100.0, 100.0)*.15), 1e-8, 1e12);
    center += before-screen_to_world(cursor, viewport);
    fit = false;
}

void SizeView::frame(glm::dvec4 bounds, glm::dvec4 available, glm::dvec2 viewport) {
    double w = std::max(viewport.x*.3, available.z-available.x);
    double h = std::max(viewport.y*.4, available.w-available.y);
    height = std::max({bounds.w-bounds.y, (bounds.z-bounds.x)*h/w, .01})*viewport.y/h*1.15;
    auto midpoint = glm::dvec2(bounds.x+bounds.z, bounds.y+bounds.w)*.5;
    center = midpoint-(glm::dvec2(available.x+available.z, available.y+available.w)-viewport)*.5*
        glm::dvec2(1, -1)*(height/viewport.y);
    fit = false;
}

void SizeView::focus(const SizeBody& body, glm::dvec4 available, glm::dvec2 viewport) {
    height = std::max(body.radius*5, 1e-6)*viewport.y/std::max(viewport.y*.4, available.w-available.y);
    center = body.position-(glm::dvec2(available.x+available.z, available.y+available.w)-viewport)*.5*
        glm::dvec2(1, -1)*(height/viewport.y);
    fit = false;
}

bool SizeComparison::synchronize(const ephemeris::SystemManager& manager, const ephemeris::SystemSnapshot& live) {
    std::vector<std::shared_ptr<const ephemeris::SystemDefinition>> definitions;
    std::vector<SizeBody> next;
    errors.clear();
    std::vector<std::string> unique;
    for (const auto& name : selected_systems) {
        if (std::find(unique.begin(), unique.end(), name) != unique.end()) continue;
        unique.push_back(name);
        try {
            auto& entry = cache_[name];
            bool active = name == live.name();
            std::shared_ptr<const ephemeris::SystemDefinition> definition;
            if (active) definition = live.share_definition();
            else {
                if (!manager.system_exists(name)) throw std::runtime_error("System is no longer available.");
                auto path = paths::external_path("data", "systems", name, "system.json");
                std::error_code ec;
                auto stamp = std::filesystem::last_write_time(path, ec);
                if (!entry.definition || entry.stamp != stamp) {
                    auto loaded = manager.load_system(name);
                    auto candidate = std::make_shared<const ephemeris::SystemDefinition>(std::move(loaded.definition));
                    std::vector<std::shared_ptr<const physics::PreparedAtmosphere>> atmospheres;
                    for (const auto& b : candidate->bodies)
                        atmospheres.push_back(b.atmosphere.enabled && b.atmosphere.physical ?
                            physics::prepare_atmosphere(b.atmosphere, b.mass_msun, b.radius_au()*AU_TO_KM) : nullptr);
                    entry.definition = std::move(candidate);
                    entry.atmospheres = std::move(atmospheres);
                    entry.stamp = stamp;
                }
                definition = entry.definition;
            }
            definitions.push_back(definition);
            int offset = int(next.size());
            for (size_t i = 0; i < definition->bodies.size(); ++i) {
                const auto& b = definition->bodies[i];
                SizeBody body;
                body.system = definition;
                body.index = i;
                body.parent = b.parent_idx < 0 ? -1 : b.parent_idx+offset;
                body.spin = active ? live.body(i).render.spin_angle_rad : b.w0_deg*DEG_TO_RAD;
                body.atmosphere = active ? live.body(i).render.atmosphere : entry.atmospheres[i];
                next.push_back(std::move(body));
            }
        } catch (const std::exception& error) { errors[name] = error.what(); }
    }
    selected_systems = std::move(unique);
    bool changed = definitions != definitions_;
    if (changed) {
        bodies = std::move(next);
        definitions_ = std::move(definitions);
        bounds = arrange_bodies(bodies);
        ++revision;
    } else {
        for (size_t i = 0; i < bodies.size(); ++i) {
            bodies[i].spin = next[i].spin;
            bodies[i].atmosphere = std::move(next[i].atmosphere);
        }
    }
    std::erase_if(cache_, [&](const auto& item) {
        return std::find(selected_systems.begin(), selected_systems.end(), item.first) == selected_systems.end();
    });
    if (selection() < 0) { selected_system.clear(); selected_body.clear(); }
    return changed;
}

void SizeComparison::refresh() { cache_.clear(); }

int SizeComparison::selection() const {
    for (size_t i = 0; i < bodies.size(); ++i)
        if (bodies[i].system->name == selected_system && bodies[i].definition().name == selected_body) return int(i);
    return -1;
}

int SizeComparison::pick(glm::dvec2 point, double minimum_radius) const {
    int result = -1;
    double nearest = std::numeric_limits<double>::max();
    for (size_t i = 0; i < bodies.size(); ++i) {
        double distance = glm::length(bodies[i].position-point);
        if (distance <= std::max(bodies[i].radius, minimum_radius) && distance < nearest) {
            nearest = distance;
            result = int(i);
        }
    }
    return result;
}

void SizeComparison::select(int index) {
    if (index < 0 || size_t(index) >= bodies.size()) { selected_system.clear(); selected_body.clear(); return; }
    selected_system = bodies[index].system->name;
    selected_body = bodies[index].definition().name;
}

}
