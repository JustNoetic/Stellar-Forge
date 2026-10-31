#pragma once

#include "benchmark/audit_fixture.hpp"
#include "terrain/terrain_source.hpp"
#include <nlohmann/json.hpp>
#include <limits>
#include <vector>

namespace sf::benchmark {

inline bool terrain_navigation_scene(std::string_view scene) {
    return scene == "earth-terrain-still" || scene == "earth-terrain-walk" ||
        scene == "earth-terrain-fly" || scene == "earth-terrain-turn";
}

struct TerrainNavigation {
    struct Pose {
        glm::dvec3 offset;
        glm::dvec3 forward;
        glm::dvec3 up;
        void apply(Camera& camera) const {
            camera.cam_pos_rel = offset;
            audit_look(camera, forward, up);
        }
    };
    std::vector<Pose> poses;
    nlohmann::json fixture;
};

inline TerrainNavigation terrain_navigation(const ephemeris::SystemSnapshot& system, const terrain::HeightSource& source,
                                            const rendering::RenderSettings& settings, std::string_view scene,
                                            uint32_t frames, double step_seconds) {
    int earth = system.find_body("Earth");
    if (earth < 0 || system.star_index() < 0) throw std::runtime_error("Terrain navigation requires Earth and Sun.");
    const auto& body = system.body(earth);
    const auto& info = source.info();
    if (!source.compatible(body.opaque_radius_au()*AU_TO_KM*1000, 1-body.definition.oblateness))
        throw std::runtime_error("Terrain navigation requires a compatible Earth elevation source.");
    auto rotation = terrain::body_rotation(body.render.pole, body.render.spin_angle_rad);
    auto sun = glm::transpose(rotation)*glm::normalize(system.body(system.star_index()).render.position.value-body.render.position.value);
    auto scale = glm::dvec3(1,info.polar_ratio,1);
    const terrain::HeightRecord* peak = nullptr;
    for (const auto& record : source.records()) {
        if (record.tile.level != info.data_level) continue;
        double width = std::ldexp(2.,-int(record.tile.level));
        auto direction = glm::normalize(terrain::cube_vector(record.tile.face,-1+(record.tile.x+.5)*width,-1+(record.tile.y+.5)*width));
        if (glm::dot(glm::normalize(direction/scale),sun) < .3) continue;
        if (!peak || record.maximum_m > peak->maximum_m) peak = &record;
    }
    if (!peak) throw std::runtime_error("No daylight elevation tile for terrain navigation.");
    auto payload = source.read(peak->tile);
    double width = std::ldexp(2.,-int(peak->tile.level));
    double maximum = -std::numeric_limits<double>::infinity();
    glm::dvec3 start{0};
    for (uint32_t y=0; y<=payload.cells; ++y) for (uint32_t x=0; x<=payload.cells; ++x) {
        auto direction = glm::normalize(terrain::cube_vector(peak->tile.face,
            -1+(peak->tile.x+double(x)/payload.cells)*width,-1+(peak->tile.y+double(y)/payload.cells)*width));
        if (glm::dot(glm::normalize(direction/scale),sun) < .25) continue;
        double height = terrain::height_sample(payload,double(x)*terrain::GRID_CELLS/payload.cells,double(y)*terrain::GRID_CELLS/payload.cells);
        if (height > maximum) { maximum = height; start = direction; }
    }
    if (!std::isfinite(maximum)) throw std::runtime_error("No daylight terrain sample for navigation.");
    auto normal = glm::normalize(start/scale);
    auto tangent = sun-normal*glm::dot(sun,normal);
    if (glm::length(tangent) < 1e-6) tangent = glm::cross(normal,std::abs(normal.y)<.9 ? glm::dvec3(0,1,0) : glm::dvec3(1,0,0));
    tangent = glm::normalize(tangent);
    auto path = glm::normalize(tangent-start*glm::dot(start,tangent));
    double speed = scene == "earth-terrain-walk" ? 5 : scene == "earth-terrain-fly" ? 250 : 0;
    double clearance = scene == "earth-terrain-fly" ? 2000 : 1.7;
    double floor = settings.oceans_enabled && body.definition.ocean.enabled
        ? body.definition.ocean.sea_level_km*1000 : -std::numeric_limits<double>::infinity();
    TerrainNavigation result;
    result.poses.reserve(frames);
    for (uint32_t frame=0; frame<frames; ++frame) {
        double seconds = frame*step_seconds;
        double angle = speed*seconds/info.equatorial_radius_m;
        auto direction = start*std::cos(angle)+path*std::sin(angle);
        auto tile = terrain::tile_at(direction,info.data_level);
        auto data = tile == peak->tile ? payload : source.read(tile);
        auto axis = terrain::cube_vector(tile.face,0,0);
        double divisor = glm::dot(direction,axis);
        double u = glm::dot(direction,terrain::cube_vector(tile.face,1,0)-axis)/divisor;
        double v = glm::dot(direction,terrain::cube_vector(tile.face,0,1)-axis)/divisor;
        double n = double(1u << tile.level);
        double height = std::max(floor,terrain::height_sample(data,((u+1)*.5*n-tile.x)*terrain::GRID_CELLS,
            ((v+1)*.5*n-tile.y)*terrain::GRID_CELLS));
        auto up = glm::normalize(direction/scale);
        auto heading = glm::normalize(path-up*glm::dot(path,up));
        if (scene == "earth-terrain-turn") {
            double turn = 15*DEG_TO_RAD*seconds;
            heading = heading*std::cos(turn)+glm::cross(up,heading)*std::sin(turn);
        }
        auto position = direction*scale+up*((height+clearance)/info.equatorial_radius_m);
        auto forward = heading*std::cos(3*DEG_TO_RAD)+up*std::sin(3*DEG_TO_RAD);
        result.poses.push_back({rotation*position*(info.equatorial_radius_m*.001*KM_TO_AU),rotation*forward,rotation*up});
    }
    result.fixture = {{"version",1},{"height_identity",info.domain.identity},{"height_revision",info.domain.revision},
        {"start_direction_local",{start.x,start.y,start.z}},{"start_height_m",maximum},{"clearance_m",clearance},
        {"speed_m_s",speed},{"turn_deg_s",scene=="earth-terrain-turn"?15:0},{"pitch_above_horizon_deg",3},
        {"distance_m",speed*(frames-1)*step_seconds},{"camera_path","precomputed against decoded triangle heights; warmup holds start"}};
    return result;
}

}
