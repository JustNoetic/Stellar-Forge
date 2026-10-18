#include "ephemeris/simulation.hpp"
#include "core/math_utils.hpp"
#include "core/path_utils.hpp"
#include "core/time_utils.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/ring_optics.hpp"
#include "rendering/ring_generator.hpp"
#include "ui/inspector_state.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace sf;
using namespace sf::ephemeris;

static_assert(sizeof(units::Seconds) == sizeof(double));
static_assert(sizeof(RenderPosition) == sizeof(glm::dvec3));
static_assert(!std::is_convertible_v<units::Seconds, units::Years>);
static_assert(!std::is_assignable_v<RenderPosition&, EclipticPosition>);
static_assert(!std::is_assignable_v<EclipticVelocity&, EclipticPosition>);
static_assert(std::is_same_v<decltype(std::declval<const SystemSnapshot&>().body(0).definition), const BodyDefinition&>);

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

nlohmann::json fixture() {
    return nlohmann::json::parse(R"([
        {"name":"Planet","parentId":"Sun","m":0.000003,"r":0.00915,"a":1.0,"e":0.1,"inc":0.0,"rotation_period":24.0,"custom":{"preserve":true}},
        {"name":"Sun","type":"Star","m":1.0,"r":1.0}
    ])");
}

void units_and_coordinates() {
    check(units::to_years(units::Seconds{SECONDS_PER_YEAR}).value == 1.0, "Seconds/year conversion");
    check(std::abs(units::to_km(units::to_au(units::Kilometres{1000.0})).value - 1000.0) < 1e-10, "Distance round trip");
    EclipticPosition p{glm::dvec3(1.0, 2.0, 3.0)};
    auto render = ecliptic_to_render(p);
    check(render.value == glm::dvec3(1.0, 3.0, -2.0), "Coordinate convention");
    check(render_to_ecliptic(render).value == p.value, "Coordinate round trip");
    RenderPosition origin{glm::dvec3(1e8, 0.0, 0.0)};
    RenderPosition nearby{glm::dvec3(1e8 + 1e-4, 0.0, 0.0)};
    check(camera_relative(nearby, origin).x > 0.0, "Camera subtraction precedes float conversion");
}

void clock_and_snapshots() {
    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "Test")).accepted, "Load fixture");
    auto captured = sim.snapshot();
    int planet = captured.find_body("Planet");
    auto original = captured.body(planet).motion.position.value;
    auto revision = captured.state_revision();
    check(sim.apply(SetPaused{true}).accepted, "Pause");
    check(!sim.advance(units::Seconds{86400.0}).changed, "Paused ticks are inert");
    check(sim.snapshot().state_revision() == revision, "Paused ticks preserve revision");
    sim.apply(SetPaused{false});
    check(sim.apply(SetTimeWarp{2.0}).accepted, "Set warp");
    check(sim.advance(units::Seconds{86400.0}).accepted, "Advance fixture");
    check(std::abs(sim.clock().time.value - 172800.0 / SECONDS_PER_YEAR) < 1e-16, "Typed clock advances once");
    check(sim.snapshot().body(planet).motion.position.value != original, "Propagation changes state");
    check(captured.body(planet).motion.position.value == original && captured.time().value == 0.0, "Captured snapshots stay stable");
    check(&captured.definition() == &sim.snapshot().definition(), "Ticks share immutable metadata");
    check(!sim.apply(SetTimeWarp{std::numeric_limits<double>::infinity()}).accepted, "Reject invalid warp");
    check(sim.clock().warp == 2.0, "Rejected warp preserves clock");
    check(sim.apply(SetTimeWarp{0.1}).accepted, "Accept slow-motion warp");
    check(sim.clock().warp == 0.1, "Warp updated to slow-motion");
    check(sim.apply(SetTimeWarp{1e15}).accepted, "Accept high warp");
    check(sim.clock().warp == 1e15, "Warp updated to high warp");
    check(!sim.apply(SetTimeWarp{-1.0}).accepted, "Reject negative warp");
    check(!sim.apply(SetTimeWarp{0.0}).accepted, "Reject zero warp");
    sim.apply(SetTimeWarp{2.0});
    check(!sim.advance(units::Seconds{-1.0}).accepted, "Reject negative frame time");
    check(sim.apply(SetTimeDirection{-1.0}).accepted, "Set reverse direction");
    check(sim.clock().direction == -1.0, "Direction set to reverse");
    check(sim.clock().effective_warp() == -2.0, "Effective warp reflects reverse direction");
    check(!sim.apply(SetTimeDirection{0.0}).accepted, "Reject zero direction");
    check(!sim.apply(SetTimeDirection{2.0}).accepted, "Reject direction != 1 or -1");
    check(sim.advance(units::Seconds{86400.0}).accepted, "Rewind step");
    check(std::abs(sim.clock().time.value) < 1e-12, "Rewind returned clock to starting time");
    check(glm::length(sim.snapshot().body(planet).motion.position.value - original) < 1e-10, "Rewind restores orbital position");
    check(sim.apply(ReverseTimeDirection{}).accepted, "Toggle time direction");
    check(sim.clock().direction == 1.0, "Direction toggled to forward");
    auto body = sim.snapshot().body(planet);
    check(body.render.position.value == ecliptic_to_render(body.motion.position).value, "Derived coordinates stay synchronized");
}

void edits_and_invalidation() {
    Simulation sim;
    sim.replace_system(SystemManager::parse_system(fixture(), "Test"));
    int index = sim.snapshot().find_body("Planet");
    ui::BodyEditDraft draft;
    draft.reload(sim.snapshot(), index);
    auto captured = sim.snapshot();
    auto before = captured.body(index).revisions;
    draft.values.radius = units::Kilometres{10000.0};
    draft.values.mass = units::SolarMasses{-1.0};
    draft.dirty = true;
    check(!sim.apply(draft.command()).accepted, "Reject entire invalid draft");
    check(sim.snapshot().definition_revision() == captured.definition_revision(), "Invalid edit preserves all revisions");
    check(&captured.definition() == &sim.snapshot().definition(), "Invalid edit preserves definitions");
    draft.values.mass = units::SolarMasses{0.000003};
    check(sim.apply(draft.command()).changed, "Commit valid draft");
    auto body = sim.snapshot().body(index);
    check(std::abs(body.definition.radius_au() * AU_TO_KM - 10000.0) < 1e-9, "Radius recalculated immediately");
    check(body.revisions.geometry == before.geometry + 1 && body.revisions.lighting == before.lighting + 1, "Radius invalidates geometry and lighting");
    check(body.revisions.dynamics == before.dynamics, "Spherical radius preserves dynamics cache key");
    check(captured.body(index).definition.radius_rsun != body.definition.radius_rsun, "Old snapshot retains old definition");
    check(!sim.apply(draft.command()).accepted, "Reject stale draft");
    draft.synchronize(sim.snapshot(), index);
    check(draft.body_revision == before.definition, "Dirty draft survives concurrent edit for explicit conflict resolution");
    draft.reload(sim.snapshot(), index);
    check(!sim.apply(draft.command()).changed, "Applying identical values is inert");
    draft.values.mass = units::SolarMasses{0.000006};
    auto geometry = body.revisions.geometry;
    check(sim.apply(draft.command()).changed, "Commit mass");
    check(sim.snapshot().body(index).revisions.dynamics == before.dynamics + 1, "Mass invalidates dynamics");
    check(sim.snapshot().body(index).revisions.geometry == geometry, "Mass preserves geometry");
    auto stale_system_command = draft.command();
    sim.replace_system(SystemManager::parse_system(fixture(), "Other"));
    check(!sim.apply(stale_system_command).accepted, "Reject draft from previous system");
    draft.synchronize(sim.snapshot(), index);
    check(!draft.dirty && draft.system_generation == sim.snapshot().generation(), "System switch resets draft");
}

void serialization_and_validation() {
    Simulation sim;
    sim.replace_system(SystemManager::parse_system(fixture(), "Test"));
    int index = sim.snapshot().find_body("Planet");
    ui::BodyEditDraft draft;
    draft.reload(sim.snapshot(), index);
    draft.values.radius = units::Kilometres{12345.0};
    sim.apply(draft.command());
    auto serialized = SystemManager::serialize_definition(sim.snapshot().definition());
    auto loaded = SystemManager::parse_system(serialized, "Reloaded");
    check(std::abs(loaded.definition.bodies[index].radius_rsun * SOLAR_RADIUS_KM - 12345.0) < 1e-9, "Persistence uses committed definitions");
    check(serialized[index]["custom"]["preserve"].get<bool>(), "Unknown JSON fields survive");
    for (auto invalid : {
        R"([{"name":"A","parentId":"B"},{"name":"B","parentId":"A"}])",
        R"([{"name":"A","parentId":"Missing"}])",
        R"([{"name":"A"},{"name":"A"}])"
    }) {
        bool rejected = false;
        try { SystemManager::parse_system(nlohmann::json::parse(invalid), "Invalid"); }
        catch (const std::exception&) { rejected = true; }
        check(rejected, "Reject invalid hierarchy");
    }
    auto bad = SystemManager::parse_system(fixture(), "Invalid");
    bad.definition.bodies[0].mass_msun = -1.0;
    auto generation = sim.snapshot().generation();
    check(!sim.replace_system(std::move(bad)).accepted, "Reject invalid replacement");
    check(sim.snapshot().generation() == generation, "Invalid replacement preserves active system");
}

void render_settings() {
    rendering::RenderSettingsStore store;
    auto revision = store.revision();
    auto draft = store.values();
    draft.exposure = 2.0f;
    draft.star_intensity = std::numeric_limits<float>::quiet_NaN();
    check(!store.apply(draft).accepted, "Reject entire invalid render edit");
    check(store.values().exposure == 1.0f && store.revision() == revision, "Render edit rollback");
    draft.star_intensity = 3.0f;
    check(store.apply(draft).changed && store.revision() == revision + 1, "Commit render settings once");
    check(!store.apply(draft).changed && store.revision() == revision + 1, "Unchanged settings do not invalidate");
    draft.vsync = true;
    check(store.apply(draft).changed && store.revision() == revision + 2, "Commit vsync setting update");
    check(store.values().vsync, "Store reflects vsync state");
}

void ias15_contracts() {
    // 1. One full orbit energy and closure test
    Simulation sim;
    auto sys_json = nlohmann::json::parse(R"([
        {"name":"Planet","parentId":"Sun","m":0.000003,"r":0.00915,"a":1.0,"e":0.0,"inc":0.0,"rotation_period":24.0},
        {"name":"Sun","type":"Star","m":1.0,"r":1.0}
    ])");
    check(sim.replace_system(SystemManager::parse_system(sys_json, "KeplerTest")).accepted, "Load circular test");
    check(sim.apply(SetMotionModel{MotionModel::NBody}).changed, "Switch to NBody");
    check(sim.snapshot().definition().motion_model == MotionModel::NBody, "MotionModel is NBody");

    auto initial_pos = sim.snapshot().body(sim.snapshot().find_body("Planet")).motion.position.value;
    auto initial_vel = sim.snapshot().body(sim.snapshot().find_body("Planet")).motion.velocity.value;

    // Advance 1 year in N-body mode
    units::Seconds one_year{SECONDS_PER_YEAR};
    auto step_result = sim.advance(one_year);
    check(step_result.accepted, "Advance 1 year with IAS15");

    auto final_pos = sim.snapshot().body(sim.snapshot().find_body("Planet")).motion.position.value;
    auto final_vel = sim.snapshot().body(sim.snapshot().find_body("Planet")).motion.velocity.value;

    double pos_err = glm::length(final_pos - initial_pos);
    double vel_err = glm::length(final_vel - initial_vel);
    double r_init = glm::length(initial_pos);
    double r_final = glm::length(final_pos);
    double r_err = std::abs(r_final - r_init);
    check(r_err < 1e-6, "IAS15 circular orbit radius conservation < 1e-6 AU");
    check(pos_err < 5e-4, "IAS15 1-year orbit closure position error < 5e-4 AU");
    check(vel_err < 5e-3, "IAS15 1-year orbit closure velocity error < 5e-3 AU/yr");

    // 2. Mode switching: NBody -> Keplerian -> NBody
    check(sim.apply(SetMotionModel{MotionModel::Keplerian}).changed, "Switch back to Keplerian");
    check(sim.snapshot().definition().motion_model == MotionModel::Keplerian, "MotionModel is Keplerian");
    check(sim.advance(units::Seconds{86400.0}).accepted, "Keplerian advances after NBody");

    check(sim.apply(SetMotionModel{MotionModel::NBody}).changed, "Switch back to NBody");
    check(sim.advance(units::Seconds{86400.0}).accepted, "NBody advances after Keplerian");
}

void kepler_perturbation_contracts() {
    {
        Simulation sim;
        double r_sun_val = 0.005 / SOLAR_RADII_TO_AU;
        nlohmann::json sys = nlohmann::json::array({
            {
                {"name", "Primary"},
                {"type", "Planet"},
                {"m", 1.0},
                {"r", r_sun_val},
                {"J2", 0.001}
            },
            {
                {"name", "Satellite"},
                {"parentId", "Primary"},
                {"m", 1e-6},
                {"r", 0.0001},
                {"a", 0.05},
                {"e", 0.0},
                {"inc", 45.0},
                {"Omega", 0.0},
                {"omega", 0.0},
                {"M", 0.0}
            }
        });
        check(sim.replace_system(SystemManager::parse_system(sys, "J2Test")).accepted, "Load J2 system");
        int sat_idx = sim.snapshot().find_body("Satellite");
        const auto& orb = sim.kepler().orbits()[sat_idx];

        double mu = G_CONST * (1.0 + 1e-6);
        double a = 0.05;
        double n_mean = std::sqrt(mu / (a * a * a));
        double req = 0.005;
        double j2_ratio = req / a;
        double j2_factor = 1.5 * 0.001 * (j2_ratio * j2_ratio) * n_mean;
        double cos_i = std::cos(45.0 * DEG_TO_RAD);
        double expected_dOmega = -j2_factor * cos_i;
        double expected_domega = 0.5 * j2_factor * (5.0 * cos_i * cos_i - 1.0);
        double gr_factor = 3.0 * mu * n_mean / (C_AU_YR * C_AU_YR * a);
        expected_domega += gr_factor;

        check(std::abs(orb.prec_vec.z - expected_dOmega) < 1e-12, "J2 nodal precession rate matches theory");
        check(std::abs(orb.prec_vec.x) < 1e-15 && std::abs(orb.prec_vec.y) < 1e-15, "J2 precession axis aligned with pole");
        check(std::abs(orb.domega_peri - expected_domega) < 1e-12, "J2 apsidal precession rate matches theory");

        glm::dvec3 initial_n = orb.n;
        units::Years advance_time{10.0};
        check(sim.advance(units::to_seconds(advance_time)).accepted, "Advance 10 years with J2 precession");

        int prim_idx = sim.snapshot().find_body("Primary");
        glm::dvec3 rel_pos = sim.snapshot().body(sat_idx).motion.position.value - sim.snapshot().body(prim_idx).motion.position.value;
        glm::dvec3 rel_vel = sim.snapshot().body(sat_idx).motion.velocity.value - sim.snapshot().body(prim_idx).motion.velocity.value;

        double r_len = glm::length(rel_pos);
        check(std::abs(r_len - a) < 1e-9, "J2 circular orbit radius conserved");
        double v_len = glm::length(rel_vel);
        check(std::abs(v_len - std::sqrt(mu / a)) < 1e-8, "J2 circular orbit speed conserved");

        glm::dvec3 current_n = glm::normalize(glm::cross(rel_pos, rel_vel));
        glm::dvec3 expected_n = math::axis_angle_rotation(initial_n, glm::dvec3{0.0, 0.0, 1.0}, expected_dOmega * advance_time.value);
        check(glm::length(current_n - expected_n) < 1e-9, "J2 orbit normal precesses around spin axis at predicted rate");
        check(std::abs(current_n.z - cos_i) < 1e-10, "J2 inclination to pole conserved");
    }

    {
        Simulation sim;
        nlohmann::json sys = nlohmann::json::array({
            {
                {"name", "Star"},
                {"type", "Star"},
                {"m", 1.0},
                {"r", 1.0}
            },
            {
                {"name", "Planet"},
                {"parentId", "Star"},
                {"m", 0.001},
                {"r", 0.01},
                {"a", 1.0},
                {"e", 0.0},
                {"inc", 0.0},
                {"Omega", 0.0},
                {"omega", 0.0},
                {"M", 0.0}
            },
            {
                {"name", "Moon"},
                {"parentId", "Planet"},
                {"m", 1e-7},
                {"r", 0.001},
                {"a", 0.01},
                {"e", 0.0},
                {"inc", 30.0},
                {"Omega", 0.0},
                {"omega", 0.0},
                {"M", 0.0}
            }
        });
        check(sim.replace_system(SystemManager::parse_system(sys, "ThirdBodyTest")).accepted, "Load 3-body system");
        int moon_idx = sim.snapshot().find_body("Moon");
        int planet_idx = sim.snapshot().find_body("Planet");
        const auto& orb = sim.kepler().orbits()[moon_idx];
        const auto& gp_orb = sim.kepler().orbits()[planet_idx];

        double n_3rd = std::sqrt(gp_orb.mu / (gp_orb.a * gp_orb.a * gp_orb.a));
        glm::dvec3 gp_pole = gp_orb.n;
        double mu_moon = orb.mu;
        double a_moon = orb.a;
        double n_moon = std::sqrt(mu_moon / (a_moon * a_moon * a_moon));
        double cos_i = glm::dot(orb.n, gp_pole);
        double e_fac = std::sqrt(std::max(0.0001, 1.0 - orb.e * orb.e));
        double tb_factor = 0.75 * (n_3rd * n_3rd / n_moon) / e_fac;
        double expected_dOmega = -tb_factor * cos_i;
        glm::dvec3 expected_prec = expected_dOmega * gp_pole;
        double expected_domega = tb_factor * (5.0 * cos_i * cos_i - 1.0);
        double gr_factor = 3.0 * mu_moon * n_moon / (C_AU_YR * C_AU_YR * a_moon * (1.0 - orb.e * orb.e));
        expected_domega += gr_factor;

        check(glm::length(orb.prec_vec - expected_prec) < 1e-12, "Third-body precession vector matches theory");
        check(std::abs(orb.domega_peri - expected_domega) < 1e-12, "Third-body apsidal precession rate matches theory");
        check(std::abs(gp_orb.a - 1.0) < 1e-4, "Planet semi-major axis close to 1.0 AU");
        check(std::abs(gp_pole.z - 1.0) < 1e-4, "Grandparent pole nearly perpendicular to ecliptic");

        glm::dvec3 initial_n = orb.n;
        units::Years advance_time{2.0};
        check(sim.advance(units::to_seconds(advance_time)).accepted, "Advance 2 years with 3rd body precession");

        glm::dvec3 rel_pos = sim.snapshot().body(moon_idx).motion.position.value - sim.snapshot().body(planet_idx).motion.position.value;
        glm::dvec3 rel_vel = sim.snapshot().body(moon_idx).motion.velocity.value - sim.snapshot().body(planet_idx).motion.velocity.value;

        double r_len = glm::length(rel_pos);
        check(std::abs(r_len - a_moon) < 1e-9, "Moon circular orbit radius conserved under third-body perturbation");

        glm::dvec3 current_n = glm::normalize(glm::cross(rel_pos, rel_vel));
        glm::dvec3 expected_n = math::axis_angle_rotation(initial_n, gp_pole, expected_dOmega * advance_time.value);
        check(glm::length(current_n - expected_n) < 1e-9, "Moon orbit normal precesses around grandparent orbit normal at predicted rate");
        check(std::abs(glm::dot(current_n, gp_pole) - cos_i) < 1e-10, "Moon inclination to grandparent normal conserved");
    }
}

void time_and_date_formatting() {
    using namespace sf::time;
    check(format_sim_date("2026-06-03", units::Years{0.0}) == "2026-06-03 00:00:00 UTC", "Initial epoch date");
    check(format_sim_date("2026-06-03 12:30:45", units::Years{0.0}) == "2026-06-03 12:30:45 UTC", "Full epoch date");
    check(format_sim_date("2026-06-03", units::to_years(units::Seconds{86400.0})) == "2026-06-04 00:00:00 UTC", "Forward 1 day");
    check(format_sim_date("2026-06-03", units::to_years(units::Seconds{-86400.0})) == "2026-06-02 00:00:00 UTC", "Rewind 1 day");
    check(format_sim_date("", units::Years{0.0}) == "2026-01-01 12:00:00 UTC", "Default fallback epoch");

    check(format_time_speed(1.0) == "Realtime", "Realtime speed");
    check(format_time_speed(-1.0) == "-Realtime", "Negative realtime speed");
    check(format_time_speed(0.1) == "0.10x", "Slow-motion speed");
    check(format_time_speed(-0.5) == "-0.50x", "Negative slow-motion speed");
    check(format_time_speed(0.0) == "Paused", "Paused speed");
    check(format_time_speed(10.0) == "10.0 sec/s", "Seconds speed");
    check(format_time_speed(-60.0) == "-1.0 min/s", "Negative minutes speed");
    check(format_time_speed(7200.0) == "2.0 hr/s", "Hours speed");
    check(format_time_speed(86400.0) == "1.0 days/s", "Days speed");
    check(format_time_speed(-604800.0) == "-1.0 weeks/s", "Weeks speed");
    check(format_time_speed(2629800.0) == "1.0 months/s", "Months speed");
    check(format_time_speed(31557600.0) == "1.0 years/s", "Years speed");

    // Year > 32767 overflow bug fix tests
    auto dt30k = calculate_sim_date("2026-01-01 12:00:00", units::Years{30000.0}, false);
    check(dt30k.year == 32026, "Year 30,000 exceeds 32,000");

    auto dt32768 = calculate_sim_date("2026-01-01 12:00:00", units::Years{31000.0}, false);
    check(dt32768.year == 33026, "Year past 32767 does NOT wrap to negative");

    auto dt50k = calculate_sim_date("2026-01-01 12:00:00", units::Years{50000.0}, false);
    check(dt50k.year == 52027, "Year 50,000 Julian years is positive 52,027 in Gregorian calendar");

    auto dt1m = calculate_sim_date("2026-01-01 12:00:00", units::Years{1000000.0}, false);
    check(dt1m.year > 1000000, "Year 1,000,000 is positive > 1,000,000");

    // Proleptic Gregorian round-trip
    DateTimeUTC test_dt{35000, 7, 15, 14, 30, 0, "UTC"};
    auto sim_t = sim_time_from_date("2026-01-01 12:00:00", test_dt, true);
    auto rt_dt = calculate_sim_date("2026-01-01 12:00:00", sim_t, false);
    check(rt_dt.year == test_dt.year && rt_dt.month == test_dt.month &&
          rt_dt.day == test_dt.day && rt_dt.hour == test_dt.hour &&
          rt_dt.minute == test_dt.minute && rt_dt.second == test_dt.second,
          "Year 35,000 date round-trip is exact");

    // Negative BCE years
    auto dt_bc = calculate_sim_date("2026-01-01 12:00:00", units::Years{-3000.0}, false);
    check(dt_bc.year == -975, "Negative simulation time yields correct BCE year (-975)");

    // Timezone tests
    auto tz_info = get_local_tz_info(1767268800);
    check(!tz_info.name.empty(), "Local timezone name is detected");
    auto local_dt = calculate_sim_date("2026-01-01 12:00:00", units::Years{0.0}, true);
    check(local_dt.tz == tz_info.name, "Local sim date displays detected timezone");
}

void jump_and_timeline_contracts() {
    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "Test")).accepted, "Load fixture");

    // Test JumpToTime in Keplerian mode
    check(sim.apply(SetMotionModel{MotionModel::Keplerian}).accepted, "Switch to Keplerian");
    auto res_jump = sim.apply(JumpToTime{units::Years{5.0}});
    check(res_jump.accepted, "Keplerian JumpToTime accepted");
    check(std::abs(sim.clock().time.value - 5.0) < 1e-9, "Clock time is 5.0 yr after jump");
    check(std::abs(sim.snapshot().time().value - 5.0) < 1e-9, "Snapshot time is 5.0 yr after jump");

    // Test instant jump rejected in N-Body
    check(sim.apply(SetMotionModel{MotionModel::NBody}).accepted, "Switch to NBody");
    auto res_jump_nb = sim.apply(JumpToTime{units::Years{10.0}});
    check(!res_jump_nb.accepted, "N-Body rejects instant jump");

    // Test SyncTimelineState
    std::vector<BodyMotion> motions(sim.snapshot().motions().begin(), sim.snapshot().motions().end());
    motions[1].position = EclipticPosition{motions[1].position.value + glm::dvec3(0.1, 0.0, 0.0)};
    check(sim.apply(SyncTimelineState{units::Years{7.5}, motions}).accepted, "SyncTimelineState accepted");
    check(std::abs(sim.clock().time.value - 7.5) < 1e-9, "Clock time synced to 7.5 yr");

    // Test SpkExporter with populated TimelineRecording
    TimelineRecording rec;
    rec.start_time = units::Years{0.0};
    rec.end_time = units::Years{1.0};
    rec.source_model = MotionModel::Keplerian;
    const size_t num_steps = 50;
    const size_t num_bodies = sim.snapshot().size();
    rec.times.resize(num_steps);
    rec.motions.resize(num_steps * num_bodies);
    for (size_t s = 0; s < num_steps; ++s) {
        rec.times[s] = units::Years{static_cast<double>(s + 1) / static_cast<double>(num_steps)};
        for (size_t b = 0; b < num_bodies; ++b) {
            rec.motions[s * num_bodies + b] = motions[b];
        }
    }

    std::filesystem::path temp_bsp = paths::external_path("scratch", "test_export.bsp");
    std::filesystem::create_directories(temp_bsp.parent_path());
    std::filesystem::remove(temp_bsp);
    std::filesystem::remove(temp_bsp.string() + ".json");

    std::string spice_err;
    auto spice = SpiceEphemeris::open(spice_err);
    if (spice && spice->can_write_spk()) {
        SpkExportOptions opts;
        opts.spk_type = 2;
        opts.cheb_degree = 7;
        opts.cheb_tol_km = 10.0;
        auto export_res = SpkExporter::export_timeline(temp_bsp, rec, sim.snapshot().definition(),
                                                       sim.snapshot().reference_hierarchy(), spice, opts);
        check(export_res.success, "SPK Type 2 export succeeded");
        check(std::filesystem::exists(temp_bsp), "Exported .bsp exists");
        check(std::filesystem::exists(temp_bsp.string() + ".json"), "Exported .bsp.json exists");

        auto inspect = spice->inspect_bsp(temp_bsp);
        check(inspect.valid, "inspect_bsp validates exported BSP");
        check(inspect.bodies.size() == num_bodies, "inspect_bsp found all bodies");
    }
}

void keplerian_and_hydrostatic_math() {
    auto earth_shape = math::compute_hydrostatic_shape(3.003e-6, 6371.0, 23.93447, "Terrestrial");
    check(earth_shape.oblateness > 0.003 && earth_shape.oblateness < 0.004, "Earth flattening roughly 1/300");
    check(earth_shape.j2 > 0.001 && earth_shape.j2 < 0.002, "Earth J2 roughly 0.001");
    check(earth_shape.j4 < 0.0, "Earth J4 is negative");

    double mu = G_CONST * 1.0;
    auto cart = math::get_cartesian_from_keplerian(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    check(std::abs(glm::length(cart.pos) - 1.0) < 1e-6, "Circular orbit pos length is 1 AU");
    auto elem = math::compute_keplerian_elements(cart.pos, cart.vel, mu);
    check(std::abs(elem.a - 1.0) < 1e-4, "Recovered semi-major axis 1 AU");
    check(elem.e < 1e-4, "Recovered eccentricity ~ 0");

    auto cart_ell = math::get_cartesian_from_keplerian(1.0, 0.0, 1.5, 0.2, 10.0, 30.0, 45.0, 60.0);
    auto elem_ell = math::compute_keplerian_elements(cart_ell.pos, cart_ell.vel, mu);
    check(std::abs(elem_ell.a - 1.5) < 1e-3, "Recovered semi-major axis 1.5 AU");
    check(std::abs(elem_ell.e - 0.2) < 1e-3, "Recovered eccentricity 0.2");
    check(std::abs(elem_ell.inc_deg - 10.0) < 1e-2, "Recovered inclination 10 deg");
}

void orbital_element_edits() {
    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "Test")).accepted, "Load fixture");
    int planet = sim.snapshot().find_body("Planet");
    auto rev_def_before = sim.snapshot().body(planet).revisions.definition;
    auto rev_dyn_before = sim.snapshot().body(planet).revisions.dynamics;

    auto draft = ui::BodyEditDraft{};
    draft.reload(sim.snapshot(), planet);
    draft.values.update_orbit = true;
    draft.values.a_au = 2.5;
    draft.values.e = 0.05;
    draft.values.inc_deg = 5.0;

    auto result = sim.apply(draft.command());
    check(result.accepted, "Apply orbital edit");
    check(sim.snapshot().body(planet).definition.semi_major_au == 2.5, "New semi-major axis updated");
    check(sim.snapshot().body(planet).definition.eccentricity == 0.05, "New eccentricity updated");
    check(sim.snapshot().body(planet).revisions.definition > rev_def_before, "Definition revision incremented");
    check(sim.snapshot().body(planet).revisions.dynamics > rev_dyn_before, "Dynamics revision incremented");
}

void bring_moons_contracts() {
    Simulation sim;
    auto sys_json = nlohmann::json::parse(R"([
        {"name":"Star","type":"Star","m":1.0,"r":1.0},
        {"name":"Planet","parentId":"Star","type":"Gas Giant","m":0.001,"r":0.05,"a":2.0,"e":0.0,"inc":0.0},
        {"name":"Moon","parentId":"Planet","type":"Moon","m":1e-6,"r":0.001,"a":0.01,"e":0.0,"inc":0.0},
        {"name":"EscapedChild","parentId":"Planet","type":"Moon","m":1e-8,"r":0.0001,"a":1.0,"e":0.0,"inc":0.0},
        {"name":"DistantAsteroid","parentId":"Star","type":"Terrestrial","m":1e-9,"r":0.0001,"a":5.0,"e":0.0,"inc":0.0}
    ])");
    check(sim.replace_system(SystemManager::parse_system(sys_json, "MoonSystemTest")).accepted, "Load moon system fixture");

    int planet_idx = sim.snapshot().find_body("Planet");
    int moon_idx = sim.snapshot().find_body("Moon");
    int escaped_idx = sim.snapshot().find_body("EscapedChild");
    int asteroid_idx = sim.snapshot().find_body("DistantAsteroid");

    auto p_pos_init = sim.snapshot().body(planet_idx).motion.position.value;
    auto m_pos_init = sim.snapshot().body(moon_idx).motion.position.value;
    auto e_pos_init = sim.snapshot().body(escaped_idx).motion.position.value;
    auto a_pos_init = sim.snapshot().body(asteroid_idx).motion.position.value;
    auto rel_pos_init = m_pos_init - p_pos_init;
    auto rel_vel_init = sim.snapshot().body(moon_idx).motion.velocity.value - sim.snapshot().body(planet_idx).motion.velocity.value;

    auto draft = ui::BodyEditDraft{};
    draft.reload(sim.snapshot(), planet_idx);
    check(draft.values.bring_moons, "bring_moons is enabled by default");
    draft.values.update_orbit = true;
    draft.values.a_au = 3.5;
    draft.values.mean_anom_deg = 45.0;

    auto res = sim.apply(draft.command());
    check(res.accepted, "Apply orbit edit with bring_moons");

    auto p_pos_new = sim.snapshot().body(planet_idx).motion.position.value;
    auto m_pos_new = sim.snapshot().body(moon_idx).motion.position.value;
    auto e_pos_new = sim.snapshot().body(escaped_idx).motion.position.value;
    auto a_pos_new = sim.snapshot().body(asteroid_idx).motion.position.value;

    check(glm::length(p_pos_new - p_pos_init) > 0.5, "Planet moved to new position");
    auto rel_pos_new = m_pos_new - p_pos_new;
    auto rel_vel_new = sim.snapshot().body(moon_idx).motion.velocity.value - sim.snapshot().body(planet_idx).motion.velocity.value;

    check(glm::length(rel_pos_new - rel_pos_init) < 1e-12, "Moon relative position to parent preserved");
    check(glm::length(rel_vel_new - rel_vel_init) < 1e-12, "Moon relative velocity to parent preserved");
    check(glm::length(e_pos_new - e_pos_init) < 1e-12, "Child outside Hill sphere not moved");
    check(glm::length(a_pos_new - a_pos_init) < 1e-12, "Body outside Hill sphere untouched");
    check(sim.snapshot().body(moon_idx).revisions.dynamics > 0, "Moon dynamics revision incremented");

    check(sim.advance(units::Seconds{86400.0}).accepted, "Keplerian advances with moved moons");
    double moon_dist_after = glm::length(sim.snapshot().body(moon_idx).motion.position.value - sim.snapshot().body(planet_idx).motion.position.value);
    check(std::abs(moon_dist_after - 0.01) < 1e-4, "Moon circular orbit around parent continues after move");

    draft.reload(sim.snapshot(), planet_idx);
    draft.values.bring_moons = false;
    draft.values.update_orbit = true;
    draft.values.a_au = 4.5;
    draft.values.mean_anom_deg = 90.0;

    auto m_pos_before_optout = sim.snapshot().body(moon_idx).motion.position.value;
    auto res_optout = sim.apply(draft.command());
    check(res_optout.accepted, "Apply orbit edit with bring_moons disabled");

    auto m_pos_after_optout = sim.snapshot().body(moon_idx).motion.position.value;
    check(glm::length(m_pos_after_optout - m_pos_before_optout) < 1e-12, "Moon position unchanged when bring_moons is false");
}

void add_and_delete_contracts() {
    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "AddDeleteTest")).accepted, "Load fixture");
    check(sim.snapshot().size() == 2, "Fixture starts with 2 bodies (Sun and Planet)");

    auto def_rev = sim.snapshot().definition_revision();
    auto state_rev = sim.snapshot().state_revision();

    // 1. Add Planet Two
    BodyDefinition p2{};
    p2.name = "Planet Two";
    p2.type = "Terrestrial";
    p2.parent_idx = sim.snapshot().find_body("Sun");
    p2.mass_msun = 0.000005;
    p2.radius_rsun = 0.01;
    p2.semi_major_au = 2.0;
    p2.eccentricity = 0.05;
    p2.inc_deg = 2.0;
    p2.omega_node_deg = 45.0;
    p2.arg_peri_deg = 30.0;
    p2.mean_anom_deg = 10.0;
    p2.rotation_period_hr = 20.0;

    AddBody add_cmd{sim.snapshot().generation(), p2, false, 5.0};
    auto add_res = sim.apply(add_cmd);
    check(add_res.accepted && add_res.changed, "Add Planet Two accepted");
    check(sim.snapshot().size() == 3, "Size increased to 3");
    int p2_idx = sim.snapshot().find_body("Planet Two");
    check(p2_idx == 2, "Planet Two placed at index 2");
    check(sim.snapshot().definition_revision() == def_rev + 1, "Definition revision incremented");
    check(sim.snapshot().state_revision() == state_rev + 1, "State revision incremented");
    check(sim.snapshot().body(p2_idx).definition.parent_id == "Sun", "Parent ID correctly recorded as Sun");

    // Advance in Keplerian mode
    check(sim.advance(units::Seconds{86400.0}).accepted, "Advance 1 day with added body");

    // 2. Add Moon to Planet Two
    BodyDefinition moon{};
    moon.name = "Moon Two";
    moon.type = "Moon";
    moon.parent_idx = p2_idx;
    moon.mass_msun = 1e-8;
    moon.radius_rsun = 0.002;
    moon.semi_major_au = 0.0025;
    moon.eccentricity = 0.01;
    moon.inc_deg = 1.0;
    moon.rotation_period_hr = 100.0;
    moon.tidally_locked = true;

    AddBody add_moon_cmd{sim.snapshot().generation(), moon, false, 0.0};
    check(sim.apply(add_moon_cmd).accepted, "Add Moon Two accepted");
    check(sim.snapshot().size() == 4, "Size increased to 4");
    int moon_idx = sim.snapshot().find_body("Moon Two");
    check(moon_idx == 3, "Moon placed at index 3");
    check(sim.snapshot().body(moon_idx).definition.parent_id == "Planet Two", "Moon parent recorded as Planet Two");

    // Advance further
    check(sim.advance(units::Seconds{86400.0 * 5.0}).accepted, "Advance 5 days with moon");

    // 3. Reject duplicate name
    BodyDefinition dup = moon;
    AddBody dup_cmd{sim.snapshot().generation(), dup};
    check(!sim.apply(dup_cmd).accepted, "Reject duplicate name");

    // 4. Reject invalid parameters
    BodyDefinition invalid_b = p2;
    invalid_b.name = "Bad Planet";
    invalid_b.radius_rsun = -1.0;
    check(!sim.apply(AddBody{sim.snapshot().generation(), invalid_b}).accepted, "Reject negative radius");
    invalid_b.radius_rsun = 0.01;
    invalid_b.eccentricity = 1.5;
    check(!sim.apply(AddBody{sim.snapshot().generation(), invalid_b}).accepted, "Reject hyperbolic eccentricity in AddBody");

    // 5. Delete Planet Two -> Moon Two must reparent to Sun!
    check(sim.snapshot().body(moon_idx).definition.parent_idx == p2_idx, "Moon parent is Planet Two before delete");
    DeleteBody del_cmd{sim.snapshot().generation(), p2_idx};
    auto del_res = sim.apply(del_cmd);
    check(del_res.accepted && del_res.changed, "Delete Planet Two accepted");
    check(sim.snapshot().size() == 3, "Size decreased back to 3");
    check(sim.snapshot().find_body("Planet Two") == -1, "Planet Two no longer found");
    int new_moon_idx = sim.snapshot().find_body("Moon Two");
    check(new_moon_idx >= 0, "Moon Two still exists");
    check(sim.snapshot().body(new_moon_idx).definition.parent_id == "Sun", "Moon Two reparented to Sun");
    check(sim.snapshot().body(new_moon_idx).definition.parent_idx == sim.snapshot().find_body("Sun"), "Moon parent index updated to Sun");

    // Advance after deletion
    check(sim.advance(units::Seconds{86400.0 * 10.0}).accepted, "Advance 10 days after deletion");

    // 6. Test IAS15 N-Body mode with added body
    check(sim.apply(SetMotionModel{MotionModel::NBody}).accepted, "Switch to N-Body mode");
    check(sim.advance(units::Seconds{86400.0 * 10.0}).accepted, "Advance 10 days in N-Body mode");

    // Add another asteroid in N-Body mode
    BodyDefinition ast{};
    ast.name = "Asteroid";
    ast.parent_idx = sim.snapshot().find_body("Sun");
    ast.mass_msun = 1e-12;
    ast.radius_rsun = 0.0001;
    ast.semi_major_au = 2.5;
    ast.eccentricity = 0.1;
    check(sim.apply(AddBody{sim.snapshot().generation(), ast}).accepted, "Add Asteroid in N-Body mode");
    check(sim.snapshot().size() == 4, "Size is 4 in N-Body mode");
    check(sim.advance(units::Seconds{86400.0 * 10.0}).accepted, "Advance in N-Body mode after addition");

    // Delete Asteroid in N-Body mode
    int ast_idx = sim.snapshot().find_body("Asteroid");
    check(sim.apply(DeleteBody{sim.snapshot().generation(), ast_idx}).accepted, "Delete Asteroid in N-Body mode");
    check(sim.snapshot().size() == 3, "Size back to 3 in N-Body mode");
    check(sim.advance(units::Seconds{86400.0 * 10.0}).accepted, "Advance in N-Body mode after deletion");

    // 7. Reject deleting primary star
    int sun_idx = sim.snapshot().find_body("Sun");
    check(!sim.apply(DeleteBody{sim.snapshot().generation(), sun_idx}).accepted, "Reject deleting primary star");

    // 8. Reject invalid generation
    check(!sim.apply(DeleteBody{99999, new_moon_idx}).accepted, "Reject stale generation on DeleteBody");
    check(!sim.apply(AddBody{99999, ast}).accepted, "Reject stale generation on AddBody");
}

void ring_state_and_serialization_contracts() {
    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "RingTest")).accepted, "Load fixture for ring tests");
    int planet = sim.snapshot().find_body("Planet");
    int sun = sim.snapshot().find_body("Sun");
    check(planet >= 0 && sun >= 0, "Fixture bodies exist");
    check(sim.snapshot().body(planet).definition.rings.empty(), "Planet initially has no rings");

    RingLayerDefinition layer{};
    layer.inner = 1.35;
    layer.outer = 2.45;
    layer.color_rgb = {0.82f, 0.76f, 0.60f};
    layer.opacity = 0.85f;
    layer.scatter = 0.65f;
    layer.asymmetry = 0.78f;
    layer.backscatter = -0.42f;
    layer.unlit_factor = 1.1f;
    layer.is_textured = true;
    layer.hue_shift = 0.12f;
    layer.saturation = 1.25f;
    layer.brightness = 0.95f;
    layer.alpha_boost = 1.4f;
    layer.gradient = {{0.0f, 0.1f}, {0.5f, 0.9f}, {1.0f, 0.2f}};

    auto gen = sim.snapshot().generation();
    auto rev_before = sim.snapshot().body(planet).revisions;
    auto def_rev_before = sim.snapshot().definition_revision();

    // Reject applying rings to a star
    check(!sim.apply(ApplyBodyRings{gen, sun, sim.snapshot().body(sun).revisions.definition, {layer}}).accepted,
          "Stars cannot have planetary rings");

    // Reject invalid inner/outer radii
    RingLayerDefinition bad_layer = layer;
    bad_layer.inner = 0.95;
    check(!sim.apply(ApplyBodyRings{gen, planet, rev_before.definition, {bad_layer}}).accepted,
          "Reject ring inner radius <= 1.0");
    bad_layer.inner = 2.0;
    bad_layer.outer = 1.8;
    check(!sim.apply(ApplyBodyRings{gen, planet, rev_before.definition, {bad_layer}}).accepted,
          "Reject ring outer radius <= inner radius");

    // Apply valid ring layer
    auto res = sim.apply(ApplyBodyRings{gen, planet, rev_before.definition, {layer}, 1.35, 2.45});
    check(res.accepted && res.changed, "ApplyBodyRings accepted");
    const auto& updated = sim.snapshot().body(planet);
    check(updated.definition.rings.size() == 1, "Ring layer stored on body definition");
    check(updated.definition.rings[0] == layer, "Ring layer fields match applied values");
    check(updated.definition.ring_texture_inner == 1.35 && updated.definition.ring_texture_outer == 2.45,
          "Ring texture bounds stored");
    check(updated.revisions.definition == rev_before.definition + 1, "Ring edit increments definition revision");
    check(updated.revisions.geometry == rev_before.geometry + 1, "Ring edit increments geometry revision");
    check(updated.revisions.lighting == rev_before.lighting + 1, "Ring edit increments lighting revision");
    check(updated.revisions.dynamics == rev_before.dynamics, "Ring edit preserves dynamics revision");
    check(sim.snapshot().definition_revision() == def_rev_before + 1, "Ring edit increments system definition revision");

    // Idempotent re-apply does not mark changed
    auto res_same = sim.apply(ApplyBodyRings{gen, planet, updated.revisions.definition, {layer}, 1.35, 2.45});
    check(res_same.accepted && !res_same.changed, "Identical ApplyBodyRings is a no-op");

    // Reject stale revision
    check(!sim.apply(ApplyBodyRings{gen, planet, rev_before.definition, {}}).accepted,
          "Stale body revision rejected on ApplyBodyRings");

    // Serialization round-trip
    auto serialized = SystemManager::serialize_definition(sim.snapshot().definition());
    auto reloaded = SystemManager::parse_system(serialized, "RingTestReloaded");
    int r_planet = reloaded.definition.find_body("Planet");
    check(r_planet >= 0, "Reloaded planet found");
    const auto& r_def = reloaded.definition.bodies[r_planet];
    check(r_def.rings.size() == 1, "Reloaded planet has 1 ring layer");
    check(std::abs(r_def.rings[0].inner - 1.35) < 1e-6 && std::abs(r_def.rings[0].outer - 2.45) < 1e-6,
          "Reloaded ring bounds preserved");
    check(std::abs(r_def.rings[0].scatter - 0.65f) < 1e-5f &&
          std::abs(r_def.rings[0].asymmetry - 0.78f) < 1e-5f &&
          std::abs(r_def.rings[0].backscatter - (-0.42f)) < 1e-5f &&
          std::abs(r_def.rings[0].unlit_factor - 1.1f) < 1e-5f,
          "Reloaded ring phase parameters preserved");
    check(r_def.rings[0].is_textured &&
          std::abs(r_def.rings[0].hue_shift - 0.12f) < 1e-5f &&
          std::abs(r_def.rings[0].saturation - 1.25f) < 1e-5f &&
          std::abs(r_def.rings[0].brightness - 0.95f) < 1e-5f &&
          std::abs(r_def.rings[0].alpha_boost - 1.4f) < 1e-5f,
          "Reloaded ring texture HSBA adjustments preserved");
    check(r_def.rings[0].gradient.size() == 3, "Reloaded ring gradient stops preserved");

    // PNG round-trip contract
    auto tmp_png = paths::external_path("scratch", "ring_png_contract_test.png");
    std::filesystem::create_directories(tmp_png.parent_path());
    auto profile = rendering::generate_ring_segment_profile(layer, {}, 256);
    check(rendering::save_ring_png_2d(tmp_png, profile, 8), "Save 2D ring PNG");
    auto decoded = rendering::load_ring_png_1d(tmp_png, 256);
    check(decoded.has_value() && decoded->size() == 256, "Decode 1D ring PNG");
    for (size_t i = 0; i < 256; ++i) {
        check(glm::length((*decoded)[i] - profile[i]) < 0.02f, "PNG encode/decode preserves 8-bit RGBA profile within quantization");
    }
    std::error_code ec;
    std::filesystem::remove(tmp_png, ec);

    // Check Saturn's Rings.png if present in workspace
    if (auto saturn_tex = find_ring_texture_path("Solar System", "Saturn")) {
        auto sat_samples = rendering::load_ring_png_1d(*saturn_tex, rendering::RING_ATLAS_WIDTH);
        check(sat_samples.has_value() && sat_samples->size() == rendering::RING_ATLAS_WIDTH,
              "Saturn Rings.png decodes to RING_ATLAS_WIDTH samples");
    }
}

void procedural_ring_generator_contracts() {
    auto presets = rendering::get_ring_presets();
    check(presets.size() == 7, "Exactly 7 procedural ring presets defined");

    for (size_t i = 0; i < presets.size(); ++i) {
        const auto& p = presets[i];
        check(!p.name.empty(), "Preset name non-empty");
        check(p.band_count >= 1 && p.band_count <= 8, "Preset band count in [1, 8]");
        check(p.gap_count >= 0 && p.gap_count <= 6, "Preset gap count in [0, 6]");
        check(p.striation_freq >= 20.0f, "Preset striation frequency valid");
        check(p.contrast >= 0.5f, "Preset contrast valid");
        check(p.base_density >= 0.01f, "Preset base density positive");
        check(p.core_boost >= 0.1f, "Preset core boost positive");
        check(p.inner_radius_ratio >= 1.05f, "Preset inner radius ratio valid");
        check(p.outer_radius_ratio > p.inner_radius_ratio, "Preset outer radius ratio exceeds inner");
        check(p.opacity > 0.0f && p.opacity <= 1.0f, "Preset opacity in (0, 1]");
        check(p.asymmetry >= -0.99f && p.asymmetry <= 0.99f, "Preset asymmetry valid");
        check(p.backscatter >= -0.99f && p.backscatter <= 0.0f, "Preset backscatter valid");
        check(p.scatter >= 0.0f && p.scatter <= 1.0f, "Preset phase balance valid");
        check(rendering::get_ring_preset(p.name).name == p.name, "Preset lookup by name matches");
        check(rendering::get_ring_preset(i).name == p.name, "Preset lookup by index matches");
    }

    const char* expected_names[] = {
        "Saturnian Ice", "Jovian Silicate Dust", "Uranian Charcoal",
        "Neptunian Arcs", "Chariklo Centaur", "Warm Tholin Ice", "Custom"
    };
    for (const char* name : expected_names) {
        check(rendering::get_ring_preset(name).name == name, "Expected preset exists");
    }

    // Generate Saturnian Ice profile
    auto saturn_cfg = rendering::ProceduralRingConfig::from_preset(rendering::get_ring_preset("Saturnian Ice"), 42);
    auto saturn_res = rendering::generate_procedural_ring_profile(saturn_cfg, 4096);
    check(saturn_res.profile.size() == 4096, "Profile size matches requested resolution");
    check(saturn_res.inner_radius_ratio == saturn_cfg.inner_radius_ratio, "Result inner ratio matches config");
    check(saturn_res.outer_radius_ratio == saturn_cfg.outer_radius_ratio, "Result outer ratio matches config");

    float alpha_sum = 0.0f;
    for (size_t i = 0; i < saturn_res.profile.size(); ++i) {
        const auto& sample = saturn_res.profile[i];
        check(std::isfinite(sample.r) && sample.r >= 0.0f && sample.r <= 1.0f, "Profile red in [0, 1]");
        check(std::isfinite(sample.g) && sample.g >= 0.0f && sample.g <= 1.0f, "Profile green in [0, 1]");
        check(std::isfinite(sample.b) && sample.b >= 0.0f && sample.b <= 1.0f, "Profile blue in [0, 1]");
        check(std::isfinite(sample.a) && sample.a >= 0.0f && sample.a <= 1.0f, "Profile alpha in [0, 1]");
        alpha_sum += sample.a;
    }
    check(alpha_sum > 10.0f, "Profile has non-trivial optical depth");
    check(saturn_res.profile.front().a < 0.1f, "Inner boundary fade suppresses extreme edge");
    check(saturn_res.profile.back().a < 0.1f, "Outer boundary fade suppresses extreme edge");

    // Determinism test: identical seed -> identical output bit-for-bit
    auto saturn_res2 = rendering::generate_procedural_ring_profile(saturn_cfg, 4096);
    for (size_t i = 0; i < saturn_res.profile.size(); ++i) {
        check(saturn_res.profile[i] == saturn_res2.profile[i], "Procedural ring generator is deterministic");
    }

    // Seed variation test: different seed -> distinct profile
    saturn_cfg.seed = 43;
    auto saturn_res_diff = rendering::generate_procedural_ring_profile(saturn_cfg, 4096);
    float diff_sum = 0.0f;
    for (size_t i = 0; i < saturn_res.profile.size(); ++i) {
        diff_sum += glm::length(saturn_res.profile[i] - saturn_res_diff.profile[i]);
    }
    check(diff_sum > 1.0f, "Different seeds produce distinctly varied profiles");

    // Ring layer definition and simulation integration
    ephemeris::RingLayerDefinition proc_layer{};
    rendering::apply_procedural_ring_layer(saturn_res, proc_layer);
    check(proc_layer.inner == saturn_res.inner_radius_ratio, "Applied layer inner radius matches");
    check(proc_layer.outer == saturn_res.outer_radius_ratio, "Applied layer outer radius matches");
    check(proc_layer.is_textured, "Applied procedural layer marked as textured");
    check(proc_layer.gradient.size() == 2 && proc_layer.gradient[0].a == 1.0f && proc_layer.gradient[1].a == 1.0f,
          "Applied procedural layer has full gradient stops");

    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "ProcRingSimTest")).accepted, "Load fixture for procedural rings");
    int planet = sim.snapshot().find_body("Planet");
    auto app_res = sim.apply(ApplyBodyRings{
        sim.snapshot().generation(), planet, sim.snapshot().body(planet).revisions.definition,
        {proc_layer}, proc_layer.inner, proc_layer.outer
    });
    check(app_res.accepted, "Simulation successfully accepts applied procedural ring layer");
    check(sim.snapshot().body(planet).definition.rings.size() == 1, "Planet has 1 procedural ring layer");

    // PNG disk bake and reload round-trip
    auto test_png_path = paths::external_path("textures", "ProcRingBakeTest", "Planet", "Rings.png");
    std::filesystem::create_directories(test_png_path.parent_path());
    ephemeris::RingLayerDefinition baked_layer{};
    check(rendering::bake_and_save_procedural_ring_texture("ProcRingBakeTest", "Planet", saturn_res, baked_layer, 16),
          "Bake procedural ring texture to PNG disk");
    check(std::filesystem::exists(test_png_path), "Baked Rings.png exists on disk");
    auto reloaded_png = rendering::load_ring_png_1d(test_png_path, 4096);
    check(reloaded_png.has_value() && reloaded_png->size() == 4096, "Reload baked procedural ring PNG");
    for (size_t i = 0; i < 4096; i += 64) {
        check(glm::length((*reloaded_png)[i] - saturn_res.profile[i]) < 0.02f,
              "Baked and reloaded procedural profile matches original within 8-bit quantization");
    }
    std::error_code ec;
    std::filesystem::remove_all(paths::external_path("textures", "ProcRingBakeTest"), ec);
}

void atmosphere_physics_contracts() {
    auto a = physics::earth_physical_atmosphere();
    constexpr double radius = 6356.766;
    double mass = 9.80665*radius*radius*1e6/(6.67430e-11*1.98847e30);
    auto surface = physics::sample_atmosphere(a, mass, radius, 0);
    check(std::abs(surface.density_kg_m3-1.225) < .001, "Earth gas density matches standard atmosphere");
    check(std::abs(surface.scale_height_km-8.434) < .02, "Ideal-gas pressure scale height");
    constexpr double heights[] = {11, 20, 30, 50};
    constexpr double pressures[] = {22632.1, 5474.89, 1171.87, 75.9448};
    for (int i = 0; i < 4; ++i) {
        double h = radius*heights[i]/(radius-heights[i]);
        auto state = physics::sample_atmosphere(a, mass, radius, h);
        check(std::abs(state.pressure_pa/pressures[i]-1) < .004, "Sparse layer queries match U.S. Standard Atmosphere pressures");
    }
    auto profile = physics::prepare_atmosphere(a, mass, radius);
    for (double height : {-20.0, -1.0, 0.0, 11.0, 20.0, 30.0, 50.0, 99.0}) {
        double index = std::clamp((std::asinh(height/profile->profile_domain.z)-profile->profile_domain.x)*profile->profile_domain.y, 0.0, 1.0)*255;
        size_t lower = std::min(size_t(index), size_t(254));
        double relative = std::exp(std::lerp(double(profile->log_density[lower].x), double(profile->log_density[lower+1].x), index-lower));
        auto state = physics::sample_atmosphere(a, mass, radius, height);
        check(std::abs(relative*surface.number_density_m3/state.number_density_m3-1) < .003, "Prepared log density retains hydrostatic layers above and below datum");
    }
    glm::dvec3 ozone_tau(0);
    for (int i = 0; i < 10000; ++i) {
        double height = (i+.5)*a.height_km/10000;
        ozone_tau += glm::dvec3(profile->ozone_absorption)*std::exp(-.5*std::pow((height-a.ozone_peak_km)/a.ozone_width_km, 2))*double(a.height_km)/10000.0;
    }
    auto expected = glm::dvec3(1.209e-25, 3.5e-25, 1.582e-26)*(300*2.687e20);
    check(glm::length(ozone_tau-expected) < 1e-8, "Gaussian ozone conserves the supplied DU column");
    a.temperature_layers.clear();
    auto isothermal = physics::sample_atmosphere(a, mass, radius, 40);
    double expected_p = a.pressure_pa*std::exp(-radius*40/(radius+40)/surface.scale_height_km);
    check(std::abs(isothermal.pressure_pa/expected_p-1) < 1e-12, "Exact spherical isothermal hydrostatic solution");
    a.pressure_pa = 0;
    auto vacuum = physics::prepare_atmosphere(a, mass, radius);
    check(vacuum->rayleigh == glm::vec3(0) && vacuum->ozone_absorption == glm::vec3(0), "Vacuum has no molecular or ozone opacity");
    a.pressure_pa = -1;
    check(!valid_atmosphere(a), "Reject negative physical pressure");
    a = physics::earth_physical_atmosphere();
    a.composition["Tholin"] = .1;
    check(!valid_atmosphere(a), "Aerosol particles are not gas mole fractions");

    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(fixture(), "Physical atmosphere")).accepted, "Physical state fixture");
    int body = sim.snapshot().find_body("Planet");
    auto apply = [&](const AtmosphereDefinition& values) {
        return sim.apply(ApplyBodyAtmosphere{sim.snapshot().generation(), body, sim.snapshot().body(body).revisions.definition, values});
    };
    a = physics::earth_physical_atmosphere();
    check(apply(a).changed, "Apply physical atmosphere");
    auto captured = sim.snapshot();
    auto prepared = captured.body(body).render.atmosphere;
    sim.advance(units::Seconds{1});
    check(sim.snapshot().body(body).render.atmosphere == prepared, "Playback reuses immutable physics profiles");
    check(!apply(a).changed && sim.snapshot().body(body).render.atmosphere == prepared, "Unchanged physical inputs do no work");
    a.pressure_pa *= 2;
    check(apply(a).changed && sim.snapshot().body(body).render.atmosphere != prepared, "Pressure edit rebuilds prepared optics");
    check(captured.body(body).render.atmosphere == prepared, "Existing snapshots retain their original prepared atmosphere");
    auto loaded = SystemManager::parse_system(SystemManager::serialize_definition(sim.snapshot().definition()), "Physical roundtrip");
    check(loaded.definition.bodies[loaded.definition.find_body("Planet")].atmosphere == a, "Physical layers, ozone, mixture and manual aerosols survive persistence");
    auto properties = BodyPhysicalProperties::from(sim.snapshot().body(body).definition);
    properties.mass.value = 0;
    check(!sim.apply(ApplyBodyPhysicalProperties{sim.snapshot().generation(), body, sim.snapshot().body(body).revisions.definition, properties}).accepted,
        "Reject mass edits that remove hydrostatic support");
}

void atmosphere_customization_contracts() {
    nlohmann::json legacy{{"surface_pressure", 1.45}, {"temperature", 94.0},
        {"composition", {{"N2", .9435}, {"CH4", .0565}}}};
    auto atmosphere = legacy.get<AtmosphereDefinition>();
    check(atmosphere.enabled && atmosphere.physical && atmosphere.automatic_aerosols, "Complete legacy gas data enables a calculated atmosphere");
    check(atmosphere.pressure_pa == 1.45*101325 && atmosphere.temperature_k == 94, "Legacy pressure units and supplied temperature are retained");
    auto haze = physics::prepare_atmosphere(atmosphere, 6.763e-8, 2574.73);
    check(haze->aerosols.size() == 1 && haze->aerosols[0].scattering.b/haze->aerosols[0].extinction.b <
        haze->aerosols[0].scattering.r/haze->aerosols[0].extinction.r,
        "Legacy methane-rich gas migrates to an explicit absorbing particle column");
    atmosphere.pressure_pa = 0;
    auto vacuum = physics::prepare_atmosphere(atmosphere, 6.763e-8, 2574.73);
    check(vacuum->mie_extinction == glm::vec3(0) && vacuum->aerosols.empty(), "Zero pressure disables particle optics");
    legacy["enabled"] = false;
    check(!legacy.get<AtmosphereDefinition>().enabled, "Explicit disabled atmosphere survives complete physical inputs");
    legacy.erase("enabled");
    legacy.erase("surface_pressure");
    check(!legacy.get<AtmosphereDefinition>().enabled, "Partial gas data does not silently enable an atmosphere");
    check(!nlohmann::json::object().get<AtmosphereDefinition>().enabled, "Empty atmosphere remains disabled");
    legacy["pressure_pa"] = 101325.0;
    check(legacy.get<AtmosphereDefinition>().physical && legacy.get<AtmosphereDefinition>().enabled, "Modern physical inputs also infer physical mode and activation");
    legacy["aerosol_mode"] = "manual";
    legacy["beta_mie"] = 2e-6;
    legacy["h_mie"] = 1.7;
    legacy["mie_g"] = .76;
    legacy["mie_albedo"] = {.9, .8, .7};
    auto manual = legacy.get<AtmosphereDefinition>();
    check(!manual.automatic_aerosols && std::abs(manual.mie_extinction.g-.002f) < 1e-8f &&
        std::abs(manual.mie_scattering.g-.0016f) < 1e-8f && std::abs(manual.mie_scale_height_km-1.7f) < 1e-6f,
        "Legacy manual aerosol units, scale height and colored absorption migrate correctly");
    auto roundtrip = nlohmann::json(manual).get<AtmosphereDefinition>();
    check(roundtrip == manual, "Automatic activation and imported overrides survive save and reload");

    ui::AtmosphereEditDraft draft;
    draft.set_gas_percentage("CO2", 25);
    check(draft.values.physical && draft.values.automatic_height && draft.values.automatic_aerosols &&
        std::abs(draft.values.composition.at("CO2")-.25) < 1e-12, "Composition edits select calculated properties and preserve the entered percentage");
    double total = 0;
    for (auto [_, weight] : draft.values.composition) total += weight;
    check(std::abs(total-1) < 1e-12, "Changing a percentage balances the remaining gas fractions");
    draft.remove_gas("CO2");
    total = 0;
    for (auto [_, weight] : draft.values.composition) total += weight;
    check(!draft.values.composition.contains("CO2") && std::abs(total-1) < 1e-12, "Removing a gas retains a complete normalized mixture");
}

void atmosphere_state_contracts() {
    Simulation sim;
    auto source = fixture();
    source[0]["height_range_km"] = {-10.984, 8.848};
    check(sim.replace_system(SystemManager::parse_system(source, "Atmosphere")).accepted, "Atmosphere state fixture");
    int index = sim.snapshot().find_body("Planet");
    auto captured = sim.snapshot();
    auto original = captured.body(index).revisions;
    ui::AtmosphereEditDraft draft;
    draft.synchronize(captured, index);
    draft.values.enabled = true;
    draft.dirty = true;
    check(sim.apply(draft.command()).changed, "Enable atmosphere through command");
    const auto& changed = sim.snapshot().body(index);
    check(!captured.body(index).definition.atmosphere.enabled && changed.definition.atmosphere.enabled, "Atmosphere edits preserve snapshots");
    check(changed.revisions.lighting == original.lighting+1 && changed.revisions.dynamics == original.dynamics && changed.revisions.geometry == original.geometry,
          "Atmosphere edits invalidate lighting without invalidating geometry or motion");
    check(!sim.apply(draft.command()).accepted, "Stale atmosphere draft is rejected");
    draft.reload(sim.snapshot(), index);
    check(!sim.apply(draft.command()).changed, "Identical atmosphere command is inert");
    auto valid = draft.values;
    draft.values.mie_extinction.x = -1.0f;
    check(!sim.apply(draft.command()).accepted, "Reject negative atmosphere extinction");
    draft.values = valid;
    draft.values.mie_extinction.x = draft.values.mie_scattering.x*0.5f;
    check(!sim.apply(draft.command()).accepted, "Reject nonphysical single scattering albedo");
    draft.values = valid;
    draft.values.rayleigh_scale_height_km = std::numeric_limits<float>::quiet_NaN();
    check(!sim.apply(draft.command()).accepted, "Reject nonfinite scale height");
    draft.values = valid;
    draft.values.mie_asymmetry = 1.0f;
    check(!sim.apply(draft.command()).accepted, "Reject singular phase function");
    draft.values = valid;
    draft.generation += 1;
    check(!sim.apply(draft.command()).accepted, "Reject atmosphere edit from another system");
    auto json = SystemManager::serialize_definition(sim.snapshot().definition());
    auto roundtrip = SystemManager::parse_system(json, "Roundtrip");
    check(roundtrip.definition.bodies[roundtrip.definition.find_body("Planet")].atmosphere == valid, "Atmosphere JSON roundtrip");
    check(roundtrip.definition.bodies[roundtrip.definition.find_body("Planet")].height_range_km == glm::dvec2(-10.984, 8.848), "Established terrain range survives definition serialization");
    auto exported = SystemManager::serialize_snapshot_to_nbody(sim.snapshot());
    auto nbody = SystemManager::parse_system(exported, "NBody roundtrip");
    check(nbody.definition.bodies[nbody.definition.find_body("Planet")].atmosphere == valid, "Atmosphere survives snapshot export");
    check(nbody.definition.bodies[nbody.definition.find_body("Planet")].height_range_km == glm::dvec2(-10.984, 8.848), "Terrain range survives snapshot export");
    auto invalid_range = SystemManager::parse_system(source, "Invalid terrain range");
    invalid_range.definition.bodies[invalid_range.definition.find_body("Planet")].height_range_km = {1.0, -1.0};
    check(!sim.replace_system(invalid_range).accepted, "Reject reversed terrain ranges");
    invalid_range.definition.bodies[invalid_range.definition.find_body("Planet")].height_range_km = {-1e10, 0.0};
    check(!sim.replace_system(invalid_range).accepted, "Reject terrain minimum beyond the body centre");
    auto invalid = SystemManager::parse_system(fixture(), "Invalid");
    invalid.definition.bodies[invalid.definition.find_body("Planet")].atmosphere.height_km = -1.0f;
    check(!sim.replace_system(std::move(invalid)).accepted, "Validate atmospheres on system load");
    draft.reload(sim.snapshot(), sim.snapshot().find_body("Sun"));
    draft.values.enabled = true;
    check(!sim.apply(draft.command()).accepted, "Stars cannot receive planetary atmosphere");
}

#include "particle_optics_tests.hpp"

int main(int argc, char** argv) {
    try {
        units_and_coordinates();
        atmosphere_physics_contracts();
        atmosphere_customization_contracts();
        particle_optics_contracts();
        giant_atmosphere_contracts();
        atmosphere_state_contracts();
        clock_and_snapshots();
        time_and_date_formatting();
        jump_and_timeline_contracts();
        keplerian_and_hydrostatic_math();
        orbital_element_edits();
        bring_moons_contracts();
        edits_and_invalidation();
        add_and_delete_contracts();
        ring_state_and_serialization_contracts();
        procedural_ring_generator_contracts();
        serialization_and_validation();
        render_settings();
        ias15_contracts();
        kepler_perturbation_contracts();
        size_t presets = 0;
        if (argc == 2) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(argv[1])) {
                if (entry.path().filename() != "system.json") continue;
                std::ifstream file(entry.path());
                auto json = nlohmann::json::parse(file);
                Simulation sim;
                auto loaded = SystemManager::parse_system(json, entry.path().parent_path().filename().string());
#ifndef STELLAR_TEST_EPHEMERIS
                if (loaded.definition.motion_model == MotionModel::Spice) continue;
#endif
                auto result = sim.replace_system(std::move(loaded));
                if (!result) throw std::runtime_error(entry.path().string() + ": " + result.error);
                for (size_t bi = 0; bi < sim.snapshot().size(); ++bi) {
                    const auto& b = sim.snapshot().body(bi);
                    if (!b.definition.rings.empty()) {
                        auto rings = b.definition.rings;
                        rings[0].opacity = std::max(0.01f, rings[0].opacity * 0.95f);
                        auto r_res = sim.apply(ApplyBodyRings{
                            sim.snapshot().generation(), static_cast<int>(bi), b.revisions.definition,
                            rings, b.definition.ring_texture_inner, b.definition.ring_texture_outer
                        });
                        if (!r_res) throw std::runtime_error(entry.path().string() + " (" + b.definition.name + " rings): " + r_res.error);
                    }
                }
                check(sim.advance(units::Seconds{1.0}).accepted, "Preset propagation");
                ++presets;
            }
        }
        std::cout << "PASS: units, coordinates, clock, snapshot isolation, atomic edits, revision dependencies, stale drafts, persistence, hierarchy validation, render settings; " << presets << " presets checked.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
