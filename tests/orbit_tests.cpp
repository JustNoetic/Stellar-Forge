#include "core/path_utils.hpp"
#include "ephemeris/orbit_curves.hpp"
#include "ephemeris/simulation.hpp"
#include "ephemeris/spice_ephemeris.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace sf;
using namespace sf::ephemeris;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

glm::dvec3 vector(const nlohmann::json& value) {
    return {value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
}

bool near(double a, double b) {
    return std::abs(a - b) < 2e-9 * std::max({1.0, std::abs(a), std::abs(b)});
}

LoadedSystem multi_moon_fixture(double inner_eccentricity = 0.1) {
    LoadedSystem loaded;
    const char* names[] = {"Star", "Planet", "Inner", "Submoon", "Outer"};
    const int parents[] = {-1, 0, 1, 2, 1};
    const double masses[] = {1.0, 0.005, 0.0008, 0.0002, 0.0006};
    for (int i = 0; i < 5; ++i) {
        BodyDefinition body;
        body.name = names[i];
        body.mass_msun = masses[i];
        body.parent_idx = parents[i];
        if (parents[i] >= 0) body.parent_id = names[parents[i]];
        loaded.definition.bodies.push_back(body);
        loaded.motions.emplace_back();
    }
    auto place = [&](bool velocity) {
        glm::dvec3 planet{5.0, 0.0, 0.0}, inner{0.02, 0.0, 0.0}, outer{0.06, 0.0, 0.0}, submoon{0.001, 0.0, 0.0};
        if (velocity) {
            planet = {0.0, std::sqrt(G_CONST * 1.0066 * 1.05 / 5.0), 0.0};
            inner = {0.0, std::sqrt(G_CONST * 0.006 * (1.0 + inner_eccentricity) / 0.02), 0.0};
            outer = {0.0, std::sqrt(G_CONST * 0.0066 * 1.2 / 0.06), 0.0};
            submoon = {0.0, std::sqrt(G_CONST * 0.001 * 1.15 / 0.001), 0.0};
        }
        auto star = -0.0066 / 1.0066 * planet;
        auto planet_system = star + planet;
        auto inner_group = planet_system - 0.0006 / 0.0066 * outer;
        auto planet_body = inner_group - 0.001 / 0.006 * inner;
        auto inner_system = planet_body + inner;
        auto inner_body = inner_system - 0.0002 / 0.001 * submoon;
        glm::dvec3 values[] = {star, planet_body, inner_body, inner_body + submoon, inner_group + outer};
        for (int i = 0; i < 5; ++i) {
            if (velocity) loaded.motions[i].velocity = EclipticVelocity{values[i]};
            else loaded.motions[i].position = EclipticPosition{values[i]};
        }
    };
    place(false);
    place(true);
    return loaded;
}

void multi_moon_stability(double inner_eccentricity) {
    Simulation sim;
    check(sim.replace_system(multi_moon_fixture(inner_eccentricity)).accepted, "Load nested multi-moon system");
    auto initial = compute_orbit_curves(sim.snapshot());
    double worst_eccentricity = 0.0, worst_size = 0.0;
    bool radial_order_crossed = false;
    for (int step = 0; step < 1000; ++step) {
        check(sim.advance(units::Seconds{SECONDS_PER_YEAR / 1000.0}).accepted, "Advance multi-moon system");
        auto curves = compute_orbit_curves(sim.snapshot());
        auto planet = sim.snapshot().body(1).motion.position.value;
        radial_order_crossed |= glm::length(sim.snapshot().body(2).motion.position.value - planet) >
                                glm::length(sim.snapshot().body(4).motion.position.value - planet);
        for (const auto& reference : initial) {
            if (reference.reflex || reference.patched) continue;
            auto current = std::find_if(curves.begin(), curves.end(), [&](const auto& curve) {
                return curve.body_index == reference.body_index && !curve.reflex && !curve.patched;
            });
            check(current != curves.end(), "Multi-moon curve remains present");
            worst_eccentricity = std::max(worst_eccentricity, std::abs(current->eccentricity - reference.eccentricity));
            worst_size = std::max(worst_size, std::abs(current->semilatus_au / reference.semilatus_au - 1.0));
        }
    }
    std::cout << "Multi-moon maximum eccentricity/relative size drift: " << worst_eccentricity << " / " << worst_size << '\n';
    check(worst_eccentricity < 1e-9 && worst_size < 1e-9, "Jacobi moon conics remain stable over one year");
    Simulation direct;
    direct.replace_system(multi_moon_fixture(inner_eccentricity));
    direct.advance(units::Seconds{SECONDS_PER_YEAR});
    for (size_t i = 0; i < sim.snapshot().size(); ++i) {
        check(glm::length(sim.snapshot().body(i).motion.position.value - direct.snapshot().body(i).motion.position.value) < 1e-9,
              "Jacobi propagation is independent of frame partitioning");
    }
    if (inner_eccentricity > 0.5) check(radial_order_crossed, "Crossing sibling fixture actually changes radial order");
    auto captured = sim.snapshot();
    auto physical = BodyPhysicalProperties::from(captured.body(2).definition);
    physical.mass.value *= 1.5;
    check(sim.apply(ApplyBodyPhysicalProperties{captured.generation(), 2, captured.body(2).revisions.definition, physical}).accepted,
          "Changing a moon mass reseeds Jacobi motion");
    for (size_t i = 0; i < captured.size(); ++i) {
        check(sim.snapshot().body(i).motion.position.value == captured.body(i).motion.position.value &&
              sim.snapshot().body(i).motion.velocity.value == captured.body(i).motion.velocity.value,
              "Mass edits preserve the complete instantaneous Cartesian state");
    }
    auto edited_reference = compute_orbit_curves(sim.snapshot());
    sim.advance(units::Seconds{86400.0});
    auto edited_curves = compute_orbit_curves(sim.snapshot());
    for (const auto& reference : edited_reference) {
        if (reference.reflex || reference.patched) continue;
        auto current = std::find_if(edited_curves.begin(), edited_curves.end(), [&](const auto& curve) {
            return curve.body_index == reference.body_index && !curve.reflex && !curve.patched;
        });
        check(current != edited_curves.end() && std::abs(current->eccentricity - reference.eccentricity) < 1e-9,
              "Mass edit propagation uses the new masses and current epoch");
    }
}

void pluto_charon_stability(const std::filesystem::path& path) {
    std::ifstream file(path);
    Simulation sim;
    check(sim.replace_system(SystemManager::parse_system(nlohmann::json::parse(file), "Solar System")).accepted,
          "Load actual Solar System for Pluto/Charon regression");
    check(sim.apply(SetMotionModel{MotionModel::Keplerian}).accepted, "Pluto/Charon Keplerian mode");
    int pluto = sim.snapshot().find_body("Pluto"), charon = sim.snapshot().find_body("Charon");
    auto initial = compute_orbit_curves(sim.snapshot());
    auto select = [](const auto& curves, int body, bool reflex) -> const OrbitCurve& {
        auto it = std::find_if(curves.begin(), curves.end(), [&](const auto& curve) {
            return curve.body_index == body && curve.reflex == reflex && !curve.patched;
        });
        check(it != curves.end(), "Pluto/Charon local curve is present");
        return *it;
    };
    double initial_e = select(initial, charon, false).eccentricity;
    double initial_p = select(initial, charon, false).semilatus_au;
    double initial_system_e = select(initial, pluto, false).eccentricity;
    double initial_system_p = select(initial, pluto, false).semilatus_au;
    double worst_e = 0.0, worst_size = 0.0;
    double worst_system_e = 0.0, worst_system_size = 0.0;
    for (int step = 0; step < 512; ++step) {
        check(sim.advance(units::Seconds{21600.0}).accepted, "Advance real Pluto/Charon through twenty lunar periods");
        auto curves = compute_orbit_curves(sim.snapshot());
        auto moon = select(curves, charon, false), reflex = select(curves, pluto, true);
        auto system = select(curves, pluto, false);
        worst_e = std::max(worst_e, std::abs(moon.eccentricity - initial_e));
        worst_size = std::max(worst_size, std::abs(moon.semilatus_au / initial_p - 1.0));
        worst_system_e = std::max(worst_system_e, std::abs(system.eccentricity - initial_system_e));
        worst_system_size = std::max(worst_system_size, std::abs(system.semilatus_au / initial_system_p - 1.0));
        check(glm::length(moon.focus.value - reflex.focus.value) < 1e-10, "Pluto and Charon share their actual pair barycenter");
        double ratio = sim.snapshot().body(charon).definition.mass_msun / sim.snapshot().body(pluto).definition.mass_msun;
        check(std::abs(reflex.semilatus_au / moon.semilatus_au - ratio) < 1e-8, "Pluto reflex has the correct Charon/Pluto mass ratio");
    }
    std::cout << "Actual Pluto/Charon maximum eccentricity/relative size drift: " << worst_e << " / " << worst_size << '\n';
    std::cout << "Pluto system orbit maximum eccentricity/relative size drift: " << worst_system_e << " / " << worst_system_size << '\n';
    check(worst_e < 5e-8 && worst_size < 5e-8, "Real Pluto/Charon curves remain stable");
    check(worst_system_e < 1e-9 && worst_system_size < 1e-9, "Pluto system barycenter follows one consistent outer conic");
}

void original_orbit_fixtures(const std::filesystem::path& path) {
    std::ifstream file(path);
    auto fixtures = nlohmann::json::parse(file);
    size_t checked = 0;
    for (const auto& fixture : fixtures) {
        LoadedSystem loaded;
        for (size_t i = 0; i < fixture["masses"].size(); ++i) {
            BodyDefinition body;
            body.name = "Body" + std::to_string(i);
            body.mass_msun = fixture["masses"][i];
            body.parent_idx = fixture["parents"][i];
            if (body.parent_idx >= 0) body.parent_id = "Body" + std::to_string(body.parent_idx);
            loaded.definition.bodies.push_back(body);
            loaded.motions.push_back({EclipticPosition{vector(fixture["positions"][i])}, EclipticVelocity{vector(fixture["velocities"][i])}});
        }
        Simulation sim;
        check(sim.replace_system(std::move(loaded)).accepted, "Load original fixture");
        auto curves = compute_orbit_curves(sim.snapshot());
        const auto& expected = fixture["expected"];
        std::string name = fixture["name"];
        check(curves.size() == expected.size(), name + ": curve count");
        glm::dvec3 center{0.0};
        double mass = 0.0;
        for (size_t i = 0; i < fixture["masses"].size(); ++i) {
            double m = fixture["masses"][i];
            center += m * vector(fixture["positions"][i]);
            mass += m;
        }
        center /= mass;
        for (size_t i = 0; i < curves.size(); ++i) {
            const auto& c = curves[i];
            const auto& ref = expected[i];
            for (int k = 0; k < 3; ++k) {
                check(near(c.periapsis[k], ref[k]), name + ": periapsis basis");
                check(near(c.transverse[k], ref[k+4]), name + ": transverse basis");
                check(near(c.focus.value[k] + center[k], ref[k+8]), name + ": barycenter");
            }
            check(near(c.semilatus_au, ref[3]) && near(c.eccentricity, ref[7]), name + ": conic elements");
            check(near(std::cos(c.arc.start), ref[11]) && near(std::sin(c.arc.start), ref[15]), name + ": start angle");
            check(near(c.arc.phase, ref[16]) && near(c.arc.span, ref[17].get<double>() > 9 ? TWO_PI : ref[17].get<double>()), name + ": arc domain");
            check(c.patched == (ref[18].get<double>() > 0.5), name + ": patched flag");
            for (int sample = 0; sample <= 100; ++sample) {
                auto p = c.point(sample / 100.0).value;
                check(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z), name + ": finite samples");
            }
            if (c.arc.span == TWO_PI && c.eccentricity < 1.0) check(c.point(0.0).value == c.point(1.0).value, "Exact closed ellipse");
            ++checked;
        }
    }
    auto circular = compute_orbit_arc(0.0, 0.0, 1.0, 1.0, 2.0, 1.0);
    check(!circular.escape && circular.span == TWO_PI, "Circular Hill-contained orbit");
    auto outside = compute_orbit_arc(0.0, 0.0, 1.0, 1.0, 0.5, 1.0);
    check(outside.outside && outside.escape && outside.span == 0.0, "Circular outside-Hill arc");
    std::cout << "PASS: " << fixtures.size() << " original Python fixtures, " << checked << " curves, conic endpoints and Hill domains.\n";
}

void spice_oracles(const std::filesystem::path& fixture_path, const std::filesystem::path& preset_path) {
    std::ifstream fixture_file(fixture_path), preset_file(preset_path);
    auto fixture = nlohmann::json::parse(fixture_file);
    auto preset = nlohmann::json::parse(preset_file);
    if (preset.is_object() && preset.value("source_preset", std::string{}) == "Solar System") {
        auto solar = SystemManager{}.load_system("Solar System");
        SpiceKernelManager manager;
        std::string error;
        auto shared = manager.build_ephemeris_system(&solar.definition, "2026-06-03", error);
        check(!shared.definition.bodies.empty(), "Build shared preset for SPICE oracles: " + error);
        preset = SystemManager::serialize_definition(shared.definition);
    }
    for (auto& body : preset) {
        body.erase("sv");
        for (const char* element : {"a", "e", "inc", "Omega", "omega", "M"}) body.erase(element);
    }
    Simulation sim;
    auto loaded = SystemManager::parse_system(preset, "Ephemeris Mode");
    for (auto& motion : loaded.motions) motion = {};
    auto status = sim.replace_system(std::move(loaded));
    check(status.accepted, "SPICE load: " + status.error);
    check(sim.snapshot().definition().motion_model == MotionModel::Spice, "Ephemeris has explicit motion model");
    auto serialized = SystemManager::serialize_definition(sim.snapshot().definition());
    auto round_trip = SystemManager::parse_system(serialized, "Saved ephemeris");
    check(round_trip.definition.motion_model == MotionModel::Spice && round_trip.definition.epoch_utc == "2026-06-03",
          "Motion model and epoch survive saving under another preset name");
    double previous_days = 0.0;
    for (const auto& sample : fixture) {
        double days = sample["days"];
        check(sim.advance(units::Seconds{(days - previous_days) * 86400.0}).accepted, "Advance SPICE epoch");
        previous_days = days;
        for (const auto& reference : sample["states"]) {
            int body_index = sim.snapshot().find_body(reference["body"].get<std::string>());
            int parent_index = sim.snapshot().find_body(reference["parent"].get<std::string>());
            check(body_index >= 0 && parent_index >= 0, "Oracle body available");
            auto body = sim.snapshot().body(body_index), parent = sim.snapshot().body(parent_index);
            auto r = body.motion.position.value - parent.motion.position.value;
            auto v = body.motion.velocity.value - parent.motion.velocity.value;
            check(glm::length(r - vector(reference["position"])) < 1e-12, "Position agrees with independent SpiceyPy oracle");
            check(glm::length(v - vector(reference["velocity"])) < 1e-11, "Velocity agrees with independent SpiceyPy oracle");
        }
    }
    auto before = sim.snapshot();
    auto time = sim.clock().time;
    check(!sim.advance(units::Seconds{SECONDS_PER_YEAR * 1e6}).accepted, "Missing kernel coverage rejects step");
    check(sim.clock().time == time && sim.snapshot().state_revision() == before.state_revision(), "Coverage failure preserves time and revisions");
    for (size_t i = 0; i < before.size(); ++i) {
        check(sim.snapshot().body(i).motion.position.value == before.body(i).motion.position.value &&
              sim.snapshot().body(i).motion.velocity.value == before.body(i).motion.velocity.value, "Coverage failure restores every state");
    }
    auto invalid = SystemManager::parse_system(preset, "Invalid ephemeris");
    invalid.definition.bodies[0].spice_id = 0;
    check(!sim.replace_system(std::move(invalid)).accepted && sim.snapshot().generation() == before.generation(),
          "Missing SPICE ID rejects replacement without changing active playback");
    check(sim.advance(units::Seconds{1.0}).accepted, "Playback recovers after coverage and replacement failures");

    std::string sp_err;
    auto sp = SpiceEphemeris::open(sp_err);
    check(sp != nullptr, "Open SpiceEphemeris for BSP inspection: " + sp_err);

    auto artemis_path = sf::paths::external_path("data", "kernels", "additional", "artemis2.bsp");
    auto artemis_inspect = sp->inspect_bsp(artemis_path);
    check(artemis_inspect.valid && !artemis_inspect.bodies.empty(), "Inspect artemis2.bsp succeeds: " + artemis_inspect.error);
    bool found_orion = false;
    for (const auto& b : artemis_inspect.bodies) {
        if (b.id == -1024) {
            found_orion = true;
            check(!b.utc_start.empty() && !b.utc_end.empty(), "Artemis II Orion spacecraft segment metadata");
        }
    }
    check(found_orion, "artemis2.bsp contains NAIF ID -1024");

    auto plu_path = sf::paths::external_path("data", "kernels", "default", "plu060.bsp");
    auto plu_inspect = sp->inspect_bsp(plu_path);
    check(plu_inspect.valid && plu_inspect.bodies.size() >= 5, "Inspect plu060.bsp discovers Pluto system bodies");

    SpiceKernelManager kmgr;
    auto avail_bsps = kmgr.list_available_bsp_files();
    check(!avail_bsps.empty(), "SpiceKernelManager discovers local .bsp files");

    std::string bsp_err;
    auto generic_artemis = kmgr.build_generic_bsp_system(artemis_path, bsp_err);
    check(!generic_artemis.definition.bodies.empty(), "Build generic BSP system for artemis2.bsp: " + bsp_err);
    Simulation generic_sim;
    auto gen_res = generic_sim.replace_system(std::move(generic_artemis));
    check(gen_res.accepted, "Load generic artemis2.bsp system into Simulation: " + gen_res.error);
    check(generic_sim.advance(units::Seconds{60.0}).accepted, "Advance generic artemis2.bsp playback");

    auto ephem_manifest_path = sf::paths::external_path("data", "ephemeris.json");
    auto saved_manifest = sf::paths::read_json(ephem_manifest_path);

    std::string ephem_err;
    auto base_tmpl = SystemManager::parse_system(preset, "Ephemeris Mode");
    auto built_ephem = kmgr.build_ephemeris_system(&base_tmpl.definition, "2026-06-03", ephem_err);
    check(!built_ephem.definition.bodies.empty(), "Build dynamic ephemeris system: " + ephem_err);
    check(generic_sim.replace_system(std::move(built_ephem)).accepted, "Load dynamic ephemeris system into Simulation");

    if (saved_manifest) {
        sf::paths::write_json(ephem_manifest_path, *saved_manifest, 2);
    }

    std::cout << "PASS: independent SPICE state oracles, BSP inspection, generic BSP playback, and atomic coverage failure.\n";
}

}

int main(int argc, char** argv) {
    try {
        check(argc == 3 || argc == 4, "Expected fixture directory and systems directory, optional --spice");
        original_orbit_fixtures(std::filesystem::path(argv[1]) / "orbit_reference.json");
        multi_moon_stability(0.1);
        multi_moon_stability(0.8);
        pluto_charon_stability(std::filesystem::path(argv[2]) / "Solar System/system.json");
        if (argc == 4) spice_oracles(std::filesystem::path(argv[1]) / "spice_reference.json",
                                    std::filesystem::path(argv[2]) / "Ephemeris Mode/system.json");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
