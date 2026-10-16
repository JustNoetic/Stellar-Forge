#include "ephemeris/simulation_runner.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace sf;
using namespace sf::ephemeris;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require(CommandResult result) {
    if (!result) throw std::runtime_error(result.error);
}

LoadedSystem fixture(MotionModel model) {
    LoadedSystem loaded;
    loaded.definition.motion_model = model;
    BodyDefinition sun;
    sun.name = "Sun";
    sun.mass_msun = 1.0;
    sun.is_star = true;
    BodyDefinition earth;
    earth.name = "Earth";
    earth.parent_id = "Sun";
    earth.parent_idx = 0;
    earth.mass_msun = 3e-6;
    earth.radius_rsun = 0.009;
    loaded.definition.bodies = {sun, earth};
    loaded.motions = {{}, {EclipticPosition{{1.0, 0.0, 0.0}}, EclipticVelocity{{0.0, std::sqrt(G_CONST), 0.0}}}};
    return loaded;
}

void same(const SystemSnapshot& a, const SystemSnapshot& b) {
    check(a.time().value == b.time().value && a.size() == b.size(), "Snapshot epoch and size differ");
    check(a.state_revision() == b.state_revision() && a.hierarchy_revision() == b.hierarchy_revision(), "Snapshot revisions differ");
    for (size_t i = 0; i < a.size(); ++i) {
        check(a.body(i).motion.position.value == b.body(i).motion.position.value, "Threading changed physics positions");
        check(a.body(i).motion.velocity.value == b.body(i).motion.velocity.value, "Threading changed physics velocities");
        check(a.body(i).render.position.value == b.body(i).render.position.value, "Prepared render position differs");
        check(a.reference_parent(i) == b.reference_parent(i), "Reference parent differs");
    }
}

void deterministic_pipeline(MotionModel model, bool solar = false) {
    Simulation direct;
    SimulationRunner runner;
    auto loaded = solar ? SystemManager{}.load_system(SystemManager::SOLAR_SYSTEM_NAME) : fixture(model);
    loaded.definition.motion_model = model;
    require(direct.replace_system(loaded));
    require(runner.replace_system(std::move(loaded)));
    runner.set_threaded(true);
    require(direct.apply(SetTimeWarp{3600.0}));
    require(runner.apply(SetTimeWarp{3600.0}));
    auto dt = units::Seconds{1.0 / 1700.0};
    require(runner.advance_async(dt));
    for (int i = 0; i < 500; ++i) {
        require(direct.advance(dt));
        require(runner.wait());
        same(direct.snapshot(), runner.snapshot());
        auto captured = runner.snapshot();
        if (i < 499) require(runner.advance_async(dt));
        for (int read = 0; read < 50; ++read) same(captured, runner.snapshot());
    }
    auto before = runner.snapshot();
    require(runner.apply(SetPaused{true}));
    require(runner.advance_async(units::Seconds{10.0}));
    require(runner.wait());
    check(runner.snapshot().time().value == before.time().value, "Paused worker advanced");
    check(!runner.apply(SetTimeWarp{-1.0}), "Rejected edit accepted");
    require(runner.poll());
    require(runner.apply(SetPaused{false}));
    runner.set_threaded(false);
    require(direct.advance(dt));
    require(runner.advance(dt));
    check(direct.snapshot().body(1).motion.position.value == runner.snapshot().body(1).motion.position.value, "Switching back to inline lost state");
    for (int i = 0; i < 3; ++i) {
        runner.set_threaded(true);
        require(runner.advance_async(dt));
        runner.set_threaded(false);
    }
}

void backlog_and_commands() {
    SimulationRunner runner;
    require(runner.replace_system(fixture(MotionModel::Keplerian)));
    runner.set_threaded(true);
    double expected_seconds = 0.0;
    for (int i = 0; i < 2000; ++i) {
        require(runner.advance_async(units::Seconds{0.001}));
        expected_seconds += 0.001;
    }
    require(runner.apply(SetPaused{true}));
    check(std::abs(units::to_seconds(runner.clock().time).value - expected_seconds) < 1e-10, "Coalescing dropped elapsed time or pause overtook steps");
    auto captured = runner.snapshot();
    auto properties = BodyPhysicalProperties::from(captured.body(1).definition);
    properties.mass.value *= 2.0;
    auto edit = ApplyBodyPhysicalProperties{captured.generation(), 1, captured.body(1).revisions.definition, properties};
    require(runner.apply(edit));
    check(!runner.apply(edit), "Stale edit accepted by worker");
    check(captured.body(1).definition.mass_msun != runner.snapshot().body(1).definition.mass_msun, "Captured definitions changed");
    require(runner.replace_system(fixture(MotionModel::NBody)));
    check(runner.snapshot().generation() == captured.generation() + 1, "Worker replacement did not update generation");
    check(!runner.advance_async(units::Seconds{std::numeric_limits<double>::quiet_NaN()}), "Invalid frame time accepted");
    check(!runner.advance(units::Seconds{-1.0}), "Negative frame time accepted");
    require(runner.apply(SetPaused{false}));
    require(runner.advance_async(units::Seconds{0.01}));
}

void rejected_step() {
    SimulationRunner runner;
    require(runner.replace_system(fixture(MotionModel::NBody)));
    runner.set_threaded(true);
    auto before = runner.snapshot();
    require(runner.apply(SetTimeWarp{1e16}));
    require(runner.advance_async(units::Seconds{std::numeric_limits<double>::max()}));
    check(!runner.wait(), "Worker lost an integration error");
    check(runner.clock().paused, "Worker did not pause after a rejected step");
    same(before, runner.snapshot());
    require(runner.apply(SetTimeWarp{1.0}));
    require(runner.apply(SetPaused{false}));
    require(runner.advance(units::Seconds{0.01}));
    check(runner.clock().time.value > 0.0, "Worker did not recover after a rejected step");
}

}

int main() {
    try {
        deterministic_pipeline(MotionModel::NBody);
        deterministic_pipeline(MotionModel::Keplerian);
        deterministic_pipeline(MotionModel::NBody, true);
        backlog_and_commands();
        rejected_step();
        std::cout << "Simulation worker contracts passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
