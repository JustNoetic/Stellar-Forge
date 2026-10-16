#include "ephemeris/simulation.hpp"
#include "ephemeris/orbit_curves.hpp"
#include "ui/inspector_state.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sf;
using namespace sf::ephemeris;

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void add(LoadedSystem& system, const char* name, int parent, double mass, glm::dvec3 position,
         glm::dvec3 velocity = glm::dvec3(0.0), bool spacecraft = false) {
    BodyDefinition body;
    body.name = name;
    body.parent_idx = parent;
    if (parent >= 0) body.parent_id = system.definition.bodies[parent].name;
    body.mass_msun = mass;
    body.is_spacecraft = spacecraft;
    system.definition.bodies.push_back(body);
    system.motions.push_back({EclipticPosition{position}, EclipticVelocity{velocity}});
}

LoadedSystem encounter() {
    LoadedSystem system;
    system.definition.motion_model = MotionModel::NBody;
    add(system, "Sun", -1, 1.0, {0.0, 0.0, 0.0});
    add(system, "Probe", 0, 0.0, {1.0026, 0.0, 0.0}, {0.0, 6.6, 0.0}, true);
    add(system, "Earth", 0, 3e-6, {1.0, 0.0, 0.0}, {0.0, 6.28, 0.0});
    add(system, "Moon", 2, 3e-8, {1.0025, 0.0, 0.0}, {0.0, 6.48, 0.0});
    add(system, "Other", 0, 1e-5, {5.0, 0.0, 0.0}, {0.0, 3.0, 0.0});
    system.definition.bodies[2].radius_rsun = 0.009;
    system.definition.bodies[3].radius_rsun = 0.0025;
    return system;
}

void transitions_and_hysteresis() {
    auto system = encounter();
    ReferenceHierarchyTracker tracker;
    ReferenceHierarchy previous;
    auto evaluate = [&] { previous = tracker.evaluate(system.definition, system.motions, previous); };
    evaluate();
    check(previous.parents[1] == 3, "Find a moon parent even when its index follows the probe");
    check(previous.parents[0] == -1 && previous.parents[2] == 0 && previous.parents[3] == 2, "Preserve the nested planetary hierarchy");
    double hill = 0.0025 * std::cbrt(3e-8 / (3.0 * 3e-6));
    auto place = [&](double factor) {
        system.motions[1].position.value.x = system.motions[3].position.value.x + factor * hill;
        evaluate();
    };
    place(1.05);
    check(previous.parents[1] == 3, "Retain the current parent within the exit margin");
    place(1.11);
    check(previous.parents[1] == 2, "Leave the moon for the planet beyond the exit margin");
    place(0.95);
    check(previous.parents[1] == 2, "Do not reenter the moon within the hysteresis band");
    place(0.89);
    check(previous.parents[1] == 3, "Enter the smaller influence region beyond the entry margin");
    system.motions[1].position.value = {1.02, 0.0, 0.0};
    evaluate();
    check(previous.parents[1] == 0, "Escaping the planetary region falls back to the system root");
    system.motions[1].position.value = {5.001, 0.0, 0.0};
    evaluate();
    check(previous.parents[1] == 4, "A large endpoint jump finds a different planet");
    system.motions[2].position.value = {5.002, 0.0, 0.0};
    system.motions[3].position.value = {5.0021, 0.0, 0.0};
    system.motions[1].position.value = {5.00211, 0.0, 0.0};
    evaluate();
    check(previous.parents[2] == 4 && previous.parents[3] == 2 && previous.parents[1] == 3,
          "Resolve a moved subsystem from its new host down in one update");
    system.definition.bodies[2].mass_msun = 1e-5;
    evaluate();
    for (size_t i = 0; i < previous.parents.size(); ++i) {
        int ancestor = static_cast<int>(i);
        size_t depth = 0;
        while (ancestor >= 0 && depth <= previous.parents.size()) {
            ancestor = previous.parents[ancestor];
            ++depth;
        }
        check(depth <= previous.parents.size(), "Equal masses and overlapping regions cannot form a cycle");
    }
}

void snapshot_orbits_and_edits() {
    Simulation simulation;
    check(simulation.replace_system(encounter()).accepted, "Load a dynamic reference fixture");
    auto captured = simulation.snapshot();
    check(captured.reference_parent(1) == 3 && captured.body(1).definition.parent_idx == 0, "Reference parents are separate from stored orbital seeds");
    check(captured.jacobi_hierarchy().parents[1] == 0 && captured.orbit_hierarchy().parents[1] == 3, "Keep the analytical hierarchy separate from orbit references");
    auto coords = extract_jacobi(captured.definition(), captured.motions(), captured.orbit_hierarchy());
    check(std::abs(coords.frames[1].relative_position.x - 0.0001) < 1e-12, "Extract coordinates using a later-index reference parent");
    auto curves = compute_orbit_curves(captured);
    auto probe = std::find_if(curves.begin(), curves.end(), [](const auto& curve) {
        return curve.body_index == 1 && !curve.reflex && !curve.patched;
    });
    check(probe != curves.end(), "Build a local orbit for the dynamically assigned parent");
    check(glm::length(probe->focus.value - captured.body(3).motion.position.value) < 1e-12, "Probe orbit follows the moon's center");
    check(captured.orbit_hierarchy().traversal.size() == captured.size(), "Traversal includes every body after arbitrary reparenting");

    auto moon_properties = BodyPhysicalProperties::from(captured.body(3).definition);
    moon_properties.mass.value *= 1.25;
    check(simulation.apply(ApplyBodyPhysicalProperties{captured.generation(), 3, captured.body(3).revisions.definition, moon_properties}).accepted,
          "Edit the mass of a dynamic reference parent");
    check(simulation.snapshot().orbit_hierarchy().subsystem_mass[3] == moon_properties.mass.value,
          "A mass edit rebuilds dynamic reference masses even without reparenting");
    check(captured.orbit_hierarchy().subsystem_mass[3] == captured.body(3).definition.mass_msun,
          "Captured reference masses remain immutable after edits");

    ui::BodyEditDraft draft;
    draft.reload(captured, 1);
    draft.values.update_orbit = true;
    draft.values.a_au = 0.00012;
    draft.values.e = 0.0;
    draft.values.inc_deg = draft.values.omega_node_deg = draft.values.arg_peri_deg = draft.values.mean_anom_deg = 0.0;
    check(simulation.apply(draft.command()).accepted, "Apply an orbit edit in the displayed reference frame");
    auto relative = simulation.snapshot().body(1).motion.position.value - simulation.snapshot().body(3).motion.position.value;
    check(std::abs(glm::length(relative) - 0.00012) < 1e-12, "Edited orbit is centered on the dynamic parent");
    auto original_command = draft.command();
    draft.reload(simulation.snapshot(), 1);
    draft.values.update_orbit = true;
    draft.values.a_au = 0.02;
    draft.values.e = 0.0;
    draft.values.inc_deg = draft.values.omega_node_deg = draft.values.arg_peri_deg = draft.values.mean_anom_deg = 0.0;
    check(simulation.apply(draft.command()).accepted, "Move the probe outside the planetary region");
    check(simulation.snapshot().reference_parent(1) == 0, "Orbit edits immediately refresh parents even while paused");
    draft.reload(simulation.snapshot(), 1);
    original_command.expected_body_revision = draft.body_revision;
    check(!simulation.apply(original_command).accepted, "Reject an orbit draft prepared for an obsolete reference parent");
    check(captured.reference_parent(1) == 3 && captured.orbit_hierarchy().parents[1] == 3, "Captured snapshots retain the old reference topology");

    auto position = simulation.snapshot().body(1).motion.position.value;
    check(simulation.apply(SetMotionModel{MotionModel::Keplerian}).accepted, "Switch back to analytical propagation");
    check(simulation.snapshot().reference_parent(1) == 0, "Analytical mode uses its declared propagation parents");
    check(simulation.snapshot().body(1).motion.position.value == position, "Switching hierarchy modes does not move a body");
}

void cadence_and_sorting() {
    LoadedSystem system;
    system.definition.motion_model = MotionModel::NBody;
    add(system, "Root", -1, 1e-20, {0.0, 0.0, 0.0});
    add(system, "A", 0, 0.0, {1.0, 0.0, 0.0}, {}, true);
    add(system, "B", 0, 0.0, {1.02, 0.0, 0.0}, {-SECONDS_PER_YEAR, 0.0, 0.0}, true);
    for (double warp : {0.01, 10.0}) {
        auto warped_system = system;
        warped_system.motions[2].position.value.x = 1.0 + 0.02 * warp;
        Simulation warped;
        check(warped.replace_system(std::move(warped_system)).accepted, "Load a warp-independent cadence fixture");
        warped.apply(SetTimeWarp{warp});
        check(warped.advance(units::Seconds{0.03}).accepted, "Cross sibling radii with a non-unit warp");
        check(warped.snapshot().reference_hierarchy().children[0] == std::vector<int>({1, 2}), "Warp does not trigger a refresh before the wall-time deadline");
        check(warped.advance(units::Seconds{0.04}).accepted, "Reach the wall-time deadline with a non-unit warp");
        check(warped.snapshot().reference_hierarchy().children[0] == std::vector<int>({2, 1}), "Slow and fast warp refresh at the same wall-time deadline");
    }
    Simulation simulation;
    check(simulation.replace_system(std::move(system)).accepted, "Load the wall-time cadence fixture");
    auto original = simulation.snapshot();
    auto revision = original.hierarchy_revision();
    check(original.reference_hierarchy().children[0] == std::vector<int>({1, 2}), "Initial children are ordered by current distance");
    check(simulation.advance(units::Seconds{0.03}).accepted, "Advance past a radial crossing before the refresh deadline");
    check(simulation.snapshot().body(2).motion.position.value.x < simulation.snapshot().body(1).motion.position.value.x, "The sibling crossing actually occurred");
    check(simulation.snapshot().hierarchy_revision() == revision, "Subinterval ticks reuse the cached display hierarchy");
    check(simulation.advance(units::Seconds{0.04}).accepted, "Reach the 15 Hz refresh deadline");
    check(simulation.snapshot().reference_hierarchy().children[0] == std::vector<int>({2, 1}), "Refresh sibling ordering at the deadline");
    check(original.reference_hierarchy().children[0] == std::vector<int>({1, 2}), "A display update preserves captured child lists");
    check(&original.definition() == &simulation.snapshot().definition(), "Regular hierarchy updates share physical definitions without copying JSON");
    simulation.apply(SetTimeDirection{-1.0});
    check(simulation.advance(units::Seconds{0.06}).accepted, "Reverse time through the crossing");
    check(simulation.snapshot().reference_hierarchy().children[0] == std::vector<int>({2, 1}), "Reverse motion observes the same wall-time throttle");
    auto state_revision = simulation.snapshot().state_revision();
    simulation.apply(SetPaused{true});
    check(simulation.snapshot().reference_hierarchy().children[0] == std::vector<int>({1, 2}), "Pausing flushes the final ordering");
    check(simulation.snapshot().state_revision() == state_revision, "Flushing references does not advance physics");
    revision = simulation.snapshot().hierarchy_revision();
    simulation.advance(units::Seconds{1.0});
    check(simulation.snapshot().hierarchy_revision() == revision, "Paused frames do not rebuild references");
    check(!simulation.advance(units::Seconds{-1.0}).accepted && simulation.snapshot().hierarchy_revision() == revision, "Rejected ticks preserve hierarchy state");
    simulation.apply(SetPaused{false});
    simulation.apply(SetTimeWarp{0.01});
    check(simulation.advance(units::Seconds{0.1}).accepted, "Slow-motion tick completes");
    auto fresh = ReferenceHierarchyTracker{}.evaluate(simulation.snapshot().definition(), simulation.snapshot().motions(), {});
    check(simulation.snapshot().reference_hierarchy().children == fresh.children, "Refresh cadence is independent of simulation warp");
}

void spatial_scaling() {
    constexpr int planets = 4096;
    LoadedSystem system;
    system.definition.motion_model = MotionModel::NBody;
    add(system, "Root", -1, 1.0, {0.0, 0.0, 0.0});
    for (int i = 0; i < planets; ++i) {
        auto name = "Planet" + std::to_string(i);
        add(system, name.c_str(), 0, 1e-9, {1000.0 + (i % 64) * 20.0, (i / 64) * 20.0, 0.0});
    }
    for (int i = 0; i < planets; ++i) {
        auto name = "Probe" + std::to_string(i);
        add(system, name.c_str(), 0, 0.0, system.motions[i + 1].position.value + glm::dvec3(0.1, 0.0, 0.0), {}, true);
    }
    ReferenceHierarchyTracker tracker;
    auto start = std::chrono::steady_clock::now();
    auto result = tracker.evaluate(system.definition, system.motions, {});
    auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    for (int i = 0; i < planets; ++i) check(result.parents[planets + 1 + i] == i + 1, "Find each separated planetary region through the spatial index");
    check(tracker.candidate_checks() < planets * 4, "Spatial culling avoids all-pairs influence tests in a separated scene");
    std::cout << "8193-body reference update: " << elapsed << " ms, " << tracker.candidate_checks() << " candidate checks\n";
}

}

int main() {
    try {
        transitions_and_hysteresis();
        snapshot_orbits_and_edits();
        cadence_and_sorting();
        spatial_scaling();
        std::cout << "PASS: reference hierarchy transitions, hysteresis, arbitrary parent indices, orbit frames, edits, snapshot isolation, 15 Hz cadence, reverse time and spatial culling\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
