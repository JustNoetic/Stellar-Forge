#include "comparison/system_comparison.hpp"
#include <algorithm>
#include <cmath>

namespace sf::comparison {

using namespace ephemeris;

namespace {

CommandResult align_clock(SimulationRunner& runner, const SimulationClock& primary) {
    const auto& clock = runner.clock();
    CommandResult result;
    if (clock.warp != primary.warp) result = runner.apply(SetTimeWarp{primary.warp});
    if (result && clock.direction != primary.direction) result = runner.apply(SetTimeDirection{primary.direction});
    if (result && clock.paused != primary.paused) result = runner.apply(SetPaused{primary.paused});
    return result;
}

}

double system_extent_au(const SystemSnapshot& snapshot) {
    const auto& bodies = snapshot.definition().bodies;
    std::vector<double> reach(bodies.size(), -1.0);
    auto body_reach = [&](auto& self, size_t index, int depth) -> double {
        if (reach[index] >= 0.0) return reach[index];
        const auto& body = bodies[index];
        double apoapsis = body.semi_major_au * (1.0 + std::clamp(body.eccentricity, 0.0, 1.0));
        double parent = body.parent_idx >= 0 && size_t(body.parent_idx) < bodies.size() && depth < 64
            ? self(self, size_t(body.parent_idx), depth + 1) : 0.0;
        return reach[index] = parent + (std::isfinite(apoapsis) ? apoapsis : 0.0);
    };
    glm::dvec3 center = snapshot.empty() || snapshot.star_index() < 0 || size_t(snapshot.star_index()) >= snapshot.size()
        ? glm::dvec3{0.0} : snapshot.body(size_t(snapshot.star_index())).motion.position.value;
    double extent = 0.0;
    for (size_t i = 0; i < bodies.size(); ++i) {
        double current = glm::length(snapshot.body(i).motion.position.value - center);
        extent = std::max({extent, body_reach(body_reach, i, 0), std::isfinite(current) ? current : 0.0});
    }
    return extent;
}

glm::dvec3 barycentre_au(const SystemSnapshot& snapshot) {
    double mass = 0.0;
    glm::dvec3 weighted{0.0};
    for (size_t i = 0; i < snapshot.size(); ++i) {
        double m = std::max(snapshot.body(i).definition.mass_msun, 0.0);
        mass += m;
        weighted += m * snapshot.body(i).motion.position.value;
    }
    return mass > 0.0 && std::isfinite(mass) ? weighted / mass : glm::dvec3{0.0};
}

uint32_t ComparisonSet::primary_lit_by() const noexcept {
    uint32_t mask = 0;
    for (size_t i = 0; i < slots_.size() && !overlay_; ++i)
        if (slots_[i].slot_lights_primary) mask |= 1u << (i + 1);
    return mask;
}

uint32_t ComparisonSet::slot_lit_by(size_t slot) const {
    return !overlay_ && slots_.at(slot).primary_lights_slot ? 1u : 0u;
}

glm::dvec3 ComparisonSet::placement_au(size_t slot) const {
    return overlay_ ? -barycentre_au(slots_.at(slot).snapshot()) : slots_.at(slot).offset_au;
}

void ComparisonSet::set_threaded(bool enabled) {
    threaded_ = enabled;
    for (auto& slot : slots_) slot.runner->set_threaded(enabled);
}

CommandResult ComparisonSet::merge(const ComparisonSlot& slot, CommandResult combined, CommandResult result) {
    if (!combined) return combined;
    if (!result) return {false, combined.changed || result.changed, slot.system_name + ": " + result.error};
    combined.changed = combined.changed || result.changed;
    return combined;
}

CommandResult ComparisonSet::add(std::string system_name, LoadedSystem loaded, const SimulationClock& primary) {
    ComparisonSlot slot;
    slot.system_name = std::move(system_name);
    slot.offset_au = {DEFAULT_COMPARISON_OFFSET_AU * double(slots_.size() + 1), 0.0, 0.0};
    slot.runner = std::make_unique<SimulationRunner>();
    auto result = slot.runner->replace_system(std::move(loaded));
    if (result && force_keplerian_) result = slot.runner->apply(SetMotionModel{MotionModel::Keplerian});
    if (result) result = align_clock(*slot.runner, primary);
    if (result && slot.clock().time.value != primary.time.value) result = slot.runner->apply(JumpToTime{primary.time});
    if (!result) return {false, false, slot.system_name + ": " + result.error};
    slot.runner->set_threaded(threaded_);
    slots_.push_back(std::move(slot));
    return {true, true, {}};
}

void ComparisonSet::remove(size_t slot) {
    if (slot < slots_.size()) slots_.erase(slots_.begin() + std::ptrdiff_t(slot));
}

void ComparisonSet::clear() {
    slots_.clear();
}

CommandResult ComparisonSet::swap_primary(size_t slot, SimulationRunner& primary) {
    if (slot >= slots_.size()) return {false, false, "No compared system in that slot."};
    auto& target = slots_[slot];
    if (primary.is_rendering_timeline() || target.runner->is_rendering_timeline())
        return {false, false, "Cancel the timeline render before swapping systems."};
    std::string primary_name = primary.snapshot().name();
    primary.swap_simulation(*target.runner);
    target.system_name = std::move(primary_name);
    std::swap(target.primary_lights_slot, target.slot_lights_primary);
    return {true, true, {}};
}

void ComparisonSet::auto_space(const SystemSnapshot& primary, double margin) {
    double previous_x = 0.0, previous_extent = system_extent_au(primary);
    for (auto& slot : slots_) {
        double extent = system_extent_au(slot.snapshot());
        previous_x += std::max(1.0, margin * (previous_extent + extent));
        slot.offset_au = {previous_x, 0.0, 0.0};
        previous_extent = extent;
    }
}

CommandResult ComparisonSet::synchronize(const SimulationClock& primary) {
    CommandResult combined;
    for (auto& slot : slots_) combined = merge(slot, combined, align_clock(*slot.runner, primary));
    return combined;
}

CommandResult ComparisonSet::jump(units::Years target) {
    CommandResult combined;
    for (auto& slot : slots_) {
        if (slot.clock().time.value == target.value) continue;
        combined = merge(slot, combined, slot.runner->apply(JumpToTime{target}));
    }
    return combined;
}

CommandResult ComparisonSet::advance_async(units::Seconds elapsed) {
    CommandResult combined;
    for (auto& slot : slots_) combined = merge(slot, combined, slot.runner->advance_async(elapsed));
    return combined;
}

CommandResult ComparisonSet::wait() {
    CommandResult combined;
    for (auto& slot : slots_) combined = merge(slot, combined, slot.runner->wait());
    return combined;
}

double ComparisonSet::max_time_skew_years(units::Years primary) const {
    double skew = 0.0;
    for (const auto& slot : slots_) skew = std::max(skew, std::abs(slot.clock().time.value - primary.value));
    return skew;
}

double ComparisonSet::skew_tolerance_years(const SimulationClock& primary, units::Seconds frame_dt) {
    return units::to_years(units::Seconds{std::max(1.0, 4.0 * frame_dt.value * primary.warp)}).value;
}

}
