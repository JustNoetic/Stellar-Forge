#pragma once

#include "ephemeris/simulation_runner.hpp"
#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace sf::comparison {

inline constexpr size_t COMPARISON_SOFT_CAP = 3;
inline constexpr double DEFAULT_COMPARISON_OFFSET_AU = 10.0;
inline constexpr double AUTO_SPACING_MARGIN = 1.25;

double system_extent_au(const ephemeris::SystemSnapshot& snapshot);
glm::dvec3 barycentre_au(const ephemeris::SystemSnapshot& snapshot);

struct ComparisonSlot {
    std::string system_name;
    glm::dvec3 offset_au{DEFAULT_COMPARISON_OFFSET_AU, 0.0, 0.0};
    bool primary_lights_slot = false;
    bool slot_lights_primary = false;
    std::unique_ptr<ephemeris::SimulationRunner> runner;

    const ephemeris::SystemSnapshot& snapshot() const noexcept { return runner->snapshot(); }
    const ephemeris::SimulationClock& clock() const noexcept { return runner->clock(); }
};

class ComparisonSet {
public:
    void set_threaded(bool enabled);
    void set_force_keplerian(bool enabled) noexcept { force_keplerian_ = enabled; }
    bool force_keplerian() const noexcept { return force_keplerian_; }

    CommandResult add(std::string system_name, ephemeris::LoadedSystem loaded, const ephemeris::SimulationClock& primary);
    void remove(size_t slot);
    void clear();
    CommandResult swap_primary(size_t slot, ephemeris::SimulationRunner& primary);
    void auto_space(const ephemeris::SystemSnapshot& primary, double margin = AUTO_SPACING_MARGIN);
    void set_overlay(bool enabled) noexcept { overlay_ = enabled; }
    bool overlay() const noexcept { return overlay_; }
    uint32_t primary_lit_by() const noexcept;
    uint32_t slot_lit_by(size_t slot) const;
    glm::dvec3 placement_au(size_t slot) const;

    bool empty() const noexcept { return slots_.empty(); }
    size_t size() const noexcept { return slots_.size(); }
    ComparisonSlot& slot(size_t index) { return slots_.at(index); }
    const ComparisonSlot& slot(size_t index) const { return slots_.at(index); }

    template<class Command> CommandResult broadcast(const Command& command) {
        CommandResult combined;
        for (auto& slot : slots_) combined = merge(slot, combined, slot.runner->apply(command));
        return combined;
    }
    CommandResult synchronize(const ephemeris::SimulationClock& primary);
    CommandResult jump(units::Years target);
    CommandResult advance_async(units::Seconds elapsed);
    CommandResult wait();

    double max_time_skew_years(units::Years primary) const;
    static double skew_tolerance_years(const ephemeris::SimulationClock& primary, units::Seconds frame_dt);

private:
    static CommandResult merge(const ComparisonSlot& slot, CommandResult combined, CommandResult result);

    std::vector<ComparisonSlot> slots_;
    bool threaded_ = true;
    bool force_keplerian_ = true;
    bool overlay_ = false;
};

}
