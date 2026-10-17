#pragma once

#include "core/command_result.hpp"
#include "core/units.hpp"
#include "ephemeris/system_manager.hpp"
#include "ephemeris/reference_hierarchy.hpp"
#include "ephemeris/spice_ephemeris.hpp"
#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sf::ephemeris {

struct TimelineRecording {
    units::Years start_time{};
    units::Years end_time{};
    MotionModel source_model = MotionModel::NBody;
    std::vector<units::Years> times;
    std::vector<BodyMotion> motions;

    bool empty() const noexcept { return times.empty(); }
    std::size_t num_steps() const noexcept { return times.size(); }
    std::size_t num_bodies() const noexcept { return times.empty() ? 0 : motions.size() / times.size(); }
};

struct SpkExportOptions {
    int spk_type = 2;
    int cheb_degree = 13;
    double cheb_tol_km = 1.0;
    int hermite_degree = 3;
    bool use_hierarchical_center = true;
    std::function<void(float progress, std::string_view status)> progress_callback = nullptr;
    const std::atomic<bool>* cancel_flag = nullptr;
};

struct SpkExportResult {
    bool success = false;
    std::string error;
    std::filesystem::path bsp_path;
    std::filesystem::path json_path;
    nlohmann::json metadata;
};

class SpkExporter {
public:
    static SpkExportResult export_timeline(
        const std::filesystem::path& destination_bsp,
        const TimelineRecording& recording,
        const SystemDefinition& system_def,
        const ReferenceHierarchy& hierarchy,
        std::shared_ptr<SpiceEphemeris> spice,
        const SpkExportOptions& options = {}
    );
};

} // namespace sf::ephemeris
