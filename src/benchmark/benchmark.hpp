#pragma once

#include "core/timing.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace sf::benchmark {

struct Options {
    bool enabled = false;
    bool hidden = false;
    bool ui = false;
    bool detailed = false;
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t warmup = 300;
    uint32_t frames = 1200;
    uint32_t repeats = 3;
    double step_seconds = 1.0 / 60.0;
    double warp = 1.0;
    std::string simulation_mode = "inline";
    std::string variant = "baseline";
    std::vector<std::string> scenes{"overview", "approach", "analytical", "nbody"};
    std::filesystem::path output;
};

struct LaunchOptions {
    Options benchmark;
    bool self_test = false;
    bool simulation_thread = true;
    bool help = false;
};

struct FrameSample {
    uint64_t frame_id = 0;
    uint32_t index = 0;
    double simulation_years = 0.0;
    std::array<double, 3> eye_au{};
    double frame_ms = 0.0;
    double cpu_simulation_ms = 0.0;
    double cpu_camera_ms = 0.0;
    double worker_advance_ms = 0.0;
    double snapshot_copy_ms = 0.0;
    double cpu_record_ms = 0.0;
    double cpu_ui_ms = 0.0;
    double cpu_celestial_ms = 0.0;
    double cpu_gaia_ms = 0.0;
    double cpu_orbits_ms = 0.0;
    FrameHostTimings host;
    std::map<std::string, double> cpu;
    std::map<std::string, double> counters;
    std::map<std::string, double> gpu;
};

struct Run {
    std::string scene;
    uint32_t repetition = 0;
    nlohmann::json fixture;
    std::vector<FrameSample> frames;
};

LaunchOptions parse_options(int argc, char** argv);
std::string_view help();
nlohmann::json statistics(std::vector<double> values);
std::map<std::string, double> metrics(const FrameSample& sample);
nlohmann::json summarize(const std::vector<FrameSample>& frames);
std::string fingerprint(const std::filesystem::path& path);
nlohmann::json host_info();
std::string summary_text(const nlohmann::json& report);
void write_report(const Options& options, const nlohmann::json& metadata, const std::vector<Run>& runs);

}
