#include "benchmark/benchmark.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <numeric>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>
#include <cstring>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace sf::benchmark {
namespace {

uint32_t integer(std::string_view value, bool allow_zero = false) {
    uint32_t number = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && number == 0))
        throw std::runtime_error("Expected a positive integer: " + std::string(value));
    return number;
}

double positive(std::string_view value) {
    double number = 0.0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
    if (error != std::errc{} || end != value.data() + value.size() || !std::isfinite(number) || number <= 0.0)
        throw std::runtime_error("Expected a finite positive number: " + std::string(value));
    return number;
}

}

LaunchOptions parse_options(int argc, char** argv) {
    LaunchOptions launch;
    bool benchmark_option = false;
    bool simulation_option = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        auto value = [&]() -> std::string_view {
            if (++i >= argc) throw std::runtime_error("Missing value for " + std::string(arg));
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") { launch.help = true; continue; }
        if (arg == "--self-test") { launch.self_test = true; continue; }
        if (arg == "--simulation-thread" || arg == "--no-simulation-thread") {
            launch.simulation_thread = arg == "--simulation-thread";
            simulation_option = true;
            continue;
        }
        if (arg == "--benchmark") { launch.benchmark.enabled = true; continue; }
        benchmark_option = true;
        auto& o = launch.benchmark;
        if (arg == "--benchmark-hidden") o.hidden = true;
        else if (arg == "--benchmark-ui") o.ui = true;
        else if (arg == "--benchmark-detailed") o.detailed = true;
        else if (arg == "--benchmark-variant") {
            o.variant = value();
            const std::set<std::string_view> variants{"baseline", "no-atmosphere", "no-aerial", "no-shadows", "no-refraction", "no-secondary",
                "no-planetshine", "no-ringshine", "high-exposure", "no-orbits", "no-gaia", "smooth", "no-textures", "no-normals", "no-rings", "no-ring-shadows", "cached-shadow-weights", "native-shadow-weights", "fast-ring-shadows", "analytic-shadows", "jittered-shadows", "no-clouds", "legacy-clouds", "no-bloom", "fast-bloom", "terrain-detail"};
            if (o.variant.empty()) throw std::runtime_error("Unknown benchmark variant: " + o.variant);
            for (size_t start = 0; start <= o.variant.size();) {
                size_t plus = o.variant.find('+', start);
                std::string_view token = std::string_view(o.variant).substr(start, plus == std::string::npos ? std::string_view::npos : plus - start);
                if (token.empty() || !variants.contains(token))
                    throw std::runtime_error("Unknown benchmark variant: " + o.variant);
                if (plus == std::string::npos) break;
                start = plus + 1;
            }
        }
        else if (arg == "--benchmark-width") o.width = integer(value());
        else if (arg == "--benchmark-height") o.height = integer(value());
        else if (arg == "--benchmark-warmup") o.warmup = integer(value(), true);
        else if (arg == "--benchmark-frames") o.frames = integer(value());
        else if (arg == "--benchmark-repeats") o.repeats = integer(value());
        else if (arg == "--benchmark-step") o.step_seconds = positive(value());
        else if (arg == "--benchmark-warp") o.warp = positive(value());
        else if (arg == "--benchmark-simulation") {
            o.simulation_mode = value();
            if (o.simulation_mode != "inline" && o.simulation_mode != "worker" && o.simulation_mode != "pipeline")
                throw std::runtime_error("Unknown simulation mode: " + o.simulation_mode);
        }
        else if (arg == "--benchmark-output") { o.output = value(); if (o.output.empty()) throw std::runtime_error("Empty output directory."); }
        else if (arg == "--benchmark-capture") { o.capture = value(); if (o.capture.empty()) throw std::runtime_error("Empty capture path."); }
        else if (arg == "--benchmark-scene") {
            auto scene = value();
            if (scene == "all") o.scenes = {"overview", "approach", "analytical", "nbody"};
            else if (scene == "terrain-navigation") o.scenes = {"earth-terrain-still", "earth-terrain-walk", "earth-terrain-fly", "earth-terrain-turn"};
            else if (scene == "audit") o.scenes = {"overview", "analytical", "nbody", "approach", "earth-ground",
                "earth-ground-rings", "earth-ground-turn", "earth-ground-rings-turn", "earth-disc", "earth-full",
                "earth-limb", "earth-full-rings", "earth-eclipse", "saturn", "earth-ground-live", "earth-ground-rings-live", "earth-full-live"};
            else if (scene == "overview" || scene == "approach" || scene == "analytical" || scene == "nbody" || scene == "spice"
                     || scene == "earth-full" || scene == "earth-eclipse" || scene == "earth-ground" || scene == "earth-ground-rings"
                     || scene == "earth-ground-turn" || scene == "earth-ground-rings-turn" || scene == "earth-disc"
                     || scene == "earth-limb" || scene == "earth-full-rings" || scene == "saturn" || scene == "saturn-midnight" || scene == "earth-ground-live"
                     || scene == "earth-ground-rings-live" || scene == "earth-full-live" || scene == "earth-terrain-still"
                     || scene == "earth-terrain-walk" || scene == "earth-terrain-fly" || scene == "earth-terrain-turn")
                o.scenes = {std::string(scene)};
            else throw std::runtime_error("Unknown benchmark scene: " + std::string(scene));
        } else throw std::runtime_error("Unknown argument: " + std::string(arg));
    }
    const auto& o = launch.benchmark;
    if (benchmark_option && !o.enabled) throw std::runtime_error("Benchmark options require --benchmark.");
    if (o.enabled && launch.self_test) throw std::runtime_error("--benchmark and --self-test are mutually exclusive.");
    if (o.enabled && simulation_option) throw std::runtime_error("Use --benchmark-simulation inline|worker|pipeline for benchmarks.");
    if (o.width > 16384 || o.height > 16384 || o.frames > 1000000 || o.warmup > 1000000 || o.repeats > 100)
        throw std::runtime_error("Benchmark dimensions/counts exceed supported limits.");
    if (!std::isfinite(o.step_seconds * o.warp)) throw std::runtime_error("Simulation step overflows.");
    return launch;
}

std::string_view help() {
    return R"(StellarForge [--self-test | --benchmark] [options]
  --simulation-thread      Asynchronous simulation worker (interactive default)
  --no-simulation-thread   Run simulation on the main thread for comparison/debugging
  --benchmark-simulation inline|worker|pipeline  Inline, wait per step, or overlap rendering (inline)
  --benchmark-scene all|overview|approach|analytical|nbody|spice|earth-full|earth-eclipse
                    audit|earth-ground|earth-ground-rings|earth-ground-turn|earth-ground-rings-turn
                    earth-disc|earth-limb|earth-full-rings|saturn|saturn-midnight
                    earth-ground-live|earth-ground-rings-live|earth-full-live
                    terrain-navigation|earth-terrain-still|earth-terrain-walk|earth-terrain-fly|earth-terrain-turn
  --benchmark-variant baseline|no-atmosphere|no-aerial|no-shadows|no-refraction|no-secondary|no-orbits
                      no-planetshine|no-ringshine|high-exposure (x10^4, image checks)|no-gaia|smooth|no-textures|no-normals|no-rings|no-ring-shadows
                      cached-shadow-weights|native-shadow-weights|fast-ring-shadows|no-clouds|legacy-clouds|no-bloom|fast-bloom
                      terrain-detail (2-pixel geometry error; other defaults retained)
  --benchmark-width N       Framebuffer width (1920)
  --benchmark-height N      Framebuffer height (1080)
  --benchmark-warmup N      Unmeasured frames per run (300; zero allowed)
  --benchmark-frames N      Measured frames per run (1200)
  --benchmark-repeats N     Repetitions per scene (3)
  --benchmark-step SECONDS  Fixed simulation/camera delta (1/60)
  --benchmark-warp FACTOR   Simulation warp (1)
  --benchmark-output PATH   New report directory (default: exports/benchmarks/<UTC timestamp>)
  --benchmark-capture FILE  Save the last frame's linear HDR scene (before halo and tone mapping) as PFM
  --benchmark-hidden        Hidden window; still requires Vulkan/window system
  --benchmark-ui            Include the fixed, disabled application workspace
  --benchmark-detailed      Also export raw frames and per-run statistics
SPICE is opt-in and requires the configured local runtime/kernels.
)";
}

nlohmann::json statistics(std::vector<double> values) {
    if (values.empty()) return nullptr;
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p) {
        double offset = p * static_cast<double>(values.size() - 1);
        auto lo = static_cast<size_t>(offset);
        auto hi = std::min(lo + 1, values.size() - 1);
        return std::lerp(values[lo], values[hi], offset - static_cast<double>(lo));
    };
    return {{"count", values.size()}, {"mean", std::accumulate(values.begin(), values.end(), 0.0) / values.size()},
            {"median", percentile(0.5)}, {"p95", percentile(0.95)}, {"p99", percentile(0.99)},
            {"min", values.front()}, {"max", values.back()}};
}

std::map<std::string, double> metrics(const FrameSample& s) {
    std::map<std::string, double> result{
        {"frame_ms", s.frame_ms}, {"cpu_simulation_ms", s.cpu_simulation_ms}, {"cpu_camera_ms", s.cpu_camera_ms},
        {"worker_advance_ms", s.worker_advance_ms}, {"snapshot_copy_ms", s.snapshot_copy_ms},
        {"cpu_work_ms", s.cpu_simulation_ms + s.cpu_camera_ms + s.cpu_record_ms},
        {"wait_ms", s.host.fence_wait_ms + s.host.acquire_wait_ms + s.host.present_ms},
        {"cpu_record_ms", s.cpu_record_ms}, {"cpu_ui_ms", s.cpu_ui_ms}, {"cpu_celestial_ms", s.cpu_celestial_ms},
        {"cpu_gaia_ms", s.cpu_gaia_ms}, {"cpu_orbits_ms", s.cpu_orbits_ms}, {"fence_wait_ms", s.host.fence_wait_ms},
        {"acquire_wait_ms", s.host.acquire_wait_ms}, {"submit_ms", s.host.submit_ms}, {"present_ms", s.host.present_ms}
    };
    for (const auto& [name, ms] : s.gpu) result[name + "_ms"] = ms;
    result.insert(s.cpu.begin(), s.cpu.end());
    return result;
}

nlohmann::json summarize(const std::vector<FrameSample>& frames) {
    std::map<std::string, std::vector<double>> columns;
    for (const auto& frame : frames)
        for (const auto& [key, ms] : metrics(frame)) columns[key].push_back(ms);
    nlohmann::json result = nlohmann::json::object();
    for (auto& [key, values] : columns) result[key] = statistics(std::move(values));
    return result;
}

std::string fingerprint(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot fingerprint " + path.string());
    uint64_t hash = 14695981039346656037ull;
    std::array<char, 65536> buffer;
    while (stream) {
        stream.read(buffer.data(), buffer.size());
        for (std::streamsize i = 0; i < stream.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
            hash *= 1099511628211ull;
        }
    }
    if (!stream.eof()) throw std::runtime_error("Cannot read " + path.string());
    return std::format("fnv1a64:{:016x}", hash);
}

nlohmann::json host_info() {
    const char* cpu = std::getenv("PROCESSOR_IDENTIFIER");
    std::string brand = cpu ? cpu : "unavailable";
#if (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
    std::array<int, 4> registers{};
    auto query = [&](unsigned leaf) {
#if defined(__GNUC__)
        unsigned a, b, c, d;
        __cpuid(leaf, a, b, c, d);
        registers = {static_cast<int>(a), static_cast<int>(b), static_cast<int>(c), static_cast<int>(d)};
#else
        __cpuid(registers.data(), static_cast<int>(leaf));
#endif
    };
    query(0x80000000);
    if (static_cast<unsigned>(registers[0]) >= 0x80000004) {
        std::array<char, 49> text{};
        for (unsigned i = 0; i < 3; ++i) {
            query(0x80000002 + i);
            std::memcpy(text.data() + i * 16, registers.data(), 16);
        }
        brand = text.data();
    }
#endif
    return {{"cpu_name", brand}, {"cpu_identifier", cpu ? cpu : "unavailable"}, {"logical_processors", std::thread::hardware_concurrency()},
#ifdef _WIN32
            {"os", "Windows"}
#else
            {"os", "Unix"}
#endif
    };
}

std::string summary_text(const nlohmann::json& report) {
    const auto& config = report.at("configuration");
    std::string text = std::format("Stellar Forge benchmark - average timings\n{}x{}, {} frames x {} repeats per scene; {} warmup frames excluded\n\n",
        config.at("width").get<uint32_t>(), config.at("height").get<uint32_t>(), config.at("frames").get<uint32_t>(),
        config.at("repeats").get<uint32_t>(), config.at("warmup").get<uint32_t>());
    auto timing = [](const nlohmann::json& averages, const char* name) {
        return averages.contains(name) ? std::format("{:.3f}", averages.at(name).get<double>()) : std::string("n/a");
    };
    text += std::format("{:<12} {:>10} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "Scene", "Frame ms", "FPS", "CPU ms", "GPU ms", "Sim ms", "Record ms", "Wait ms");
    for (const auto& scene : report.at("scenes")) {
        const auto& a = scene.at("averages_ms");
        double frame = a.at("frame_ms").get<double>();
        text += std::format("{:<12} {:>10} {:>9.1f} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
            scene.at("scene").get<std::string>(), timing(a, "frame_ms"), frame > 0 ? 1000.0 / frame : 0.0,
            timing(a, "cpu_work_ms"), timing(a, "gpu_frame_ms"), timing(a, "cpu_simulation_ms"), timing(a, "cpu_record_ms"), timing(a, "wait_ms"));
    }
    text += "\nGPU pass averages (ms)\n";
    text += std::format("{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}\n", "Scene", "Celestial", "Gaia", "Orbits", "Composite", "UI");
    for (const auto& scene : report.at("scenes")) {
        const auto& a = scene.at("averages_ms");
        text += std::format("{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}\n", scene.at("scene").get<std::string>(),
            timing(a, "gpu_celestial_ms"), timing(a, "gpu_gaia_ms"), timing(a, "gpu_orbits_ms"), timing(a, "gpu_post_composite_ms"), timing(a, "gpu_ui_ms"));
    }
    text += "\nCPU = simulation + camera + render recording (wall time). Wait = fence + acquire + present.\nCPU and GPU overlap; do not add them. FPS = 1000 / average frame ms.\n";
    return text;
}

void write_report(const Options& o, const nlohmann::json& metadata, const std::vector<Run>& runs) {
    nlohmann::json report{{"schema_version", 2}, {"metadata", metadata},
        {"configuration", {{"fixture_version", 1}, {"width", o.width}, {"height", o.height}, {"warmup", o.warmup},
            {"frames", o.frames}, {"repeats", o.repeats}, {"step_seconds", o.step_seconds}, {"warp", o.warp},
            {"ui", o.ui}, {"hidden", o.hidden}, {"scenes", o.scenes}, {"simulation_mode", o.simulation_mode}, {"variant", o.variant}}}, {"scenes", nlohmann::json::array()}};
    for (const auto& name : o.scenes) {
        std::map<std::string, double> totals;
        std::map<std::string, uint64_t> counts;
        uint64_t samples = 0;
        nlohmann::json fixture;
        for (const auto& run : runs) {
            if (run.scene != name) continue;
            if (fixture.is_null()) fixture = run.fixture;
            for (const auto& frame : run.frames) {
                ++samples;
                for (const auto& [key, value] : metrics(frame)) {
                    totals[key] += value;
                    ++counts[key];
                }
            }
        }
        if (samples == 0) continue;
        for (auto& [key, total] : totals) total /= static_cast<double>(counts.at(key));
        report["scenes"].push_back({{"scene", name}, {"samples", samples}, {"fixture", fixture},
            {"averages_ms", totals}, {"timing_counts", counts}});
    }
    auto text = summary_text(report);
    if (!paths::write_binary_file(o.output / "report.txt", text.data(), text.size()))
        throw std::runtime_error("Cannot write benchmark summary.");
    std::cout << '\n' << text << '\n';
    if (!o.detailed) {
        if (!paths::write_json(o.output / "report.json", report)) throw std::runtime_error("Cannot write benchmark report.");
        return;
    }
    report["runs"] = nlohmann::json::array();
    std::set<std::string> columns;
    for (const auto& run : runs)
        for (const auto& frame : run.frames)
            for (const auto& [name, ms] : metrics(frame)) columns.insert(name);
    std::ofstream csv(o.output / "frames.csv", std::ios::binary);
    csv.exceptions(std::ios::badbit | std::ios::failbit);
    csv.precision(17);
    csv << "scene,repetition,index,frame_id,simulation_years,eye_x_au,eye_y_au,eye_z_au";
    for (const auto& column : columns) csv << ',' << column;
    csv << '\n';
    for (const auto& run : runs) {
        nlohmann::json entry{{"scene", run.scene}, {"repetition", run.repetition}, {"fixture", run.fixture},
            {"summary", summarize(run.frames)}, {"frames", nlohmann::json::array()}};
        for (const auto& frame : run.frames) {
            auto times = metrics(frame);
            entry["frames"].push_back({{"index", frame.index}, {"frame_id", frame.frame_id},
                {"simulation_years", frame.simulation_years}, {"eye_au", frame.eye_au}, {"timings", times}, {"counters", frame.counters}});
            csv << run.scene << ',' << run.repetition << ',' << frame.index << ',' << frame.frame_id << ',' << frame.simulation_years;
            for (double coordinate : frame.eye_au) csv << ',' << coordinate;
            for (const auto& column : columns) {
                csv << ',';
                if (auto it = times.find(column); it != times.end()) csv << it->second;
            }
            csv << '\n';
        }
        report["runs"].push_back(std::move(entry));
    }
    csv.close();
    if (!paths::write_json(o.output / "report.json", report)) throw std::runtime_error("Cannot write benchmark report.");
}

}
