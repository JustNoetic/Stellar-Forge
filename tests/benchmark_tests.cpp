#include "benchmark/benchmark.hpp"
#include "core/path_utils.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

sf::benchmark::LaunchOptions parse(std::initializer_list<const char*> args) {
    std::vector<char*> argv;
    for (auto arg : args) argv.push_back(const_cast<char*>(arg));
    return sf::benchmark::parse_options(static_cast<int>(argv.size()), argv.data());
}

void reject(std::initializer_list<const char*> args) {
    bool rejected = false;
    try { parse(args); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "Invalid CLI accepted");
}

}

int main(int argc, char** argv) {
    try {
        auto options = parse({"app", "--benchmark", "--benchmark-scene", "spice", "--benchmark-warmup", "0"});
        check(options.benchmark.enabled && options.benchmark.warmup == 0 && options.benchmark.scenes == std::vector<std::string>{"spice"}, "CLI values");
        reject({"app", "--benchmark", "--self-test"});
        reject({"app", "--benchmark", "--benchmark-frames", "0"});
        reject({"app", "--benchmark", "--benchmark-step", "nan"});
        reject({"app", "--benchmark", "--benchmark-warp", "inf"});
        reject({"app", "--benchmark", "--benchmark-width", "-1"});
        reject({"app", "--benchmark", "--benchmark-frames", "12x"});
        reject({"app", "--benchmark", "--benchmark-output"});
        reject({"app", "--benchmark-scene", "overview"});
        reject({"app", "--benchmark", "--benchmark-scene", "missing"});
        check(parse({"app", "--benchmark", "--benchmark-scene", "earth-full"}).benchmark.scenes[0] == "earth-full", "Earth close-up scene");
        check(parse({"app", "--benchmark", "--benchmark-scene", "earth-eclipse"}).benchmark.scenes[0] == "earth-eclipse", "Earth eclipse scene");
        check(parse({"app", "--simulation-thread"}).simulation_thread, "Interactive simulation worker flag");
        check(parse({"app"}).simulation_thread, "Interactive simulation worker is not the default");
        check(!parse({"app", "--no-simulation-thread"}).simulation_thread, "Inline simulation fallback flag");
        check(parse({"app", "--benchmark"}).benchmark.simulation_mode == "inline", "Benchmark baseline default changed");
        check(parse({"app", "--benchmark", "--benchmark-simulation", "pipeline"}).benchmark.simulation_mode == "pipeline", "Pipeline mode");
        reject({"app", "--benchmark", "--benchmark-simulation", "missing"});
        reject({"app", "--benchmark", "--simulation-thread"});
        reject({"app", "--benchmark", "--no-simulation-thread"});
        auto stats = sf::benchmark::statistics({9, 1, 5, 3});
        check(stats["count"] == 4 && stats["median"] == 4 && stats["mean"] == 4.5, "Statistics center");
        check(std::abs(stats["p95"].get<double>() - 8.4) < 1e-12, "Statistics percentile interpolation");
        check(sf::benchmark::statistics({}).is_null(), "Empty statistics");
        check(sf::benchmark::statistics({7})["p99"] == 7, "Singleton statistics");
        sf::benchmark::Run run;
        run.scene = "overview";
        run.frames.resize(2);
        run.frames[0].gpu["gpu_frame"] = 2.0;
        run.frames[0].frame_id = 7;
        run.frames[1].frame_id = 8;
        run.frames[1].index = 1;
        auto summary = sf::benchmark::summarize(run.frames);
        check(summary["gpu_frame_ms"]["count"] == 1 && summary["gpu_frame_ms"]["mean"] == 2, "Missing GPU samples were zero-filled");
        check(summary["frame_ms"]["count"] == 2, "CPU sample count");
        check(argc == 2, "Expected scratch output directory");
        options.benchmark.scenes = {"overview"};
        options.benchmark.output = std::filesystem::path(argv[1]) / "compact";
        std::filesystem::create_directories(options.benchmark.output);
        sf::benchmark::write_report(options.benchmark, {{"test", true}}, {run});
        auto report = sf::paths::read_json(options.benchmark.output / "report.json");
        check(report && !report->contains("runs") && (*report)["scenes"][0]["averages_ms"]["gpu_frame_ms"] == 2, "Compact scene averages");
        check((*report)["scenes"][0]["timing_counts"]["gpu_frame_ms"] == 1, "Compact missing GPU count");
        check(!std::filesystem::exists(options.benchmark.output / "frames.csv"), "Default exported raw CSV");
        auto text = sf::paths::read_text_file(options.benchmark.output / "report.txt");
        check(text && text->find("average timings") != std::string::npos && text->find("overview") != std::string::npos, "Readable summary");
        options.benchmark.detailed = true;
        options.benchmark.output = std::filesystem::path(argv[1]) / "detailed";
        std::filesystem::create_directories(options.benchmark.output);
        sf::benchmark::write_report(options.benchmark, {{"test", true}}, {run});
        report = sf::paths::read_json(options.benchmark.output / "report.json");
        check(report && (*report)["runs"][0]["frames"][1]["frame_id"] == 8, "JSON frame identity");
        check(!(*report)["runs"][0]["frames"][1]["timings"].contains("gpu_frame_ms"), "JSON missing GPU value");
        auto csv = sf::paths::read_text_file(options.benchmark.output / "frames.csv");
        check(csv && csv->find("gpu_frame_ms") != std::string::npos, "CSV columns");
        std::cout << "Benchmark contracts passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
