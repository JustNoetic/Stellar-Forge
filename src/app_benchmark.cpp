#include "app.hpp"
#include "benchmark/build_info.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <stdexcept>

namespace sf {
namespace {

void require(CommandResult result) {
    if (!result) throw std::runtime_error(result.error);
}

std::array<double, 3> coordinates(const glm::dvec3& value) {
    return {value.x, value.y, value.z};
}

std::string present_mode_name(VkPresentModeKHR mode) {
    switch (mode) {
        case VK_PRESENT_MODE_IMMEDIATE_KHR: return "immediate";
        case VK_PRESENT_MODE_MAILBOX_KHR: return "mailbox";
        case VK_PRESENT_MODE_FIFO_KHR: return "fifo";
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "fifo_relaxed";
        default: return std::to_string(mode);
    }
}

}

void App::run_benchmark() {
    auto& options = benchmark_;
    auto utc = std::format("{:%Y%m%dT%H%M%SZ}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
    if (options.output.empty()) options.output = paths::external_path("exports", "benchmarks", utc);
    else if (options.output.is_relative()) options.output = paths::external_path(options.output);
    std::filesystem::create_directories(options.output.parent_path());
    if (!std::filesystem::create_directory(options.output))
        throw std::runtime_error("Report directory already exists; choose a new --benchmark-output path: " + options.output.string());

    auto check_window = [&]() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        if (width != static_cast<int>(options.width) || height != static_cast<int>(options.height))
            throw std::runtime_error(std::format("Requested {}x{} framebuffer, got {}x{}.", options.width, options.height, width, height));
        if (glfwWindowShouldClose(window_)) throw std::runtime_error("Benchmark cancelled; no complete report was written.");
        if (!options.hidden && glfwGetWindowAttrib(window_, GLFW_ICONIFIED))
            throw std::runtime_error("Benchmark window was minimized; capture rejected.");
    };
    check_window();
    if (!star_catalog_.loaded()) throw std::runtime_error("Benchmark requires the Gaia catalog.");
    auto settings = rendering::RenderSettings{};
    require(render_settings_.apply(settings));
    vk_.set_vsync(false);

    nlohmann::json assets = nlohmann::json::object();
    assets["data/gaia/stars.bin"] = benchmark::fingerprint(paths::external_path("data", "gaia", "stars.bin"));
    for (const auto& entry : std::filesystem::recursive_directory_iterator(paths::bundled_path("shaders"))) {
        if (!entry.is_regular_file()) continue;
        auto key = std::filesystem::relative(entry.path(), paths::project_root()).generic_string();
        assets[key] = benchmark::fingerprint(entry.path());
    }
    if (std::find(options.scenes.begin(), options.scenes.end(), "spice") != options.scenes.end()) {
        auto manifest_path = paths::external_path("data", "ephemeris.json");
        auto manifest = paths::read_json(manifest_path);
        if (!manifest) throw std::runtime_error("Missing SPICE manifest.");
        assets["data/ephemeris.json"] = benchmark::fingerprint(manifest_path);
        log::info("Fingerprinting SPICE runtime and kernels before capture.");
        assets["spice_runtime"] = benchmark::fingerprint(paths::external_path("data", "kernels", manifest->at("runtime").get<std::string>()));
        for (const auto& kernel : manifest->at("kernels")) {
            auto name = kernel.get<std::string>();
            assets["kernel/" + name] = benchmark::fingerprint(paths::external_path("data", "kernels", name));
        }
    }
    const auto& device = vk_.device_properties();
    const auto& driver = vk_.driver_properties();
    nlohmann::json metadata{
        {"started_utc", utc}, {"host", benchmark::host_info()},
        {"build", {{"type", SF_BUILD_TYPE}, {"compiler", SF_COMPILER}, {"git_commit", SF_GIT_COMMIT},
                   {"git_dirty_at_configure", SF_GIT_DIRTY}, {"identity_recorded_at", "CMake configure"}}},
        {"gpu", {{"name", device.deviceName}, {"vendor_id", device.vendorID}, {"device_id", device.deviceID},
                 {"driver_version_raw", device.driverVersion}, {"driver_name", driver.driverName}, {"driver_info", driver.driverInfo},
                 {"vulkan_api_version", device.apiVersion}, {"timestamp_period_ns", device.limits.timestampPeriod},
                 {"timestamps_supported", vk_.timestamps_supported()}, {"bresenham_lines", vk_.bresenham_lines_enabled()}}},
        {"validation_enabled", false}, {"assets", assets},
        {"timing_notes", {
            "CPU scopes are wall time, not thread CPU execution time. cpu_record_ms includes UI and renderer child scopes.",
            "GPU frame/scene scopes include child scopes; do not sum inclusive timings or CPU and GPU timings.",
            "GPU intervals can include pipeline overlap and CPU submission gaps; they are not GPU busy-time measurements.",
            "GPU results are matched by submitted frame ID; absent/unsupported values are omitted, never zero-filled.",
            "Warmup, initialization, asset fingerprinting, final GPU drain and report writing are excluded from frame_ms."
        }}
    };
    metadata["simulation_execution"] = options.simulation_mode;
    metadata["timing_notes"].push_back("worker_advance_ms and snapshot_copy_ms run on the simulation worker and overlap main-thread scopes; do not add them to CPU work. Pipeline prefetches the next step while rendering the same state sequence as inline.");
    log::info("Benchmark: {} scenes, {} warmup + {} measured frames, {} repetitions.",
              options.scenes.size(), options.warmup, options.frames, options.repeats);
    if (std::string_view(SF_BUILD_TYPE) != "Release" && std::string_view(SF_BUILD_TYPE) != "RelWithDebInfo")
        log::warn("Benchmark build type is {}; use Release for comparisons.", SF_BUILD_TYPE);

    std::vector<benchmark::Run> runs;
    for (const auto& scene : options.scenes) {
        for (uint32_t repetition = 0; repetition < options.repeats; ++repetition) {
            vk_.drain_gpu_timings();
            vk_.capture_gpu_timings(true, options.warmup + options.frames);
            bool spice = scene == "spice";
            auto system_name = spice ? std::string_view("Ephemeris Mode") : ephemeris::SystemManager::SOLAR_SYSTEM_NAME;
            auto system_path = paths::external_path("data", "systems", system_name, "system.json");
            require(simulation_.replace_system(sys_mgr_.load_system(system_name)));
            if (scene == "nbody") require(simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::NBody}));
            else if (!spice) require(simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::Keplerian}));
            if (spice && system().definition().motion_model != ephemeris::MotionModel::Spice)
                throw std::runtime_error("SPICE fixture did not load SPICE playback.");
            bool paused = scene == "overview" || scene == "approach";
            require(simulation_.apply(ephemeris::SetPaused{paused}));
            require(simulation_.apply(ephemeris::SetTimeWarp{options.warp}));
            camera_ = Camera{};
            camera_.tracking_idx = scene == "approach" ? system().find_body("Earth") : system().star_index();
            if (camera_.tracking_idx < 0) throw std::runtime_error("Benchmark target is missing.");
            camera_.horizon_align = false;
            ui_state_ = {};
            ui_state_.inspected_body = system().find_body("Earth");
            frame_dt_ = units::Seconds{options.step_seconds};
            benchmark::Run run;
            run.scene = scene;
            run.repetition = repetition;
            run.frames.reserve(options.frames);
            auto target_radius = system().body(camera_.tracking_idx).definition.radius_au();
            auto direction = glm::normalize(glm::dvec3{0.0, 19.02, 40.78});
            run.fixture = {
                {"system", system_name}, {"system_fingerprint", benchmark::fingerprint(system_path)},
                {"body_count", system().size()}, {"gaia_count", star_catalog_.count()}, {"paused", paused},
                {"motion_model", spice ? "spice" : scene == "nbody" ? "ias15" : "analytical"},
                {"epoch_utc", system().definition().epoch_utc}, {"fov_deg", camera_.fov_deg},
                {"tracking_body", system().body(camera_.tracking_idx).definition.name},
                {"camera_direction", coordinates(direction)}, {"overview_offset_au", {0.0, 19.02, 40.78}},
                {"approach_start_au", 0.1}, {"approach_end_au", target_radius * 4.0},
                {"approach_path", "logarithmic distance, fixed direction; warmup holds start"},
                {"render_settings", {{"exposure", settings.exposure}, {"hdr_enabled", settings.hdr_enabled},
                    {"show_gaia_stars", settings.show_gaia_stars}, {"show_orbits", settings.show_orbits},
                    {"orbit_opacity", settings.orbit_opacity}, {"orbit_segments", settings.orbit_segments},
                    {"star_point_size", settings.star_point_size}, {"star_intensity", settings.star_intensity}, {"vsync", false}}}
            };
            if (scene == "nbody") run.fixture["ias15_epsilon"] = simulation_.nbody_epsilon();
            log::info("{}: repetition {}/{}", scene, repetition + 1, options.repeats);
            for (uint32_t index = 0; index < options.warmup + options.frames; ++index) {
                benchmark::FrameSample sample;
                sample.frame_id = vk_.submitted_frames();
                sample.index = index >= options.warmup ? index - options.warmup : 0;
                auto frame_start = SteadyClock::now();
                glfwPollEvents();
                check_window();
                {
                    ScopedTimer timer(&sample.cpu_simulation_ms);
                    if (options.simulation_mode == "pipeline" && index > 0) require(simulation_.wait());
                    else require(simulation_.advance(frame_dt_));
                    if (simulation_.threaded() && !paused) {
                        sample.worker_advance_ms = simulation_.publication().advance_ms;
                        sample.snapshot_copy_ms = simulation_.publication().copy_ms;
                    }
                    if (options.simulation_mode == "pipeline" && index + 1 < options.warmup + options.frames)
                        require(simulation_.advance_async(frame_dt_));
                }
                {
                    ScopedTimer timer(&sample.cpu_camera_ms);
                    if (scene == "approach") {
                        double progress = index < options.warmup || options.frames == 1 ? 0.0
                            : static_cast<double>(sample.index) / (options.frames - 1);
                        double distance = std::exp(std::lerp(std::log(0.1), std::log(target_radius * 4.0), progress));
                        camera_.cam_pos_rel = direction * distance;
                    }
                    camera_.update(frame_dt_, system());
                }
                if (!render_frame(&sample)) throw std::runtime_error("Swapchain changed during benchmark; capture rejected.");
                sample.frame_ms = elapsed_ms(frame_start);
                auto extent = vk_.swapchain_extent();
                if (extent.width != options.width || extent.height != options.height)
                    throw std::runtime_error("Swapchain extent differs from the benchmark framebuffer.");
                sample.simulation_years = system().time().value;
                sample.eye_au = coordinates(camera_.eye_pos_render().value);
                if (index >= options.warmup) run.frames.push_back(std::move(sample));
            }
            auto samples = vk_.drain_gpu_timings();
            auto first_id = run.frames.front().frame_id;
            for (auto& sample : samples) {
                if (sample.frame_id < first_id || sample.frame_id - first_id >= run.frames.size()) continue;
                auto& frame = run.frames[static_cast<size_t>(sample.frame_id - first_id)];
                frame.gpu.insert(sample.timings.begin(), sample.timings.end());
            }
            if (vk_.timestamps_supported()) {
                for (const auto& frame : run.frames)
                    for (const char* name : {"gpu_frame", "gpu_scene_hdr", "gpu_celestial", "gpu_gaia", "gpu_orbits", "gpu_post_composite", "gpu_ui"})
                        if (!frame.gpu.contains(name)) throw std::runtime_error("Incomplete GPU timestamps; capture rejected.");
            }
            run.fixture["present_mode"] = present_mode_name(vk_.present_mode());
            run.fixture["framebuffer"] = {options.width, options.height};
            run.fixture["simulation_start_years"] = run.frames.front().simulation_years;
            run.fixture["simulation_end_years"] = run.frames.back().simulation_years;
            runs.push_back(std::move(run));
        }
    }
    vk_.capture_gpu_timings(false);
    benchmark::write_report(options, metadata, runs);
    log::info("Benchmark reports: {}", options.output.string());
}

}
