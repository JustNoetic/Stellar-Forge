#include "app.hpp"
#include "benchmark/build_info.hpp"
#include "benchmark/earth_closeup.hpp"
#include "benchmark/audit_fixture.hpp"
#include "benchmark/terrain_navigation.hpp"
#include "core/log.hpp"
#include "core/path_utils.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <stdexcept>

namespace sf {
namespace {

void require(CommandResult result) {
    if (!result) throw std::runtime_error(result.error);
}

std::array<double, 3> coordinates(const glm::dvec3& value) {
    return {value.x, value.y, value.z};
}

void write_pfm(const std::filesystem::path& path, uint32_t width, uint32_t height, const uint16_t* rgba_half) {
    std::vector<float> rgb(size_t(width)*height*3);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            const uint16_t* source = rgba_half+(size_t(height-1-y)*width+x)*4;
            float* target = rgb.data()+(size_t(y)*width+x)*3;
            for (int c = 0; c < 3; ++c) target[c] = glm::unpackHalf1x16(source[c]);
        }
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::binary);
    file << "PF\n" << width << ' ' << height << "\n-1.0\n";
    file.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()*sizeof(float)));
    if (!file) throw std::runtime_error("Failed to write benchmark capture: " + path.string());
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

void App::save_hdr_capture(const std::filesystem::path& path) {
    vk_.wait_idle();
    auto readback = vk_.create_host_buffer(VkDeviceSize(hdr_color_.extent.width)*hdr_color_.extent.height*8,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    vk_.immediate_submit([&](VkCommandBuffer cmd) {
        VkImageLayout restore = hdr_color_.layout;
        vk_.transition_image(cmd, hdr_color_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {hdr_color_.extent.width, hdr_color_.extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, hdr_color_.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle, 1, &copy);
        vk_.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        vk_.transition_image(cmd, hdr_color_, restore);
    });
    vmaInvalidateAllocation(vk_.allocator(), readback.allocation, 0, VK_WHOLE_SIZE);
    write_pfm(path, hdr_color_.extent.width, hdr_color_.extent.height, static_cast<const uint16_t*>(readback.mapped));
    vk_.destroy_buffer(readback);
    log::info("Benchmark capture: {}", path.string());
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
    auto settings = benchmark::audit_settings(options.variant);
    require(render_settings_.apply(settings));
    vk_.set_vsync(false);

    nlohmann::json assets = nlohmann::json::object();
    assets["data/gaia/stars.bin"] = benchmark::fingerprint(paths::external_path("data", "gaia", "stars.bin"));
    assets["data/surfaces.json"] = benchmark::fingerprint(paths::external_path("data", "surfaces.json"));
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
            auto loaded = sys_mgr_.load_system(system_name);
            bool audit_rings = scene.find("rings") != std::string::npos;
            if (audit_rings) {
                auto earth = std::find_if(loaded.definition.bodies.begin(), loaded.definition.bodies.end(), [](const auto& b) { return b.name == "Earth"; });
                if (earth == loaded.definition.bodies.end()) throw std::runtime_error("Ring audit requires Earth.");
                auto profile = rendering::generate_procedural_ring_profile(rendering::ProceduralRingConfig{});
                earth->rings = {benchmark::audit_earth_ring(profile)};
                earth->ring_texture_inner = profile.inner_radius_ratio;
                earth->ring_texture_outer = profile.outer_radius_ratio;
                celestial_renderer_.set_custom_ring_texture(loaded.definition.name, "Earth", std::move(profile.profile));
            }
            require(simulation_.replace_system(std::move(loaded)));
            if (scene == "nbody") require(simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::NBody}));
            else if (!spice) require(simulation_.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::Keplerian}));
            if (spice && system().definition().motion_model != ephemeris::MotionModel::Spice)
                throw std::runtime_error("SPICE fixture did not load SPICE playback.");
            bool earth_closeup = scene.starts_with("earth-") || scene.starts_with("saturn");
            if (scene == "earth-eclipse") {
                require(simulation_.apply(ephemeris::SetPaused{false}));
                require(simulation_.apply(ephemeris::SetTimeWarp{1.0}));
                require(simulation_.advance(units::Seconds{benchmark::ECLIPSE_FIXTURE_YEARS * SECONDS_PER_YEAR}));
            }
            bool paused = scene == "overview" || scene == "approach" || (earth_closeup && !scene.ends_with("-live"));
            require(simulation_.apply(ephemeris::SetPaused{paused}));
            require(simulation_.apply(ephemeris::SetTimeWarp{options.warp}));
            camera_ = Camera{};
            camera_.tracking_idx = scene == "approach" ? system().find_body("Earth") : system().star_index();
            if (camera_.tracking_idx < 0) throw std::runtime_error("Benchmark target is missing.");
            camera_.horizon_align = false;
            bool navigation = benchmark::terrain_navigation_scene(scene);
            benchmark::TerrainNavigation navigation_path;
            if (navigation) {
                auto source = celestial_renderer_.height_source("Earth");
                if (!source) throw std::runtime_error("Terrain navigation benchmark requires Earth elevation data.");
                navigation_path = benchmark::terrain_navigation(system(), *source, settings, scene, options.frames, options.step_seconds);
                camera_.tracking_idx = system().find_body("Earth");
                camera_.horizon_align = true;
                navigation_path.poses.front().apply(camera_);
            } else if (earth_closeup) camera_ = benchmark::audit_camera(system(), scene);
            double base_yaw = camera_.yaw_deg;
            ui_state_ = {};
            ui_state_.inspected_body = system().find_body(scene.starts_with("saturn") ? "Saturn" : "Earth");
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
            if (earth_closeup) {
                run.fixture["tracking_body"] = system().body(camera_.tracking_idx).definition.name;
                run.fixture["camera_direction"] = coordinates(glm::normalize(camera_.cam_pos_rel));
                run.fixture["distance_radii"] = glm::length(camera_.cam_pos_rel)/target_radius;
                run.fixture["forward"] = coordinates(camera_.forward_dir());
                run.fixture["up"] = coordinates(camera_.up);
                run.fixture["synthetic_earth_rings"] = audit_rings;
                if (audit_rings) run.fixture["procedural_ring"] = {{"seed", 42}, {"config", "ProceduralRingConfig defaults"},
                    {"inner_radius_ratio", 1.2}, {"outer_radius_ratio", 2.3}};
                run.fixture["turning"] = scene.ends_with("-turn");
                run.fixture["altitude_km"] = (glm::length(camera_.cam_pos_rel)-ellipsoid_surface_radius(target_radius,
                    system().body(camera_.tracking_idx).definition.oblateness, system().body(camera_.tracking_idx).render.pole, camera_.cam_pos_rel))*AU_TO_KM;
            }
            run.fixture["variant"] = options.variant;
            if (navigation) run.fixture["navigation"] = navigation_path.fixture;
            run.fixture["render_settings"]["terrain_error_pixels"] = settings.terrain_error_pixels;
            run.fixture["render_settings"]["terrain_patch_limit"] = settings.terrain_patch_limit;
            run.fixture["render_settings"]["bloom"] = {{"mode", settings.bloom_mode}, {"strength", settings.bloom_strength},
                {"highlight_limit", settings.bloom_highlight_limit},
                {"radius", settings.bloom_radius}, {"diffraction", settings.bloom_diffraction}, {"rotation_deg", settings.bloom_rotation_deg},
                {"spike_threshold", settings.bloom_spike_threshold}};
            run.fixture["render_settings"]["features"] = {{"atmospheres", settings.show_atmospheres}, {"aerial_perspective", settings.aerial_perspective_enabled},
                {"atmosphere_shadows", settings.atmosphere_shadows_enabled},
                {"refraction", settings.refraction_enabled}, {"terrain", settings.terrain_imported_enabled}, {"normals", settings.terrain_normal_maps_enabled},
                {"textures", settings.surface_textures_enabled}, {"rings", settings.show_rings}, {"ring_shadows", settings.ring_shadows_enabled},
                {"direct_sunlight_only", settings.direct_sunlight_only},
                {"clouds", settings.clouds_enabled}, {"cloud_mode", settings.cloud_mode}, {"legacy_transparency", settings.transparency_reference_legacy},
                {"planetshine", settings.planetshine_enabled}, {"ringshine", settings.ringshine_enabled}, {"shadow_temporal", settings.atmosphere_shadow_temporal},
                {"shadow_cached_weights", settings.atmosphere_shadow_cached_weights}, {"shadow_fast_rings", settings.atmosphere_shadow_fast_rings}, {"shadow_analytic", settings.atmosphere_shadow_analytic}};
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
                    if (navigation) navigation_path.poses[index < options.warmup ? 0 : sample.index].apply(camera_);
                    else if (scene.ends_with("-turn")) camera_.yaw_deg = camera_.yaw_actual_deg = base_yaw+15*std::sin(index*options.step_seconds*0.7);
                    if (scene == "approach") {
                        double progress = index < options.warmup || options.frames == 1 ? 0.0
                            : static_cast<double>(sample.index) / (options.frames - 1);
                        double distance = std::exp(std::lerp(std::log(0.1), std::log(target_radius * 4.0), progress));
                        camera_.cam_pos_rel = direction * distance;
                    }
                }
                if (!render_frame(&sample)) throw std::runtime_error("Swapchain changed during benchmark; capture rejected.");
                sample.frame_ms = elapsed_ms(frame_start);
                const auto& terrain = celestial_renderer_.terrain_stats();
                const auto& textures = celestial_renderer_.texture_stats();
                const auto& normals = celestial_renderer_.normal_stats();
                const auto& atmosphere = celestial_renderer_.atmosphere_stats();
                sample.counters = {{"terrain_patches", terrain.patches}, {"terrain_bakes", terrain.cache_bakes},
                    {"terrain_selection_rebuilds", terrain.selection_rebuilds}, {"terrain_budget_limited", terrain.budget_limited},
                    {"terrain_selection_reuses", terrain.selection_reuses}, {"terrain_selection_checks", terrain.selection_checks},
                    {"terrain_dependency_rebuilds", terrain.dependency_rebuilds}, {"terrain_patch_updates", terrain.patch_updates},
                    {"terrain_loading_bodies", terrain.loading_bodies}, {"terrain_deepest_level", terrain.deepest_level},
                    {"terrain_imported_patches", terrain.imported_patches}, {"terrain_reference_patches", terrain.reference_patches},
                    {"terrain_stream_pending", terrain.stream.pending}, {"terrain_stream_failed", terrain.stream.failed},
                    {"texture_uploads", textures.uploads}, {"texture_pending", textures.pending}, {"texture_failed", textures.failed},
                    {"normal_uploads", normals.uploads}, {"normal_pending", normals.pending}, {"normal_failed", normals.failed},
                    {"atmospheres_visible", atmosphere.visible}, {"optical_updates", atmosphere.optical_lut_updates},
                    {"sky_updates", atmosphere.sky_lut_updates}, {"aerial_updates", atmosphere.aerial_lut_updates},
                    {"secondary_updates", atmosphere.secondary_lut_updates},
                    {"secondary_source_updates", atmosphere.secondary_source_lut_updates},
                    {"shadow_updates", atmosphere.shadow_updates}, {"shadow_vrs_updates", atmosphere.shadow_vrs_updates},
                    {"shadow_resets", atmosphere.shadow_history_resets},
                    {"shadow_rotation_updates", atmosphere.shadow_rotation_updates},
                    {"transparency_fragments", static_cast<double>(celestial_renderer_.transparency().last_stats().fragments)},
                    {"transparency_overflow", static_cast<double>(celestial_renderer_.transparency().last_stats().overflow)},
                    {"ringshine_bakes", celestial_renderer_.ringshine_frame_bake_count()},
                    {"observer_builds", celestial_renderer_.refraction_observer_builds()},
                    {"camera_yaw_deg", camera_.yaw_actual_deg}, {"camera_pitch_deg", camera_.pitch_actual_deg}};
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
    if (!options.capture.empty()) save_hdr_capture(options.capture);
    benchmark::write_report(options, metadata, runs);
    log::info("Benchmark reports: {}", options.output.string());
}

}
