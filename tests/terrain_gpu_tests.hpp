#pragma once

#include "core/timing.hpp"
#include "benchmark/build_info.hpp"
#include "terrain/geometry_cache.hpp"
#include <cstdlib>
#include <numeric>

namespace {

ephemeris::LoadedSystem precision_fixture(glm::dvec3 origin = {}, double flattening = 1.0 / 298.257223563) {
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{origin + glm::dvec3(0, 0, 10)});
    loaded.definition.bodies[0].lum_eq = 100.0;
    loaded.definition.bodies[0].lum_pole = 100.0;
    ephemeris::BodyDefinition earth;
    earth.name = "Precision terrain";
    earth.type = "Terrestrial";
    earth.radius_rsun = 6378.137 / SOLAR_RADIUS_KM;
    earth.oblateness = flattening;
    earth.color_rgb = {0.45f, 0.65f, 0.3f};
    loaded.definition.bodies.push_back(earth);
    ephemeris::BodyMotion motion;
    motion.position = render_to_ecliptic(RenderPosition{origin});
    loaded.motions.push_back(motion);
    loaded.definition.name_to_idx[earth.name] = 1;
    return loaded;
}

bool declares_float64(const std::vector<uint32_t>& spirv) {
    for (size_t offset = 5; offset < spirv.size();) {
        uint32_t words = spirv[offset] >> 16;
        check(words > 0 && offset + words <= spirv.size(), "Valid SPIR-V instruction stream");
        constexpr uint32_t op_capability = 17, capability_float64 = 10;
        if ((spirv[offset] & 0xffffu) == op_capability && words == 2 && spirv[offset + 1] == capability_float64) return true;
        offset += words;
    }
    return false;
}

void geometry_bake_precision(RenderProbe& probe) {
    check(!declares_float64(rhi::ShaderCache::compile_to_spirv("terrain/terrain.vert", VK_SHADER_STAGE_VERTEX_BIT)),
          "Cached near-ground draw shader contains no FP64 capability");
    check(declares_float64(rhi::ShaderCache::compile_to_spirv("terrain/terrain_reference.vert", VK_SHADER_STAGE_VERTEX_BIT)),
          "Reference shader retains FP64 precision");
    const std::pair<std::string, std::string> cached_double[] = {{"TERRAIN_CACHED_FP64", "1"}};
    check(declares_float64(rhi::ShaderCache::compile_to_spirv("terrain/terrain.vert", VK_SHADER_STAGE_VERTEX_BIT, cached_double)),
          "Controlled cached-FP64 benchmark shader uses FP64 arithmetic");
    terrain::GeometryCache cache;
    cache.init(probe.vk);
    std::vector<terrain::GeometryKey> keys;
    std::vector<terrain::GeometryBakeGPU> requests;
    for (uint32_t face = 0; face < 6; ++face) {
        for (uint32_t x : {0u, 1u << 23, (1u << 24) - 1u}) {
            terrain::GeometryBakeGPU job;
            double width = std::ldexp(2.0, -24);
            auto anchor = terrain::ellipsoid_point(face, -1.0 + (x + 0.5) * width, -1.0 + (x + 0.5) * width, 0.8);
            job.anchor_polar_ratio = {anchor, 0.8};
            job.address = {face, 24u, x, x};
            requests.push_back(job);
            keys.push_back({1, 1, terrain::TileId{face, 24, x, x}.key(), 0, 0.8, 0.0});
        }
    }
    struct Vertex { glm::vec4 position, normal, direction; };
    static_assert(sizeof(Vertex) == terrain::GEOMETRY_VERTEX_BYTES);
    VkDeviceSize block_size = terrain::PATCH_VERTICES * sizeof(Vertex);
    auto readback = probe.vk.create_host_buffer(keys.size() * block_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto plan = cache.prepare(probe.vk, keys, requests);
    check(plan.bakes.size() == keys.size(), "Precision probe bakes every requested geometry block");
    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
        cache.bake(cmd, probe.vk);
        probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        std::vector<VkBufferCopy> copies;
        for (size_t i = 0; i < keys.size(); ++i) copies.push_back({VkDeviceSize(plan.slots[i]) * block_size, i * block_size, block_size});
        vkCmdCopyBuffer(cmd, cache.buffer().handle, readback.handle, uint32_t(copies.size()), copies.data());
        probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });
    vmaInvalidateAllocation(probe.vk.allocator(), readback.allocation, 0, VK_WHOLE_SIZE);
    auto* vertices = static_cast<const Vertex*>(readback.mapped);
    double maximum_error_m = 0.0;
    for (size_t i = 0; i < keys.size(); ++i) {
        auto job = requests[i];
        double width = std::ldexp(2.0, -24);
        for (uint32_t y = 0; y <= 16; ++y) {
            for (uint32_t x = 0; x <= 16; ++x) {
                auto expected = terrain::ellipsoid_point(job.address.x, -1.0 + (job.address.z + x / 16.0) * width,
                                                         -1.0 + (job.address.w + y / 16.0) * width, 0.8) - glm::dvec3(job.anchor_polar_ratio);
                const auto& vertex = vertices[i * terrain::PATCH_VERTICES + y * 17 + x];
                double error = glm::length(glm::dvec3(vertex.position) - expected) * 70000000.0;
                maximum_error_m = std::max(maximum_error_m, error);
                check(error < 0.00001, "LOD24 patch-local positions retain submillimeter precision even at giant-planet radius");
                check(std::abs(glm::length(glm::vec3(vertex.normal)) - 1.0f) < 1e-5f, "Cached normals are finite unit vectors");
                if (x > 0) {
                    auto previous = vertices[i * terrain::PATCH_VERTICES + y * 17 + x - 1].position;
                    check(glm::length(glm::vec3(vertex.position - previous)) * 6378137.0 > 0.001,
                          "Adjacent fine vertices never collapse in FP32 storage");
                }
            }
        }
    }
    check(cache.prepare(probe.vk, keys, requests).bakes.empty(), "Unchanged GPU geometry is reused");
    probe.vk.destroy_buffer(readback);
    cache.shutdown(probe.vk);
    std::cout << "LOD24 cache error at 70,000 km radius: " << maximum_error_m << " m.\n";
}

std::vector<uint16_t> capture_pixels(const RenderProbe& probe) {
    auto* data = static_cast<const uint16_t*>(probe.readback.mapped);
    return {data, data + size_t(probe.extent.width) * probe.extent.height * 4};
}

void cached_reference_image(RenderProbe& probe, const Camera& camera, const ephemeris::SystemSnapshot& system,
                            rendering::RenderSettings settings) {
    settings.terrain_reference_fp64 = true;
    probe.render(camera, system, settings);
    auto reference = capture_pixels(probe);
    settings.terrain_reference_fp64 = false;
    for (uint32_t frame = 0; frame <= terrain::MAX_PATCHES / terrain::MAX_GEOMETRY_BAKES; ++frame) {
        probe.render(camera, system, settings);
        check(probe.renderer.terrain_stats().cache_bakes <= terrain::MAX_GEOMETRY_BAKES, "GPU bake budget enforced");
        if (probe.renderer.terrain_stats().reference_patches == 0) break;
    }
    auto stats = probe.renderer.terrain_stats();
    check(stats.patches > 0 && stats.reference_patches == 0, "Cache converges to the FP32 draw path");
    probe.render(camera, system, settings);
    check(probe.renderer.terrain_stats().cache_bakes == 0 && probe.renderer.terrain_stats().selection_rebuilds == 0,
          "Stationary draws reuse geometry and selection");
    auto cached = capture_pixels(probe);
    double error_sum = 0.0;
    size_t changed = 0;
    for (size_t i = 0; i < reference.size(); i += 4) {
        float largest = 0.0f;
        for (size_t c = 0; c < 3; ++c) {
            float a = glm::unpackHalf2x16(reference[i + c]).x;
            float b = glm::unpackHalf2x16(cached[i + c]).x;
            check(std::isfinite(b), "Cached HDR output remains finite");
            largest = std::max(largest, std::abs(a - b));
        }
        error_sum += largest;
        changed += largest > 0.03f;
    }
    double pixel_count = reference.size() / 4;
    if (!(changed < pixel_count * 0.005 && error_sum / pixel_count < 0.002))
        std::cerr << "Cached/reference difference: " << changed << " / " << pixel_count << ", mean " << error_sum / pixel_count << '\n';
    check(changed < pixel_count * 0.005 && error_sum / pixel_count < 0.002,
          "Cached and FP64 images agree within subpixel edge and HDR rounding tolerances");
}

void terrain_precision(RenderProbe& probe) {
    geometry_bake_precision(probe);
    probe.resize({512, 512});
    ephemeris::Simulation simulation;
    rendering::RenderSettings settings;
    settings.terrain_enabled = true;
    settings.terrain_debug_tiles = true;
    constexpr double radius = 6378.137 * KM_TO_AU;
    for (auto origin : {glm::dvec3(0), glm::dvec3(100, -150, 200)}) {
        check(simulation.replace_system(precision_fixture(origin, 0.15)).accepted, "Precision system load");
        for (auto direction : {glm::dvec3(0, 0, 1), glm::dvec3(0, 1, 0), glm::normalize(glm::dvec3(1, 0, 1)), glm::normalize(glm::dvec3(1, 1, 1))}) {
            auto camera = camera_at(radius * 3.0);
            camera.origin_render = origin;
            camera.cam_pos_rel = direction * radius * 3.0;
            camera.up = std::abs(direction.y) > 0.9 ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
            camera.center_body(1, simulation.snapshot());
            cached_reference_image(probe, camera, simulation.snapshot(), settings);
        }
        check(simulation.replace_system(precision_fixture(origin)).accepted, "Near-ground precision system load");
        for (double height : {0.0, 2.0}) {
            settings.terrain_height_km = float(height);
            double ground = height * (0.30 * std::sin(-11.0) + 0.15 * std::sin(61.0));
            auto camera = camera_at(radius + (ground + 0.001) * KM_TO_AU);
            camera.origin_render = origin;
            camera.tracking_idx = 1;
            camera.is_near_surface = true;
            cached_reference_image(probe, camera, simulation.snapshot(), settings);
            settings.terrain_debug_tiles = false;
            cached_reference_image(probe, camera, simulation.snapshot(), settings);
            settings.terrain_debug_tiles = true;
        }
        settings.terrain_height_km = 2.0f;
        for (auto direction : {glm::dvec3(0, 0, 1), glm::normalize(glm::dvec3(1, 2, 3))}) {
            auto camera = camera_at(200.0);
            camera.origin_render = origin;
            camera.cam_pos_rel = direction * 200.0;
            camera.cam_look_aim = true;
            camera.fov_deg = 0.00003;
            cached_reference_image(probe, camera, simulation.snapshot(), settings);
            check(probe.red_at(256, 256) > 0.01f, "Distant narrow-angle terrain remains aligned with the camera target");
        }
    }
    settings.terrain_height_km = 2;
    auto camera = camera_at(radius * 3.0);
    camera.origin_render = {100, -150, 200};
    cached_reference_image(probe, camera, simulation.snapshot(), settings);
    settings.exposure = 2.0f;
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.renderer.terrain_stats().cache_bakes == 0, "Lighting-only changes retain geometry");
    settings.terrain_height_km = 3.0f;
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.renderer.terrain_stats().cache_bakes > 0, "Height changes invalidate geometry");
    probe.resize(EXTENT);
    std::cout << "Cached/reference images, near-ground precision, source invalidation and reuse passed.\n";
}

nlohmann::json terrain_timing_stats(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return {{"mean", std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()},
            {"median", samples[samples.size() / 2]}, {"p95", samples[size_t((samples.size() - 1) * .95)]},
            {"p99", samples[size_t((samples.size() - 1) * .99)]}, {"min", samples.front()}, {"max", samples.back()}};
}

struct TerrainSample {
    double cpu_prepare_ms, cpu_record_ms, gpu_prepare_ms, gpu_total_ms, gpu_draw_ms;
    double cpu_ms() const { return cpu_prepare_ms + cpu_record_ms; }
};

TerrainSample measure_terrain(RenderProbe& probe, const Camera& camera, const ephemeris::SystemSnapshot& system,
                              const rendering::RenderSettings& settings) {
    TerrainSample sample{};
    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
        probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        probe.vk.transition_image(cmd, probe.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        auto timer = probe.vk.begin_gpu_timer(cmd, "terrain_total");
        auto prepare_timer = probe.vk.begin_gpu_timer(cmd, "terrain_prepare");
        auto start = SteadyClock::now();
        probe.renderer.prepare(cmd, probe.vk, camera, system, settings, probe.extent);
        sample.cpu_prepare_ms = elapsed_ms(start);
        probe.vk.end_gpu_timer(cmd, prepare_timer);
        auto color = rhi::RenderingAttachment::clear_color(probe.color);
        auto depth = rhi::RenderingAttachment::clear_depth_reverse_z(probe.depth);
        probe.vk.begin_rendering(cmd, probe.extent, {&color, 1}, &depth);
        auto draw_timer = probe.vk.begin_gpu_timer(cmd, "terrain_draw");
        start = SteadyClock::now();
        probe.renderer.render(cmd, probe.vk, camera, system, settings, probe.extent);
        sample.cpu_record_ms = elapsed_ms(start);
        probe.vk.end_gpu_timer(cmd, draw_timer);
        probe.vk.end_rendering(cmd);
        probe.vk.end_gpu_timer(cmd, timer);
    });
    probe.vk.drain_gpu_timings();
    sample.gpu_prepare_ms = probe.vk.gpu_time_ms("terrain_prepare");
    sample.gpu_total_ms = probe.vk.gpu_time_ms("terrain_total");
    sample.gpu_draw_ms = probe.vk.gpu_time_ms("terrain_draw");
    return sample;
}

Camera ground_trace_camera(double speed_m_s, double clearance_m, bool horizon, uint32_t frame, double start_angle,
                           double height_km = 2.0, bool follow_surface = true) {
    constexpr double radius_m = 6378137.0;
    double angle = start_angle + speed_m_s * double(frame) / (60.0 * radius_m);
    glm::dvec3 radial{std::sin(angle), 0, std::cos(angle)};
    glm::vec3 d{radial};
    float relief = .55f * std::sin(18.f * d.x + 5.f * d.y) * std::cos(14.f * d.z)
                 + .30f * std::sin(37.f * d.y - 11.f * d.z) * std::cos(23.f * d.x)
                 + .15f * std::sin(61.f * d.z + 19.f * d.x) * std::cos(47.f * d.y);
    auto camera = camera_at(0.0);
    camera.cam_pos_rel = radial * ((radius_m + (follow_surface ? height_km * 1000.0 * relief : 0.0) + clearance_m) * .001 * KM_TO_AU);
    camera.tracking_idx = 1;
    camera.is_near_surface = true;
    camera.cam_look_aim = !horizon;
    camera.up = horizon ? radial : glm::dvec3{0, 1, 0};
    camera.yaw_actual_deg = -angle * RAD_TO_DEG;
    return camera;
}

void terrain_ground_benchmark(RenderProbe& probe, const std::filesystem::path& output, bool relief_comparison = false) {
    check(!output.empty() && !std::filesystem::exists(output), "Choose a new moving-ground benchmark output path");
    check(probe.vk.timestamps_supported(), "Ground benchmark requires hardware GPU timestamps");
    probe.resize({1920, 1080});
    ephemeris::Simulation simulation;
    rendering::RenderSettings settings;
    settings.terrain_enabled = true;
    settings.terrain_height_km = 2.f;
    constexpr uint32_t warmup = 60, measured = 240, repetitions = 3;
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"framebuffer", {1920, 1080}}, {"fov_degrees", 60},
        {"build_type", SF_BUILD_TYPE}, {"compiler", SF_COMPILER},
        {"cpu_identifier", std::getenv("PROCESSOR_IDENTIFIER") ? std::getenv("PROCESSOR_IDENTIFIER") : "unavailable"},
        {"warmup_per_block", warmup}, {"measured_per_block", measured}, {"repetitions", repetitions},
        {"selector", "balanced-refinement-local-metadata"},
        {"path_hz", 60}, {"height_amplitude_km", 2}, {"error_pixels", settings.terrain_error_pixels},
        {"patch_budget", settings.terrain_patch_limit},
        {"geometry_cache_bytes", terrain::GEOMETRY_CACHE_CAPACITY * terrain::PATCH_VERTICES * terrain::GEOMETRY_VERTEX_BYTES},
        {"bakes_per_frame", terrain::MAX_GEOMETRY_BAKES},
        {"notes", "Release offscreen celestial pass; same equatorial path in each repetition, fresh fixture generation before each warmup. Radial altitude relative to each row's altitude_reference, fixed 60 Hz path independent of wall time. CPU excludes submit/waits; GPU includes bakes and clears. No texture streaming, physics, Gaia, orbits, UI, presentation or timed image readback. Not application FPS."},
        {"results", nlohmann::json::array()}};
    if (relief_comparison) report.erase("height_amplitude_km");
    report["metadata_capacity"] = terrain::TILE_METADATA_CAPACITY;
    struct Case {
        const char* name;
        double speed_m_s, clearance_m;
        bool horizon;
        double start_angle = 0.15, height_km = 2.0;
        bool follow_surface = true;
    };
    const std::vector<Case> cases = relief_comparison ? std::vector<Case>{
        {"2km-ellipsoid-flat-down", 100, 2000, false, .15, 0, false},
        {"2km-ellipsoid-relief-down", 100, 2000, false, .15, 20, false},
        {"2km-ellipsoid-flat-horizon", 100, 2000, true, .15, 0, false},
        {"2km-ellipsoid-relief-horizon", 100, 2000, true, .15, 20, false},
        {"2km-surface-relief-down", 100, 2000, false, .15, 20, true},
        {"2km-surface-relief-horizon", 100, 2000, true, .15, 20, true},
    } : std::vector<Case>{
        {"stationary-1m-down", 0, 1, false},
        {"walking-1m-down", 5, 1, false},
        {"walking-1m-horizon", 5, 1, true},
        {"flying-1m-down", 100, 1, false},
        {"flying-1m-horizon", 100, 1, true},
        {"fast-100m-horizon", 1000, 100, true},
        {"origin-walking-1m-down", 5, 1, false, 0.0},
        {"origin-flying-1m-horizon", 100, 1, true, 0.0},
    };
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    for (const auto& trace : cases) {
        settings.terrain_height_km = float(trace.height_km);
        std::vector<double> cpu, prepare, record, gpu, gpu_prepare, draw, patches, bakes, fallback, levels;
        std::vector<double> selection_nodes, metadata_builds, metadata_entries;
        uint32_t limited_frames = 0, fallback_frames = 0, rebuilds = 0;
        nlohmann::json frames = nlohmann::json::array(), cold = nlohmann::json::array();
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            check(simulation.replace_system(precision_fixture()).accepted, "Fresh ground trace fixture");
            for (uint32_t frame = 0; frame < warmup + measured; ++frame) {
                auto camera = ground_trace_camera(trace.speed_m_s, trace.clearance_m, trace.horizon, frame, trace.start_angle,
                                                 trace.height_km, trace.follow_surface);
                auto sample = measure_terrain(probe, camera, simulation.snapshot(), settings);
                const auto& stats = probe.renderer.terrain_stats();
                check(stats.patches > 0 && stats.cache_bakes <= terrain::MAX_GEOMETRY_BAKES, "Ground trace uses bounded terrain geometry");
                check(stats.patches <= settings.terrain_patch_limit && (relief_comparison || stats.deepest_level >= 15),
                      "Ground trace retains fine terrain within the patch budget");
                nlohmann::json row{{"repetition", repetition}, {"frame", frame}, {"cpu_ms", sample.cpu_ms()},
                    {"cpu_prepare_ms", sample.cpu_prepare_ms}, {"cpu_record_ms", sample.cpu_record_ms},
                    {"gpu_total_ms", sample.gpu_total_ms}, {"gpu_prepare_ms", sample.gpu_prepare_ms}, {"gpu_draw_ms", sample.gpu_draw_ms},
                    {"patches", stats.patches}, {"deepest_level", stats.deepest_level}, {"bakes", stats.cache_bakes},
                    {"cache_misses", stats.cache_misses}, {"fallback", stats.reference_patches}, {"budget_limited", stats.budget_limited}};
                row["selection_nodes"] = stats.selection_nodes;
                row["metadata_builds"] = stats.metadata_builds;
                row["metadata_entries"] = stats.metadata_entries;
                if (frame < 4) cold.push_back(row);
                if (frame < warmup) continue;
                frames.push_back(row);
                cpu.push_back(sample.cpu_ms()); prepare.push_back(sample.cpu_prepare_ms); record.push_back(sample.cpu_record_ms);
                gpu.push_back(sample.gpu_total_ms); gpu_prepare.push_back(sample.gpu_prepare_ms); draw.push_back(sample.gpu_draw_ms);
                patches.push_back(stats.patches); bakes.push_back(stats.cache_bakes); fallback.push_back(stats.reference_patches);
                levels.push_back(stats.deepest_level);
                selection_nodes.push_back(stats.selection_nodes);
                metadata_builds.push_back(stats.metadata_builds);
                metadata_entries.push_back(stats.metadata_entries);
                limited_frames += stats.budget_limited; fallback_frames += stats.reference_patches > 0;
                rebuilds += stats.selection_rebuilds;
            }
        }
        nlohmann::json result{{"scene", trace.name}, {"speed_m_s", trace.speed_m_s}, {"clearance_m", trace.clearance_m},
            {"start_angle_radians", trace.start_angle},
            {"height_km", trace.height_km}, {"altitude_reference", trace.follow_surface ? "synthetic-surface" : "ellipsoid"},
            {"view", trace.horizon ? "horizon" : "down"}, {"measured_frames", cpu.size()},
            {"measured_distance_m_per_repetition", trace.speed_m_s * measured / 60.0},
            {"cpu_ms", terrain_timing_stats(cpu)}, {"cpu_prepare_ms", terrain_timing_stats(prepare)},
            {"cpu_record_ms", terrain_timing_stats(record)}, {"gpu_total_ms", terrain_timing_stats(gpu)},
            {"gpu_prepare_ms", terrain_timing_stats(gpu_prepare)}, {"gpu_draw_ms", terrain_timing_stats(draw)},
            {"patches", terrain_timing_stats(patches)}, {"deepest_level", terrain_timing_stats(levels)},
            {"bakes", terrain_timing_stats(bakes)}, {"fallback_patches", terrain_timing_stats(fallback)},
            {"bakes_in_measured_frames", std::accumulate(bakes.begin(), bakes.end(), 0.0)},
            {"budget_limited_frames", limited_frames}, {"fallback_frames", fallback_frames}, {"selection_rebuilds", rebuilds}};
        result["selection_nodes"] = terrain_timing_stats(selection_nodes);
        result["metadata_builds"] = terrain_timing_stats(metadata_builds);
        result["metadata_entries"] = terrain_timing_stats(metadata_entries);
        std::cout << result.dump() << '\n';
        result["cold_frames"] = std::move(cold);
        result["frames"] = std::move(frames);
        report["results"].push_back(std::move(result));
        auto camera = ground_trace_camera(trace.speed_m_s, trace.clearance_m, trace.horizon, warmup + measured - 1, trace.start_angle,
                                         trace.height_km, trace.follow_surface);
        settings.terrain_debug_tiles = true;
        check(probe.render(camera, simulation.snapshot(), settings) > 1.0, "Ground trace preview contains visible terrain");
        settings.terrain_debug_tiles = false;
        probe.save_preview(output.parent_path() / (output.stem().string() + "-" + trace.name + ".ppm"), true);
        std::ofstream file(output);
        check(bool(file), "Ground benchmark output is writable");
        file << report.dump(2) << '\n';
    }
}

void terrain_benchmark(RenderProbe& probe, const std::filesystem::path& output) {
    check(output.empty() || !std::filesystem::exists(output), "Choose a new terrain benchmark output path to preserve previous captures");
    check(probe.vk.timestamps_supported(), "Terrain benchmark requires hardware GPU timestamps");
    probe.resize({1920, 1080});
    ephemeris::Simulation simulation;
    check(simulation.replace_system(precision_fixture()).accepted, "Terrain benchmark fixture");
    rendering::RenderSettings settings;
    constexpr double radius = 6378.137 * KM_TO_AU;
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"framebuffer", {1920, 1080}}, {"fov_degrees", 60},
        {"build_type", SF_BUILD_TYPE}, {"compiler", SF_COMPILER},
        {"cpu_identifier", std::getenv("PROCESSOR_IDENTIFIER") ? std::getenv("PROCESSOR_IDENTIFIER") : "unavailable"},
        {"warmup_per_block", 40}, {"measured_per_block", 120}, {"repetitions", 3},
        {"geometry_cache_bytes", terrain::GEOMETRY_CACHE_CAPACITY * terrain::PATCH_VERTICES * terrain::GEOMETRY_VERTEX_BYTES},
        {"bakes_per_frame", terrain::MAX_GEOMETRY_BAKES},
        {"notes", "Release offscreen celestial microbenchmark. CPU includes preparation and recording, excludes waits. GPU total includes bakes and HDR/depth clears. No simulation, Gaia, orbits, UI, presentation or image readback. Timings overlap; do not sum CPU and GPU."},
        {"results", nlohmann::json::array()}};
    auto measure = [&](const Camera& camera) {
        auto sample = measure_terrain(probe, camera, simulation.snapshot(), settings);
        return std::array{sample.cpu_ms(), sample.gpu_total_ms, sample.gpu_draw_ms};
    };
    for (std::string_view name : {"orbital", "30-km", "30-km-horizon", "1-meter", "1-meter-horizon", "1-meter-relief", "moving-orbit", "distant-telescope"}) {
        bool meter = name.starts_with("1-meter"), horizon = name.ends_with("horizon"), relief = name == "1-meter-relief";
        bool moving = name == "moving-orbit";
        bool distant = name == "distant-telescope";
        settings.terrain_height_km = meter && !relief ? 0.0f : 2.0f;
        double distance = name == "orbital" || moving ? radius * 3.0 : radius + (meter ? .001 : 30.0) * KM_TO_AU;
        if (relief) distance += settings.terrain_height_km * (0.30 * std::sin(-11.0) + 0.15 * std::sin(61.0)) * KM_TO_AU;
        auto camera = camera_at(distance);
        if (distant) {
            camera.cam_pos_rel = glm::normalize(glm::dvec3(1, 2, 3)) * 200.0;
            camera.cam_look_aim = true;
            camera.fov_deg = 0.00003;
        }
        camera.tracking_idx = 1;
        camera.is_near_surface = name != "orbital" && !moving && !distant;
        if (horizon) { camera.yaw_actual_deg = 0; camera.up = {0, 0, 1}; }
        std::vector<double> cpu[4], gpu[4], draw[4];
        std::array<uint32_t, 4> patches{}, bakes{}, rebuilds{};
        for (int repetition = 0; repetition < 3; ++repetition) {
            for (int block = 0; block < 4; ++block) {
                int mode = (block + repetition) % 4;
                settings.terrain_enabled = mode != 0;
                settings.terrain_reference_fp64 = mode == 1;
                settings.terrain_cached_fp64 = mode == 3;
                for (uint32_t index = 0; index < 160; ++index) {
                    if (moving) {
                        double angle = double(index) * 0.002;
                        camera.cam_pos_rel = glm::dvec3(std::sin(angle), 0, std::cos(angle)) * distance;
                        camera.cam_look_aim = true;
                    }
                    auto sample = measure(camera);
                    const auto& stats = probe.renderer.terrain_stats();
                    if (index >= 40) {
                        cpu[mode].push_back(sample[0]); gpu[mode].push_back(sample[1]); draw[mode].push_back(sample[2]);
                        patches[mode] = stats.patches;
                        bakes[mode] += stats.cache_bakes;
                        rebuilds[mode] += stats.selection_rebuilds;
                        if (mode >= 2) check(stats.reference_patches == 0, "Measured cached mode contains no fallback draws");
                    }
                }
            }
        }
        for (uint32_t mode = 0; mode < 4; ++mode) {
            nlohmann::json row{{"scene", name}, {"mode", mode == 0 ? "sphere" : mode == 1 ? "fp64-reference" : mode == 2 ? "cached" : "cached-fp64"},
                {"height_km", settings.terrain_height_km}, {"patches", patches[mode]}, {"bakes_in_measured_frames", bakes[mode]},
                {"fov_degrees", camera.fov_deg},
                {"selection_rebuilds", rebuilds[mode]}, {"cpu_ms", terrain_timing_stats(cpu[mode])},
                {"gpu_total_ms", terrain_timing_stats(gpu[mode])}, {"gpu_draw_ms", terrain_timing_stats(draw[mode])}};
            report["results"].push_back(row);
            std::cout << row.dump() << '\n';
        }
    }
    check(simulation.replace_system(precision_fixture()).accepted, "Cold cache fixture generation");
    settings.terrain_enabled = true;
    settings.terrain_reference_fp64 = false;
    settings.terrain_cached_fp64 = false;
    settings.terrain_height_km = 2.0f;
    auto camera = camera_at(radius * 3.0);
    camera.tracking_idx = 1;
    report["cold_frames"] = nlohmann::json::array();
    for (uint32_t i = 0; i < 4; ++i) {
        auto sample = measure(camera);
        const auto& stats = probe.renderer.terrain_stats();
        report["cold_frames"].push_back({{"frame", i}, {"cpu_ms", sample[0]}, {"gpu_total_ms", sample[1]},
            {"bakes", stats.cache_bakes}, {"cached", stats.cached_patches}, {"fallback", stats.reference_patches}});
    }
    if (!output.empty()) {
        if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
        std::ofstream file(output);
        check(bool(file), "Terrain benchmark output is writable");
        file << report.dump(2) << '\n';
    }
}

}
