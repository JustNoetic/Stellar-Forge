#pragma once

#include <cstring>

namespace {

ephemeris::LoadedSystem surface_ring_fixture(uint32_t hosts = 4) {
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{0, 5, 5}});
    for (uint32_t h = 0; h < hosts; ++h) {
        ephemeris::BodyDefinition host;
        host.name = "Surface ring host " + std::to_string(h);
        host.radius_rsun = 0.1;
        host.pole_ecl = render_to_ecliptic(RenderPosition{{0, 1, 0}}).value;
        ephemeris::RingLayerDefinition ring;
        ring.inner = 1.4;
        ring.outer = 2.4;
        ring.opacity = 0.9f;
        ring.gradient = {{0, 1}, {1, 1}};
        host.rings.push_back(ring);
        loaded.definition.name_to_idx[host.name] = static_cast<int>(loaded.definition.bodies.size());
        loaded.definition.bodies.push_back(host);
        ephemeris::BodyMotion motion;
        if (h + 1 < hosts) motion.position = render_to_ecliptic(RenderPosition{{100.0 + 10.0 * h, 100, -100}});
        loaded.motions.push_back(motion);
    }
    ephemeris::BodyDefinition moon;
    moon.name = "Ring shadow moon";
    moon.radius_rsun = 0.02;
    moon.parent_idx = hosts;
    moon.parent_id = loaded.definition.bodies[hosts].name;
    loaded.definition.name_to_idx[moon.name] = static_cast<int>(loaded.definition.bodies.size());
    loaded.definition.bodies.push_back(moon);
    ephemeris::BodyMotion motion;
    double radius = 0.1 * SOLAR_RADII_TO_AU;
    motion.position = render_to_ecliptic(RenderPosition{{2 * radius, -radius, 0}});
    loaded.motions.push_back(motion);
    return loaded;
}

void moon_ring_penumbra_contracts(RenderProbe& probe) {
    probe.resize({512, 512});
    auto loaded = surface_ring_fixture(1);
    double radius = loaded.definition.bodies[1].radius_au();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{5, 0.05, 1.8 * radius}});
    loaded.definition.bodies.back().radius_rsun = 0.002;
    loaded.motions.back().position = render_to_ecliptic(RenderPosition{{-6 * radius, -0.076 * radius, 1.8 * radius}});
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Grazing ring-to-moon penumbra fixture");
    double moon_radius = loaded.definition.bodies.back().radius_au();
    auto camera = camera_at(moon_radius * 6);
    camera.origin_render = ecliptic_to_render(loaded.motions.back().position).value;
    auto direction = glm::normalize(glm::dvec3(1, 0.01, 0));
    camera.cam_pos_rel = direction * moon_radius * 6.0;
    camera.yaw_actual_deg = std::atan2(-direction.z, -direction.x) * RAD_TO_DEG;
    camera.pitch_actual_deg = std::asin(-direction.y) * RAD_TO_DEG;
    rendering::RenderSettings settings;
    settings.surface_textures_enabled = false;
    settings.exposure = 50;
    settings.ring_shadows_enabled = false;
    probe.render(camera, simulation.snapshot(), settings);
    std::vector<float> lit;
    for (uint32_t y = 194; y < 318; ++y) lit.push_back(probe.red_at(256, y));
    settings.ring_shadows_enabled = true;
    for (bool single_box : {false, true}) {
        settings.ring_shadow_single_box = single_box;
        probe.render(camera, simulation.snapshot(), settings);
        double maximum_step = 0, minimum_visibility = 1, previous = 1;
        uint32_t partial = 0;
        for (uint32_t y = 194; y < 318; ++y) {
            check(lit[y - 194] > 0.01f, "Penumbra scan stays on the illuminated moon surface");
            double visibility = probe.red_at(256, y) / lit[y - 194];
            minimum_visibility = std::min(minimum_visibility, visibility);
            if (y > 194) maximum_step = std::max(maximum_step, std::abs(visibility - previous));
            partial += visibility > 0.1 && visibility < 0.9;
            previous = visibility;
        }
        std::cout << "Moon ring penumbra " << (single_box ? "single" : "three") << " box: largest pixel step "
                  << maximum_step << ", partial pixels " << partial << ".\n";
        check(minimum_visibility < 0.7 && partial > 20, "Finite star produces a broad partial shadow on the moon");
        check(maximum_step < 0.12, "Grazing ring penumbra has no clipped hard boundary on the moon");
    }
}

void surface_ring_shadow_contracts(RenderProbe& probe) {
    probe.resize({512, 512});
    auto loaded = surface_ring_fixture(16);
    ephemeris::Simulation simulation;
    rendering::RenderSettings settings;
    settings.ring_shadow_single_box = false;
    settings.surface_textures_enabled = false;
    settings.exposure = 50;
    auto camera = camera_at(0.02 * SOLAR_RADII_TO_AU * 6.0);
    camera.origin_render = ecliptic_to_render(loaded.motions.back().position).value;
    auto compare = [&]() {
        check(simulation.replace_system(loaded).accepted, "Surface ring shadow fixture");
        settings.ring_shadow_reference_all_hosts = true;
        probe.render(camera, simulation.snapshot(), settings);
        auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
        std::vector<uint16_t> reference(pixels, pixels + size_t(probe.extent.width) * probe.extent.height * 4);
        settings.ring_shadow_reference_all_hosts = false;
        double flux = probe.render(camera, simulation.snapshot(), settings);
        check(std::equal(reference.begin(), reference.end(), pixels), "Filtered ring shadows match all-host HDR reference exactly");
        return flux;
    };
    double shadowed = compare();
    auto masks = probe.renderer.surface_ring_candidate_masks();
    check(masks.back() == (1u << 15u), "Moon retains the sixteenth ring host and rejects unrelated ring systems");
    settings.ring_shadows_enabled = false;
    double unshadowed = probe.render(camera, simulation.snapshot(), settings);
    check(shadowed < unshadowed * 0.3, "Ring system casts a substantial shadow on a moon");
    check(probe.renderer.surface_ring_candidate_masks().empty(), "Disabled ring shadows build no masks");
    settings.ring_shadows_enabled = true;
    settings.ring_shadow_single_box = true;
    check(compare() < unshadowed * 0.3, "Single-box blur preserves ring-to-moon shadow");
    settings.ring_shadow_single_box = false;
    auto second_star = loaded.definition.bodies[0];
    second_star.name = "Unblocked star";
    loaded.definition.name_to_idx[second_star.name] = static_cast<int>(loaded.definition.bodies.size());
    loaded.definition.bodies.push_back(second_star);
    ephemeris::BodyMotion star_motion;
    star_motion.position = render_to_ecliptic(RenderPosition{{5, 2, 0}});
    loaded.motions.push_back(star_motion);
    double two_star_flux = compare();
    masks = probe.renderer.surface_ring_candidate_masks();
    check(masks[17 * 2] == (1u << 15u) && masks[17 * 2 + 1] == 0u,
          "Ring-to-moon candidates are independent for each star");
    check(two_star_flux > shadowed * 2.0, "Second star illuminates a ring-shadowed moon");
    loaded.definition.bodies.pop_back();
    loaded.motions.pop_back();
    loaded.definition.bodies[0].radius_rsun = 100;
    loaded.motions.back().position = render_to_ecliptic(RenderPosition{{2.2 * 0.1 * SOLAR_RADII_TO_AU, -0.1 * SOLAR_RADII_TO_AU, 0}});
    camera.origin_render = ecliptic_to_render(loaded.motions.back().position).value;
    settings.terrain_enabled = true;
    settings.terrain_height_km = 20;
    compare();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{5, 0.0001, 0}});
    compare();
    moon_ring_penumbra_contracts(probe);
    std::cout << "Surface ring shadows passed: HDR reference equality, ring-to-moon, sparse host 16, two stars, relief and grazing light.\n";
    probe.resize(EXTENT);
}

void surface_ring_benchmark(RenderProbe& probe, const std::filesystem::path& output) {
    check(!std::filesystem::exists(output), "Preserve surface ring benchmark results");
    std::filesystem::create_directories(output);
    probe.resize({1920, 1080});
    nlohmann::json report{{"gpu", probe.vk.device_properties().deviceName}, {"extent", {1920, 1080}},
        {"method", "Serialized offscreen draws; 40 warmup + 240 measured frames per mode, three rotating-order repetitions; CPU record includes candidate construction/upload and draw recording; GPU total includes prepare and rings, not application FPS."},
        {"scenes", nlohmann::json::array()}};
    for (auto scene : {"moon-shadow", "irrelevant", "penumbra"}) {
        auto loaded = surface_ring_fixture();
        double radius = 0.1 * SOLAR_RADII_TO_AU;
        if (std::string_view(scene) == "irrelevant")
            loaded.motions.back().position = render_to_ecliptic(RenderPosition{{20 * radius, -radius, 0}});
        if (std::string_view(scene) == "penumbra") {
            loaded.definition.bodies[0].radius_rsun = 100;
            loaded.motions.back().position = render_to_ecliptic(RenderPosition{{2.2 * radius, -radius, 0}});
        }
        ephemeris::Simulation simulation;
        check(simulation.replace_system(loaded).accepted, "Surface ring benchmark fixture");
        auto camera = camera_at(loaded.definition.bodies.back().radius_au() * 1.1);
        camera.origin_render = ecliptic_to_render(loaded.motions.back().position).value;
        rendering::RenderSettings settings;
        settings.surface_textures_enabled = false;
        settings.exposure = 50;
        std::array<std::vector<double>, 3> cpus, gpus;
        for (uint32_t repeat = 0; repeat < 3; ++repeat)
            for (uint32_t step = 0; step < 3; ++step) {
                uint32_t mode = (step + repeat) % 3;
                settings.ring_shadow_reference_all_hosts = mode == 0;
                settings.ring_shadow_single_box = mode == 2;
                for (uint32_t frame = 0; frame < 280; ++frame) {
                    auto sample = measure_terrain(probe, camera, simulation.snapshot(), settings);
                    if (frame < 40) continue;
                    cpus[mode].push_back(sample.cpu_record_ms);
                    gpus[mode].push_back(sample.gpu_total_ms);
                }
            }
        nlohmann::json row{{"name", scene}, {"modes", nlohmann::json::array()}};
        std::vector<uint16_t> reference;
        for (uint32_t mode = 0; mode < 3; ++mode) {
            const char* label = mode == 0 ? "all_hosts_three_box" : mode == 1 ? "filtered_three_box" : "filtered_single_box";
            settings.ring_shadow_reference_all_hosts = mode == 0;
            settings.ring_shadow_single_box = mode == 2;
            probe.render(camera, simulation.snapshot(), settings);
            auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
            size_t components = size_t(probe.extent.width) * probe.extent.height * 4;
            if (mode == 0) reference.assign(pixels, pixels + components);
            size_t changed = 0;
            double squared_error = 0, squared_reference = 0;
            for (size_t c = 0; c < components; ++c) {
                changed += reference[c] != pixels[c];
                double a = glm::unpackHalf2x16(reference[c]).x, b = glm::unpackHalf2x16(pixels[c]).x;
                if (c % 4 != 3) {
                    squared_error += (a - b) * (a - b);
                    squared_reference += a * a;
                }
            }
            if (mode == 1) check(changed == 0, "Benchmark filtered HDR equals all-host reference");
            row["modes"].push_back({{"name", label}, {"cpu_record_ms", terrain_timing_stats(cpus[mode])},
                {"gpu_total_ms", terrain_timing_stats(gpus[mode])}, {"changed_components", changed},
                {"relative_rms", std::sqrt(squared_error / std::max(squared_reference, 1e-20))}});
            probe.save_preview(output / (std::string(scene) + "-" + label + ".ppm"), true);
            std::cout << scene << '/' << label << ": CPU " << terrain_timing_stats(cpus[mode])["median"]
                      << " ms, GPU " << terrain_timing_stats(gpus[mode])["median"] << " ms, changed " << changed << " components.\n";
        }
        report["scenes"].push_back(std::move(row));
        check(paths::write_json(output / "report.json", report), "Write surface ring benchmark report");
    }
}

void ring_filtering_contracts(RenderProbe& probe) {
    probe.resize(EXTENT);
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{5, 2, 3}});
    ephemeris::BodyDefinition planet;
    planet.name = "Filtered rings";
    planet.radius_rsun = 0.1;
    planet.pole_ecl = {0, -1, 0};
    ephemeris::RingLayerDefinition layer;
    layer.inner = 1.4;
    layer.outer = 2.4;
    layer.color_rgb = {0.9f, 0.8f, 0.7f};
    layer.opacity = 0.7f;
    layer.gradient = {{0, 0.5f}, {1, 0.5f}};
    planet.rings.push_back(layer);
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    ephemeris::Simulation smooth, fine;
    check(smooth.replace_system(loaded).accepted, "Smooth ring filtering reference");
    auto& gradient = loaded.definition.bodies[1].rings[0].gradient;
    gradient.clear();
    for (uint32_t i = 0; i < 128; ++i) gradient.push_back({float(i) / 127.0f, i % 2 ? 0.9f : 0.1f});
    check(fine.replace_system(loaded).accepted, "Fine radial band filtering fixture");
    rendering::RenderSettings settings;
    settings.exposure = 50;
    settings.surface_textures_enabled = false;
    settings.ring_shadows_enabled = false;
    double errors[2]{};
    for (uint32_t phase = 0; phase < 8; ++phase) {
        auto camera = camera_at(planet.radius_au() * 6.0, double(phase) / 8.0);
        probe.render(camera, smooth.snapshot(), settings);
        std::vector<float> reference;
        for (uint32_t y = 116; y < 140; ++y)
            for (uint32_t x = 184; x < 208; ++x) reference.push_back(probe.red_at(x, y));
        for (uint32_t mode = 0; mode < 2; ++mode) {
            settings.ring_reference_base_level = mode == 0;
            probe.render(camera, fine.snapshot(), settings);
            size_t pixel = 0;
            for (uint32_t y = 116; y < 140; ++y)
                for (uint32_t x = 184; x < 208; ++x) {
                    double delta = probe.red_at(x, y) - reference[pixel++];
                    errors[mode] += delta * delta;
                }
        }
    }
    check(errors[0] > 1e-4, "Fine bands exercise visible base-level aliasing");
    check(errors[1] < errors[0] * 0.1, "Footprint-selected mips suppress fine-band error across subpixel camera phases");
    std::cout << "Ring minification error ratio (mips/base): " << errors[1] / errors[0] << ".\n";
    settings.ring_reference_base_level = false;
    auto& layers = loaded.definition.bodies[1].rings;
    layers[0].color_rgb = {1, 0, 0};
    layers[0].inner = 1.2;
    layers[0].outer = 1.8;
    layers[0].gradient = {{0, 1}, {1, 1}};
    auto blue = layers[0];
    blue.inner = 2.0;
    blue.outer = 2.6;
    blue.color_rgb = {0, 0, 1};
    layers.push_back(blue);
    check(fine.replace_system(loaded).accepted, "Separate red and blue profile arrays");
    probe.render(camera_at(planet.radius_au() * 40.0), fine.snapshot(), settings);
    auto channel = [&](uint32_t x, uint32_t c) {
        auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
        return glm::unpackHalf2x16(pixels[4 * (128 * probe.extent.width + x) + c]).x;
    };
    check(channel(136, 0) > 5 * channel(136, 2) && channel(141, 2) > 5 * channel(141, 0),
          "Minified array mipmaps keep independent red and blue ring profiles isolated");
}

void ring_system_cache_contracts(RenderProbe& probe) {
    probe.resize(EXTENT);
    for (uint32_t scene = 0; scene < 12; ++scene) {
        auto loaded = fixture();
        loaded.motions[0].position = render_to_ecliptic(RenderPosition{{5, 2, 2}});
        if (scene % 2) {
            ephemeris::BodyDefinition planet;
            planet.name = "Independent ring system";
            planet.radius_rsun = 0.1;
            planet.pole_ecl = glm::normalize(glm::dvec3(0, -0.5, 0.8660254037844386));
            ephemeris::RingLayerDefinition layer;
            layer.inner = 1.4;
            layer.outer = 2.4;
            layer.opacity = 0.8f;
            layer.gradient = {{0, 1}, {1, 1}};
            planet.rings.push_back(layer);
            loaded.definition.bodies.push_back(planet);
            loaded.motions.emplace_back();
        }
        ephemeris::Simulation simulation;
        check(simulation.replace_system(std::move(loaded)).accepted, "Independent system ring cache fixture");
        rendering::RenderSettings settings;
        settings.surface_textures_enabled = false;
        settings.ring_shadows_enabled = false;
        settings.exposure = 50;
        probe.render(camera_at(0.1 * SOLAR_RADII_TO_AU * 6), simulation.snapshot(), settings);
        check(probe.renderer.ring_eclipse_candidate_counts().size() == scene % 2,
              "Independent systems with equal revision counters invalidate the ring cache");
        if (scene % 2) check(probe.red_at(184, 128) > 0.01f, "Independent ring system preserves visible rings");
    }
}

void ring_specialization_contracts(RenderProbe& probe) {
    probe.resize({512, 512});
    auto loaded = fixture();
    ephemeris::BodyDefinition planet;
    planet.name = "Specialized rings";
    planet.type = "Gas Giant";
    planet.radius_rsun = 0.1;
    for (uint32_t i = 0; i < 8; ++i) {
        ephemeris::RingLayerDefinition layer;
        layer.inner = 1.2 + i * 0.18;
        layer.outer = layer.inner + (i % 2 ? 0.27 : 0.08);
        layer.opacity = i == 3 ? 0.0f : 0.2f + i * 0.07f;
        layer.color_rgb = {0.9f, 0.7f, 0.5f};
        layer.gradient = {{0.0f, 0.2f}, {0.5f, 0.9f}, {1.0f, 0.3f}};
        planet.rings.push_back(layer);
    }
    loaded.definition.name_to_idx[planet.name] = 1;
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    auto second_star = loaded.definition.bodies[0];
    second_star.name = "Second star";
    loaded.definition.name_to_idx[second_star.name] = 2;
    loaded.definition.bodies.push_back(second_star);
    loaded.motions.emplace_back();
    ephemeris::Simulation simulation;
    rendering::RenderSettings settings;
    settings.surface_textures_enabled = false;
    settings.ring_shadows_enabled = false;
    settings.exposure = 50.0f;
    uint32_t largest_delta = 0, cases = 0;
    for (auto pole : {glm::dvec3(0, 1, 0), glm::normalize(glm::dvec3(0.3, 0.8, 0.5))}) {
        auto tangent = glm::normalize(glm::cross(pole, glm::dvec3(1, 0, 0)));
        auto bitangent = glm::cross(pole, tangent);
        loaded.definition.bodies[1].pole_ecl = render_to_ecliptic(RenderPosition{pole}).value;
        for (double elevation : {0.6, -0.6, 0.0144, 0.0146, 0.0001}) {
            auto direction = glm::normalize(tangent + pole * elevation);
            auto camera = camera_at(planet.radius_au() * 6.0);
            camera.cam_pos_rel = direction * planet.radius_au() * 6.0;
            camera.yaw_actual_deg = std::atan2(-direction.z, -direction.x) * RAD_TO_DEG;
            camera.pitch_actual_deg = std::asin(-direction.y) * RAD_TO_DEG;
            for (double sun_height : {0.0, 0.5, 0.9999, 1.0001, 2.0, -2.0}) {
                double star_radius = loaded.definition.bodies[0].radius_au();
                loaded.motions[0].position = render_to_ecliptic(RenderPosition{tangent * 5.0 + pole * (star_radius * sun_height)});
                loaded.motions[2].position = render_to_ecliptic(RenderPosition{bitangent * 5.0 - pole * (star_radius * 3.0)});
                auto scene = loaded;
                if (elevation < 0.0 || elevation == 0.0146) {
                    scene.definition.bodies.pop_back();
                    scene.motions.pop_back();
                    scene.definition.name_to_idx.erase(second_star.name);
                }
                check(simulation.replace_system(std::move(scene)).accepted, "Ring specialization fixture");
                settings.ring_reference_full_annulus = true;
                settings.ring_reference_finite_source = true;
                probe.render(camera, simulation.snapshot(), settings);
                auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
                std::vector<uint16_t> reference(pixels, pixels + size_t(probe.extent.width) * probe.extent.height * 4);
                settings.ring_reference_full_annulus = false;
                settings.ring_reference_finite_source = false;
                probe.render(camera, simulation.snapshot(), settings);
                for (size_t block = 0; block < reference.size(); block += 1024) {
                    size_t end = std::min(block + 1024, reference.size());
                    if (std::memcmp(reference.data() + block, pixels + block, (end - block) * sizeof(uint16_t)) == 0) continue;
                    for (size_t i = block; i < end; ++i) {
                        uint32_t delta = uint32_t(std::abs(int(reference[i]) - int(pixels[i])));
                        largest_delta = std::max(largest_delta, delta);
                        check(delta <= 2, "Specialized rings match full-annulus finite-source reference within two FP16 ULPs");
                    }
                }
                ++cases;
            }
            std::cout << "Ring reference comparisons complete: " << cases << "/60.\n";
        }
    }
    std::cout << "Ring specialization reference comparisons passed: " << cases
              << " equinox, mixed-star, grazing and tilted views; maximum FP16 ULP delta " << largest_delta << ".\n";
    probe.resize(EXTENT);
}

void ring_benchmark(RenderProbe& probe, const std::filesystem::path& output, std::string_view only_scene = {},
                    VkExtent2D extent = {1920, 1080}, uint32_t draw_count = 1, std::string_view render_mode = "optimized") {
    check(render_mode == "optimized" || render_mode == "reference" || render_mode == "full" || render_mode == "finite" || render_mode == "base", "Known ring rendering mode");
    check(extent.width > 0 && extent.height > 0 && extent.width <= 8192 && extent.height <= 8192, "Valid ring benchmark extent");
    check(draw_count > 0 && draw_count <= 32, "Ring benchmark batch fits timestamp capacity");
    check(!std::filesystem::exists(output), "Preserve previous ring benchmark captures");
    check(probe.vk.timestamps_supported(), "Ring benchmark requires GPU timestamps");
    std::filesystem::create_directories(output);
    probe.resize(extent);
    ephemeris::Simulation simulation;
    check(simulation.replace_system(ephemeris::SystemManager{}.load_system("Solar System")).accepted,
          "Ring benchmark Solar System fixture");
    const auto& system = simulation.snapshot();
    int saturn = system.find_body("Saturn");
    check(saturn >= 0, "Ring benchmark Saturn exists");
    auto body = system.body(saturn);
    double radius = body.definition.radius_au();
    glm::dvec3 pole = glm::normalize(body.render.pole);
    glm::dvec3 light = glm::normalize(system.body(system.star_index()).render.position.value - body.render.position.value);
    glm::dvec3 planar = glm::normalize(light - pole * glm::dot(light, pole));
    rendering::RenderSettings settings;
    settings.surface_textures_enabled = false;
    settings.ring_reference_full_annulus = render_mode == "reference" || render_mode == "full";
    settings.ring_reference_finite_source = render_mode == "reference" || render_mode == "finite";
    settings.ring_reference_base_level = render_mode == "base";
    settings.exposure = 100.0f;
    nlohmann::json report{
        {"gpu", probe.vk.device_properties().deviceName}, {"extent", {extent.width, extent.height}},
        {"draws_per_submit", draw_count}, {"pipeline_statistics", probe.vk.pipeline_statistics_enabled()},
        {"render_mode", render_mode},
        {"build_type", SF_BUILD_TYPE}, {"compiler", SF_COMPILER},
        {"system_definition", ephemeris::SystemManager::serialize_definition(system.definition())},
        {"method", "Serialized offscreen celestial pass; fixed Solar System epoch; 40 warmup and 120 measured frames per mode, three alternating-order repetitions. Batched mode prepares once, repeats the celestial draw in one submit, reports amortized total/CPU time and the final ring draw timestamp. No presentation, simulation, UI, Gaia, orbits, or timed image readback. GPU rings is nested in GPU total; not application FPS."},
        {"scenes", nlohmann::json::array()}};
    for (auto name : {"lit", "transmitted", "occluded", "distant"}) {
        if (!only_scene.empty() && only_scene != name) continue;
        glm::dvec3 direction = glm::normalize(planar + pole * (glm::dot(light, pole) >= 0.0 ? 0.65 : -0.65));
        if (std::string_view(name) == "transmitted") direction = -direction;
        double distance = std::string_view(name) == "occluded" ? 1.1 : std::string_view(name) == "distant" ? 20000.0 : 5.0;
        auto camera = camera_at(radius * distance);
        camera.origin_render = body.render.position.value;
        camera.cam_pos_rel = direction * radius * distance;
        camera.yaw_actual_deg = std::atan2(-direction.z, -direction.x) * RAD_TO_DEG;
        camera.pitch_actual_deg = std::asin(-direction.y) * RAD_TO_DEG;
        std::array<std::vector<double>, 3> totals, rings, cpus;
        for (int repeat = 0; repeat < 3; ++repeat) {
            for (int step = 0; step < 3; ++step) {
                int mode = repeat % 2 == 0 ? step : 2 - step;
                settings.show_rings = mode != 0;
                settings.ring_shadows_enabled = mode == 2;
                for (int frame = 0; frame < 160; ++frame) {
                    auto sample = measure_terrain(probe, camera, system, settings, draw_count);
                    if (frame < 40) continue;
                    totals[mode].push_back(sample.gpu_total_ms / draw_count);
                    rings[mode].push_back(mode == 0 ? 0.0 : probe.vk.gpu_time_ms("rings"));
                    cpus[mode].push_back((sample.cpu_prepare_ms + sample.cpu_record_ms) / draw_count);
                }
            }
        }
        nlohmann::json scene{{"name", name}, {"distance_radii", distance},
                              {"view_direction", {direction.x, direction.y, direction.z}},
                              {"modes", nlohmann::json::array()}};
        for (int mode = 0; mode < 3; ++mode) {
            auto label = mode == 0 ? "disabled" : mode == 1 ? "rings" : "rings_and_shadows";
            scene["modes"].push_back({{"name", label}, {"gpu_total_ms", terrain_timing_stats(totals[mode])},
                                      {"gpu_rings_ms", terrain_timing_stats(rings[mode])},
                                      {"cpu_ms", terrain_timing_stats(cpus[mode])}});
            std::cout << name << '/' << label << ": GPU total " << terrain_timing_stats(totals[mode])["median"]
                      << " ms, rings " << terrain_timing_stats(rings[mode])["median"] << " ms\n";
        }
        report["scenes"].push_back(std::move(scene));
        settings.show_rings = true;
        settings.ring_shadows_enabled = true;
        probe.render(camera, system, settings);
        auto candidate_counts = probe.renderer.ring_eclipse_candidate_counts();
        report["scenes"].back()["ring_eclipse_candidate_counts"] = std::vector<uint32_t>(candidate_counts.begin(), candidate_counts.end());
        std::ofstream pixels(output / (std::string(name) + ".rgba16"), std::ios::binary);
        pixels.write(static_cast<const char*>(probe.readback.mapped), size_t(probe.extent.width) * probe.extent.height * 8);
        check(bool(pixels), "Write ring benchmark HDR capture");
        probe.save_preview(output / (std::string(name) + ".ppm"), true);
    }
    check(!report["scenes"].empty(), "Known ring benchmark scene");
    check(paths::write_json(output / "report.json", report), "Write ring benchmark report");
}

}
