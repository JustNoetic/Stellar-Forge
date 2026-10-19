#pragma once

namespace {

ephemeris::LoadedSystem atmosphere_shadow_fixture(bool rings = false) {
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{rings ? glm::dvec3(0, 1, 1) : glm::dvec3(0, 0, 1)});
    ephemeris::BodyDefinition planet;
    planet.name = "Atmosphere shadow receiver";
    planet.radius_rsun = 1000.0/SOLAR_RADIUS_KM;
    planet.color_rgb = glm::vec3(0.0f);
    planet.pole_ecl = render_to_ecliptic(RenderPosition{{0, 1, 0}}).value;
    planet.atmosphere.enabled = true;
    planet.atmosphere.height_km = 100;
    planet.atmosphere.rayleigh_scale_height_km = 10;
    planet.atmosphere.rayleigh_scattering = {0.003f, 0.006f, 0.01f};
    planet.atmosphere.mie_scattering = planet.atmosphere.mie_extinction = glm::vec3(0.0f);
    planet.atmosphere.ground_albedo = glm::vec3(0.0f);
    if (rings) {
        ephemeris::RingLayerDefinition ring;
        ring.inner = 1.2;
        ring.outer = 2.4;
        ring.opacity = 0.95f;
        ring.gradient = {{0, 1}, {.25f, 1}, {.27f, 0}, {.4f, 0}, {.42f, 1}, {1, 1}};
        planet.rings.push_back(ring);
    }
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    if (!rings) {
        ephemeris::BodyDefinition moon;
        moon.name = "Offscreen shadow caster";
        moon.radius_rsun = 20.0/SOLAR_RADIUS_KM;
        loaded.definition.bodies.push_back(moon);
        loaded.motions.emplace_back();
        loaded.motions.back().position = render_to_ecliptic(RenderPosition{{0, 0, 2000*KM_TO_AU}});
    }
    return loaded;
}

std::vector<glm::vec3> atmosphere_shadow_pixels(const RenderProbe& probe) {
    const auto* data = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<glm::vec3> result(size_t(probe.extent.width)*probe.extent.height);
    for (size_t i = 0; i < result.size(); ++i)
        for (int c = 0; c < 3; ++c) result[i][c] = glm::unpackHalf2x16(data[i*4+c]).x;
    return result;
}

void atmosphere_shadow_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    probe.resize({129, 129});
    if (!output.empty()) std::filesystem::create_directories(output);
    auto loaded = atmosphere_shadow_fixture();
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Atmosphere eclipse fixture");
    auto camera = camera_at(1100*KM_TO_AU);
    rendering::RenderSettings settings;
    settings.surface_textures_enabled = false;
    settings.ringshine_enabled = false;
    settings.exposure = 8;
    settings.atmosphere_shadow_reference = true;
    settings.atmosphere_shadow_temporal = false;
    rendering::RenderSettingsStore validation;
    for (uint32_t samples : {4u, 8u, 16u}) {
        auto draft = validation.values();
        draft.atmosphere_shadow_samples = samples;
        check(validation.apply(draft).accepted, "Validated atmospheric shadow sample budget");
    }
    auto invalid = validation.values();
    invalid.atmosphere_shadow_samples = 8192;
    check(!validation.apply(invalid).accepted, "Adaptive-sized atmospheric shadow budgets are rejected");
    auto render = [&](bool enabled, float depth = 0.0f, glm::vec3 background = glm::vec3(0)) {
        settings.atmosphere_shadows_enabled = enabled;
        probe.render(camera, simulation.snapshot(), settings, nullptr, false, background, true, depth);
        return surface_center(probe);
    };
    auto oracle = [&](double offset, double end) {
        glm::dvec3 deficit(0);
        constexpr int steps = 32768;
        for (int i = 0; i < steps; ++i) {
            double t = end*(i+0.5)/steps;
            double h = 100-t;
            double z = 1100-t;
            double source = SOLAR_RADIUS_KM*((2000-z)/(AU_TO_KM-z));
            double visibility = eclipse_circle_oracle(20/source, offset/source);
            double column = 10*(std::exp(-h/10)-std::exp(-10.0));
            for (int c = 0; c < 3; ++c) {
                double beta = loaded.definition.bodies[1].atmosphere.rayleigh_scattering[c];
                double irradiance = PI*settings.exposure;
                deficit[c] += irradiance*(3/(8*PI))*beta*std::exp(-h/10)*std::exp(-2*beta*column)
                            *(1-visibility)*end/steps;
            }
        }
        return glm::vec3(deficit);
    };
    for (bool bruneton : {false, true}) for (bool dual : {false, true}) {
        settings.aerial_perspective_bruneton = bruneton;
        settings.atmosphere_dual_source = dual;
        for (double offset : {0.0, 19.0, 23.0, 30.0}) {
            auto motions = loaded.motions;
            motions[2].position = render_to_ecliptic(RenderPosition{{offset*KM_TO_AU, 0, 2000*KM_TO_AU}});
            check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted,
                  "Move atmospheric shadow caster without changing optics");
            auto unshadowed = render(false);
            auto shadowed = render(true);
            auto expected = oracle(offset, 100);
            std::cout << "Atmosphere deficit " << bruneton << '/' << dual << " offset " << offset << ": "
                      << (unshadowed-shadowed).r << ',' << (unshadowed-shadowed).g << ',' << (unshadowed-shadowed).b
                      << " expected " << expected.r << ',' << expected.g << ',' << expected.b << '\n';
            check(glm::all(glm::greaterThanEqual(unshadowed-shadowed, glm::min(expected, unshadowed)-glm::vec3(.0025f))),
                  "Total shadow deficit includes the independently integrated single-scattering contribution");
            if (offset == 0) check(glm::length(shadowed) < .02f*glm::length(unshadowed)+.002f,
                  "Full umbra suppresses cached multiple-scattering fill");
            check(probe.renderer.atmosphere_stats().optical_lut_updates == 0 &&
                  probe.renderer.atmosphere_stats().bruneton_lut_updates == 0,
                  "Caster motion and shadow toggles reuse optical and Bruneton tables");
            if (offset == 30) check(unshadowed == shadowed, "Clear atmospheric rays remain bitwise unchanged");
        }
    }
    auto motions = loaded.motions;
    check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted, "Restore total eclipse");
    settings.aerial_perspective_bruneton = false;
    for (double end : {0.1, 10.0, 50.0}) {
        float depth = camera.compute_near_plane_au()/float(end*KM_TO_AU);
        auto clear = render(false, depth);
        auto dark = render(true, depth);
        std::cout << "Finite shadow " << end << ": " << (clear-dark).r << ',' << (clear-dark).g << ',' << (clear-dark).b
                  << " expected " << oracle(0, end).r << ',' << oracle(0, end).g << ',' << oracle(0, end).b << '\n';
        check(glm::all(glm::greaterThanEqual(clear-dark, glm::min(oracle(0, end), clear)-glm::vec3(.001f))),
              "Finite shadow deficit includes direct scattering only up to the opaque endpoint");
        auto transmission = render(true, depth, glm::vec3(1))-dark;
        auto expected = render(false, depth, glm::vec3(1))-clear;
        check(glm::length(transmission-expected) < 0.002f, "Volumetric shadow leaves RGB extinction unchanged");
    }
    for (bool bruneton : {false, true}) for (bool dual : {false, true}) {
        settings.aerial_perspective_bruneton = bruneton;
        settings.atmosphere_dual_source = dual;
        float depth = camera.compute_near_plane_au()/float(50*KM_TO_AU);
        glm::vec3 surface(.4f, .6f, .8f);
        auto sky = render(true);
        auto endpoint = render(true, depth, surface);
        TransparencyProbe transparent(probe);
        TransparentTestLayer layer{{surface, .6f}, {depth, 0, 0, 0}};
        auto actual = transparent.render(camera, simulation.snapshot(), settings, {&layer, 1});
        check(glm::length(actual-(.6f*endpoint+.4f*sky)) < .003f,
              "Transparent endpoints receive the same shadow correction in both LUT and blend paths");
    }
    auto eclipse = render(true);
    auto overlapped = loaded;
    overlapped.definition.bodies.push_back(loaded.definition.bodies[2]);
    overlapped.definition.bodies.back().name = "Overlapping shadow caster";
    overlapped.motions.push_back(loaded.motions[2]);
    check(simulation.replace_system(overlapped).accepted, "Overlapping atmosphere shadow fixture");
    check(glm::length(render(true)-eclipse) < .0005f, "Overlapping umbrae subtract direct radiance only once");
    check(simulation.replace_system(loaded).accepted, "Restore single atmosphere caster");
    auto single_deficit = render(false)-render(true);
    auto second = loaded.definition.bodies[0];
    second.name = "Unblocked second atmosphere star";
    loaded.definition.bodies.push_back(second);
    loaded.motions.emplace_back();
    loaded.motions.back().position = render_to_ecliptic(RenderPosition{{1, 0, 0}});
    check(simulation.replace_system(loaded).accepted, "Multiple atmosphere shadow stars");
    auto two_deficit = render(false)-render(true);
    check(glm::length(single_deficit-two_deficit) < .002f, "A blocked star leaves independent stellar scattering intact");
    loaded = atmosphere_shadow_fixture(true);
    check(simulation.replace_system(loaded).accepted, "Own-ring atmosphere shadow fixture");
    probe.resize({513, 513});
    camera = camera_at(3000*KM_TO_AU);
    auto capture = [&](bool enabled, const char* name) {
        render(enabled);
        if (!output.empty()) probe.save_preview(output/name, true);
        return atmosphere_shadow_pixels(probe);
    };
    auto clear = capture(false, "rings-clear.ppm");
    auto dark = capture(true, "rings-shadow.ppm");
    size_t affected = 0;
    for (size_t i = 0; i < clear.size(); ++i) {
        check(glm::all(glm::greaterThanEqual(dark[i], glm::vec3(0))) &&
              glm::all(glm::lessThanEqual(dark[i], clear[i]+glm::vec3(0.0001f))), "Ring deficit stays finite, nonnegative and subtractive");
        if (glm::length(clear[i]-dark[i]) > 0.001f) ++affected;
    }
    std::cout << "Ring shadow affects " << affected << " full-resolution atmosphere pixels.\n";
    check(affected > 500, "Own rings shadow the atmosphere at orbital resolution");
    settings.ring_shadows_enabled = false;
    auto disabled = capture(true, "rings-disabled.ppm");
    check(disabled == clear, "Ring shadow toggle restores the exact clear-atmosphere image");
    settings.ring_shadows_enabled = true;
    settings.show_rings = false;
    check(capture(true, "rings-hidden.ppm") == clear, "Hidden rings do not shadow atmospheric scattering");
    settings.show_rings = true;
    probe.resize({129, 129});
    for (bool thin : {false, true}) {
        if (thin) {
            auto& ring = loaded.definition.bodies[1].rings[0];
            ring.gradient = {{0, 0}, {.166f, 0}, {.1667f, 1}, {.1672f, 1}, {.168f, 0}, {1, 0}};
            loaded.definition.bodies[0].radius_rsun = .001;
            check(simulation.replace_system(loaded).accepted, "Thin ring-band atmosphere fixture");
        }
        settings.atmosphere_shadow_reference = false;
        settings.atmosphere_shadow_temporal = true;
        auto thin_clear = capture(false, thin ? "thin-ring-clear.ppm" : "rings-clear-small.ppm");
        auto first = capture(true, thin ? "thin-ring-first.ppm" : "rings-first.ppm");
        for (int frame = 0; frame < 64; ++frame) render(true);
        auto optimized = capture(true, thin ? "thin-ring-shadow.ppm" : "rings-optimized.ppm");
        settings.atmosphere_shadow_reference = true;
        auto reference = capture(true, thin ? "thin-ring-reference.ppm" : "rings-reference.ppm");
        float maximum = 0, shadow = 0;
        double error = 0, first_error = 0;
        for (size_t i = 0; i < optimized.size(); ++i) {
            maximum = std::max(maximum, glm::length(optimized[i]-reference[i]));
            shadow = std::max(shadow, glm::length(thin_clear[i]-reference[i]));
            error += glm::dot(optimized[i]-reference[i], optimized[i]-reference[i]);
            first_error += glm::dot(first[i]-reference[i], first[i]-reference[i]);
        }
        double rmse = std::sqrt(error/optimized.size()), first_rmse = std::sqrt(first_error/optimized.size());
        std::cout << "Ring deficit vs dense unclipped reference (thin=" << thin << "): maximum " << maximum
                  << ", RMSE " << rmse << " (first frame " << first_rmse << ")\n";
        check(shadow > .00001f, "Dense ring reference resolves a real shadow in each band fixture");
        check(rmse < .01f,
              "Normalized fixed-budget ring integration matches the dense reference");
    }
    loaded = atmosphere_shadow_fixture();
    check(simulation.replace_system(loaded).accepted, "Temporal moving-caster fixture");
    camera = camera_at(1100*KM_TO_AU);
    settings.atmosphere_shadow_reference = false;
    settings.atmosphere_shadow_temporal = true;
    for (int frame = 0; frame < 32; ++frame) render(true);
    auto moved = loaded.motions;
    moved[2].position = render_to_ecliptic(RenderPosition{{30*KM_TO_AU, 0, 2000*KM_TO_AU}});
    check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(moved)}).accepted, "Move shadow out of the center ray");
    auto moved_shadow = render(true);
    check(moved_shadow == render(false), "Fast caster changes reject shadow history without a lingering dark center");
    check(simulation.replace_system(loaded).accepted, "Temporal viewport fixture");
    settings.atmosphere_shadows_enabled = true;
    settings.atmosphere_shadow_temporal = true;
    {
        TransparencyProbe transparent(probe);
        for (bool bruneton : {false, true}) for (bool dual : {false, true}) {
            settings.aerial_perspective_bruneton = bruneton;
            settings.atmosphere_dual_source = dual;
            float near_depth = camera.compute_near_plane_au()/float(30*KM_TO_AU);
            float far_depth = camera.compute_near_plane_au()/float(90*KM_TO_AU);
            std::array<TransparentTestLayer, 2> layers{{{{.4f, .6f, .8f, .6f}, {far_depth, 0, 0, 0}},
                                                       {{.8f, .3f, .2f, .3f}, {near_depth, 0, 0, 0}}}};
            for (size_t count : {size_t(1), size_t(2)}) {
                std::span<const TransparentTestLayer> visible(layers.data(), count);
                settings.atmosphere_shadow_reference = true;
                auto expected = transparent.render(camera, simulation.snapshot(), settings, visible);
                settings.atmosphere_shadow_reference = false;
                glm::vec3 actual(0);
                for (int frame = 0; frame < 64; ++frame)
                    actual = transparent.render(camera, simulation.snapshot(), settings, visible);
                std::cout << "Temporal transparent endpoints " << bruneton << '/' << dual << '/' << count
                          << " error " << glm::length(actual-expected) << '\n';
                check(glm::length(actual-expected) < (count == 1 ? .008f : .02f),
                      "Shared stochastic prefixes and additional-depth fallback agree with endpoint-specific dense transport");
            }
        }
    }
    settings.atmosphere_shadow_temporal = false;
    auto frozen = render(true);
    check(frozen == render(true), "Disabling temporal accumulation produces stable spatial jitter");
    probe.resize({257, 257});
    check(glm::all(glm::greaterThanEqual(render(true), glm::vec3(0))), "History resize retains finite shadow output");
    if (!output.empty()) {
        loaded = atmosphere_shadow_fixture(true);
        check(simulation.replace_system(loaded).accepted, "Stochastic ring preview fixture");
        camera = camera_at(3000*KM_TO_AU);
        settings.aerial_perspective_bruneton = true;
        settings.atmosphere_dual_source = false;
        settings.atmosphere_shadow_temporal = true;
        probe.resize({513, 513});
        render(false);
        probe.save_preview(output/"stochastic-clear.ppm", true);
        render(true);
        probe.save_preview(output/"stochastic-first.ppm", true);
        for (int frame = 0; frame < 64; ++frame) render(true);
        probe.save_preview(output/"stochastic-settled.ppm", true);
    }
    probe.resize(EXTENT);
    std::cout << "PASS: full-resolution atmospheric body/ring shadows, penumbrae, finite endpoints, extinction and cache reuse.\n";
}

void atmosphere_shadow_benchmark(RenderProbe& probe, const std::filesystem::path& output, uint32_t samples = 16) {
    check(samples == 4 || samples == 8 || samples == 16, "Shadow benchmark sample budget");
    check(probe.vk.timestamps_supported(), "Atmosphere shadow benchmark requires timestamps");
    probe.resize({1920, 1080});
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"resolution", {1920, 1080}}, {"samples", samples},
        {"method", "Serialized offscreen atmosphere composition, no readback/presentation. Three alternating-order repetitions; 8 warmup and 16 measured frames. Not application FPS."},
        {"cases", nlohmann::json::array()}};
    for (int scene = 0; scene < 3; ++scene) {
        auto loaded = atmosphere_shadow_fixture(scene == 2);
        if (scene == 0) loaded.motions[2].position = render_to_ecliptic(RenderPosition{{10000*KM_TO_AU, 0, 2000*KM_TO_AU}});
        ephemeris::Simulation simulation;
        check(simulation.replace_system(loaded).accepted, "Atmosphere shadow benchmark fixture");
        auto camera = camera_at((scene == 2 ? 3000 : 1100)*KM_TO_AU);
        rendering::RenderSettings settings;
        settings.atmosphere_shadow_samples = samples;
        settings.ringshine_enabled = false;
        settings.surface_textures_enabled = false;
        for (bool bruneton : {false, true}) {
            settings.aerial_perspective_bruneton = bruneton;
            std::array<std::vector<double>, 2> composition, prepare_cpu, shadow_gpu;
            for (int repeat = 0; repeat < 3; ++repeat) for (int order = 0; order < 2; ++order) {
                int enabled = repeat%2 ? 1-order : order;
                settings.atmosphere_shadows_enabled = enabled;
                for (int frame = 0; frame < 24; ++frame) {
                    double cpu = 0;
                    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
                        auto start = std::chrono::steady_clock::now();
                        probe.renderer.prepare(cmd, probe.vk, camera, simulation.snapshot(), settings, probe.extent);
                        cpu = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
                        probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                        probe.vk.transition_image(cmd, probe.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
                        auto color = rhi::RenderingAttachment::clear_color(probe.color);
                        auto depth = rhi::RenderingAttachment::clear_depth_reverse_z(probe.depth);
                        probe.vk.begin_rendering(cmd, probe.extent, {&color, 1}, &depth);
                        probe.vk.end_rendering(cmd);
                        probe.renderer.composite_atmospheres(cmd, probe.vk, probe.color, probe.depth);
                    });
                    probe.vk.drain_gpu_timings();
                    if (frame < 8) continue;
                    composition[enabled].push_back(probe.vk.gpu_time_ms("atmosphere_composite"));
                    shadow_gpu[enabled].push_back(probe.renderer.atmosphere_stats().shadow_updates ? probe.vk.gpu_time_ms("atmosphere_shadows") : 0.0);
                    prepare_cpu[enabled].push_back(cpu);
                }
            }
            for (int enabled = 0; enabled < 2; ++enabled) {
                nlohmann::json row{{"scene", scene == 0 ? "clear" : scene == 1 ? "eclipse" : "rings"}, {"bruneton", bruneton},
                    {"shadows", bool(enabled)}, {"samples", samples}, {"gpu_composite_ms", terrain_timing_stats(composition[enabled])},
                    {"gpu_shadow_ms", terrain_timing_stats(shadow_gpu[enabled])},
                    {"cpu_prepare_ms", terrain_timing_stats(prepare_cpu[enabled])}};
                std::cout << row.dump() << '\n';
                report["cases"].push_back(std::move(row));
            }
        }
    }
    std::filesystem::create_directories(output.parent_path());
    check(paths::write_json(output, report), "Write atmospheric shadow benchmark");
    probe.resize(EXTENT);
}

}
