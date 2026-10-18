#pragma once

namespace {

void atmosphere_floor_contracts(RenderProbe& probe) {
    probe.resize(EXTENT);
    constexpr double radius_km = 6378.137;
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1.0, 0.0, 1.0}});
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0.0;
    ephemeris::BodyDefinition body;
    body.name = "Atmosphere floor";
    body.radius_rsun = radius_km/SOLAR_RADIUS_KM;
    body.height_range_km = {-10.984, 8.848};
    body.atmosphere.enabled = true;
    terrain::TerrainRenderer terrain_bounds;
    rendering::RenderSettings bounds_settings;
    check(terrain_bounds.minimum_height_km(body, bounds_settings) == -10.984, "Terrain atmosphere bound uses established minimum without a loaded source");
    bounds_settings.terrain_enabled = true;
    bounds_settings.terrain_height_km = 20.0f;
    check(terrain_bounds.minimum_height_km(body, bounds_settings) == -20.0, "Procedural relief expands the atmospheric lower bound conservatively");
    loaded.definition.bodies.push_back(body);
    loaded.motions.emplace_back();
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Below-datum atmosphere fixture");
    rendering::RenderSettings settings;
    const auto& a = body.atmosphere;
    for (double altitude : {1.0, 0.0, -5.0, -10.0}) {
        Camera cam = camera_at((radius_km+altitude)*KM_TO_AU);
        cam.yaw_actual_deg = 90.0;
        probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f), true);
        auto transmitted = surface_center(probe);
        for (int c = 0; c < 3; ++c) {
            double rayleigh_column = a.rayleigh_scale_height_km*(std::exp(-altitude/a.rayleigh_scale_height_km)-std::exp(-a.height_km/a.rayleigh_scale_height_km));
            double mie_column = a.mie_scale_height_km*(std::exp(-std::max(altitude, 0.0)/a.mie_scale_height_km)-std::exp(-a.height_km/a.mie_scale_height_km))-std::min(altitude, 0.0);
            double expected = std::exp(-a.rayleigh_scattering[c]*rayleigh_column-a.mie_extinction[c]*mie_column);
            check(std::abs(transmitted[c]-expected) < 0.003,
                  "Below-datum transmission matches analytic molecular column and bounded aerosol continuation");
        }
        check(probe.renderer.atmosphere_stats().optical_lut_updates == (altitude == 1.0 ? 1u : 0u),
              "Changing camera altitude preserves optical tables including the below-datum domain");
    }
    body.oblateness = 0.00335;
    loaded.definition.bodies[1] = body;
    check(sim.replace_system(loaded).accepted, "Oblate below-datum atmosphere fixture");
    Camera polar = camera_at(0.0);
    polar.cam_pos_rel = {0.0, (radius_km*(1.0-body.oblateness)-5.0)*KM_TO_AU, 0.0};
    polar.pitch_actual_deg = 90.0;
    polar.up = {0.0, 0.0, 1.0};
    probe.render(polar, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f), true);
    auto transmitted = surface_center(probe);
    double model_altitude = -5.0/(1.0-body.oblateness);
    double tau = a.rayleigh_scattering.x*a.rayleigh_scale_height_km*std::exp(-model_altitude/a.rayleigh_scale_height_km)
               + a.mie_extinction.x*(a.mie_scale_height_km-model_altitude);
    check(std::abs(transmitted.r-std::exp(-tau)) < 0.003, "Oblate polar altitude uses the reference ellipsoid rather than equatorial radial distance");
    loaded.definition.bodies[1].oblateness = 0.0;
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1.0;
    loaded.definition.bodies[1].color_rgb = glm::vec3(0.0f);
    check(sim.replace_system(loaded).accepted, "Opaque surface preservation fixture");
    for (int view = 0; view < 3; ++view) {
        Camera cam = camera_at((view == 0 ? radius_km*3.0 : radius_km+1.0)*KM_TO_AU);
        if (view > 0) {
            cam.yaw_actual_deg = view == 1 ? 0.0 : 0.2;
            cam.up = {0.0, 0.0, 1.0};
        }
        settings.show_atmospheres = false;
        probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f));
        auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
        std::vector<uint16_t> bare(pixels, pixels+size_t(probe.extent.width)*probe.extent.height*4);
        settings.show_atmospheres = true;
        probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f));
        pixels = static_cast<const uint16_t*>(probe.readback.mapped);
        uint32_t protected_pixels = 0;
        for (size_t pixel = 0; pixel < bare.size(); pixel += 4) {
            if (bare[pixel] != 0 || bare[pixel+1] != 0 || bare[pixel+2] != 0) continue;
            ++protected_pixels;
            check(pixels[pixel] == bare[pixel] && pixels[pixel+1] == bare[pixel+1] && pixels[pixel+2] == bare[pixel+2],
                  "Sky atmosphere preserves every opaque surface pixel at orbit and near-ground horizon views");
        }
        check(protected_pixels > 10000, "Surface preservation regression covers the opaque disc or terrain horizon");
    }
    std::cout << "PASS: negative-altitude density, sea-level normalization, aerosol bound, oblate poles and opaque surface preservation.\n";
}

void atmosphere_surface_preview(RenderProbe& probe, const std::filesystem::path& output) {
    std::filesystem::create_directories(output);
    probe.resize({640, 480});
    auto solar = ephemeris::SystemManager{}.load_system("Solar System");
    int earth_index = solar.definition.find_body("Earth");
    check(earth_index >= 0, "Earth surface preview requires the Earth preset");
    ephemeris::LoadedSystem loaded;
    loaded.definition.name = "Solar System";
    loaded.definition.bodies = {solar.definition.bodies[solar.definition.star_idx], solar.definition.bodies[earth_index]};
    loaded.motions = {solar.motions[solar.definition.star_idx], solar.motions[earth_index]};
    loaded.definition.bodies[1].parent_idx = 0;
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Textured atmosphere preview fixture");
    auto catalog = paths::read_json(paths::external_path("data", "surfaces.json"));
    check(catalog.has_value(), "Surface preview requires the local surface catalog");
    for (const auto& entry : catalog->at("bodies")) {
        if (entry.at("body").get<std::string>() != "Earth") continue;
        probe.renderer.set_color_sources({{"Earth", terrain::ColorSource::open(paths::external_path(entry.at("albedo").get<std::string>()))}});
        probe.renderer.set_height_source("Earth", terrain::HeightSource::open(paths::external_path(entry.at("height").get<std::string>())));
    }
    auto earth = sim.snapshot().body(1);
    auto sun = sim.snapshot().body(0);
    glm::dvec3 direction = glm::normalize(sun.render.position.value-earth.render.position.value);
    glm::dvec3 tangent = glm::normalize(glm::cross(earth.render.pole, direction));
    double radius = earth.definition.radius_au();
    rendering::RenderSettings settings;
    settings.terrain_error_pixels = 4.0f;
    settings.terrain_patch_limit = 512;
    for (bool orbit : {true, false}) {
        Camera cam;
        cam.origin_render = earth.render.position.value;
        cam.tracking_idx = 1;
        cam.up = std::abs(direction.y) > 0.99 ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
        cam.cam_pos_rel = direction*(radius*2.2);
        if (!orbit) {
            double local_radius = ellipsoid_surface_radius(radius, earth.definition.oblateness, earth.render.pole, direction);
            cam.cam_pos_rel = direction*(local_radius+12.0*KM_TO_AU);
            cam.is_near_surface = true;
            cam.cam_look_aim = false;
            cam.up = direction;
            glm::dvec3 forward = glm::normalize(tangent-direction*0.03);
            cam.yaw_actual_deg = std::atan2(forward.z, forward.x)*RAD_TO_DEG;
            cam.pitch_actual_deg = std::asin(forward.y)*RAD_TO_DEG;
        }
        uint32_t quiet = 0;
        for (uint32_t frame = 0; frame < 600 && quiet < 3; ++frame) {
            probe.render(cam, sim.snapshot(), settings);
            const auto& t = probe.renderer.terrain_stats();
            const auto& c = probe.renderer.texture_stats();
            quiet = t.loading_bodies == 0 && t.cache_bakes == 0 && t.patch_updates == 0 && c.pending == 0 && c.ready == 0 ? quiet+1 : 0;
            if (quiet < 3) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        check(quiet == 3, "Textured atmosphere preview reaches stable surface residency");
        for (bool enabled : {false, true}) {
            settings.show_atmospheres = enabled;
            probe.render(cam, sim.snapshot(), settings, nullptr, true);
            probe.save_preview(output / (std::string(orbit ? "orbit" : "horizon")+(enabled ? "-atmosphere.ppm" : "-surface.ppm")), true);
        }
    }
    std::cout << "PASS: textured Earth orbital and horizon atmosphere previews exported.\n";
}

void atmosphere_benchmark(RenderProbe& probe, const std::filesystem::path& output) {
    check(probe.vk.timestamps_supported(), "Atmosphere benchmark requires GPU timestamps");
    probe.resize({1920, 1080});
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"resolution", {1920, 1080}},
        {"method", "Serialized offscreen draws; GPU atmosphere timers exclude image readback and presentation. Three repetitions, alternating floor order, 8 warmup + 24 measured frames per mode. Not application FPS."},
        {"cases", nlohmann::json::array()}};
    constexpr double radius = 6378.137*KM_TO_AU;
    for (bool orbit : {true, false}) {
        for (int mode = 0; mode < 3; ++mode) {
            std::array<std::vector<double>, 2> preparation, composition;
            for (int repeat = 0; repeat < 3; ++repeat) {
                for (int order = 0; order < 2; ++order) {
                    int floor = repeat%2 ? 1-order : order;
                    auto loaded = fixture();
                    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1.0, 0.0, 1.0}});
                    ephemeris::BodyDefinition body;
                    body.name = "Atmosphere benchmark";
                    body.radius_rsun = radius/SOLAR_RADII_TO_AU;
                    body.height_range_km = {floor ? -10.984 : 0.0, 8.848};
                    body.atmosphere.enabled = true;
                    loaded.definition.bodies.push_back(body);
                    loaded.motions.emplace_back();
                    ephemeris::Simulation sim;
                    check(sim.replace_system(loaded).accepted, "Atmosphere benchmark fixture");
                    Camera cam = camera_at(orbit ? radius*3.0 : radius+KM_TO_AU);
                    if (!orbit) cam.yaw_actual_deg = 90.0;
                    rendering::RenderSettings settings;
                    for (int frame = 0; frame < 32; ++frame) {
                        if (mode == 1) cam.cam_pos_rel.z += 0.001*KM_TO_AU;
                        if (mode == 2) {
                            auto a = sim.snapshot().body(1).definition.atmosphere;
                            a.mie_asymmetry = frame%2 ? 0.8f : 0.79f;
                            check(sim.apply(ephemeris::ApplyBodyAtmosphere{sim.snapshot().generation(), 1, sim.snapshot().body(1).revisions.definition, a}).accepted, "Benchmark optical update");
                        }
                        probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
                            probe.renderer.prepare(cmd, probe.vk, cam, sim.snapshot(), settings, probe.extent);
                            probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                            probe.vk.transition_image(cmd, probe.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
                            auto color = rhi::RenderingAttachment::clear_color(probe.color);
                            auto depth = rhi::RenderingAttachment::clear_depth_reverse_z(probe.depth);
                            probe.vk.begin_rendering(cmd, probe.extent, {&color, 1}, &depth);
                            probe.vk.set_viewport_and_scissor(cmd, probe.extent);
                            probe.renderer.render(cmd, probe.vk, cam, sim.snapshot(), settings, probe.extent);
                            probe.vk.end_rendering(cmd);
                            probe.renderer.composite_atmospheres(cmd, probe.vk, probe.color, probe.depth);
                        });
                        probe.vk.drain_gpu_timings();
                        if (frame < 8) continue;
                        preparation[floor].push_back(probe.vk.gpu_time_ms("atmosphere_luts"));
                        composition[floor].push_back(probe.vk.gpu_time_ms("atmosphere_composite"));
                    }
                }
            }
            for (int floor = 0; floor < 2; ++floor) {
                auto row = nlohmann::json{{"view", orbit ? "orbit" : "ground"}, {"mode", mode == 0 ? "cached" : mode == 1 ? "sky_update" : "optical_update"},
                    {"minimum_km", floor ? -10.984 : 0.0}, {"gpu_luts_ms", terrain_timing_stats(preparation[floor])},
                    {"gpu_composite_ms", terrain_timing_stats(composition[floor])}};
                std::cout << row.dump() << '\n';
                report["cases"].push_back(std::move(row));
            }
        }
    }
    std::filesystem::create_directories(output.parent_path());
    check(paths::write_json(output, report), "Write atmosphere benchmark report");
}

}
