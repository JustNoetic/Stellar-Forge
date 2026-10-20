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
    std::vector<uint16_t> data(size_t(probe.extent.width)*probe.extent.height*4);
    std::memcpy(data.data(), probe.readback.mapped, data.size()*sizeof(uint16_t));
    std::vector<glm::vec3> result(size_t(probe.extent.width)*probe.extent.height);
    for (size_t i = 0; i < result.size(); ++i)
        for (int c = 0; c < 3; ++c) result[i][c] = glm::unpackHalf2x16(data[i*4+c]).x;
    return result;
}

void atmosphere_adaptive_shadow_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    rendering::AtmosphereParametersGPU parameters;
    parameters.radii_heights = {1000, 1100, 10, 1.2f};
    parameters.reference_shape = {1000, 1, 1000, 0};
    parameters.rayleigh = {0.003f, 0.006f, 0.01f, 0};
    rendering::AtmosphereShadowHeaderGPU header;
    header.temporal.w = 1;
    auto uniform = probe.vk.create_host_buffer(sizeof(parameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &parameters);
    auto shadows = probe.vk.create_host_buffer(sizeof(header), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &header);
    auto results = probe.vk.create_host_buffer(12*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    auto pipeline = rhi::ShaderCache::create_compute_pipeline(probe.vk, "tests/atmosphere_shadow_budget.comp", bindings);
    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        rhi::PushDescriptorWrite writes[] = {rhi::PushDescriptorWrite::uniform_buffer(0, uniform),
            rhi::PushDescriptorWrite::storage_buffer(1, results), rhi::PushDescriptorWrite::storage_buffer(11, shadows)};
        probe.vk.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, 12, 1, 1);
        probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });
    vmaInvalidateAllocation(probe.vk.allocator(), results.allocation, 0, VK_WHOLE_SIZE);
    auto* budgets = static_cast<const glm::vec4*>(results.mapped);
    for (int first : {0, 4, 8}) {
        check(budgets[first].x == 4 && budgets[first+1].x == 4 && budgets[first+3].x == 4,
            "Vertical sky, short ground and nearby horizontal columns use the minimum shadow budget");
        check(budgets[first+2].x == budgets[first+2].w,
            "A long grazing column retains the selected maximum shadow budget");
    }
    rhi::ShaderCache::destroy_pipeline(probe.vk, pipeline);
    probe.vk.destroy_buffer(results);
    probe.vk.destroy_buffer(shadows);
    probe.vk.destroy_buffer(uniform);

    probe.resize({129, 129});
    if (!output.empty()) std::filesystem::create_directories(output);
    for (int scene = 0; scene < 8; ++scene) {
        auto loaded = atmosphere_shadow_fixture(scene != 7);
        if (scene == 6) {
            loaded.definition.bodies[1].rings[0].gradient = {{0, 0}, {.166f, 0}, {.1667f, 1}, {.1672f, 1}, {.168f, 0}, {1, 0}};
            loaded.definition.bodies[0].radius_rsun = .001;
        }
        if (scene == 7) {
            auto second = loaded.definition.bodies[0];
            second.name = "Adaptive unblocked second star";
            loaded.definition.bodies.push_back(second);
            loaded.motions.emplace_back();
            loaded.motions.back().position = render_to_ecliptic(RenderPosition{{1, 0, 0}});
        }
        ephemeris::Simulation simulation;
        check(simulation.replace_system(loaded).accepted, "Adaptive atmosphere shadow fixture");
        auto camera = camera_at((scene == 1 ? 1100.1 : scene >= 2 && scene <= 4 ? 1099.9 : scene == 5 ? 1001 : scene == 7 ? 1100 : 3000)*KM_TO_AU);
        if (scene == 1 || scene == 2 || scene == 5) {
            camera.cam_pos_rel.y = -10*KM_TO_AU;
            camera.yaw_actual_deg = -std::acos(1000/(glm::length(camera.cam_pos_rel)*AU_TO_KM))*RAD_TO_DEG;
        }
        if (scene == 3) camera.yaw_actual_deg = 90;
        rendering::RenderSettings settings;
        settings.surface_textures_enabled = settings.ringshine_enabled = false;
        settings.atmosphere_shadow_vrs = false;
        settings.atmosphere_shadow_temporal = false;
        settings.exposure = 8;
        auto capture = [&](bool dense, bool adaptive) {
            settings.atmosphere_shadow_reference = dense;
            settings.atmosphere_shadow_adaptive = adaptive;
            probe.render(camera, simulation.snapshot(), settings, nullptr, false, glm::vec3(0), true);
            return atmosphere_shadow_pixels(probe);
        };
        auto reference = capture(true, false);
        for (uint32_t maximum : {4u, 8u, 16u}) {
            settings.atmosphere_shadow_samples = maximum;
            auto fixed = capture(false, false);
            auto adaptive = capture(false, true);
            check(adaptive == capture(false, true), "Adaptive sample allocation and frozen jitter remain stable without temporal accumulation");
            if (maximum == 4) check(fixed == adaptive, "A four-sample ceiling preserves the fixed four-sample result exactly");
            double fixed_error = 0, adaptive_error = 0, difference = 0;
            for (size_t i = 0; i < adaptive.size(); ++i) {
                fixed_error += glm::dot(fixed[i]-reference[i], fixed[i]-reference[i]);
                adaptive_error += glm::dot(adaptive[i]-reference[i], adaptive[i]-reference[i]);
                difference += glm::dot(adaptive[i]-fixed[i], adaptive[i]-fixed[i]);
            }
            double fixed_rmse = std::sqrt(fixed_error/adaptive.size());
            double adaptive_rmse = std::sqrt(adaptive_error/adaptive.size());
            std::cout << "Adaptive shadow scene " << scene << " cap " << maximum << " fixed/adaptive RMSE "
                      << fixed_rmse << '/' << adaptive_rmse << " delta " << std::sqrt(difference/adaptive.size()) << '\n';
            check(adaptive_rmse <= fixed_rmse+0.0015, "Adaptive shadow quality stays close to the selected fixed budget against dense integration");
            if (!output.empty() && maximum == 16) {
                probe.save_preview(output/("adaptive-"+std::to_string(scene)+".ppm"), true);
                capture(false, false);
                probe.save_preview(output/("fixed-"+std::to_string(scene)+".ppm"), true);
            }
        }
    }
    probe.resize(EXTENT);
    std::cout << "PASS: adaptive shadow budgets, boundary views, narrow rings, independent stars and dense-reference quality.\n";
}

void atmosphere_shadow_vrs_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    rendering::AtmosphereParametersGPU parameters;
    parameters.camera = {0, 0, 1001, 1};
    parameters.axis_x = {1, 0, 0, 0}; parameters.axis_y = {0, 1, 0, 0}; parameters.axis_z = {0, 0, 1, 0};
    rendering::AtmosphereShadowHeaderGPU header;
    header.eye_focal = {0, 0, 1001, 1000};
    header.stars[0] = {0, 0, 1e8f, 1e6f};
    std::array<glm::vec4, 36> fractions, guides;
    for (int scene = 0; scene < 9; ++scene) for (int tap = 0; tap < 4; ++tap) {
        fractions[scene*4+tap] = {0.2f+0.01f*float(tap%2), 0.3f, 0.4f, 1};
        guides[scene*4+tap] = {std::log2(10.0f), 0, 0, 1};
    }
    fractions[4].x = 1;
    guides[8].z = 1;
    guides[12].x += .1f;
    guides[16].w = 0;
    fractions[20].w = 0;
    guides[24].y = 1;
    auto fraction = probe.vk.create_image_2d(2, 18, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    auto guide = probe.vk.create_image_2d(2, 18, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    probe.vk.upload_image_2d(fraction, fractions.data(), sizeof(fractions));
    probe.vk.upload_image_2d(guide, guides.data(), sizeof(guides));
    auto uniform = probe.vk.create_host_buffer(sizeof(parameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &parameters);
    auto shadows = probe.vk.create_host_buffer(sizeof(header), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &header);
    auto results = probe.vk.create_host_buffer(9*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {22, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}};
    auto pipeline = rhi::ShaderCache::create_compute_pipeline(probe.vk, "tests/atmosphere_shadow_vrs.comp", bindings);
    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        rhi::PushDescriptorWrite writes[] = {rhi::PushDescriptorWrite::uniform_buffer(0, uniform),
            rhi::PushDescriptorWrite::storage_buffer(1, results), rhi::PushDescriptorWrite::storage_buffer(11, shadows),
            rhi::PushDescriptorWrite::sampled_image(21, fraction, probe.vk.sampler_nearest_clamp()),
            rhi::PushDescriptorWrite::sampled_image(22, guide, probe.vk.sampler_nearest_clamp())};
        probe.vk.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, 9, 1, 1);
        probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });
    vmaInvalidateAllocation(probe.vk.allocator(), results.allocation, 0, VK_WHOLE_SIZE);
    auto* values = static_cast<const glm::vec4*>(results.mapped);
    check(values[0].w == 1 && std::abs(values[0].x-.205f) < 1e-5f, "VRS interpolates compatible smooth shadow fractions");
    for (int scene : {1, 2, 3, 4, 5, 6, 8})
        check(values[scene].w == 0, "VRS rejects shadow edges, geometry/segment/coverage mismatches and missing border taps");
    check(values[7].w == 1 && std::abs(values[7].x-.2f) < 1e-5f, "Native anchor pixels reuse their exact shadow sample");
    rhi::ShaderCache::destroy_pipeline(probe.vk, pipeline);
    probe.vk.destroy_buffer(results); probe.vk.destroy_buffer(shadows); probe.vk.destroy_buffer(uniform);
    probe.vk.destroy_image(fraction); probe.vk.destroy_image(guide);

    if (!output.empty()) std::filesystem::create_directories(output);
    for (auto extent : {VkExtent2D{257, 259}, VkExtent2D{1920, 1080}}) {
        probe.resize(extent);
        for (int scene = 0; scene < 6; ++scene) {
            auto loaded = atmosphere_shadow_fixture(scene != 5);
            if (scene == 1) loaded.definition.bodies[0].radius_rsun = 10;
            if (scene == 2) {
                loaded.definition.bodies[1].rings[0].gradient = {{0, 0}, {.166f, 0}, {.1667f, 1}, {.1672f, 1}, {.168f, 0}, {1, 0}};
                loaded.definition.bodies[0].radius_rsun = .001;
            }
            if (scene == 5) {
                loaded.definition.bodies.push_back(loaded.definition.bodies[0]);
                loaded.definition.bodies.back().name = "VRS independent star";
                loaded.motions.emplace_back();
                loaded.motions.back().position = render_to_ecliptic(RenderPosition{{1, 0, 0}});
            }
            ephemeris::Simulation simulation;
            check(simulation.replace_system(loaded).accepted, "VRS shadow fixture");
            auto camera = camera_at((scene == 3 ? 1099.9 : scene == 4 ? 1001 : scene == 5 ? 1100 : 3000)*KM_TO_AU);
            if (scene == 3 || scene == 4) {
                camera.cam_pos_rel.y = -10*KM_TO_AU;
                camera.yaw_actual_deg = -std::acos(1000/(glm::length(camera.cam_pos_rel)*AU_TO_KM))*RAD_TO_DEG;
            }
            rendering::RenderSettings settings;
            settings.atmosphere_shadow_temporal = false;
            settings.surface_textures_enabled = settings.ringshine_enabled = false;
            settings.exposure = 8;
            auto capture = [&](bool vrs) {
                settings.atmosphere_shadow_vrs = vrs;
                probe.render(camera, simulation.snapshot(), settings);
                return atmosphere_shadow_pixels(probe);
            };
            for (int frame = 0; frame < 4; ++frame) capture(false);
            for (bool adaptive : {false, true}) {
                settings.atmosphere_shadow_adaptive = adaptive;
                auto native = capture(false);
                auto reconstructed = capture(true);
                check(probe.renderer.atmosphere_stats().shadow_vrs_updates == (scene == 5 ? 0u : 1u), "VRS routes one-star shadows and bypasses independent multiple stars");
                double difference = 0;
                float maximum = 0;
                for (size_t i = 0; i < native.size(); ++i) {
                    difference += glm::dot(native[i]-reconstructed[i], native[i]-reconstructed[i]);
                    maximum = std::max(maximum, glm::length(native[i]-reconstructed[i]));
                }
                double rmse = std::sqrt(difference/native.size());
                std::cout << "VRS " << extent.width << 'x' << extent.height << " scene " << scene << " adaptive " << adaptive << " native difference RMSE " << rmse << " max " << maximum << '\n';
                check(rmse < .0015 && maximum < .05f, "VRS preserves native shadow quality on rings, depth edges and atmosphere boundary views");
                if (scene == 2 || scene == 5) check(native == reconstructed, "Narrow ring features and independent stars remain exactly native");
                auto repeated = capture(true);
                if (reconstructed != repeated) {
                    float change = 0;
                    size_t count = 0;
                    for (size_t i = 0; i < repeated.size(); ++i) if (reconstructed[i] != repeated[i]) {
                        change = std::max(change, glm::length(reconstructed[i]-repeated[i]));
                        ++count;
                    }
                    std::cout << "Frozen VRS changes " << count << " maximum " << change << '\n';
                }
                check(reconstructed == repeated, "Frozen VRS reconstruction has no history dependency");
                if (!output.empty() && adaptive) {
                    probe.save_preview(output/("vrs-"+std::to_string(extent.width)+"-"+std::to_string(scene)+".ppm"), true);
                    capture(false);
                    probe.save_preview(output/("native-"+std::to_string(extent.width)+"-"+std::to_string(scene)+".ppm"), true);
                }
            }
            if (probe.vk.fragment_stores_enabled()) {
                TransparencyProbe transparent(probe);
                float depth = camera.compute_near_plane_au()/float(20*KM_TO_AU);
                std::array<TransparentTestLayer, 2> layers{{{{.4f,.6f,.8f,.6f},{depth,0,0,0}},{{.8f,.3f,.2f,.3f},{2*depth,0,0,0}}}};
                settings.atmosphere_shadow_vrs = false;
                auto native = transparent.render(camera, simulation.snapshot(), settings, layers);
                settings.atmosphere_shadow_vrs = true;
                check(native == transparent.render(camera, simulation.snapshot(), settings, layers), "Transparent shadow prefixes retain native samples with VRS enabled");
            }
        }
    }
    auto loaded = atmosphere_shadow_fixture(true);
    loaded.definition.bodies.push_back(loaded.definition.bodies[1]);
    loaded.definition.bodies.back().name = "Second VRS atmosphere";
    loaded.motions.emplace_back();
    loaded.motions.back().position = render_to_ecliptic(RenderPosition{{2500*KM_TO_AU, 0, -1000*KM_TO_AU}});
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Shared VRS scratch fixture");
    auto camera = camera_at(6000*KM_TO_AU);
    rendering::RenderSettings settings;
    settings.atmosphere_shadow_temporal = false;
    settings.surface_textures_enabled = settings.ringshine_enabled = false;
    for (auto extent : {VkExtent2D{258, 260}, VkExtent2D{127, 131}, VkExtent2D{33, 35}, VkExtent2D{257, 259}}) {
        probe.resize(extent);
        settings.atmosphere_shadow_vrs = false;
        probe.render(camera, simulation.snapshot(), settings);
        auto native = atmosphere_shadow_pixels(probe);
        settings.atmosphere_shadow_vrs = true;
        probe.render(camera, simulation.snapshot(), settings);
        check(probe.renderer.atmosphere_stats().shadow_updates == 2, "Both atmosphere bodies evaluate their own shadow endpoints");
        check(probe.renderer.atmosphere_stats().shadow_vrs_updates == (extent.width < 64 ? 0u : 2u), "VRS shares scratch across bodies and bypasses tiny viewports");
        auto reconstructed = atmosphere_shadow_pixels(probe);
        double difference = 0;
        for (size_t i = 0; i < native.size(); ++i) difference += glm::dot(native[i]-reconstructed[i], native[i]-reconstructed[i]);
        check(std::sqrt(difference/native.size()) < .0015, "Shared VRS scratch remains correct across bodies and viewport resizing");
    }
    probe.resize(EXTENT);
    std::cout << "PASS: software shadow VRS guards, odd extents, sharp features, independent stars and transparent endpoints.\n";
}

void atmosphere_progressive_shadow_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    probe.resize({129, 129});
    auto loaded = atmosphere_shadow_fixture(true);
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Progressive shadow fixture");
    auto camera = camera_at(3000*KM_TO_AU);
    rendering::RenderSettings settings;
    settings.surface_textures_enabled = settings.ringshine_enabled = false;
    settings.atmosphere_shadow_samples = 4;
    settings.exposure = 8;
    float depth = 0;
    auto capture = [&] {
        probe.render(camera, simulation.snapshot(), settings, nullptr, false, glm::vec3(0), true, depth);
        return atmosphere_shadow_pixels(probe);
    };
    auto rmse = [](const auto& a, const auto& b) {
        double error = 0;
        for (size_t i = 0; i < a.size(); ++i) error += glm::dot(a[i]-b[i], a[i]-b[i]);
        return std::sqrt(error/a.size());
    };
    for (bool dual : {false, true}) for (uint32_t noise : {0u, 1u}) {
        settings.atmosphere_dual_source = dual;
        settings.atmosphere_shadow_noise_type = noise;
        settings.atmosphere_shadow_reference = true;
        auto reference = capture();
        settings.atmosphere_shadow_reference = false;
        settings.atmosphere_shadow_temporal = false;
        settings.atmosphere_shadow_vrs = false;
        auto native = capture();
        settings.atmosphere_shadow_temporal = true;
        settings.atmosphere_shadow_vrs = true;
        auto first = capture();
        check(first == native && probe.renderer.atmosphere_stats().shadow_history_resets == 1,
              "Progressive first frame is the native estimate with no stale history");
        check(probe.renderer.atmosphere_stats().shadow_vrs_updates == 0,
              "Progressive integration bypasses fraction reconstruction");
        for (int frame = 1; frame < 64; ++frame) capture();
        auto refined = capture();
        check(probe.renderer.atmosphere_stats().shadow_history_resets == 0,
              "Stationary progressive frames retain their history");
        check(refined == capture(), "Completed refinement remains bitwise stable");
        double first_error = rmse(first, reference), refined_error = rmse(refined, reference);
        std::cout << "Progressive shadows blend/noise " << dual << '/' << noise << " first/refined RMSE "
                  << first_error << '/' << refined_error << '\n';
        check(refined_error < first_error*0.8, "Temporal energy/loss integration reduces error against the dense reference");
        if (!output.empty()) {
            std::filesystem::create_directories(output);
            probe.save_preview(output/("progressive-"+std::to_string(dual)+"-"+std::to_string(noise)+".ppm"), true);
        }
    }
    settings.atmosphere_shadow_vrs = false;
    auto check_reset = [&](const char* label, bool global = true) {
        auto restarted = capture();
        if (global) check(probe.renderer.atmosphere_stats().shadow_history_resets == 1, label);
        settings.atmosphere_shadow_temporal = false;
        check(restarted == capture(), label);
        settings.atmosphere_shadow_temporal = true;
        for (int frame = 0; frame < 4; ++frame) capture();
    };
    camera.cam_pos_rel.x += .5*KM_TO_AU;
    check_reset("Camera translation discards progressive history immediately");
    camera.yaw_actual_deg += .1;
    check_reset("Camera rotation discards progressive history immediately");
    camera.fov_deg += 1;
    check_reset("Projection changes discard progressive history immediately");
    auto motions = loaded.motions;
    motions[0].position = render_to_ecliptic(RenderPosition{{.1, 1, 1}});
    check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted, "Move progressive shadow light");
    check_reset("Light movement discards progressive history immediately");
    depth = camera.compute_near_plane_au()/float(2200*KM_TO_AU);
    check_reset("Opaque endpoint changes discard pixel history immediately", false);
    probe.resize({131, 127});
    check_reset("Resizing discards progressive history immediately");
    settings.show_atmospheres = false;
    capture();
    settings.show_atmospheres = true;
    check_reset("Re-enabling atmospheres discards progressive history immediately");
    loaded.definition.bodies[1].rings[0].opacity = .3f;
    check(simulation.replace_system(loaded).accepted, "Edit progressive ring profile");
    check_reset("Ring profile edits discard progressive history immediately");
    probe.renderer.set_custom_ring_texture("", "Atmosphere shadow receiver", std::vector<glm::vec4>(64, glm::vec4(.2f)));
    check_reset("Ring texture uploads discard progressive history immediately");
    loaded = atmosphere_shadow_fixture();
    check(simulation.replace_system(loaded).accepted, "Progressive eclipse fixture");
    camera = camera_at(1100*KM_TO_AU);
    depth = 0;
    for (int frame = 0; frame < 64; ++frame) capture();
    auto umbra = surface_center(probe);
    settings.atmosphere_shadow_temporal = false;
    capture();
    std::cout << "Progressive/native umbra " << umbra.r << ',' << umbra.g << ',' << umbra.b << " / "
              << surface_center(probe).r << ',' << surface_center(probe).g << ',' << surface_center(probe).b << '\n';
    settings.atmosphere_shadow_temporal = true;
    for (int frame = 0; frame < 64; ++frame) capture();
    check(surface_center(probe) == glm::vec3(0), "Progressive full umbra retains exact zero radiance");
    motions = loaded.motions;
    motions[2].position = render_to_ecliptic(RenderPosition{{20*KM_TO_AU, 0, 2000*KM_TO_AU}});
    check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted, "Move progressive eclipse caster");
    check_reset("Caster movement discards progressive history immediately");
    if (probe.vk.fragment_stores_enabled()) {
        loaded = atmosphere_shadow_fixture(true);
        loaded.definition.bodies[1].name = "Progressive transparent receiver";
        check(simulation.replace_system(loaded).accepted, "Progressive transparent fixture");
        camera = camera_at(3000*KM_TO_AU);
        TransparencyProbe transparent(probe);
        for (bool dual : {false, true}) {
            settings.atmosphere_dual_source = dual;
            float near_depth = camera.compute_near_plane_au()/float(1930*KM_TO_AU);
            TransparentTestLayer layer{{.4f, .6f, .8f, .6f}, {near_depth, 0, 0, 0}};
            auto transparent_capture = [&](bool present) {
                transparent.render(camera, simulation.snapshot(), settings,
                    present ? std::span<const TransparentTestLayer>{&layer, 1} : std::span<const TransparentTestLayer>{});
                return atmosphere_shadow_pixels(probe);
            };
            for (int frame = 0; frame < 64; ++frame) transparent_capture(true);
            for (bool present : {true, false, true}) {
                layer.parameters.x = camera.compute_near_plane_au()/float(1960*KM_TO_AU);
                auto restarted = transparent_capture(present);
                settings.atmosphere_shadow_temporal = false;
                check(restarted == transparent_capture(present),
                      "Transparent endpoint motion, removal and arrival discard pixel history in both blend paths");
                settings.atmosphere_shadow_temporal = true;
                for (int frame = 0; frame < 4; ++frame) transparent_capture(present);
            }
        }
    }
    probe.resize(EXTENT);
    std::cout << "PASS: progressive shadow convergence, frozen completion and immediate history invalidation.\n";
}

void atmosphere_shadow_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    atmosphere_shadow_vrs_contracts(probe, output);
    atmosphere_adaptive_shadow_contracts(probe, output);
    atmosphere_progressive_shadow_contracts(probe, output);
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
    check(!validation.apply(invalid).accepted, "Unbounded atmospheric shadow sample ceilings are rejected");
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
    for (bool dual : {false, true}) {
        settings.atmosphere_dual_source = dual;
        for (double offset : {0.0, 19.0, 23.0, 30.0}) {
            auto motions = loaded.motions;
            motions[2].position = render_to_ecliptic(RenderPosition{{offset*KM_TO_AU, 0, 2000*KM_TO_AU}});
            check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted,
                  "Move atmospheric shadow caster without changing optics");
            auto unshadowed = render(false);
            auto shadowed = render(true);
            auto expected = oracle(offset, 100);
            std::cout << "Atmosphere deficit " << dual << " offset " << offset << ": "
                      << (unshadowed-shadowed).r << ',' << (unshadowed-shadowed).g << ',' << (unshadowed-shadowed).b
                      << " expected " << expected.r << ',' << expected.g << ',' << expected.b << '\n';
            check(glm::all(glm::greaterThanEqual(unshadowed-shadowed, glm::min(expected, unshadowed)-glm::vec3(.0025f))),
                  "Total shadow deficit includes the independently integrated single-scattering contribution");
            if (offset == 0) check(glm::length(shadowed) < .02f*glm::length(unshadowed)+.002f,
                  "Full umbra suppresses cached multiple-scattering fill");
            check(probe.renderer.atmosphere_stats().optical_lut_updates == 0 &&
                  probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
                  "Caster motion and shadow toggles reuse optical and Horizon tables");
            if (offset == 30) check(unshadowed == shadowed, "Clear atmospheric rays remain bitwise unchanged");
        }
    }
    auto motions = loaded.motions;
    check(simulation.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted, "Restore total eclipse");
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
    for (bool dual : {false, true}) {
        settings.atmosphere_dual_source = dual;
        float depth = camera.compute_near_plane_au()/float(50*KM_TO_AU);
        glm::vec3 surface(.4f, .6f, .8f);
        auto sky = render(true);
        auto endpoint = render(true, depth, surface);
        TransparencyProbe transparent(probe);
        TransparentTestLayer layer{{surface, .6f}, {depth, 0, 0, 0}};
        auto actual = transparent.render(camera, simulation.snapshot(), settings, {&layer, 1});
        check(glm::length(actual-(.6f*endpoint+.4f*sky)) < .003f,
              "Transparent endpoints receive the same shadow correction in both blend paths");
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
        for (bool dual : {false, true}) {
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
                std::cout << "Temporal transparent endpoints " << dual << '/' << count
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
        {"method", "Serialized offscreen atmosphere composition, no readback/presentation. Three alternating-order repetitions; 8 warmup (64 for completed progressive history) and 16 measured frames. Not application FPS."},
        {"cases", nlohmann::json::array()}};
    for (int scene = 0; scene < 7; ++scene) {
        auto loaded = atmosphere_shadow_fixture(scene >= 2);
        if (scene == 0) loaded.motions[2].position = render_to_ecliptic(RenderPosition{{10000*KM_TO_AU, 0, 2000*KM_TO_AU}});
        ephemeris::Simulation simulation;
        check(simulation.replace_system(loaded).accepted, "Atmosphere shadow benchmark fixture");
        auto camera = camera_at((scene == 2 ? 3000 : scene == 3 ? 1100.1 : scene == 4 || scene == 5 ? 1099.9 : scene == 6 ? 1001 : 1100)*KM_TO_AU);
        if (scene == 3 || scene == 4) {
            camera.cam_pos_rel.y = -10*KM_TO_AU;
            camera.yaw_actual_deg = -std::acos(1000/(glm::length(camera.cam_pos_rel)*AU_TO_KM))*RAD_TO_DEG;
        }
        if (scene == 5) camera.yaw_actual_deg = 90;
        const char* names[] = {"clear", "eclipse", "rings_orbit", "rings_above", "rings_inside_horizon", "rings_inside_zenith", "rings_inside_ground"};
        rendering::RenderSettings settings;
        settings.atmosphere_shadow_samples = samples;
        settings.ringshine_enabled = false;
        settings.surface_textures_enabled = false;
        for (int method = 0; method < 5; ++method) {
            bool adaptive = method != 0;
            bool vrs = method == 2;
            settings.atmosphere_shadow_adaptive = adaptive;
            settings.atmosphere_shadow_vrs = vrs;
            settings.atmosphere_shadow_temporal = method >= 3;
            std::array<std::vector<double>, 2> composition, prepare_cpu, shadow_gpu, coarse_gpu, resolve_gpu;
            for (int repeat = 0; repeat < 3; ++repeat) for (int order = 0; order < 2; ++order) {
                int enabled = repeat%2 ? 1-order : order;
                settings.atmosphere_shadows_enabled = enabled;
                int warmup = method == 4 ? 64 : 8;
                for (int frame = 0; frame < warmup+16; ++frame) {
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
                    if (frame < warmup) continue;
                    composition[enabled].push_back(probe.vk.gpu_time_ms("atmosphere_composite"));
                    shadow_gpu[enabled].push_back(probe.renderer.atmosphere_stats().shadow_updates ? probe.vk.gpu_time_ms("atmosphere_shadows") : 0.0);
                    coarse_gpu[enabled].push_back(probe.renderer.atmosphere_stats().shadow_vrs_updates ? probe.vk.gpu_time_ms("atmosphere_shadow_coarse") : 0.0);
                    resolve_gpu[enabled].push_back(probe.renderer.atmosphere_stats().shadow_updates ? probe.vk.gpu_time_ms("atmosphere_shadow_resolve") : 0.0);
                    prepare_cpu[enabled].push_back(cpu);
                }
            }
            for (int enabled = 0; enabled < 2; ++enabled) {
                nlohmann::json row{{"scene", names[scene]}, {"adaptive", adaptive}, {"vrs", vrs}, {"model", "horizon"},
                    {"temporal", settings.atmosphere_shadow_temporal}, {"history_complete", method == 4},
                    {"shadows", bool(enabled)}, {"samples", samples}, {"gpu_composite_ms", terrain_timing_stats(composition[enabled])},
                    {"gpu_shadow_ms", terrain_timing_stats(shadow_gpu[enabled])},
                    {"gpu_shadow_coarse_ms", terrain_timing_stats(coarse_gpu[enabled])},
                    {"gpu_shadow_resolve_ms", terrain_timing_stats(resolve_gpu[enabled])},
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
