#pragma once

namespace {

void atmosphere_cache_order_contracts(RenderProbe& probe) {
    auto loaded = fixture();
    loaded.definition.bodies[0].mass_msun = 1.0;
    constexpr double radius_km = 6378.137;
    const double radius = radius_km*KM_TO_AU;
    for (int i = 0; i < 3; ++i) {
        ephemeris::BodyDefinition planet;
        planet.name = "Cache atmosphere " + std::to_string(i);
        planet.radius_rsun = radius_km/SOLAR_RADIUS_KM;
        planet.mass_msun = 3.0034896154649684e-6;
        planet.atmosphere = physics::earth_physical_atmosphere();
        planet.atmosphere.automatic_height = false;
        planet.atmosphere.pressure_pa *= i+1;
        loaded.definition.bodies.push_back(planet);
        loaded.motions.emplace_back();
        loaded.motions.back().position = render_to_ecliptic(RenderPosition{
            {i == 0 ? -3*radius : 3*radius, 0.0, i == 2 ? 100*radius : 0.0}});
    }
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Atmosphere cache order fixture");
    rendering::AtmosphereRenderer renderer;
    renderer.init(probe.vk);
    rendering::RenderSettings settings;
    auto cam = camera_at(20*radius);
    cam.cam_pos_rel.x = -4*radius;
    auto prepare = [&] {
        probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
            renderer.prepare(cmd, probe.vk, cam, sim.snapshot(), settings, EXTENT);
        });
        return renderer.stats();
    };
    auto records = [&] {
        const auto* data = static_cast<const glm::vec4*>(renderer.surface_records(probe.vk.frame_index()).mapped);
        return std::array{data[1], data[2], data[3]};
    };
    auto cold = prepare();
    check(cold.visible == 2 && cold.optical_lut_updates == 2 && cold.aerial_lut_updates == 2 &&
        cold.surface_lut_updates == 2, "Both visible atmospheres warm their optical and surface caches");
    auto layers = records();
    check(layers[0].w > 0 && layers[1].w > 0 && layers[0].w != layers[1].w,
        "Physical atmospheres own distinct surface irradiance layers");
    auto reused = [&](uint32_t visible) {
        auto stats = prepare();
        if (stats.visible != visible || stats.optical_lut_updates || stats.surface_lut_updates)
            std::cerr << "Cache reuse: visible " << stats.visible << '/' << visible << ", optical " << stats.optical_lut_updates
                << ", surface " << stats.surface_lut_updates << '\n';
        check(stats.visible == visible && stats.optical_lut_updates == 0 &&
            stats.surface_lut_updates == 0, "Visibility and distance ordering reuse expensive atmosphere tables");
    };
    cam.cam_pos_rel.x = 4*radius;
    reused(2);
    check(records()[0] == layers[0] && records()[1] == layers[1],
        "Distance order swaps preserve each body's surface irradiance layer");
    cam.cam_pos_rel.x = -4*radius;
    cam.fov_deg = 10;
    cam.yaw_actual_deg = std::atan2(-20.0, 1.0)*RAD_TO_DEG;
    reused(1);
    cam.fov_deg = 60;
    cam.yaw_actual_deg = -90;
    reused(2);
    cam.pitch_actual_deg = 90;
    reused(0);
    cam.pitch_actual_deg = 0;
    reused(2);
    auto motions = loaded.motions;
    motions[3].position = render_to_ecliptic(RenderPosition{{0.0, 0.0, -10*radius}});
    check(sim.apply(ephemeris::SyncTimelineState{units::Years{0}, std::move(motions)}).accepted,
        "A third atmosphere enters ahead of the warmed entries in distance order");
    auto added = prepare();
    check(added.visible == 3 && added.optical_lut_updates == 1 && added.aerial_lut_updates >= 1 &&
        added.surface_lut_updates == 1, "Only the newly visible atmosphere builds optical and surface tables");
    check(records()[0] == layers[0] && records()[1] == layers[1],
        "A newly visible atmosphere does not steal existing surface layers");
    auto optical = sim.snapshot().body(1).definition.atmosphere;
    optical.pressure_pa *= 1.1;
    check(sim.apply(ephemeris::ApplyBodyAtmosphere{sim.snapshot().generation(), 1,
        sim.snapshot().body(1).revisions.definition, optical}).accepted, "Optical edit after reordering");
    auto edited = prepare();
    check(edited.optical_lut_updates >= 1 && edited.aerial_lut_updates >= 1 && edited.surface_lut_updates >= 1,
        "Real optical edits still invalidate cached transport and surface lighting");
    loaded = fixture();
    for (int i = 0; i < 20; ++i) {
        ephemeris::BodyDefinition planet;
        planet.name = "Cache pressure " + std::to_string(i);
        planet.radius_rsun = radius_km/SOLAR_RADIUS_KM;
        planet.atmosphere.enabled = true;
        loaded.definition.bodies.push_back(planet);
        loaded.motions.emplace_back();
        double angle = i*18*DEG_TO_RAD;
        double distance = (i == 0 ? 120 : 100)*radius;
        loaded.motions.back().position = render_to_ecliptic(RenderPosition{
            {std::cos(angle)*distance, 0.0, std::sin(angle)*distance}});
    }
    check(sim.replace_system(loaded).accepted, "Atmosphere cache pressure fixture");
    settings.aerial_perspective_enabled = false;
    cam = camera_at(0);
    cam.fov_deg = 1;
    for (int i = 0; i < 20; ++i) {
        cam.yaw_actual_deg = i*18;
        auto stats = prepare();
        check(stats.visible == 1 && stats.optical_lut_updates == 1,
            "Cache pressure warms one newly visible atmosphere at a time");
    }
    reused(1);
    cam.fov_deg = 20;
    cam.yaw_actual_deg = -9;
    auto pressure = prepare();
    check(pressure.visible == 2 && pressure.optical_lut_updates == 1,
        "An evicted atmosphere entering first does not evict a later visible cache hit");
    cam.fov_deg = 1;
    cam.yaw_actual_deg = 342;
    reused(1);
    renderer.shutdown(probe.vk);
    std::cout << "PASS: atmosphere caches survive distance swaps, hidden bodies, visibility insertion and cache pressure.\n";
}

void curved_surface_depth_contracts(RenderProbe& probe) {
    rendering::SceneCelestialUBO scene;
    scene.view_proj = camera_at(2.0).proj_matrix(1.0f, 0.1f);
    auto ubo = probe.vk.create_host_buffer(sizeof(scene), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &scene);
    auto output = probe.vk.create_host_buffer(6*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1}};
    auto pipeline = rhi::ShaderCache::create_compute_pipeline(probe.vk, "tests/surface_depth.comp", bindings);
    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        rhi::PushDescriptorWrite writes[] = {rhi::PushDescriptorWrite::uniform_buffer(0, ubo),
            rhi::PushDescriptorWrite::storage_buffer(4, output)};
        probe.vk.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, 6, 1, 1);
        probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });
    vmaInvalidateAllocation(probe.vk.allocator(), output.allocation, 0, VK_WHOLE_SIZE);
    auto* mapped = static_cast<const glm::vec4*>(output.mapped);
    std::vector<glm::vec4> depths(mapped, mapped+6);
    rhi::ShaderCache::destroy_pipeline(probe.vk, pipeline);
    probe.vk.destroy_buffer(output);
    probe.vk.destroy_buffer(ubo);
    auto near_surface = scene.view_proj*glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);
    for (size_t index = 0; index < depths.size(); index += 2) {
        check(std::abs(depths[index].x-near_surface.z/near_surface.w) < 1e-6f,
              "Near-side chord depth retains analytic surface curvature on spheres and oblate bodies");
        check(depths[index+1].x == depths[index+1].y && depths[index+1].x < depths[index].x,
              "Far-side terrain stays behind the horizon instead of moving to the near ellipsoid intersection");
    }
}

void aerial_perspective_contracts(RenderProbe& probe) {
    probe.resize({257, 257});
    constexpr double radius_km = 6378.137;
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1.0, 0.0, 1.0}});
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0.0;
    ephemeris::BodyDefinition body;
    body.name = "Finite atmospheric transport";
    body.color_rgb = glm::vec3(0.0f);
    body.color_pole_rgb = glm::vec3(0.0f);
    body.radius_rsun = radius_km/SOLAR_RADIUS_KM;
    body.height_range_km = {-10.984, 8.848};
    body.atmosphere.enabled = true;
    loaded.definition.bodies.push_back(body);
    loaded.motions.emplace_back();
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Finite transport fixture");
    rendering::RenderSettings settings;
    auto sample = [&](const Camera& camera, double distance, glm::vec3 background = glm::vec3(1.0f)) {
        float depth = camera.compute_near_plane_au()/float(distance*KM_TO_AU);
        probe.render(camera, sim.snapshot(), settings, nullptr, false, background, true, depth);
        return surface_center(probe);
    };
    for (double altitude : {-5.0, 1.0, 99.99, 100.01}) {
        Camera cam = camera_at((radius_km+altitude)*KM_TO_AU);
        cam.yaw_actual_deg = 90.0;
        cam.is_near_surface = true;
        for (double distance : {0.01, 0.1, 1.0, 10.0, 100.0}) {
            auto actual = sample(cam, distance);
            double span = std::max(0.0, std::min(distance, 100.0-altitude));
            const auto& a = body.atmosphere;
            double r = a.rayleigh_scale_height_km*std::exp(-altitude/a.rayleigh_scale_height_km)*(-std::expm1(-span/a.rayleigh_scale_height_km));
            double below = std::min(span, std::max(0.0, -altitude));
            double m = below+a.mie_scale_height_km*std::exp(-std::max(altitude, 0.0)/a.mie_scale_height_km)*(-std::expm1(-(span-below)/a.mie_scale_height_km));
            for (int c = 0; c < 3; ++c) {
                double expected = std::exp(-a.rayleigh_scattering[c]*r-a.mie_extinction[c]*m);
                check(std::abs(actual[c]-expected) < 0.004, "Finite RGB extinction matches analytic columns above/below datum and atmosphere top");
            }
            if (distance != 0.01) check(probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
                "Changing only opaque endpoint depth reuses the aerial volume");
        }
    }
    Camera cam = camera_at((radius_km+1.0)*KM_TO_AU);
    cam.yaw_actual_deg = 0.0;
    cam.is_near_surface = true;
    for (double distance : {0.1, 1.0, 10.0, 100.0}) {
        auto actual = sample(cam, distance);
        glm::dvec3 tau(0.0);
        for (int i = 0; i < 8192; ++i) {
            double t = distance*(i+0.5)/8192.0;
            double h = std::hypot(radius_km+1.0, t)-radius_km;
            tau += (glm::dvec3(body.atmosphere.rayleigh_scattering)*std::exp(-h/8.0)
                  + glm::dvec3(body.atmosphere.mie_extinction)*std::exp(-h/1.2))*(distance/8192.0);
        }
        for (int c = 0; c < 3; ++c) {
            if (std::abs(actual[c]-std::exp(-tau[c])) >= 0.01)
                std::cerr << "Horizon distance " << distance << " channel " << c << " actual " << actual[c] << " expected " << std::exp(-tau[c]) << '\n';
            check(std::abs(actual[c]-std::exp(-tau[c])) < 0.01,
                "Near-horizon finite extinction agrees with independent numerical integration");
        }
    }
    cam = camera_at(radius_km*3.0*KM_TO_AU);
    check(sample(cam, 100.0) == glm::vec3(1.0f), "Foreground in vacuum receives no atmosphere");
    auto orbital = sample(cam, radius_km*2.0-1.0);
    const auto& a = body.atmosphere;
    for (int c = 0; c < 3; ++c) {
        double tau = a.rayleigh_scattering[c]*8.0*(std::exp(-1.0/8.0)-std::exp(-100.0/8.0))
                   + a.mie_extinction[c]*1.2*(std::exp(-1.0/1.2)-std::exp(-100.0/1.2));
        check(std::abs(orbital[c]-std::exp(-tau)) < 0.008, "Orbital volume allocates finite transport after atmosphere entry");
    }
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1.0;
    check(sim.replace_system(loaded).accepted, "Lit aerial transport fixture");
    cam = camera_at((radius_km+1.0)*KM_TO_AU);
    cam.yaw_actual_deg = 0.0;
    cam.is_near_surface = true;
    auto L = sample(cam, 100.0, glm::vec3(0.0f));
    check(L.r > 0.001f && L.b > L.r, "Finite-distance surfaces receive colored atmospheric scattering");
    auto stable = sample(cam, 100.0, glm::vec3(0.0f));
    check(stable == L && probe.renderer.atmosphere_stats().aerial_lut_updates == 0, "Paused aerial transport reuses its volume exactly");
    settings.exposure = 2.0f;
    auto exposed = sample(cam, 100.0, glm::vec3(0.0f));
    check(glm::length(exposed-2.0f*L) < 0.002f && probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
          "Aerial radiance exposure applies once without rebaking");
    cam.yaw_actual_deg += 0.2;
    sample(cam, 100.0);
    check(probe.renderer.atmosphere_stats().aerial_lut_updates == 0 && probe.renderer.atmosphere_stats().sky_lut_updates == 0,
          "Rotation reuses the horizon volume and sky LUT");
    settings.aerial_perspective_enabled = false;
    auto extinction_only = sample(cam, 100.0);
    check(extinction_only.r > 0.0f && extinction_only.r < 1.0f &&
        sample(cam, 100.0, glm::vec3(0)) == glm::vec3(0),
        "Disabling aerial perspective retains extinction without adding surface inscattering");
    cam.cam_pos_rel.z += 0.1*KM_TO_AU;
    sample(cam, 100.0);
    settings.aerial_perspective_enabled = true;
    sample(cam, 100.0);
    check(probe.renderer.atmosphere_stats().aerial_lut_updates == 1, "Re-enabling aerial perspective refreshes changes made while disabled");
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{0.0, 0.0, 1.0}});
    check(sim.replace_system(loaded).accepted, "Illuminated curved-surface limb fixture");
    settings.exposure = 1.0f;
    cam = camera_at(radius_km*3.0*KM_TO_AU);
    settings.terrain_enabled = false;
    probe.render(cam, sim.snapshot(), settings);
    auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> sphere(pixels, pixels+size_t(probe.extent.width)*probe.extent.height*4);
    settings.terrain_enabled = true;
    settings.terrain_height_km = 0.0f;
    for (float error : {16.0f, 1.0f, 4.0f}) {
        settings.terrain_error_pixels = error;
        for (int frame = 0; frame < 4; ++frame) probe.render(cam, sim.snapshot(), settings);
        pixels = static_cast<const uint16_t*>(probe.readback.mapped);
        float curvature_error = 0;
        for (int y = 70; y < 187; ++y) {
            for (int x = 70; x < 187; ++x) {
                if ((x-128)*(x-128)+(y-128)*(y-128) > 55*55) continue;
                size_t offset = 4*(size_t(y)*probe.extent.width+x);
                for (int c = 0; c < 3; ++c) {
                    curvature_error = std::max(curvature_error, std::abs(glm::unpackHalf2x16(pixels[offset+c]).x-glm::unpackHalf2x16(sphere[offset+c]).x));
                }
            }
        }
        if (curvature_error >= .0025f) std::cerr << "Curvature maximum error at LOD error " << error << ": " << curvature_error << '\n';
        check(curvature_error < .0025f, "Aerial transport preserves curved-surface depth across sphere mesh, quadtree LOD and geomorph changes");
    }
    probe.resize({640, 480});
    cam = camera_at(radius_km*2.2*KM_TO_AU);
    settings.show_atmospheres = false;
    probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f));
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> mask(pixels, pixels+size_t(probe.extent.width)*probe.extent.height*4);
    settings.show_atmospheres = true;
    settings.aerial_perspective_enabled = false;
    probe.render(cam, sim.snapshot(), settings);
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> sky_only(pixels, pixels+mask.size());
    settings.aerial_perspective_enabled = true;
    probe.render(cam, sim.snapshot(), settings);
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    uint32_t background_checks = 0;
    for (size_t pixel = 0; pixel < mask.size(); pixel += 4) {
        if (mask[pixel] == 0 && mask[pixel+1] == 0 && mask[pixel+2] == 0) continue;
        for (int c = 0; c < 4; ++c) check(pixels[pixel+c] == sky_only[pixel+c],
            "Horizon aerial transport never spills onto background sky or limb pixels");
        ++background_checks;
    }
    check(background_checks > 10000, "Full-resolution surface mask protects the surrounding sky");
    auto direct_reference = [&](uint32_t x, uint32_t y) {
        double tan_y = std::tan(cam.fov_deg*DEG_TO_RAD*0.5);
        glm::dvec3 d = glm::normalize(cam.forward_dir()+cam.right_dir()*((2.0*(x+0.5)/640.0-1.0)*tan_y*640.0/480.0)
                                    -cam.up_dir()*((2.0*(y+0.5)/480.0-1.0)*tan_y));
        glm::dvec3 p(0.0, 0.0, radius_km*2.2);
        auto roots = [](glm::dvec3 origin, glm::dvec3 ray, double radius) {
            double b = glm::dot(origin, ray);
            auto perpendicular = glm::cross(origin, ray);
            double discriminant = radius*radius-glm::dot(perpendicular, perpendicular);
            check(discriminant >= 0.0, "Reference limb ray intersects its analytic surface");
            return glm::dvec2(-b-std::sqrt(discriminant), -b+std::sqrt(discriminant));
        };
        double start = roots(p, d, radius_km+100.0).x;
        double end = roots(p, d, radius_km).x;
        const auto& a = body.atmosphere;
        auto extinction = [&](double h) {
            return a.rayleigh_scattering.r*std::exp(-h/a.rayleigh_scale_height_km)
                 + a.mie_extinction.r*std::exp(-std::max(h, 0.0)/a.mie_scale_height_km);
        };
        double mu = d.z, g = a.mie_asymmetry;
        double phase_r = 3.0*(1.0+mu*mu)/(16.0*PI);
        double phase_m = 3.0*(1.0-g*g)*(1.0+mu*mu)/(8.0*PI*(2.0+g*g)*std::pow(1.0+g*g-2.0*g*mu, 1.5));
        double L = 0.0, T = 1.0, dt = (end-start)/256.0;
        for (int i = 0; i < 256; ++i) {
            auto q = p+d*(start+(i+0.5)*dt);
            double solar_end = roots(q, {0.0, 0.0, 1.0}, radius_km+100.0).y;
            double solar_tau = 0.0;
            for (int j = 0; j < 64; ++j) {
                double t0 = double(j)/64.0, t1 = double(j+1)/64.0;
                t0 *= t0; t1 *= t1;
                auto point = q+glm::dvec3(0.0, 0.0, solar_end*(t0+t1)*0.5);
                solar_tau += extinction(glm::length(point)-radius_km)*solar_end*(t1-t0);
            }
            double h = glm::length(q)-radius_km;
            double e = extinction(h);
            double source = std::exp(-solar_tau)*(a.rayleigh_scattering.r*std::exp(-h/a.rayleigh_scale_height_km)*phase_r
                          +a.mie_scattering.r*std::exp(-std::max(h, 0.0)/a.mie_scale_height_km)*phase_m);
            L += T*source*(-std::expm1(-e*dt))/e;
            T *= std::exp(-e*dt);
        }
        return L;
    };
    uint32_t limb_checks = 0;
    for (uint32_t y = 120; y < 360; ++y) {
        for (uint32_t x = 1; x < 320; ++x) {
            size_t offset = 4*(size_t(y)*probe.extent.width+x);
            if (mask[offset] != 0 || mask[offset+1] != 0 || mask[offset+2] != 0) continue;
            float outer = probe.red_at(x-1, y), edge = probe.red_at(x, y);
            if (outer > 0.005f) {
                double reference = direct_reference(x, y);
                double normalized = reference*probe.red_at(x+8, y)/direct_reference(x+8, y);
                if (edge < normalized*0.5) std::cerr << "Limb (" << x << ',' << y << ") edge " << edge << " reference profile " << normalized << '\n';
                check(edge >= normalized*0.5, "Opaque limb preserves the independently integrated scattering profile without dark volume sampling dots");
                ++limb_checks;
            }
            break;
        }
    }
    check(limb_checks > 40, "Limb regression samples a resolved curved silhouette");
    std::cout << "PASS: finite-distance aerial RGB extinction/scattering, near geometry, vacuum entry, horizon, cache and exposure.\n";
}

void horizon_perspective_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    curved_surface_depth_contracts(probe);
    rendering::AtmosphereParametersGPU parameters;
    parameters.radii_heights = {6378.137f, 6478.137f, 8.0f, 1.2f};
    parameters.reference_shape = {6378.137f, 1.0f, 6378.137f, 0.0f};
    auto uniform = probe.vk.create_host_buffer(sizeof(parameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &parameters);
    auto visibility = probe.vk.create_host_buffer(45*sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rhi::DescriptorLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
    auto pipeline = rhi::ShaderCache::create_compute_pipeline(probe.vk, "tests/atmosphere_visibility.comp", bindings);
    probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        rhi::PushDescriptorWrite writes[] = {rhi::PushDescriptorWrite::uniform_buffer(0, uniform),
            rhi::PushDescriptorWrite::storage_buffer(1, visibility)};
        probe.vk.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, 45, 1, 1);
        probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    });
    vmaInvalidateAllocation(probe.vk.allocator(), visibility.allocation, 0, VK_WHOLE_SIZE);
    auto* values = static_cast<const glm::vec4*>(visibility.mapped);
    for (int row = 0; row < 45; row += 5) {
        check(values[row].x == 0.0f && values[row+4].x == 1.0f, "Host shadow converges to full night and daylight");
        check(values[row+1].y > 0.0f && values[row+1].x > 0.0f && values[row+1].x < 0.5f,
            "Partially visible stars remain lit even when the central solar ray intersects the planet");
        if (std::abs(values[row+2].x-0.5f) >= 0.002f || values[row+3].x <= 0.5f || values[row+3].x >= 1.0f)
            std::cerr << "Solar visibility row " << row << ": " << values[row+1].x << ',' << values[row+2].x << ',' << values[row+3].x << '\n';
        check(std::abs(values[row+2].x-0.5f) < 0.002f && values[row+3].x > 0.5f && values[row+3].x < 1.0f,
            "Solar visibility crosses the altitude-dependent horizon smoothly for different stellar sizes");
    }
    rhi::ShaderCache::destroy_pipeline(probe.vk, pipeline);
    probe.vk.destroy_buffer(visibility);
    probe.vk.destroy_buffer(uniform);
    probe.resize({257, 257});
    constexpr double radius_km = 6378.137;
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1.0, 0.0, 1.0}});
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0.0;
    ephemeris::BodyDefinition body;
    body.name = "Horizon transport contracts";
    body.color_rgb = glm::vec3(0.0f);
    body.color_pole_rgb = glm::vec3(0.0f);
    body.radius_rsun = radius_km/SOLAR_RADIUS_KM;
    body.height_range_km = {-10.984, 8.848};
    body.atmosphere.enabled = true;
    loaded.definition.bodies.push_back(body);
    loaded.motions.emplace_back();
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Horizon transport fixture");
    rendering::RenderSettings settings;
    auto sample = [&](const Camera& camera, double distance, glm::vec3 background = glm::vec3(1.0f)) {
        float depth = camera.compute_near_plane_au()/float(distance*KM_TO_AU);
        probe.render(camera, sim.snapshot(), settings, nullptr, false, background, true, depth);
        return surface_center(probe);
    };
    for (double altitude : {-5.0, 1.0, 99.99, 100.01}) {
        Camera cam = camera_at((radius_km+altitude)*KM_TO_AU);
        cam.yaw_actual_deg = 90.0;
        cam.is_near_surface = true;
        for (double distance : {0.01, 0.1, 1.0, 10.0, 100.0}) {
            auto actual = sample(cam, distance);
            double span = std::max(0.0, std::min(distance, 100.0-altitude));
            const auto& a = body.atmosphere;
            double r = a.rayleigh_scale_height_km*std::exp(-altitude/a.rayleigh_scale_height_km)*(-std::expm1(-span/a.rayleigh_scale_height_km));
            double below = std::min(span, std::max(0.0, -altitude));
            double m = below+a.mie_scale_height_km*std::exp(-std::max(altitude, 0.0)/a.mie_scale_height_km)*(-std::expm1(-(span-below)/a.mie_scale_height_km));
            for (int c = 0; c < 3; ++c) {
                double expected = std::exp(-a.rayleigh_scattering[c]*r-a.mie_extinction[c]*m);
                check(std::abs(actual[c]-expected) < 0.005, "Horizon finite RGB extinction matches analytic columns");
            }
            if (distance != 0.01) check(probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
                "Changing distance reuses the horizon LUT without updating volume");
        }
    }
    Camera cam = camera_at((radius_km+1.0)*KM_TO_AU);
    cam.yaw_actual_deg = 0.0;
    cam.is_near_surface = true;
    for (double distance : {0.1, 1.0, 10.0, 100.0}) {
        auto actual = sample(cam, distance);
        glm::dvec3 tau(0.0);
        for (int i = 0; i < 8192; ++i) {
            double t = distance*(i+0.5)/8192.0;
            double h = std::hypot(radius_km+1.0, t)-radius_km;
            tau += (glm::dvec3(body.atmosphere.rayleigh_scattering)*std::exp(-h/8.0)
                  + glm::dvec3(body.atmosphere.mie_extinction)*std::exp(-h/1.2))*(distance/8192.0);
        }
        for (int c = 0; c < 3; ++c) {
            check(std::abs(actual[c]-std::exp(-tau[c])) < 0.012,
                "Horizon near-horizon finite extinction agrees with numerical integration");
        }
    }
    cam = camera_at(radius_km*3.0*KM_TO_AU);
    check(sample(cam, 100.0) == glm::vec3(1.0f), "Horizon vacuum foreground receives no atmosphere");
    auto orbital = sample(cam, radius_km*2.0-1.0);
    const auto& a = body.atmosphere;
    for (int c = 0; c < 3; ++c) {
        double tau = a.rayleigh_scattering[c]*8.0*(std::exp(-1.0/8.0)-std::exp(-100.0/8.0))
                   + a.mie_extinction[c]*1.2*(std::exp(-1.0/1.2)-std::exp(-100.0/1.2));
        check(std::abs(orbital[c]-std::exp(-tau)) < 0.008, "Horizon orbital volume entry matches analytic transport");
    }
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1.0;
    check(sim.replace_system(loaded).accepted, "Lit Horizon fixture");
    cam = camera_at((radius_km+1.0)*KM_TO_AU);
    cam.yaw_actual_deg = 0.0;
    cam.is_near_surface = true;
    auto L = sample(cam, 100.0, glm::vec3(0.0f));
    check(L.r > 0.001f && L.b > L.r, "Horizon finite surfaces receive colored atmospheric scattering");
    auto stable = sample(cam, 100.0, glm::vec3(0.0f));
    check(stable == L && probe.renderer.atmosphere_stats().aerial_lut_updates == 0, "Paused Horizon reuses LUT");
    settings.exposure = 2.0f;
    auto exposed = sample(cam, 100.0, glm::vec3(0.0f));
    check(glm::length(exposed-2.0f*L) < 0.002f && probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
          "Exposure applies to Horizon transport without rebaking");
    settings.exposure = 1.0f;
    for (double altitude : {0.01, 1.0, 12.0}) {
        cam = camera_at((radius_km+altitude)*KM_TO_AU);
        cam.is_near_surface = true;
        double bottom = radius_km+body.height_range_km.x;
        double horizon = -std::acos(bottom/(radius_km+altitude))*RAD_TO_DEG;
        glm::vec3 previous(0.0f);
        float maximum_jump = 0.0f;
        for (int step = -20; step <= 20; ++step) {
            cam.yaw_actual_deg = horizon+step*0.002;
            auto current = sample(cam, 100.0, glm::vec3(0.0f));
            if (step != -20) maximum_jump = std::max(maximum_jump, glm::length(current-previous));
            if (step % 20 == 0) {
                glm::dvec3 origin(0.0, 0.0, radius_km+altitude);
                auto direction = cam.forward_dir();
                glm::dvec3 tau(0.0);
                for (int i = 0; i < 8192; ++i) {
                    double t = 100.0*(i+0.5)/8192.0;
                    double h = glm::length(origin+direction*t)-radius_km;
                    tau += (glm::dvec3(body.atmosphere.rayleigh_scattering)*std::exp(-h/8.0)
                          + glm::dvec3(body.atmosphere.mie_extinction)*std::exp(-std::max(h, 0.0)/1.2))*(100.0/8192.0);
                }
                auto transmission = sample(cam, 100.0)-current;
                for (int c = 0; c < 3; ++c) check(std::abs(transmission[c]-std::exp(-tau[c])) < 0.015,
                    "Grazing finite extinction matches numerical integration even when boundary transmittances underflow");
            }
            previous = current;
        }
        std::cout << "Horizon scattering altitude " << altitude << " maximum jump " << maximum_jump << '\n';
        check(maximum_jump < 0.004f, "Horizon finite scattering stays continuous across the ground/sky LUT boundary");
    }
    cam = camera_at((radius_km+1.0)*KM_TO_AU);
    cam.is_near_surface = true;
    cam.yaw_actual_deg = 0.0;
    sample(cam, 100.0);
    cam.yaw_actual_deg += 0.2;
    sample(cam, 100.0);
    check(probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
          "Rotation reuses the horizon table without rebaking");
    cam.fov_deg = 75.0;
    sample(cam, 100.0);
    check(probe.renderer.atmosphere_stats().aerial_lut_updates == 0 && probe.renderer.atmosphere_stats().sky_lut_updates == 0,
          "Changing FOV reuses the full angular transport domain");
    cam.fov_deg = 60.0;
    cam.cam_pos_rel.z += 10.0*KM_TO_AU;
    sample(cam, 100.0);
    check(probe.renderer.atmosphere_stats().aerial_lut_updates == 1,
          "Translation rebuilds the camera-position-dependent horizon table");
    auto edited = sim.snapshot().body(1).definition.atmosphere;
    edited.mie_asymmetry = 0.75f;
    check(sim.apply(ephemeris::ApplyBodyAtmosphere{sim.snapshot().generation(), 1, sim.snapshot().body(1).revisions.definition, edited}).accepted, "Apply optical change");
    sample(cam, 100.0);
    check(probe.renderer.atmosphere_stats().aerial_lut_updates == 1, "Optical edit rebakes Horizon table");
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1.0, 0.5, 1.0}});
    check(sim.replace_system(loaded).accepted, "Asymmetric azimuth lighting fixture");
    cam = camera_at((radius_km+1.0)*KM_TO_AU);
    cam.is_near_surface = true;
    cam.yaw_actual_deg = 180.0;
    cam.pitch_actual_deg = -0.0001;
    auto seam_left = sample(cam, 100.0, glm::vec3(0));
    cam.pitch_actual_deg = 0.0001;
    auto seam_right = sample(cam, 100.0, glm::vec3(0));
    check(glm::length(seam_right-seam_left) < 0.0001f && probe.renderer.atmosphere_stats().aerial_lut_updates == 0,
          "Periodic bicubic filtering preserves continuity across the azimuth seam");
    probe.resize({640, 480});
    cam = camera_at(radius_km*2.2*KM_TO_AU);
    settings.show_atmospheres = false;
    probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f));
    auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> mask(pixels, pixels+size_t(probe.extent.width)*probe.extent.height*4);
    settings.show_atmospheres = true;
    settings.aerial_perspective_enabled = false;
    probe.render(cam, sim.snapshot(), settings);
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> sky_only(pixels, pixels+mask.size());
    settings.aerial_perspective_enabled = true;
    probe.render(cam, sim.snapshot(), settings);
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    uint32_t background_checks = 0;
    for (size_t pixel = 0; pixel < mask.size(); pixel += 4) {
        if (mask[pixel] == 0 && mask[pixel+1] == 0 && mask[pixel+2] == 0) continue;
        for (int c = 0; c < 4; ++c) check(pixels[pixel+c] == sky_only[pixel+c],
            "Horizon transport never spills onto background sky or limb pixels");
        ++background_checks;
    }
    check(background_checks > 10000, "Horizon full-resolution surface mask protects surrounding sky");
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{-1.0, 0.0, 0.0}});
    check(sim.replace_system(loaded).accepted, "Terminator fixture");
    cam = camera_at(radius_km*3.0*KM_TO_AU);
    probe.resize({256, 256});
    probe.render(cam, sim.snapshot(), settings);
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    float prev_luma = 1e9f;
    for (int x = 110; x <= 146; ++x) {
        size_t idx = 4*(128*256+x);
        float r = glm::unpackHalf2x16(pixels[idx]).x;
        float g = glm::unpackHalf2x16(pixels[idx+1]).x;
        float b = glm::unpackHalf2x16(pixels[idx+2]).x;
        float luma = 0.2126f*r + 0.7152f*g + 0.0722f*b;
        check(luma <= prev_luma + 0.001f, "Horizon terminator profile is monotonically decreasing across day-night boundary");
        if (prev_luma < 1e8f) {
            check(std::abs(luma - prev_luma) < 0.08f, "Horizon terminator luminance transitions continuously without stair-step banding");
        }
        prev_luma = luma;
    }
    if (!output.empty()) {
        std::filesystem::create_directories(output);
        probe.resize({768, 768});
        probe.render(cam, sim.snapshot(), settings, nullptr, true);
        probe.save_preview(output/"terminator.ppm", true);
    }
    std::cout << "PASS: Horizon-mapped cumulative aerial perspective contracts.\n";
}

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
    settings.aerial_perspective_enabled = false;
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
        settings.show_atmospheres = false;
        probe.render(cam, sim.snapshot(), settings, nullptr, true);
        probe.save_preview(output / (std::string(orbit ? "orbit" : "horizon")+"-surface.ppm"), true);
        settings.show_atmospheres = true;
        probe.render(cam, sim.snapshot(), settings, nullptr, true);
        probe.save_preview(output / (std::string(orbit ? "orbit" : "horizon")+"-atmosphere.ppm"), true);
    }
    std::cout << "PASS: textured Earth orbital and horizon atmosphere previews exported.\n";
}

void physical_atmosphere_contracts(RenderProbe& probe) {
    probe.resize({129, 129});
    constexpr double radius = 6378.137;
    auto loaded = fixture();
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0;
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{0, 0, 1}});
    ephemeris::BodyDefinition planet;
    planet.name = "Physical atmosphere";
    planet.radius_rsun = radius/SOLAR_RADIUS_KM;
    planet.mass_msun = 3.0034896154649684e-6;
    planet.color_rgb = glm::vec3(.5f);
    planet.atmosphere = physics::earth_physical_atmosphere();
    planet.atmosphere.automatic_height = false;
    planet.atmosphere.temperature_layers.clear();
    planet.atmosphere.mie_scattering = planet.atmosphere.mie_extinction = glm::vec3(0);
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Physical atmosphere GPU fixture");
    rendering::RenderSettings settings;
    auto cam = camera_at((radius+1)*KM_TO_AU);
    cam.yaw_actual_deg = 90;
    cam.is_near_surface = true;
    auto physical = sim.snapshot().body(1).render.atmosphere;
    glm::dvec3 tau(0);
    for (int i = 0; i < 16384; ++i) {
        double h = 1+(i+.5)*99/16384;
        double relative = std::exp(-radius*h/(radius+h)/physical->surface.scale_height_km);
        double ozone = std::exp(-.5*std::pow((h-25)/7, 2));
        tau += (glm::dvec3(physical->rayleigh+physical->mixed_absorption)*relative+
            glm::dvec3(physical->ozone_absorption)*ozone)*(99.0/16384);
    }
    for (bool dual : {false, true}) {
        settings.atmosphere_dual_source = dual;
        probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1), true);
        auto actual = surface_center(probe);
        for (int c = 0; c < 3; ++c) check(std::abs(actual[c]-std::exp(-tau[c])) < .004,
            "Physical RGB transmission and ozone match independent spherical integration in both composites");
    }
    check(probe.renderer.atmosphere_stats().optical_lut_updates == 0 && probe.renderer.atmosphere_stats().surface_lut_updates == 0,
        "Changing compositor reuses physical optics and surface irradiance");
    auto first = sim.snapshot().body(1).render.atmosphere;
    cam.yaw_actual_deg += 2;
    probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1), true);
    check(sim.snapshot().body(1).render.atmosphere == first && probe.renderer.atmosphere_stats().surface_lut_updates == 0,
        "Moving camera reuses physics and surface light tables");

    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1;
    loaded.definition.bodies[1].atmosphere = physics::earth_physical_atmosphere();
    check(sim.replace_system(loaded).accepted, "Lit physical atmosphere fixture");
    cam = camera_at(radius*3*KM_TO_AU);
    settings.aerial_perspective_enabled = false;
    settings.atmosphere_surface_lighting = false;
    probe.render(cam, sim.snapshot(), settings);
    auto unattenuated = surface_center(probe);
    settings.atmosphere_surface_lighting = true;
    probe.render(cam, sim.snapshot(), settings);
    auto attenuated = surface_center(probe);
    check(attenuated.b < unattenuated.b && attenuated.r > 0, "Physical atmosphere attenuates direct surface sunlight with colored transmission");
    check(probe.renderer.atmosphere_stats().surface_lut_updates == 0, "Surface lighting switch reuses its irradiance cache");
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1, 0, -.05}});
    check(sim.replace_system(loaded).accepted, "Physical twilight fixture");
    probe.render(cam, sim.snapshot(), settings);
    auto twilight = surface_center(probe);
    check(twilight.b > 1e-6f, "Cached diffuse surface irradiance persists after local sunset");
    settings.atmosphere_surface_lighting = false;
    probe.render(cam, sim.snapshot(), settings);
    check(surface_center(probe) == glm::vec3(0), "Surface without atmosphere lighting has no artificial night floor");

    settings.atmosphere_surface_lighting = true;
    settings.aerial_perspective_enabled = true;
    settings.atmosphere_dual_source = false;
    probe.render(cam, sim.snapshot(), settings, nullptr, false, {.2f, .3f, .4f});
    auto* mapped = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> compute(mapped, mapped+size_t(probe.extent.width)*probe.extent.height*4);
    settings.atmosphere_dual_source = true;
    probe.render(cam, sim.snapshot(), settings, nullptr, false, {.2f, .3f, .4f});
    mapped = static_cast<const uint16_t*>(probe.readback.mapped);
    for (size_t i = 0; i < compute.size(); ++i) check(std::abs(glm::unpackHalf2x16(mapped[i]).x-glm::unpackHalf2x16(compute[i]).x) < .001,
        "Dual-source and compute transport agree over opaque surfaces, sky and RGB backgrounds");
    loaded.definition.bodies[1].atmosphere.pressure_pa = 0;
    check(sim.replace_system(loaded).accepted, "Physical vacuum fixture");
    cam = camera_at((radius+1)*KM_TO_AU);
    cam.yaw_actual_deg = 90;
    cam.is_near_surface = true;
    probe.render(cam, sim.snapshot(), settings, nullptr, false, {.25f, .5f, 1}, true);
    check(surface_center(probe) == glm::vec3(.25f, .5f, 1), "Zero pressure suppresses gas, ozone and supplied aerosols");
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0;
    loaded.definition.bodies[1].atmosphere = nlohmann::json{{"surface_pressure", 1.45}, {"temperature", 94.0},
        {"composition", {{"N2", .9435}, {"CH4", .0565}}}}.get<ephemeris::AtmosphereDefinition>();
    check(sim.replace_system(loaded).accepted, "Automatically enabled legacy haze fixture");
    physical = sim.snapshot().body(1).render.atmosphere;
    tau = glm::dvec3(0);
    double path = physical->height_km-1;
    for (int i = 0; i < 8192; ++i) {
        double h = 1+(i+.5)*path/8192;
        double relative = std::exp(-radius*h/(radius+h)/physical->surface.scale_height_km);
        tau += (glm::dvec3(physical->rayleigh+physical->mixed_absorption)*relative+
            glm::dvec3(physical->mie_extinction)*std::exp(-h/physical->mie_scale_height_km))*(path/8192);
        for (const auto& layer : physical->aerosols)
            tau += glm::dvec3(layer.extinction)*physics::aerosol_density(layer.shape, h)*(path/8192);
    }
    probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1), true);
    auto automatic = surface_center(probe);
    for (int c = 0; c < 3; ++c) check(std::abs(automatic[c]-std::exp(-tau[c])) < .004,
        "GPU transport consumes calculated aerosol density, height and colored extinction from inferred legacy physical data");
}

void particle_atmosphere_contracts(RenderProbe& probe) {
    probe.resize({129, 129});
    constexpr double radius = 3396.19;
    auto loaded = fixture();
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0;
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1, 0, 0}});
    ephemeris::BodyDefinition planet;
    planet.name = "Layered particle transport";
    planet.radius_rsun = radius/SOLAR_RADIUS_KM;
    planet.mass_msun = 3.227156e-7;
    auto& a = planet.atmosphere;
    a.enabled = a.physical = true;
    a.pressure_pa = 636;
    a.temperature_k = 214;
    a.composition = {{"CO2", 1}};
    a.aerosols = {{"mineral_dust", .48, 1.5, .3, 12}, {"organic_haze", .08, .04, .1, 12, 80, 1}};
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Layered physical atmosphere fixture");
    auto physical = sim.snapshot().body(1).render.atmosphere;
    auto camera = camera_at((radius+1)*KM_TO_AU);
    camera.yaw_actual_deg = 90;
    camera.is_near_surface = true;
    rendering::RenderSettings settings;
    for (double path : {.1, 2.0, 20.0, 100.0, 200.0}) {
        glm::dvec3 tau(0);
        for (int i = 0; i < 16384; ++i) {
            double height = 1+(i+.5)*path/16384;
            double relative = std::exp(-radius*height/(radius+height)/physical->surface.scale_height_km);
            tau += glm::dvec3(physical->rayleigh)*relative*(path/16384);
        }
        for (const auto& layer : physical->aerosols) {
            double column;
            if (layer.shape.z > 0) {
                double scale = std::sqrt(2.0)*layer.shape.z;
                column = layer.shape.z*std::sqrt(std::numbers::pi/2)*
                    (std::erf((1+path-layer.shape.y)/scale)-std::erf((1-layer.shape.y)/scale));
            } else column = layer.shape.x*(std::exp(-1/layer.shape.x)-std::exp(-(1+path)/layer.shape.x));
            tau += glm::dvec3(layer.extinction)*column;
        }
        float depth = camera.compute_near_plane_au()/float(path*KM_TO_AU);
        probe.render(camera, sim.snapshot(), settings, nullptr, false, glm::vec3(1), true, depth);
        auto actual = surface_center(probe);
        for (int c = 0; c < 3; ++c) {
            if (std::abs(actual[c]-std::exp(-tau[c])) >= .008)
                std::cerr << "Particle transport " << " path " << path << " channel " << c << ": " << actual[c] << " vs " << std::exp(-tau[c]) << '\n';
            check(std::abs(actual[c]-std::exp(-tau[c])) < .008,
                "Layered GPU extinction agrees with analytic exponential and Gaussian columns in the horizon aerial model");
        }
    }
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1;
    check(sim.replace_system(loaded).accepted, "Illuminated particle atmosphere fixture");
    camera.yaw_actual_deg = 0;
    settings.atmosphere_dual_source = false;
    probe.render(camera, sim.snapshot(), settings, nullptr, false, glm::vec3(.2f), true);
    auto* mapped = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> compute(mapped, mapped+size_t(probe.extent.width)*probe.extent.height*4);
    settings.atmosphere_dual_source = true;
    probe.render(camera, sim.snapshot(), settings, nullptr, false, glm::vec3(.2f), true);
    mapped = static_cast<const uint16_t*>(probe.readback.mapped);
    for (size_t i = 0; i < compute.size(); ++i) check(std::abs(glm::unpackHalf2x16(mapped[i]).x-glm::unpackHalf2x16(compute[i]).x) < .002,
        "Layered phase transport agrees between compute and dual-source composition");
    check(probe.renderer.atmosphere_stats().optical_lut_updates == 0, "Compositor changes reuse particle optics");
    auto changed = sim.snapshot().body(1).definition.atmosphere;
    changed.aerosols[0].effective_radius_um = .8;
    check(sim.apply(ephemeris::ApplyBodyAtmosphere{sim.snapshot().generation(), 1,
        sim.snapshot().body(1).revisions.definition, changed}).changed, "Particle radius edit is applied atomically");
    probe.render(camera, sim.snapshot(), settings, nullptr, false, glm::vec3(.2f), true);
    check(probe.renderer.atmosphere_stats().optical_lut_updates == 1 && probe.renderer.atmosphere_stats().aerial_lut_updates == 1,
        "Particle phase edits invalidate optics and Horizon transport");
    probe.render(camera, sim.snapshot(), settings, nullptr, false, glm::vec3(.2f), true);
    check(probe.renderer.atmosphere_stats().optical_lut_updates == 0, "Unchanged particles reuse optical tables");
}

void particle_atmosphere_previews(RenderProbe& probe, const std::filesystem::path& output = {}) {
    probe.resize({513, 257});
    if (!output.empty()) std::filesystem::create_directories(output);
    auto data = paths::read_json(paths::external_path("data", "systems", "Solar System", "system.json"));
    check(data.has_value(), "Particle preview preset exists");
    auto solar = ephemeris::SystemManager::parse_system(*data, "Particle previews");
    nlohmann::json report = nlohmann::json::array();
    for (const char* name : {"Mars", "Titan"}) for (int view = 0; view < 3; ++view) {
        auto loaded = fixture();
        loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1;
        loaded.definition.bodies[0].color_rgb = glm::vec3(1);
        auto planet = solar.definition.bodies[solar.definition.find_body(name)];
        planet.parent_id.clear();
        planet.parent_idx = -1;
        planet.semi_major_au = 0;
        loaded.definition.bodies.push_back(planet);
        loaded.motions.emplace_back();
        glm::dvec3 sun = view == 1 ? glm::dvec3(1, 0, 0) : glm::normalize(glm::dvec3(1, 0, 1));
        loaded.motions[0].position = render_to_ecliptic(RenderPosition{sun});
        ephemeris::Simulation sim;
        check(sim.replace_system(loaded).accepted, "Particle preview fixture");
        double radius = planet.radius_au()*AU_TO_KM;
        auto camera = camera_at((view == 2 ? radius*3 : radius+.01)*KM_TO_AU);
        camera.up = {0, 0, 1};
        if (view < 2) { camera.yaw_actual_deg = view == 0 ? 135 : 0; camera.is_near_surface = true; }
        else camera.up = {0, 1, 0};
        rendering::RenderSettings settings;
        settings.terrain_enabled = false;
        probe.render(camera, sim.snapshot(), settings);
        std::string label = view == 0 ? "day" : view == 1 ? "sunset" : "orbit";
        if (!output.empty()) probe.save_preview(output/(std::string(name)+"-"+label+".ppm"), true);
        auto* mapped = static_cast<const uint16_t*>(probe.readback.mapped);
        size_t offset = 4*(size_t(128)*513+256+(view == 1 ? 10 : 0));
        glm::vec3 sample;
        for (int c = 0; c < 3; ++c) sample[c] = glm::unpackHalf2x16(mapped[offset+c]).x;
        if (view == 0) check(sample.r > sample.b*1.3f, "Bundled dusty Mars and hazy Titan have warm daytime sky radiance");
        if (view == 1 && std::string(name) == "Mars") check(sample.b > sample.r*1.2f,
            "Martian particle optics produce a blue halo near the setting Sun");
        if (view == 2 && std::string(name) == "Titan") check(sample.r > sample.b*1.5f,
            "Titan's main atmospheric disc is orange in the reference orbital geometry");
        report.push_back({{"body", name}, {"view", label}, {"linear_rgb", {sample.r, sample.g, sample.b}}});
    }
    if (!output.empty()) paths::write_json(output/"samples.json", report, 2);
    std::cout << "PASS: Mars warm daytime sky and blue sunset halo; Titan orange main haze in matched reference geometries.\n";
}

void pressure_reference_atmosphere_contracts(RenderProbe& probe) {
    probe.resize({129, 129});
    constexpr double radius = 70000;
    auto loaded = fixture();
    loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 0;
    ephemeris::BodyDefinition body;
    body.name = "One-bar pressure reference";
    body.radius_rsun = radius/SOLAR_RADIUS_KM;
    body.mass_msun = .00095;
    auto& a = body.atmosphere;
    a.enabled = a.physical = a.pressure_reference = true;
    a.pressure_pa = 100000;
    a.temperature_k = 165;
    a.composition = {{"H2", .9}, {"He", .1}};
    a.bottom_altitude_km = -80;
    a.temperature_layers = {{-80, 250}, {0, 165}, {100, 110}};
    a.aerosols = {{"ammonia_ice", .3, 1, .2, 12, -20, 10}};
    loaded.definition.bodies.push_back(body);
    loaded.motions.emplace_back();
    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Pressure reference GPU fixture");
    auto physical = sim.snapshot().body(1).render.atmosphere;
    auto camera = camera_at((radius-30)*KM_TO_AU);
    camera.yaw_actual_deg = 90;
    camera.is_near_surface = true;
    rendering::RenderSettings settings;
    for (double path : {1.0, 25.0, 120.0}) {
        glm::dvec3 tau(0);
        for (int i = 0; i < 32768; ++i) {
            double height = -30+(i+.5)*path/32768;
            auto gas = physics::sample_atmosphere(a, body.mass_msun, radius, height);
            tau += glm::dvec3(physical->rayleigh)*(gas.number_density_m3/physical->surface.number_density_m3)*(path/32768);
        }
        const auto& particles = physical->aerosols[0];
        double column = 10*std::sqrt(std::numbers::pi/2)*
            (std::erf((-30+path+20)/(std::sqrt(2.0)*10))-std::erf((-30+20)/(std::sqrt(2.0)*10)));
        tau += glm::dvec3(particles.extinction)*column;
        float depth = camera.compute_near_plane_au()/float(path*KM_TO_AU);
        probe.render(camera, sim.snapshot(), settings, nullptr, false, glm::vec3(1), true, depth);
        auto actual = surface_center(probe);
        for (int c = 0; c < 3; ++c) {
            if (std::abs(actual[c]-std::exp(-tau[c])) >= .012)
                std::cerr << "Pressure reference transport " << " path " << path << " c " << c << ": " << actual[c] << " vs " << std::exp(-tau[c]) << '\n';
            check(std::abs(actual[c]-std::exp(-tau[c])) < .012,
                "GPU molecular and cloud transmission below one bar matches independent signed-column integration");
        }
    }
    settings.terrain_enabled = true;
    camera = camera_at(radius*3*KM_TO_AU);
    probe.render(camera, sim.snapshot(), settings);
    const auto& terrain = probe.renderer.terrain_stats();
    check(terrain.drawn_bodies == 1 && terrain.patches > 0 && terrain.smooth_patches == terrain.patches,
          "Pressure-defined atmospheres render a smooth quadtree at the opaque boundary");
    std::cout << "PASS: pressure-reference atmospheres and finite transport below the one-bar datum.\n";
}

void giant_atmosphere_previews(RenderProbe& probe, const std::filesystem::path& output = {}) {
    probe.resize({257, 257});
    if (!output.empty()) std::filesystem::create_directories(output);
    auto solar = ephemeris::SystemManager{}.load_system("Solar System");
    nlohmann::json report = nlohmann::json::array();
    for (const char* name : {"Venus", "Jupiter", "Saturn", "Uranus", "Neptune"}) {
        auto loaded = fixture();
        loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = 1;
        loaded.definition.bodies[0].color_rgb = glm::vec3(1);
        loaded.motions[0].position = render_to_ecliptic(RenderPosition{glm::normalize(glm::dvec3(1, 0, 2))});
        auto planet = solar.definition.bodies[solar.definition.find_body(name)];
        planet.parent_id.clear(); planet.parent_idx = -1; planet.semi_major_au = 0;
        planet.rings.clear();
        loaded.definition.bodies.push_back(planet);
        loaded.motions.emplace_back();
        ephemeris::Simulation sim;
        check(sim.replace_system(loaded).accepted, "Giant reference view fixture");
        auto camera = camera_at(planet.radius_au()*2.5);
        rendering::RenderSettings settings;
        settings.terrain_enabled = false;
        settings.show_rings = false;
        probe.render(camera, sim.snapshot(), settings);
        auto center = surface_center(probe);
        if (!output.empty()) std::cout << name << " reference RGB: " << center.r << ' ' << center.g << ' ' << center.b << '\n';
        for (int c = 0; c < 3; ++c) check(std::isfinite(center[c]) && center[c] > 1e-6f && center[c] < 10,
            "Venus and giant atmospheres render finite illuminated discs");
        if (std::string(name) == "Venus") check(center.r > center.b*1.015f,
            "Venus upper cloud absorption yields a pale warm reference disc");
        if (std::string(name) == "Uranus" || std::string(name) == "Neptune") check(center.b > center.r*1.3f,
            "Methane-rich ice giants retain blue-green reference-disc radiance");
        if (!output.empty()) probe.save_preview(output/(std::string(name)+"-orbit.ppm"), true);
        report.push_back({{"body", name}, {"linear_rgb", {center.r, center.g, center.b}}});
        auto stats = probe.renderer.atmosphere_stats();
        probe.render(camera, sim.snapshot(), settings);
        check(probe.renderer.atmosphere_stats().optical_lut_updates == 0 &&
            probe.renderer.atmosphere_stats().sky_lut_updates == 0, "Static atmosphere profiles reuse atmosphere tables");
        check(stats.optical_lut_updates >= 1, "Reference atmospheres prepare their optics");
        if (planet.atmosphere.cloud_deck_pressure_pa > 0) {
            const auto& terrain = probe.renderer.terrain_stats();
            check(terrain.drawn_bodies == 1 && terrain.patches > 6 && terrain.smooth_patches == terrain.patches,
                "Solar System gas and ice giants refine their derived cloud boundary with quadtree LOD");
            auto dimmed = loaded;
            dimmed.definition.bodies[1].color_rgb *= .5f;
            ephemeris::Simulation dimmed_sim;
            check(dimmed_sim.replace_system(dimmed).accepted, "Dim cloud reflectance fixture");
            probe.render(camera, dimmed_sim.snapshot(), settings);
            auto dim = surface_center(probe);
            check(glm::length(center-dim)/glm::length(center) > .2f,
                "Clear gas preserves visible cloud-deck reflectance contrast");
        }
        if (!output.empty()) {
            auto catalog = paths::read_json(paths::external_path("data", "surfaces.json"));
            if (catalog) for (const auto& entry : catalog->at("bodies")) {
                if (entry.at("body").get<std::string>() != name) continue;
                probe.renderer.set_color_sources({{name, terrain::ColorSource::open(paths::external_path(entry.at("albedo").get<std::string>()))}});
            }
            int quiet = 0;
            for (int frame = 0; frame < 180 && quiet < 3; ++frame) {
                probe.render(camera, sim.snapshot(), settings);
                const auto& streaming = probe.renderer.texture_stats();
                quiet = streaming.pending == 0 && streaming.ready == 0 ? quiet+1 : 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            check(quiet == 3, "Giant preview cloud textures reach stable residency");
            probe.save_preview(output/(std::string(name)+"-textured.ppm"), true);
            probe.renderer.set_color_sources({});
        }
    }
    if (!output.empty()) paths::write_json(output/"samples.json", report, 2);
    std::cout << "PASS: Venus and four giant reference views, finite cloud radiance and cache reuse.\n";
}

void atmosphere_physics_benchmark(RenderProbe& probe, const std::filesystem::path& output) {
    probe.resize({1920, 1080});
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"resolution", {1920, 1080}},
        {"method", "Serialized offscreen GPU timers exclude readback/presentation. Two repetitions, 6 warmup + 16 measured frames. CPU preparation excludes submission/waits; physics edits timed separately. Models differ physically; compare costs, not images."},
        {"cases", nlohmann::json::array()}};
    for (bool orbit : {true, false}) for (int mode = 0; mode < 3; ++mode) for (bool physical : {false, true}) for (bool dual : {false, true}) {
        std::vector<double> physics_cpu, prepare_cpu, luts, composition, surface;
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto loaded = fixture();
            loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1, 0, 1}});
            ephemeris::BodyDefinition body;
            body.name = "Atmosphere cost";
            body.mass_msun = 3.0034896154649684e-6;
            body.radius_rsun = 6378.137/SOLAR_RADIUS_KM;
            body.atmosphere = physical ? physics::earth_physical_atmosphere() : ephemeris::AtmosphereDefinition{};
            body.atmosphere.enabled = true;
            loaded.definition.bodies.push_back(body);
            loaded.motions.emplace_back();
            ephemeris::Simulation sim;
            check(sim.replace_system(loaded).accepted, "Atmosphere cost fixture");
            Camera cam = camera_at((orbit ? 6378.137*3 : 6379.137)*KM_TO_AU);
            if (!orbit) { cam.yaw_actual_deg = 0; cam.up = {0, 0, 1}; cam.is_near_surface = true; }
            rendering::RenderSettings settings;
            settings.atmosphere_dual_source = dual;
            for (int frame = 0; frame < 22; ++frame) {
                double edit_ms = 0;
                if (mode == 1) cam.cam_pos_rel.z += .001*KM_TO_AU;
                if (mode == 2) {
                    auto a = sim.snapshot().body(1).definition.atmosphere;
                    if (physical) a.pressure_pa = frame%2 ? 101325 : 102325;
                    else a.rayleigh_scattering.b = frame%2 ? .0331f : .0334f;
                    auto start = std::chrono::steady_clock::now();
                    check(sim.apply(ephemeris::ApplyBodyAtmosphere{sim.snapshot().generation(), 1, sim.snapshot().body(1).revisions.definition, a}).accepted, "Timed physical edit");
                    edit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
                }
                double prepare_ms = 0;
                probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
                    auto start = std::chrono::steady_clock::now();
                    probe.renderer.prepare(cmd, probe.vk, cam, sim.snapshot(), settings, probe.extent);
                    prepare_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
                    probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                    probe.vk.transition_image(cmd, probe.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
                    auto color = rhi::RenderingAttachment::clear_color(probe.color);
                    auto depth = rhi::RenderingAttachment::clear_depth_reverse_z(probe.depth);
                    probe.vk.begin_rendering(cmd, probe.extent, {&color, 1}, &depth);
                    probe.vk.set_viewport_and_scissor(cmd, probe.extent);
                    auto timer = probe.vk.begin_gpu_timer(cmd, "atmosphere_surface");
                    probe.renderer.render(cmd, probe.vk, cam, sim.snapshot(), settings, probe.extent);
                    probe.vk.end_gpu_timer(cmd, timer);
                    probe.vk.end_rendering(cmd);
                    probe.renderer.composite_atmospheres(cmd, probe.vk, probe.color, probe.depth);
                });
                probe.vk.drain_gpu_timings();
                if (frame < 6) continue;
                physics_cpu.push_back(edit_ms); prepare_cpu.push_back(prepare_ms);
                luts.push_back(probe.vk.gpu_time_ms("atmosphere_luts"));
                composition.push_back(probe.vk.gpu_time_ms("atmosphere_composite"));
                surface.push_back(probe.vk.gpu_time_ms("atmosphere_surface"));
                if (mode == 0) check(probe.renderer.atmosphere_stats().optical_lut_updates == 0 &&
                    probe.renderer.atmosphere_stats().surface_lut_updates == 0, "Cached benchmark performs no optical or surface rebuilds");
            }
        }
        report["cases"].push_back({{"view", orbit ? "orbit" : "ground"}, {"mode", mode == 0 ? "cached" : mode == 1 ? "translation" : "input_edit"},
            {"physical", physical}, {"dual_source", dual && probe.vk.dual_source_blend_enabled()}, {"cpu_physics_ms", terrain_timing_stats(physics_cpu)},
            {"cpu_prepare_ms", terrain_timing_stats(prepare_cpu)}, {"gpu_luts_ms", terrain_timing_stats(luts)},
            {"gpu_surface_ms", terrain_timing_stats(surface)}, {"gpu_composite_ms", terrain_timing_stats(composition)}});
    }
    std::filesystem::create_directories(output.parent_path());
    check(paths::write_json(output, report), "Write atmosphere physics cost report");
}

void atmosphere_benchmark(RenderProbe& probe, const std::filesystem::path& output) {
    check(probe.vk.timestamps_supported(), "Atmosphere benchmark requires GPU timestamps");
    probe.resize({1920, 1080});
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"resolution", {1920, 1080}},
        {"method", "Serialized offscreen draws; GPU atmosphere timers exclude image readback and presentation. Three repetitions, alternating floor order, 8 warmup + 24 measured frames per mode. Not application FPS."},
        {"cases", nlohmann::json::array()}};
    constexpr double radius = 6378.137*KM_TO_AU;
    for (bool orbit : {true, false}) {
        for (int mode = 0; mode < 4; ++mode) {
            std::array<std::vector<double>, 2> preparation, composition, aerial, surface;
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
                    if (!orbit) {
                        cam.yaw_actual_deg = 0.0;
                        cam.up = {0.0, 0.0, 1.0};
                        cam.is_near_surface = true;
                    }
                    rendering::RenderSettings settings;
                    for (int frame = 0; frame < 32; ++frame) {
                        if (mode == 1) cam.cam_pos_rel.z += 0.001*KM_TO_AU;
                        if (mode == 3) cam.yaw_actual_deg += 0.01;
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
                            uint32_t surface_timer = probe.vk.begin_gpu_timer(cmd, "atmosphere_surface");
                            probe.renderer.render(cmd, probe.vk, cam, sim.snapshot(), settings, probe.extent);
                            probe.vk.end_gpu_timer(cmd, surface_timer);
                            probe.vk.end_rendering(cmd);
                            probe.renderer.composite_atmospheres(cmd, probe.vk, probe.color, probe.depth);
                        });
                        probe.vk.drain_gpu_timings();
                        if (frame < 8) continue;
                        preparation[floor].push_back(probe.vk.gpu_time_ms("atmosphere_luts"));
                        composition[floor].push_back(probe.vk.gpu_time_ms("atmosphere_composite"));
                        aerial[floor].push_back(probe.renderer.atmosphere_stats().aerial_lut_updates ? probe.vk.gpu_time_ms("atmosphere_aerial") : 0.0);
                        surface[floor].push_back(probe.vk.gpu_time_ms("atmosphere_surface"));
                    }
                }
            }
            for (int floor = 0; floor < 2; ++floor) {
                auto row = nlohmann::json{{"view", orbit ? "orbit" : "ground"}, {"mode", mode == 0 ? "cached" : mode == 1 ? "camera_translation" : mode == 2 ? "optical_update" : "camera_rotation"},
                    {"minimum_km", floor ? -10.984 : 0.0}, {"gpu_luts_ms", terrain_timing_stats(preparation[floor])},
                    {"gpu_aerial_ms", terrain_timing_stats(aerial[floor])},
                    {"gpu_surface_ms", terrain_timing_stats(surface[floor])},
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
