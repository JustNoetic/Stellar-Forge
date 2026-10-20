#pragma once

namespace {

struct TransparentTestLayer {
    glm::vec4 color;
    glm::vec4 parameters;
};

struct TransparencyProbe {
    RenderProbe& probe;
    rhi::Pipeline capture, legacy;

    explicit TransparencyProbe(RenderProbe& p) : probe(p) {
        rhi::GraphicsPipelineDesc desc;
        desc.vert_path = "tests/transparency.vert";
        desc.frag_path = "tests/transparency.frag";
        desc.push_constant_size = sizeof(TransparentTestLayer);
        desc.depth_write = false;
        desc.premultiplied_blend = true;
        desc.cull_mode = VK_CULL_MODE_NONE;
        legacy = rhi::ShaderCache::create_graphics_pipeline(probe.vk, desc);
        desc.bindings = {{14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT},
                         {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT}};
        desc.macros = {{"TRANSPARENCY_CAPTURE", "1"}};
        capture = rhi::ShaderCache::create_graphics_pipeline(probe.vk, desc);
    }

    ~TransparencyProbe() {
        rhi::ShaderCache::destroy_pipeline(probe.vk, capture);
        rhi::ShaderCache::destroy_pipeline(probe.vk, legacy);
    }

    void record(VkCommandBuffer cmd, const Camera& camera, const ephemeris::SystemSnapshot& system,
                const rendering::RenderSettings& settings, std::span<const TransparentTestLayer> layers,
                glm::vec3 background, float opaque_depth = 0.0f, bool one_pixel = false, float coverage = 1.0f) {
        auto timer = probe.vk.begin_gpu_timer(cmd, "transparency_test_total");
        probe.renderer.prepare(cmd, probe.vk, camera, system, settings, probe.extent);
        probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        probe.vk.transition_image(cmd, probe.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        auto color = rhi::RenderingAttachment::clear_color(probe.color, background.r, background.g, background.b);
        auto depth = rhi::RenderingAttachment::clear_depth_reverse_z(probe.depth, opaque_depth);
        probe.vk.begin_rendering(cmd, probe.extent, {&color, 1}, &depth);
        probe.vk.set_viewport_and_scissor(cmd, probe.extent);
        if (one_pixel) {
            VkRect2D scissor{{int32_t(probe.extent.width/2), int32_t(probe.extent.height/2)}, {1, 1}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        } else if (coverage < 1.0f) {
            uint32_t width = std::max(1u, uint32_t(probe.extent.width*std::sqrt(coverage)));
            uint32_t height = std::max(1u, uint32_t(probe.extent.height*std::sqrt(coverage)));
            VkRect2D scissor{{int32_t((probe.extent.width-width)/2), int32_t((probe.extent.height-height)/2)}, {width, height}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        }
        const bool active = probe.renderer.transparency().active();
        auto& pipeline = active ? capture : legacy;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
        if (active) probe.vk.push_descriptors(cmd, pipeline, probe.renderer.transparency().writes());
        for (const auto& layer : layers) {
            probe.vk.push_constants(cmd, pipeline, layer);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        probe.vk.end_rendering(cmd);
        probe.renderer.composite_atmospheres(cmd, probe.vk, probe.color, probe.depth);
        probe.vk.end_gpu_timer(cmd, timer);
    }

    glm::vec3 render(const Camera& camera, const ephemeris::SystemSnapshot& system,
                     const rendering::RenderSettings& settings, std::span<const TransparentTestLayer> layers,
                     glm::vec3 background = glm::vec3(0.0f), float opaque_depth = 0.0f, bool one_pixel = false) {
        probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
            record(cmd, camera, system, settings, layers, background, opaque_depth, one_pixel);
            probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {probe.extent.width, probe.extent.height, 1};
            vkCmdCopyImageToBuffer(cmd, probe.color.handle, probe.color.layout, probe.readback.handle, 1, &copy);
            probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        });
        vmaInvalidateAllocation(probe.vk.allocator(), probe.readback.allocation, 0, VK_WHOLE_SIZE);
        return surface_center(probe);
    }
};

void transparency_contracts(RenderProbe& probe) {
    if (!probe.vk.fragment_stores_enabled()) {
        std::cout << "SKIP: transparency contracts require fragment storage atomics; legacy fallback remains active.\n";
        return;
    }
    probe.resize({257, 257});
    TransparencyProbe transparent(probe);
    auto loaded = fixture();
    ephemeris::BodyDefinition body;
    body.name = "Transparent transport host";
    body.radius_rsun = 6378.137/SOLAR_RADIUS_KM;
    loaded.definition.bodies.push_back(body);
    loaded.motions.emplace_back();
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Transparency ordering fixture");
    rendering::RenderSettings settings;
    settings.show_atmospheres = false;
    auto camera = camera_at(1.0);
    std::vector<TransparentTestLayer> crossing{
        {{1, 0, 0, 0.5f}, {0.5f, 0.2f, 0, 0}},
        {{0, 0, 1, 0.5f}, {0.5f, -0.2f, 0, 0}}};
    transparent.render(camera, simulation.snapshot(), settings, crossing);
    auto left = probe.red_at(32, 128), right = probe.red_at(224, 128);
    check(std::abs(left-0.25f) < 0.002f && std::abs(right-0.5f) < 0.002f,
          "Crossing transparent surfaces exchange foreground order across the image");
    std::vector<uint16_t> first(257*257*4);
    std::memcpy(first.data(), probe.readback.mapped, first.size()*sizeof(uint16_t));
    std::reverse(crossing.begin(), crossing.end());
    transparent.render(camera, simulation.snapshot(), settings, crossing);
    check(std::memcmp(first.data(), probe.readback.mapped, first.size()*sizeof(uint16_t)) == 0,
          "Transparent surface crossings are independent of draw submission order");
    transparent.render(camera, simulation.snapshot(), settings, crossing, glm::vec3(0.0f), 0.6f);
    check(probe.red_at(32, 128) == 0.0f && std::abs(probe.red_at(224, 128)-0.5f) < 0.002f,
          "Opaque reverse-Z depth rejects hidden transparent fragments before collection");

    std::vector<TransparentTestLayer> deep;
    glm::vec3 expected(0.0f);
    for (uint32_t i = 0; i < 48; ++i) {
        glm::vec3 color(float(i%3 == 0), float(i%3 == 1), float(i%3 == 2));
        deep.push_back({glm::vec4(color, 0.125f), {0.01f+float(i)*0.02f, 0, 0, 0}});
        expected = color*0.125f+expected*0.875f;
    }
    const auto ordered = deep;
    for (size_t i = 0; i < deep.size(); ++i) deep[i] = ordered[(i*17)%deep.size()];
    auto actual = transparent.render(camera, simulation.snapshot(), settings, deep, glm::vec3(0.0f), 0.0f, true);
    check(glm::length(actual-expected) < 0.003f, "All 48 transparent layers resolve in depth order without a per-pixel truncation");
    auto statistics = probe.renderer.transparency().stats(probe.vk);
    check(statistics.fragments == 48 && statistics.overflow == 0, "Sparse fragment counters match GPU collection");

    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1, 0, 1}});
    loaded.definition.bodies[1].atmosphere.enabled = true;
    settings.show_atmospheres = true;
    settings.exposure = 0.8f;
    glm::vec3 surface(0.8f, 0.5f, 0.3f), background(0.2f, 0.3f, 0.4f);
    for (bool lit : {false, true}) {
        loaded.definition.bodies[0].lum_eq = loaded.definition.bodies[0].lum_pole = lit ? 1.0 : 0.0;
        check(simulation.replace_system(loaded).accepted, "Transparent atmospheric transport fixture");
        for (bool dual : {false, true}) {
            settings.atmosphere_dual_source = dual;
            camera = camera_at((6378.137+1.0)*KM_TO_AU);
            camera.yaw_actual_deg = 90.0;
            camera.is_near_surface = true;
            for (double distance : {0.01, 2.0, 50.0, 150.0}) {
                float depth = camera.compute_near_plane_au()/float(distance*KM_TO_AU);
                probe.render(camera, simulation.snapshot(), settings, nullptr, false, background, true);
                auto base = surface_center(probe);
                probe.render(camera, simulation.snapshot(), settings, nullptr, false, surface, true, depth);
                auto endpoint = surface_center(probe);
                for (float alpha : {0.0f, 0.6f, 1.0f}) {
                    TransparentTestLayer layer{glm::vec4(surface, alpha), {depth, 0, 0, 0}};
                    actual = transparent.render(camera, simulation.snapshot(), settings, {&layer, 1}, background);
                    expected = alpha*endpoint+(1.0f-alpha)*base;
                    check(glm::length(actual-expected) < 0.005f,
                          "Transparent fragments receive matching finite RGB extinction and alpha-weighted inscattering in both blend paths");
                }
            }
            camera = camera_at(6378.137*3.0*KM_TO_AU);
            float depth = camera.compute_near_plane_au()/float(100.0*KM_TO_AU);
            probe.render(camera, simulation.snapshot(), settings, nullptr, false, background, true);
            auto base = surface_center(probe);
            TransparentTestLayer foreground{glm::vec4(surface, 0.6f), {depth, 0, 0, 0}};
            actual = transparent.render(camera, simulation.snapshot(), settings, {&foreground, 1}, background);
            check(glm::length(actual-(surface*0.6f+base*0.4f)) < 0.005f,
                  "Transparent foreground in vacuum occludes atmospheric light without receiving extinction");
        }
    }

    probe.resize({4, 4});
    settings.show_atmospheres = false;
    std::vector<TransparentTestLayer> overflow(40, {{1, 0.5f, 0.25f, 0.25f}, {0.5f, 0, 0, 0}});
    transparent.render(camera, simulation.snapshot(), settings, overflow, glm::vec3(0.0f), 0.0f, true);
    statistics = probe.renderer.transparency().stats(probe.vk);
    check(statistics.fragments == 40 && statistics.overflow == 8 && statistics.capacity == 32,
          "Fragment budget overflow is counted explicitly after viewport resize");
    check(std::abs(probe.red_at(2, 2)-(1.0f-std::pow(0.75f, 40.0f))) < 0.003f,
          "Overflow fragments remain visible through the bounded fallback");
    probe.resize(EXTENT);
    std::cout << "Transparency contracts passed: crossings, draw order, opaque depth, 48 layers, RGB extinction, inscattering, horizon LUT and both blend paths, foreground and resize.\n";
}

void transparency_benchmark(RenderProbe& probe, const std::filesystem::path& output) {
    check(probe.vk.fragment_stores_enabled() && probe.vk.timestamps_supported(),
          "Transparency benchmark requires fragment storage atomics and GPU timestamps");
    TransparencyProbe transparent(probe);
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{1, 0, 1}});
    ephemeris::BodyDefinition body;
    body.name = "Transparency benchmark atmosphere";
    body.radius_rsun = 6378.137/SOLAR_RADIUS_KM;
    body.atmosphere.enabled = true;
    loaded.definition.bodies.push_back(body);
    loaded.motions.emplace_back();
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Transparency benchmark fixture");
    auto camera = camera_at((6378.137+1)*KM_TO_AU);
    camera.yaw_actual_deg = 90;
    camera.is_near_surface = true;
    nlohmann::json report{{"gpu", probe.vk.device_name()}, {"method", "Cached LUTs; serialized GPU timestamps excluding readback/presentation; 128 fullscreen prewarm draws per resolution, 3 alternating repetitions, 8 warmup and 24 measured frames per mode; empty/sparse/fullscreen synthetic sheets and actual ring geometry."},
        {"cases", nlohmann::json::array()}};
    for (VkExtent2D extent : {VkExtent2D{1920, 1080}, VkExtent2D{3840, 2160}}) {
        probe.resize(extent);
        rendering::RenderSettings warmup;
        warmup.show_atmospheres = true;
        TransparentTestLayer warm_layer{{0.7f, 0.5f, 0.3f, 0.6f},
            {camera.compute_near_plane_au()/float(50.0*KM_TO_AU), 0, 0, 0}};
        for (int frame = 0; frame < 128; ++frame) {
            probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
                transparent.record(cmd, camera, simulation.snapshot(), warmup, {&warm_layer, 1}, glm::vec3(0.2f));
            });
            probe.vk.drain_gpu_timings();
        }
        for (bool atmosphere : {false, true}) {
            for (bool dual : {false, true}) {
                if (!atmosphere && dual) continue;
                for (float coverage : {0.0f, 0.01f, 1.0f}) {
                std::array<std::vector<double>, 2> total, clear, resolve, composite;
                for (int repeat = 0; repeat < 3; ++repeat) {
                    for (int order = 0; order < 2; ++order) {
                        int mode = repeat%2 ? 1-order : order;
                        rendering::RenderSettings settings;
                        settings.show_atmospheres = atmosphere;
                        settings.atmosphere_dual_source = dual;
                        settings.transparency_reference_legacy = mode == 0;
                        float depth = camera.compute_near_plane_au()/float(50.0*KM_TO_AU);
                        TransparentTestLayer layer{{0.7f, 0.5f, 0.3f, 0.6f}, {depth, 0, 0, 0}};
                        for (int frame = 0; frame < 32; ++frame) {
                            probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
                                std::span<const TransparentTestLayer> layers = coverage > 0.0f ? std::span{&layer, 1} : std::span<const TransparentTestLayer>{};
                                transparent.record(cmd, camera, simulation.snapshot(), settings, layers, glm::vec3(0.2f), 0.0f, false, coverage);
                            });
                            probe.vk.drain_gpu_timings();
                            if (frame < 8) continue;
                            total[mode].push_back(probe.vk.gpu_time_ms("transparency_test_total"));
                            clear[mode].push_back(mode ? probe.vk.gpu_time_ms("transparency_clear") : 0.0);
                            resolve[mode].push_back(mode ? probe.vk.gpu_time_ms("transparency_resolve") : 0.0);
                            composite[mode].push_back(atmosphere ? probe.vk.gpu_time_ms("atmosphere_composite") : 0.0);
                        }
                    }
                }
                auto stats = probe.renderer.transparency().stats(probe.vk);
                auto row = nlohmann::json{{"resolution", {extent.width, extent.height}}, {"coverage", coverage}, {"atmosphere", atmosphere}, {"dual_source", dual},
                    {"legacy_total_ms", terrain_timing_stats(total[0])}, {"ordered_total_ms", terrain_timing_stats(total[1])},
                    {"ordered_clear_ms", terrain_timing_stats(clear[1])}, {"ordered_resolve_ms", terrain_timing_stats(resolve[1])},
                    {"legacy_atmosphere_ms", terrain_timing_stats(composite[0])}, {"ordered_atmosphere_ms", terrain_timing_stats(composite[1])},
                    {"allocation_bytes", stats.bytes}, {"fragment_count", stats.fragments}, {"overflow", stats.overflow}};
                std::cout << row.dump() << '\n';
                report["cases"].push_back(std::move(row));
                }
            }
        }
    }
    auto ring_system = surface_ring_fixture(1);
    ephemeris::Simulation rings;
    check(rings.replace_system(ring_system).accepted, "Transparency ring benchmark fixture");
    const double radius = rings.snapshot().body(1).definition.radius_au();
    auto ring_camera = camera_at(radius*5);
    auto direction = glm::normalize(glm::dvec3(0, 0.65, 1));
    ring_camera.cam_pos_rel = direction*radius*5.0;
    ring_camera.yaw_actual_deg = -90;
    ring_camera.pitch_actual_deg = std::asin(-direction.y)*RAD_TO_DEG;
    for (VkExtent2D extent : {VkExtent2D{1920, 1080}, VkExtent2D{3840, 2160}}) {
        probe.resize(extent);
        std::array<std::vector<double>, 2> total;
        for (int repeat = 0; repeat < 3; ++repeat) for (int order = 0; order < 2; ++order) {
            int mode = repeat%2 ? 1-order : order;
            rendering::RenderSettings settings;
            settings.show_atmospheres = false;
            settings.surface_textures_enabled = false;
            settings.exposure = 100;
            settings.transparency_reference_legacy = mode == 0;
            for (int frame = 0; frame < 32; ++frame) {
                probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
                    auto timer = probe.vk.begin_gpu_timer(cmd, "transparency_ring_total");
                    probe.renderer.prepare(cmd, probe.vk, ring_camera, rings.snapshot(), settings, extent);
                    probe.vk.transition_image(cmd, probe.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                    probe.vk.transition_image(cmd, probe.depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
                    auto color = rhi::RenderingAttachment::clear_color(probe.color);
                    auto depth = rhi::RenderingAttachment::clear_depth_reverse_z(probe.depth);
                    probe.vk.begin_rendering(cmd, extent, {&color, 1}, &depth);
                    probe.renderer.render(cmd, probe.vk, ring_camera, rings.snapshot(), settings, extent);
                    probe.vk.end_rendering(cmd);
                    probe.renderer.composite_atmospheres(cmd, probe.vk, probe.color, probe.depth);
                    probe.vk.end_gpu_timer(cmd, timer);
                });
                probe.vk.drain_gpu_timings();
                if (frame >= 8) total[mode].push_back(probe.vk.gpu_time_ms("transparency_ring_total"));
            }
        }
        auto stats = probe.renderer.transparency().stats(probe.vk);
        auto row = nlohmann::json{{"resolution", {extent.width, extent.height}}, {"scene", "lit_ring"},
            {"legacy_total_ms", terrain_timing_stats(total[0])}, {"ordered_total_ms", terrain_timing_stats(total[1])},
            {"fragment_count", stats.fragments}, {"overflow", stats.overflow}};
        report["cases"].push_back(row);
        std::cout << row.dump() << '\n';
    }
    std::filesystem::create_directories(output.parent_path());
    check(paths::write_json(output, report), "Write transparency benchmark report");
}

}
