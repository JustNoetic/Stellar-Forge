#include "core/constants.hpp"
#include "benchmark/earth_closeup.hpp"
#include "ephemeris/simulation.hpp"
#include "rendering/celestial_renderer.hpp"
#include "rendering/eclipse_candidates.hpp"
#include "rendering/star_catalog.hpp"
#include "rhi/shader_cache.hpp"
#include <GLFW/glfw3.h>
#include <glm/gtc/packing.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace sf;

namespace {

constexpr VkExtent2D EXTENT{256, 256};
constexpr double FOCAL = 256.0 / (2.0 * 0.5773502691896257);

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ephemeris::LoadedSystem fixture(double caster_radius_px = 0.0, double distance = 1.0) {
    ephemeris::LoadedSystem loaded;
    ephemeris::BodyDefinition star;
    star.name = "Sun";
    star.is_star = true;
    star.type = "Star";
    loaded.definition.bodies.push_back(star);
    loaded.motions.emplace_back();
    if (caster_radius_px > 0.0) {
        ephemeris::BodyDefinition caster;
        caster.name = "Caster";
        caster.radius_rsun = caster_radius_px * distance * 0.5 / FOCAL / SOLAR_RADII_TO_AU;
        caster.color_rgb = glm::vec3(0.0f);
        loaded.definition.bodies.push_back(caster);
        ephemeris::BodyMotion motion;
        motion.position = render_to_ecliptic(RenderPosition{{0.0, 0.0, distance * 0.5}});
        loaded.motions.push_back(motion);
    }
    for (size_t i = 0; i < loaded.definition.bodies.size(); ++i)
        loaded.definition.name_to_idx[loaded.definition.bodies[i].name] = static_cast<int>(i);
    return loaded;
}

struct RenderProbe {
    rhi::VkContext vk;
    rendering::CelestialRenderer renderer;
    rhi::Image color;
    rhi::Image depth;
    rhi::Image composite;
    rhi::Pipeline composite_pipeline;
    rhi::Buffer readback;
    VkExtent2D extent = EXTENT;
    GLFWwindow* window = nullptr;

    RenderProbe() {
        check(glfwInit() != 0, "GLFW initialization");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(256, 256, "Rendering tests", nullptr, nullptr);
        check(window != nullptr, "Hidden test window");
        rhi::ShaderCache::init();
        vk.init(window, false);
        renderer.init(vk);
        rhi::GraphicsPipelineDesc desc{};
        desc.vert_path = "post/fullscreen.vert";
        desc.frag_path = "post/composite.frag";
        desc.bindings = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1}};
        desc.color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
        desc.depth_format = VK_FORMAT_UNDEFINED;
        desc.depth_test = false;
        desc.depth_write = false;
        desc.cull_mode = VK_CULL_MODE_NONE;
        composite_pipeline = rhi::ShaderCache::create_graphics_pipeline(vk, desc);
        resize(EXTENT);
    }

    void resize(VkExtent2D size) {
        vk.wait_idle();
        vk.destroy_buffer(readback);
        vk.destroy_image(composite);
        vk.destroy_image(depth);
        vk.destroy_image(color);
        extent = size;
        color = vk.create_image_2d(size.width, size.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                  | VK_IMAGE_USAGE_SAMPLED_BIT);
        depth = vk.create_image_2d(size.width, size.height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        composite = vk.create_image_2d(size.width, size.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        readback = vk.create_host_buffer(static_cast<VkDeviceSize>(size.width) * size.height * 8,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    ~RenderProbe() {
        vk.wait_idle();
        vk.destroy_buffer(readback);
        vk.destroy_image(composite);
        vk.destroy_image(depth);
        vk.destroy_image(color);
        renderer.shutdown(vk);
        rhi::ShaderCache::destroy_pipeline(vk, composite_pipeline);
        vk.shutdown();
        rhi::ShaderCache::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    double render(const Camera& camera, const ephemeris::SystemSnapshot& system,
                  const rendering::RenderSettings& settings, rendering::StarCatalog* catalog = nullptr,
                  bool tone_map = false) {
        vk.immediate_submit([&](VkCommandBuffer cmd) {
            if (!catalog) renderer.prepare(cmd, vk, camera, system, settings, extent);
            if (catalog) catalog->cull(cmd, vk, camera, units::Years{0.0}, settings, extent);
            vk.transition_image(cmd, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vk.transition_image(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            auto color_attachment = rhi::RenderingAttachment::clear_color(color);
            auto depth_attachment = rhi::RenderingAttachment::clear_depth_reverse_z(depth);
            vk.begin_rendering(cmd, extent, {&color_attachment, 1}, &depth_attachment);
            vk.set_viewport_and_scissor(cmd, extent);
            if (catalog) catalog->render(cmd, vk, camera, units::Years{0.0}, settings, extent);
            else renderer.render(cmd, vk, camera, system, settings, extent);
            vk.end_rendering(cmd);
            if (tone_map) {
                vk.transition_image(cmd, color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                vk.transition_image(cmd, composite, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                auto attachment = rhi::RenderingAttachment::clear_color(composite);
                vk.begin_rendering(cmd, extent, {&attachment, 1});
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composite_pipeline.handle);
                auto write = rhi::PushDescriptorWrite::sampled_image(0, color, vk.sampler_linear_clamp());
                vk.push_descriptors(cmd, composite_pipeline, {&write, 1});
                vkCmdDraw(cmd, 3, 1, 0, 0);
                vk.end_rendering(cmd);
            }
            auto& output = tone_map ? composite : color;
            vk.transition_image(cmd, output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            vkCmdCopyImageToBuffer(cmd, output.handle, output.layout, readback.handle, 1, &copy);
            vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        });
        vmaInvalidateAllocation(vk.allocator(), readback.allocation, 0, VK_WHOLE_SIZE);
        auto* pixels = static_cast<const uint16_t*>(readback.mapped);
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(extent.width) * extent.height; ++i) {
            float red = glm::unpackHalf2x16(pixels[4 * i]).x;
            check(std::isfinite(red), "Finite HDR output");
            sum += red;
        }
        return sum;
    }

    float red_at(uint32_t x, uint32_t y) const {
        auto* pixels = static_cast<const uint16_t*>(readback.mapped);
        return glm::unpackHalf2x16(pixels[4 * (static_cast<size_t>(y) * extent.width + x)]).x;
    }

    void check_diameter(int diameter) const {
        int min_x = static_cast<int>(extent.width), min_y = static_cast<int>(extent.height);
        int max_x = -1, max_y = -1;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                if (red_at(x, y) <= 0.0f) continue;
                min_x = std::min(min_x, static_cast<int>(x));
                max_x = std::max(max_x, static_cast<int>(x));
                min_y = std::min(min_y, static_cast<int>(y));
                max_y = std::max(max_y, static_cast<int>(y));
            }
        }
        check(max_x >= min_x, "Displayed point remains visible");
        check(max_x - min_x + 1 <= diameter && max_y - min_y + 1 <= diameter,
              "Displayed point obeys its configured pixel diameter");
    }

    void save_preview(const std::filesystem::path& path, bool full = false) const {
        std::ofstream file(path, std::ios::binary);
        uint32_t width = full ? extent.width : 64;
        uint32_t height = full ? extent.height : 64;
        uint32_t x0 = (extent.width - width) / 2;
        uint32_t y0 = (extent.height - height) / 2;
        file << "P6\n" << width << ' ' << height << "\n255\n";
        auto* pixels = static_cast<const uint16_t*>(readback.mapped);
        for (uint32_t y = y0; y < y0 + height; ++y) {
            for (uint32_t x = x0; x < x0 + width; ++x) {
                for (uint32_t c = 0; c < 3; ++c) {
                    float v = glm::unpackHalf2x16(pixels[4 * (static_cast<size_t>(y) * extent.width + x) + c]).x;
                    file.put(static_cast<char>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f));
                }
            }
        }
    }
};

Camera camera_at(double distance, double pixel_offset = 0.0) {
    Camera camera;
    camera.fov_deg = 60.0;
    camera.cam_pos_rel = {pixel_offset * distance / FOCAL, 0.0, distance};
    camera.cam_look_aim = false;
    camera.yaw_actual_deg = -90.0;
    camera.pitch_actual_deg = 0.0;
    return camera;
}

void terrain_surfaces(RenderProbe& probe, const std::filesystem::path& output = {}) {
    probe.resize({512, 512});
    if (!output.empty()) std::filesystem::create_directories(output);
    constexpr double radius = 6378.137 * KM_TO_AU;
    auto make_system = [](glm::dvec3 origin, double flattening) {
        auto loaded = fixture();
        loaded.motions[0].position = render_to_ecliptic(RenderPosition{origin + glm::dvec3(0, 0, 10)});
        loaded.definition.bodies[0].lum_eq = 100.0;
        loaded.definition.bodies[0].lum_pole = 100.0;
        ephemeris::BodyDefinition earth;
        earth.name = "Terrain probe";
        earth.type = "Terrestrial";
        earth.radius_rsun = radius / SOLAR_RADII_TO_AU;
        earth.oblateness = flattening;
        earth.color_rgb = {0.45f, 0.65f, 0.3f};
        loaded.definition.bodies.push_back(earth);
        ephemeris::BodyMotion motion;
        motion.position = render_to_ecliptic(RenderPosition{origin});
        loaded.motions.push_back(motion);
        loaded.definition.name_to_idx[earth.name] = 1;
        return loaded;
    };
    ephemeris::Simulation simulation;
    check(simulation.replace_system(make_system({}, 0.15)).accepted, "Terrain fixture");
    rendering::RenderSettings settings;
    settings.terrain_enabled = true;
    settings.terrain_debug_tiles = true;
    settings.terrain_patch_limit = 512;
    settings.terrain_height_km = 2.0f;
    Camera camera = camera_at(radius * 3.0);
    camera.tracking_idx = 1;
    check(probe.render(camera, simulation.snapshot(), settings) > 1000.0, "Oblate terrain produces visible GPU output");
    check(probe.renderer.terrain_stats().patches > 6, "GPU terrain uses adaptive patches");
    check(probe.renderer.terrain_stats().patches <= settings.terrain_patch_limit, "GPU patch count respects the budget");
    for (int y = 160; y < 352; ++y)
        for (int x = 160; x < 352; ++x)
            check(probe.red_at(x, y) > 0.01f, "No holes inside the resolved terrain disc");
    if (!output.empty()) probe.save_preview(output / "orbit-tiles.ppm", true);
    settings.terrain_debug_tiles = false;
    check(probe.render(camera, simulation.snapshot(), settings) > 1000.0, "Terrain retains celestial surface lighting");
    if (!output.empty()) probe.save_preview(output / "orbit-relief.ppm", true);
    settings.terrain_debug_tiles = true;
    for (const auto& [name, direction] : std::vector<std::pair<std::string, glm::dvec3>>{
             {"pole", {0, 1, 0}}, {"face-edge", glm::normalize(glm::dvec3(1, 0, 1))},
             {"face-corner", glm::normalize(glm::dvec3(1, 1, 1))}}) {
        camera.cam_pos_rel = direction * radius * 2.0;
        camera.up = std::abs(direction.y) > 0.9 ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
        camera.center_body(1, simulation.snapshot());
        check(probe.render(camera, simulation.snapshot(), settings) > 1000.0, "Cube edges and poles remain visible");
        check(probe.red_at(256, 256) > 0.01f, "Cube junction has no center hole");
        if (!output.empty()) probe.save_preview(output / (name + ".ppm"), true);
    }
    settings.terrain_height_km = 0.0f;
    for (glm::dvec3 origin : {glm::dvec3(0), glm::dvec3(1.0, -2.0, 3.0)}) {
        check(simulation.replace_system(make_system(origin, 0.0)).accepted, "Translated terrain fixture");
        for (double altitude_km : {1.0, 0.001}) {
            camera = camera_at(radius + altitude_km * KM_TO_AU);
            camera.origin_render = origin;
            camera.tracking_idx = 1;
            camera.is_near_surface = true;
            probe.render(camera, simulation.snapshot(), settings);
            check(probe.renderer.terrain_stats().deepest_level >= 10, "Near-surface GPU view selects fine patches");
            check(probe.renderer.terrain_stats().patches <= settings.terrain_patch_limit, "Near-surface VRAM workload stays bounded");
            for (uint32_t y = 4; y < 508; ++y)
                for (uint32_t x = 4; x < 508; ++x)
                    check(probe.red_at(x, y) > 0.01f, "Continuous meter-scale surface at local and astronomical origins");
            if (!output.empty() && glm::length(origin) == 0.0)
                probe.save_preview(output / (altitude_km == 1.0 ? "one-km.ppm" : "one-meter.ppm"), true);
        }
    }
    settings.terrain_enabled = false;
    probe.render(camera_at(radius * 3.0), simulation.snapshot(), settings);
    check(probe.renderer.terrain_stats().patches == 0, "Disabling terrain clears the prepared draw list");
    probe.resize(EXTENT);
    std::cout << "Terrain GPU contracts passed: oblate disc, lighting, cube seams, poles, meter precision and bounded patches.\n";
}

void earth_closeups(RenderProbe& probe, const std::filesystem::path& output) {
    ephemeris::SystemManager manager;
    rendering::RenderSettings settings;
    if (!output.empty()) {
        std::filesystem::create_directories(output);
        probe.resize({1920, 1080});
    }
    for (bool eclipse : {false, true}) {
        ephemeris::Simulation simulation;
        check(simulation.replace_system(manager.load_system("Solar System")).accepted, "Earth fixture load");
        check(simulation.apply(ephemeris::SetMotionModel{ephemeris::MotionModel::Keplerian}).accepted, "Earth Keplerian mode");
        if (eclipse) check(simulation.advance(units::Seconds{benchmark::ECLIPSE_FIXTURE_YEARS * SECONDS_PER_YEAR}).accepted, "Eclipse epoch");
        auto camera = benchmark::earth_closeup_camera(simulation.snapshot(), eclipse);
        camera.update(units::Seconds{1.0 / 60.0}, simulation.snapshot());
        const auto& system = simulation.snapshot();
        auto sphere = [&](int index) {
            auto body = system.body(index);
            return glm::vec4(glm::vec3(camera_relative(body.render.position, camera.eye_pos_render())), float(body.definition.radius_au()));
        };
        uint32_t candidates = 0;
        int earth = system.find_body("Earth");
        for (size_t i = 0; i < system.size(); ++i) {
            if (static_cast<int>(i) == earth || system.body(i).definition.is_star) continue;
            if (rendering::eclipse_could_shadow(sphere(earth), sphere(system.star_index()), sphere(static_cast<int>(i)))) {
                ++candidates;
                if (eclipse) check(system.body(i).definition.name == "Moon", "Eclipse fixture retains only the Moon");
            }
        }
        check(candidates == (eclipse ? 1u : 0u), "Earth close-up candidate count");
        double flux = probe.render(camera, simulation.snapshot(), settings);
        check(flux > 0.0, "Earth close-up remains illuminated");
        if (!output.empty()) {
            auto name = eclipse ? "earth-eclipse" : "earth-full";
            std::ofstream file(output / (std::string(name) + ".rgba16"), std::ios::binary);
            file.write(static_cast<const char*>(probe.readback.mapped), static_cast<std::streamsize>(probe.extent.width) * probe.extent.height * 8);
        }
    }
    probe.resize(EXTENT);
    std::cout << "Screen-filling Earth and Keplerian eclipse fixtures rendered.\n";
}

void eclipse_lighting(RenderProbe& probe) {
    rendering::RenderSettings settings;
    auto center_light = [&](glm::dvec3 caster_position, bool blocker, bool second_star, bool many_bodies = false) {
        ephemeris::LoadedSystem loaded;
        auto add = [&](const char* name, glm::dvec3 position, double radius, bool star) {
            ephemeris::BodyDefinition body;
            body.name = name;
            body.radius_rsun = radius / SOLAR_RADII_TO_AU;
            body.is_star = star;
            body.lum_eq = body.lum_pole = 10000.0;
            loaded.definition.bodies.push_back(body);
            ephemeris::BodyMotion motion;
            motion.position = render_to_ecliptic(RenderPosition{position});
            loaded.motions.push_back(motion);
        };
        add("Sun", {0, 0, 100}, 1.0, true);
        add("Receiver", {0, 0, 0}, 1.0, false);
        if (blocker) add("Caster", caster_position, 0.5, false);
        if (second_star) add("Other sun", {20, 0, 100}, 1.0, true);
        if (many_bodies) {
            for (int i = 0; i < 600; ++i)
                add(("Distant " + std::to_string(i)).c_str(), {1000.0 + i, 1000, 1000}, 0.01, false);
        }
        ephemeris::Simulation simulation;
        check(simulation.replace_system(std::move(loaded)).accepted, "Eclipse lighting fixture");
        probe.render(camera_at(3.0), simulation.snapshot(), settings);
        return probe.red_at(128, 128);
    };
    float clear = center_light({}, false, false);
    check(clear > 0.1f, "Unshadowed receiver is illuminated");
    check(center_light({0, 0, 10}, true, false) < clear * 0.001f, "Aligned blocker shadows the receiver");
    check(std::abs(center_light({30, 0, 10}, true, false) / clear - 1.0f) < 0.001f, "Rejected lateral blocker preserves illumination");
    check(std::abs(center_light({0, 0, -10}, true, false) / clear - 1.0f) < 0.001f, "Blocker behind receiver casts no shadow");
    check(std::abs(center_light({0, 0, 120}, true, false) / clear - 1.0f) < 0.001f, "Blocker beyond light casts no shadow");
    float two_clear = center_light({}, false, true);
    float two_shadow = center_light({0, 0, 10}, true, true);
    check(two_shadow > clear * 0.5f && two_shadow < two_clear * 0.75f, "Candidate ranges preserve independent lighting from two stars");
    check(std::abs(center_light({0, 0, 10}, true, true, true) / two_shadow - 1.0f) < 0.001f,
          "Growing candidate buffers preserve two-star illumination");
    check(std::abs(center_light({}, false, false) / clear - 1.0f) < 0.001f,
          "Changing body/star counts clears previous candidate ranges");
    std::cout << "Surface eclipse rejection, totality and independent two-star illumination passed.\n";
}

void transition_and_distance(RenderProbe& probe) {
    ephemeris::Simulation simulation;
    check(simulation.replace_system(fixture()).accepted, "Solar fixture");
    rendering::RenderSettings settings;
    settings.exposure = 1e-6f;
    double min_ratio = 10.0;
    double max_ratio = 0.0;
    for (double radius_px = 1.3; radius_px <= 2.71; radius_px += 0.05) {
        double distance = SOLAR_RADII_TO_AU * FOCAL / radius_px;
        for (double offset : {0.0, 0.25, 0.5, 0.75}) {
            double flux = probe.render(camera_at(distance, offset), simulation.snapshot(), settings);
            double expected = PI * (5.0 / 6.0) * FOCAL * FOCAL * settings.exposure / (distance * distance);
            double ratio = flux / expected;
            min_ratio = std::min(min_ratio, ratio);
            max_ratio = std::max(max_ratio, ratio);
            check(ratio > 0.80 && ratio < 1.20, "Continuous mesh/point integrated flux");
        }
    }
    double distance = SOLAR_RADII_TO_AU * FOCAL / 0.2;
    double near_flux = probe.render(camera_at(distance), simulation.snapshot(), settings);
    double far_flux = probe.render(camera_at(distance * 2.0), simulation.snapshot(), settings);
    check(std::abs(far_flux / near_flux - 0.25) < 0.002, "Inverse-square observer falloff");
    double baseline = near_flux;
    for (double offset : {0.1, 0.25, 0.5, 0.75, 0.9}) {
        double flux = probe.render(camera_at(distance, offset), simulation.snapshot(), settings);
        check(std::abs(flux / baseline - 1.0) < 0.005, "Stable subpixel integrated flux");
    }
    settings.star_point_size = 6.0f;
    double wide_flux = probe.render(camera_at(distance), simulation.snapshot(), settings);
    check(std::abs(wide_flux / baseline - 1.0) < 0.005, "Point size preserves total flux");
    std::cout << "Transition flux/expected range: " << min_ratio << " to " << max_ratio
              << "; doubled-distance ratio: " << far_flux / near_flux << '\n';
}

void occlusion(RenderProbe& probe) {
    rendering::RenderSettings settings;
    settings.exposure = 1e-6f;
    double distance = SOLAR_RADII_TO_AU * FOCAL / 0.2;
    ephemeris::Simulation simulation;
    simulation.replace_system(fixture());
    double baseline = probe.render(camera_at(distance), simulation.snapshot(), settings);
    for (double radius : {0.01, 0.5, 1.4, 1.5}) {
        simulation.replace_system(fixture(radius, distance));
        double flux = probe.render(camera_at(distance), simulation.snapshot(), settings);
        check(std::abs(flux / baseline - 1.0) < 0.002, "Unresolved foreground body does not occlude a point");
    }
    simulation.replace_system(fixture(2.0, distance));
    double half = probe.render(camera_at(distance), simulation.snapshot(), settings);
    check(std::abs(half / baseline - 0.5) < 0.005, "Transitioning occluder fades continuously");
    simulation.replace_system(fixture(3.0, distance));
    check(probe.render(camera_at(distance), simulation.snapshot(), settings) < baseline * 0.001,
          "Resolved foreground mesh occludes a point");
    constexpr double FAR_DISTANCE = 1e8;
    settings.exposure = 1e14f;
    simulation.replace_system(fixture());
    double far_baseline = probe.render(camera_at(FAR_DISTANCE), simulation.snapshot(), settings);
    check(far_baseline > 0.0, "Very distant Sun remains visible at sufficient exposure");
    auto far_system = fixture(0.5, 2.0);
    far_system.definition.bodies[1].radius_rsun = 0.01;
    for (double shift : {-1e-6, 0.0, 1e-6}) {
        far_system.motions[1].position = render_to_ecliptic(RenderPosition{{shift, 0.0, 1.0}});
        simulation.replace_system(far_system);
        double flux = probe.render(camera_at(FAR_DISTANCE), simulation.snapshot(), settings);
        check(std::abs(flux / far_baseline - 1.0) < 0.002, "Distant overlapping points do not flicker or occlude");
    }
    std::cout << "Unresolved, transitioning and resolved occlusion passed.\n";
}

void reflected_light(RenderProbe& probe) {
    rendering::RenderSettings settings;
    settings.exposure = 100.0f;
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{20.0, 0.0, 0.0}});
    ephemeris::BodyDefinition planet;
    planet.name = "Planet";
    loaded.definition.name_to_idx[planet.name] = 1;
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Reflected-light fixture");
    double distance = SOLAR_RADII_TO_AU * FOCAL / 0.2;
    double near_flux = probe.render(camera_at(distance), simulation.snapshot(), settings);
    double far_flux = probe.render(camera_at(distance * 2.0), simulation.snapshot(), settings);
    std::cout << "Reflected-light flux near/far: " << near_flux << "/" << far_flux << '\n';
    check(std::abs(far_flux / near_flux - 0.25) < 0.002, "Reflected light follows observer inverse-square law");
    double min_transition = 10.0;
    double max_transition = 0.0;
    for (double radius_px = 1.3; radius_px <= 2.71; radius_px += 0.05) {
        double d = SOLAR_RADII_TO_AU * FOCAL / radius_px;
        double flux = probe.render(camera_at(d), simulation.snapshot(), settings);
        double expected = near_flux * distance * distance / (d * d);
        min_transition = std::min(min_transition, flux / expected);
        max_transition = std::max(max_transition, flux / expected);
        if (flux / expected <= 0.74 || flux / expected >= 1.26)
            std::cout << "Reflected-light transition radius/ratio: " << radius_px << "/" << flux / expected << '\n';
        check(flux / expected > 0.74 && flux / expected < 1.26, "Reflected-light mesh/point continuity");
    }
    std::cout << "Reflected-light transition range: " << min_transition << " to " << max_transition << '\n';
}

void catalog_calibration(RenderProbe& probe) {
    auto path = std::filesystem::temp_directory_path() / "stellar-render-test-gaia.bin";
    {
        std::ofstream file(path, std::ios::binary);
        file.write("GAIA1", 5);
        uint32_t version = 1;
        uint32_t count = 1;
        file.write(reinterpret_cast<const char*>(&version), 4);
        file.write(reinterpret_cast<const char*>(&count), 4);
        float record[]{0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 4.67f, 0.82f};
        file.write(reinterpret_cast<const char*>(record), sizeof(record));
    }
    rendering::StarCatalog catalog;
    check(catalog.init(probe.vk, path), "Synthetic solar Gaia catalog");
    std::filesystem::remove(path);
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{2062648.0624709636, 0.0, 0.0}});
    double c = static_cast<double>(0.82f);
    auto rgb = ephemeris::temperature_to_rgb(8842.0 - 4578.875 * c + 1031.17 * c * c - 96.233 * c * c * c);
    loaded.definition.bodies[0].color_rgb = rgb;
    loaded.definition.bodies[0].color_pole_rgb = rgb;
    ephemeris::Simulation simulation;
    check(simulation.replace_system(std::move(loaded)).accepted, "Distant solar fixture");
    Camera camera = camera_at(0.0);
    camera.yaw_actual_deg = 0.0;
    rendering::RenderSettings settings;
    settings.exposure = 1e8f;
    double solar = probe.render(camera, simulation.snapshot(), settings);
    double gaia = probe.render(camera, simulation.snapshot(), settings, &catalog);
    probe.check_diameter(3);
    std::cout << "Gaia solar analogue/local Sun flux ratio: " << gaia / solar << '\n';
    check(std::abs(gaia / solar - 1.0) < 0.02, "Gaia and local Sun photometric calibration");
    settings.exposure = 1e14f;
    probe.render(camera, simulation.snapshot(), settings, &catalog, true);
    probe.check_diameter(3);
    settings.exposure = 1e8f;
    settings.star_point_size = 6.0f;
    double wide = probe.render(camera, simulation.snapshot(), settings, &catalog);
    check(std::abs(wide / gaia - 1.0) < 0.005, "Gaia point size preserves total flux");

    auto get_instance_count = [&](rendering::StarCatalog& cat) -> uint32_t {
        probe.vk.immediate_submit([&](VkCommandBuffer cmd) {
            VkBufferCopy copy{0, 0, sizeof(VkDrawIndirectCommand)};
            vkCmdCopyBuffer(cmd, cat.indirect_buffer(probe.vk.frame_index()).handle, probe.readback.handle, 1, &copy);
            probe.vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        });
        vmaInvalidateAllocation(probe.vk.allocator(), probe.readback.allocation, 0, sizeof(VkDrawIndirectCommand));
        auto* cmd_ptr = static_cast<const VkDrawIndirectCommand*>(probe.readback.mapped);
        return cmd_ptr->instanceCount;
    };
    check(get_instance_count(catalog) == 1, "Visible star has instanceCount == 1");

    Camera cam_behind = camera;
    cam_behind.yaw_actual_deg = 180.0;
    double gaia_behind = probe.render(cam_behind, simulation.snapshot(), settings, &catalog);
    check(gaia_behind == 0.0, "Star behind camera renders zero flux");
    check(get_instance_count(catalog) == 0, "Star behind camera culled with instanceCount == 0");

    Camera cam_offscreen = camera;
    cam_offscreen.yaw_actual_deg = 90.0;
    double gaia_offscreen = probe.render(cam_offscreen, simulation.snapshot(), settings, &catalog);
    check(gaia_offscreen == 0.0, "Off-screen star renders zero flux");
    check(get_instance_count(catalog) == 0, "Off-screen star culled with instanceCount == 0");

    rendering::RenderSettings faint_settings = settings;
    faint_settings.exposure = 1e-15f;
    double gaia_faint = probe.render(camera, simulation.snapshot(), faint_settings, &catalog);
    check(gaia_faint == 0.0, "Sub-threshold faint star renders zero flux");
    check(get_instance_count(catalog) == 0, "Sub-threshold faint star culled with instanceCount == 0");

    catalog.shutdown(probe.vk);
}

void bright_point_shape(RenderProbe& probe, const std::filesystem::path& preview) {
    probe.resize({256, 1080});
    ephemeris::Simulation simulation;
    check(simulation.replace_system(fixture()).accepted, "Bright Sun fixture");
    rendering::RenderSettings settings;
    double focal = 1080.0 / (2.0 * std::tan(30.0 * DEG_TO_RAD));
    double distance = SOLAR_RADII_TO_AU * focal / 0.2;
    double radius = 0.5 * settings.star_point_size;
    double baseline = probe.render(camera_at(distance), simulation.snapshot(), settings);
    for (double offset : {0.0, 0.1, 0.25, 0.5, 0.75}) {
        Camera camera = camera_at(distance, offset);
        settings.exposure = 1.0f;
        double flux = probe.render(camera, simulation.snapshot(), settings);
        check(std::abs(flux / baseline - 1.0) < 0.005, "Circular PSF preserves subpixel flux");
        auto clip = camera.proj_matrix(256.0f / 1080.0f, camera.compute_near_plane_au())
                    * camera.view_matrix_eye_relative() * glm::vec4(glm::vec3(-camera.eye_pos_render().value), 1.0f);
        double center_x = (clip.x / clip.w + 1.0) * 128.0;
        for (uint32_t y = 528; y < 552; ++y) {
            for (uint32_t x = 116; x < 140; ++x) {
                double r = std::hypot(x + 0.5 - center_x, y + 0.5 - 540.0);
                if (r > radius + 1e-4)
                    check(probe.red_at(x, y) == 0.0f, "No HDR energy outside the circular pixel footprint");
            }
        }
        settings.exposure = 1e6f;
        probe.render(camera, simulation.snapshot(), settings, nullptr, true);
        check(probe.red_at(static_cast<uint32_t>(center_x), 540) > 0.95f,
              "Overexposed Sun remains white and finite through tone mapping");
        for (uint32_t y = 528; y < 552; ++y) {
            for (uint32_t x = 116; x < 140; ++x) {
                double dx = x + 0.5 - center_x;
                double dy = y + 0.5 - 540.0;
                if (std::hypot(dx, dy) > radius + 1.5)
                    check(probe.red_at(x, y) < 1e-4f, "Overexposure does not reveal square sprite corners");
            }
        }
        if (!preview.empty() && offset == 0.0) probe.save_preview(preview);
    }
    settings.exposure = 1.0f;
    settings.star_point_size = 6.0f;
    double wide = probe.render(camera_at(distance), simulation.snapshot(), settings);
    std::cout << "Wide/narrow point flux ratio: " << wide / baseline << '\n';
    check(std::abs(wide / baseline - 1.0) < 0.005, "Different point diameters preserve total flux at 1080p");
    catalog_calibration(probe);
    std::cout << "Circular footprint, point diameters and overexposed composite passed.\n";
}

void scaling_point_footprint(RenderProbe& probe) {
    ephemeris::Simulation simulation;
    check(simulation.replace_system(fixture()).accepted, "Scaling point fixture");
    rendering::RenderSettings settings;
    settings.exposure = 1e-6f;
    for (uint32_t height : {256u, 1080u, 2160u}) {
        probe.resize({256, height});
        double focal = height / (2.0 * std::tan(30.0 * DEG_TO_RAD));
        for (double apparent_radius : {0.2, 1.4, 1.5, 1.55, 1.75, 2.0, 2.25, 2.45, 2.55}) {
            double distance = SOLAR_RADII_TO_AU * focal / apparent_radius;
            for (double offset : {0.0, 0.13, 0.5, 0.9}) {
                Camera camera = camera_at(distance);
                camera.cam_pos_rel.x = offset * distance / focal;
                probe.render(camera, simulation.snapshot(), settings);
                auto clip = camera.proj_matrix(256.0f / height, camera.compute_near_plane_au())
                            * camera.view_matrix_eye_relative()
                            * glm::vec4(glm::vec3(-camera.eye_pos_render().value), 1.0f);
                double center_x = (clip.x / clip.w + 1.0) * 128.0;
                int min_x = 256, max_x = -1, min_y = static_cast<int>(height), max_y = -1;
                double bound = std::max(apparent_radius, 0.5 * settings.star_point_size) + 0.02;
                for (uint32_t y = height / 2 - 12; y < height / 2 + 12; ++y) {
                    for (uint32_t x = 116; x < 140; ++x) {
                        if (probe.red_at(x, y) <= 0.0f) continue;
                        check(std::abs(x + 0.5 - center_x) <= bound &&
                              std::abs(y + 0.5 - height * 0.5) <= bound,
                              "Point scales within the mesh footprint until it reaches the 3 px minimum");
                        min_x = std::min(min_x, static_cast<int>(x));
                        max_x = std::max(max_x, static_cast<int>(x));
                        min_y = std::min(min_y, static_cast<int>(y));
                        max_y = std::max(max_y, static_cast<int>(y));
                    }
                }
                check(max_x >= min_x, "Point remains visible through transition");
                if (apparent_radius <= 1.5) {
                    check(max_x - min_x + 1 <= 3 && max_y - min_y + 1 <= 3,
                          "Default point occupies at most 3 by 3 pixels at every framebuffer height");
                    settings.exposure = 1e6f;
                    probe.render(camera, simulation.snapshot(), settings, nullptr, true);
                    probe.check_diameter(3);
                    settings.exposure = 1e-6f;
                }
            }
        }
    }
    catalog_calibration(probe);
    std::cout << "Mesh-sized transition and unresolved 3 px footprint passed at 256p, 1080p and 2160p.\n";
}

}

#include "terrain_gpu_tests.hpp"

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        RenderProbe probe;
        if (argc >= 2 && std::string_view(argv[1]) == "--terrain-relief-benchmark") {
            terrain_ground_benchmark(probe, argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{}, true);
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--terrain-ground-benchmark") {
            terrain_ground_benchmark(probe, argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{});
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--terrain-benchmark") {
            terrain_benchmark(probe, argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{});
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--terrain-precision") {
            terrain_precision(probe);
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--terrain-preview") {
            terrain_surfaces(probe, argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path{});
            return 0;
        }
        terrain_surfaces(probe);
        terrain_precision(probe);
        if (argc == 3 && std::string_view(argv[1]) == "--earth-closeups") {
            earth_closeups(probe, argv[2]);
            return 0;
        }
        earth_closeups(probe, {});
        eclipse_lighting(probe);
        if (argc == 2 && std::string_view(argv[1]) == "--eclipse-only") return 0;
        transition_and_distance(probe);
        occlusion(probe);
        reflected_light(probe);
        catalog_calibration(probe);
        bright_point_shape(probe, argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path{});
        scaling_point_footprint(probe);
        std::cout << "Rendering contracts passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
