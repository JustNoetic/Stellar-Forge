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
#include <cstdlib>

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
        vk.init(window, std::getenv("STELLAR_VULKAN_VALIDATION") != nullptr);
        renderer.init(vk, false);
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
                                  | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        depth = vk.create_image_2d(size.width, size.height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
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
                  bool tone_map = false, glm::vec3 background = glm::vec3(0.0f), bool atmosphere_only = false) {
        vk.immediate_submit([&](VkCommandBuffer cmd) {
            if (!catalog) renderer.prepare(cmd, vk, camera, system, settings, extent);
            if (catalog) catalog->cull(cmd, vk, camera, units::Years{0.0}, settings, extent);
            vk.transition_image(cmd, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vk.transition_image(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            auto color_attachment = rhi::RenderingAttachment::clear_color(color, background.r, background.g, background.b);
            auto depth_attachment = rhi::RenderingAttachment::clear_depth_reverse_z(depth);
            vk.begin_rendering(cmd, extent, {&color_attachment, 1}, &depth_attachment);
            vk.set_viewport_and_scissor(cmd, extent);
            if (catalog) catalog->render(cmd, vk, camera, units::Years{0.0}, settings, extent);
            else if (!atmosphere_only) renderer.render(cmd, vk, camera, system, settings, extent);
            vk.end_rendering(cmd);
            if (!catalog) renderer.composite_atmospheres(cmd, vk, color, depth);
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
    camera = camera_at(radius*3.0);
    camera.origin_render = simulation.snapshot().body(1).render.position.value;
    camera.tracking_idx = 1;
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.renderer.terrain_stats().patches > 0 &&
          probe.renderer.terrain_stats().smooth_patches == probe.renderer.terrain_stats().patches,
          "Disabling synthetic relief retains smooth quadtree geometry");
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
    auto center_light = [&](glm::dvec3 caster_position, bool blocker, bool second_star, bool many_bodies = false,
                            double caster_radius = 0.5, double oblateness = 0.0, glm::dvec3 pole = {0, 1, 0}) {
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
        if (blocker) {
            add("Caster", caster_position, caster_radius, false);
            loaded.definition.bodies.back().oblateness = oblateness;
            loaded.definition.bodies.back().pole_ecl = render_to_ecliptic(RenderPosition{pole}).value;
        }
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
    float annular = center_light({0, 0, 10}, true, false, false, 0.01);
    std::cout << "Small-moon annular eclipse visibility: " << annular / clear << '\n';
    check(annular / clear > 0.98f && annular / clear < 0.995f,
          "A moon covering about one percent of the stellar disc cannot create a deep eclipse");
    float annular_offset = center_light({0.025, 0, 10}, true, false, false, 0.01);
    check(std::abs(annular_offset / annular - 1.0f) < 0.001f,
          "A small moon wholly inside the stellar disc blocks the same area away from center");
    float oblate_equator = center_light({0, 0.35, 10}, true, false, false, 0.5, 0.6, {0, 1, 0});
    float oblate_pole = center_light({0, 0.35, 10}, true, false, false, 0.5, 0.6, {0, 0, 1});
    float oblate_tilted = center_light({0, 0.35, 10}, true, false, false, 0.5, 0.6, glm::normalize(glm::dvec3(0, 1, 1)));
    float oblate_rotated = center_light({0.35, 0, 10}, true, false, false, 0.5, 0.6, {1, 0, 0});
    float oblate_major = center_light({0.35, 0, 10}, true, false, false, 0.5, 0.6, {0, 1, 0});
    check(oblate_equator > clear * 0.99f, "Flattened parent's polar limb does not cast a false spherical eclipse on its moon");
    check(oblate_rotated > clear * 0.99f && oblate_major < clear * 0.001f,
          "Parent pole orientation rotates the ellipse and preserves its equatorial shadow extent");
    check(oblate_pole < clear * 0.001f, "Pole-on oblate parent casts a circular eclipse");
    check(oblate_tilted > clear * 0.05f && oblate_tilted < clear * 0.95f,
          "Tilted parent's projected ellipse produces partial eclipse illumination");
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
    for (double radius : {0.01, 0.05, 0.1}) {
        simulation.replace_system(fixture(radius, distance));
        double flux = probe.render(camera_at(distance), simulation.snapshot(), settings);
        double expected = 1.0 - (radius / 0.2) * (radius / 0.2);
        check(std::abs(flux / baseline - expected) < 0.003,
              "Unresolved stellar transits dim by the occulted area without a caster size cutoff");
    }
    for (double radius : {0.5, 1.4, 1.5, 2.0, 3.0}) {
        simulation.replace_system(fixture(radius, distance));
        check(probe.render(camera_at(distance), simulation.snapshot(), settings) < baseline * 0.001,
              "A caster larger than the stellar disc fully occludes an unresolved star at every mesh size");
    }
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
    std::cout << "Finite stellar transits and unresolved, transitioning and resolved occlusion passed.\n";
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
    double distance = SOLAR_RADII_TO_AU * FOCAL / 24.0;
    double near_flux = probe.render(camera_at(distance), simulation.snapshot(), settings);
    double far_flux = probe.render(camera_at(distance * 2.0), simulation.snapshot(), settings);
    std::cout << "Reflected-light flux near/far: " << near_flux << "/" << far_flux << '\n';
    check(std::abs(far_flux / near_flux - 0.25) < 0.015, "Resolved quadtree reflectance follows observer inverse-square law");
    double min_transition = 10.0;
    double max_transition = 0.0;
    for (double radius_px : {12.0, 18.0, 24.0, 30.0}) {
        double d = SOLAR_RADII_TO_AU * FOCAL / radius_px;
        double flux = probe.render(camera_at(d), simulation.snapshot(), settings);
        double expected = near_flux * distance * distance / (d * d);
        min_transition = std::min(min_transition, flux / expected);
        max_transition = std::max(max_transition, flux / expected);
        check(flux / expected > 0.9 && flux / expected < 1.1, "Quadtree reflectance remains consistent across resolved LODs");
    }
    std::cout << "Reflected-light quadtree range: " << min_transition << " to " << max_transition << '\n';
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

void atmosphere_contracts(RenderProbe& probe, const std::filesystem::path& output = {}) {
    if (!output.empty()) std::filesystem::create_directories(output);
    probe.resize(EXTENT);
    constexpr double radius = 6378.137*KM_TO_AU;
    auto make = [&](glm::dvec3 origin = glm::dvec3(0.0)) {
        auto loaded = fixture();
        loaded.motions[0].position = render_to_ecliptic(RenderPosition{origin+glm::dvec3(1.0, 0.0, 1.0)});
        ephemeris::BodyDefinition earth;
        earth.name = "Atmosphere probe";
        earth.radius_rsun = radius/SOLAR_RADII_TO_AU;
        earth.color_rgb = glm::vec3(0.0f);
        earth.atmosphere.enabled = true;
        loaded.definition.bodies.push_back(earth);
        ephemeris::BodyMotion motion;
        motion.position = render_to_ecliptic(RenderPosition{origin});
        loaded.motions.push_back(motion);
        return loaded;
    };
    ephemeris::Simulation sim;
    auto loaded = make();
    check(sim.replace_system(loaded).accepted, "Atmosphere GPU fixture");
    Camera cam = camera_at(radius*3.0);
    rendering::RenderSettings settings;
    settings.show_atmospheres = false;
    check(probe.render(cam, sim.snapshot(), settings) == 0.0, "Disabled atmosphere preserves black background");
    settings.show_atmospheres = true;
    double day = probe.render(cam, sim.snapshot(), settings);
    check(day > 1.0, "Sky View LUT produces visible atmospheric radiance from orbit");
    check(probe.renderer.atmosphere_stats().optical_lut_updates == 1, "First visible atmosphere builds optical LUTs");
    check(probe.red_at(207, 128) > 0.0f, "Atmosphere is visible beyond the opaque limb");
    check(probe.red_at(240, 128) == 0.0f, "Atmosphere does not leak beyond its outer boundary");
    if (!output.empty()) {
        probe.render(cam, sim.snapshot(), settings, nullptr, true);
        probe.save_preview(output / "orbit.ppm", true);
    }
    double stable = probe.render(cam, sim.snapshot(), settings);
    check(stable == day && probe.renderer.atmosphere_stats().sky_lut_updates == 0, "Paused frame reuses identical LUTs");
    settings.exposure = 2.0f;
    check(std::abs(probe.render(cam, sim.snapshot(), settings)/day-2.0) < 0.003, "Atmosphere exposure applies once");
    check(probe.renderer.atmosphere_stats().sky_lut_updates == 0, "Exposure does not invalidate physical LUTs");
    settings.exposure = 1.0f;
    cam.yaw_actual_deg += 4.0;
    probe.render(cam, sim.snapshot(), settings);
    check(probe.renderer.atmosphere_stats().sky_lut_updates == 0 && probe.renderer.atmosphere_stats().optical_lut_updates == 0,
          "Camera rotation reuses sky and optical LUTs");
    cam.yaw_actual_deg = 90.0;
    probe.render(cam, sim.snapshot(), settings);
    check(probe.renderer.atmosphere_stats().visible == 0, "Offscreen orbital atmospheres skip LUTs and composition");
    cam = camera_at(radius*3.0);
    auto binary = loaded;
    auto companion = binary.definition.bodies[0];
    companion.name = "Companion";
    binary.definition.bodies.push_back(companion);
    binary.motions.push_back(binary.motions[0]);
    check(sim.replace_system(binary).accepted, "Binary atmosphere fixture");
    double two = probe.render(cam, sim.snapshot(), settings);
    check(std::abs(two/day-2.0) < 0.006, "Multiple stars add independent atmospheric radiance");
    auto vacuum = loaded;
    auto& a = vacuum.definition.bodies[1].atmosphere;
    a.rayleigh_scattering = a.mie_scattering = a.mie_extinction = glm::vec3(0.0f);
    check(sim.replace_system(vacuum).accepted, "Vacuum fixture");
    check(probe.render(cam, sim.snapshot(), settings) == 0.0, "Zero-density atmosphere is finite and transparent");
    check(sim.replace_system(loaded).accepted, "Restore atmosphere after coefficient edit");
    cam = camera_at(radius+KM_TO_AU);
    cam.yaw_actual_deg = 90.0;
    double ground_day = probe.render(cam, sim.snapshot(), settings);
    check(ground_day > 1.0, "Sky View LUT works inside the atmosphere");
    if (!output.empty()) {
        probe.render(cam, sim.snapshot(), settings, nullptr, true);
        probe.save_preview(output / "ground.ppm", true);
        probe.render(cam, sim.snapshot(), settings);
    }
    auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    size_t center = 4*(128*256+128);
    check(glm::unpackHalf2x16(pixels[center+2]).x > glm::unpackHalf2x16(pixels[center]).x, "Daylight sky has wavelength-dependent Rayleigh scattering");
    for (double altitude : {0.001, 50.0, 99.99, 100.01, 101.0, 1000.0}) {
        cam = camera_at(radius+altitude*KM_TO_AU);
        cam.yaw_actual_deg = 0.0;
        check(probe.render(cam, sim.snapshot(), settings) > 0.0, "Atmosphere remains finite and visible across surface and top-boundary views");
    }
    cam = camera_at(radius+KM_TO_AU);
    cam.yaw_actual_deg = 90.0;
    auto night = loaded;
    night.motions[0].position = render_to_ecliptic(RenderPosition{{0.0, 0.0, -1.0}});
    check(sim.replace_system(night).accepted, "Night atmosphere fixture");
    check(probe.render(cam, sim.snapshot(), settings) < ground_day*0.05, "Planet blocks sunlight on the night side");
    auto absorption = loaded;
    absorption.definition.bodies[0].lum_eq = absorption.definition.bodies[0].lum_pole = 0.0;
    check(sim.replace_system(absorption).accepted, "Atmospheric transmission fixture");
    probe.render(cam, sim.snapshot(), settings, nullptr, false, glm::vec3(1.0f));
    pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    const auto& optical = absorption.definition.bodies[1].atmosphere;
    for (int c = 0; c < 3; ++c) {
        double tau = optical.rayleigh_scattering[c]*optical.rayleigh_scale_height_km*std::exp(-1.0/optical.rayleigh_scale_height_km)
                   + optical.mie_extinction[c]*optical.mie_scale_height_km*std::exp(-1.0/optical.mie_scale_height_km);
        check(std::abs(glm::unpackHalf2x16(pixels[center+c]).x-std::exp(-tau)) < 0.003,
              "Zenith RGB transmittance agrees with analytic exponential-column optical depth");
    }
    auto occluded = loaded;
    ephemeris::BodyDefinition caster;
    caster.name = "Foreground occluder";
    caster.radius_rsun = radius*0.5/SOLAR_RADII_TO_AU;
    caster.color_rgb = glm::vec3(0.0f);
    occluded.definition.bodies.push_back(caster);
    ephemeris::BodyMotion motion;
    motion.position = render_to_ecliptic(RenderPosition{{0.0, 0.0, radius*2.0}});
    occluded.motions.push_back(motion);
    check(sim.replace_system(occluded).accepted, "Occluded atmosphere fixture");
    cam = camera_at(radius*3.0);
    check(probe.render(cam, sim.snapshot(), settings) == 0.0, "Foreground depth fully occludes atmospheric shell");
    glm::dvec3 origin(1000.0, -2000.0, 3000.0);
    check(sim.replace_system(make(origin)).accepted, "Translated atmosphere fixture");
    cam.origin_render = origin;
    double translated = probe.render(cam, sim.snapshot(), settings);
    check(std::abs(translated/day-1.0) < 0.005, "Atmosphere preserves precision after large world translation");
    auto pair = loaded;
    pair.motions[1].position = render_to_ecliptic(RenderPosition{{-radius*1.5, 0.0, 0.0}});
    auto second = pair.definition.bodies[1];
    second.name = "Second atmosphere";
    pair.definition.bodies.push_back(second);
    motion.position = render_to_ecliptic(RenderPosition{{radius*1.5, 0.0, 0.0}});
    pair.motions.push_back(motion);
    check(sim.replace_system(pair).accepted, "Multiple atmosphere fixture");
    cam = camera_at(radius*6.0);
    check(probe.render(cam, sim.snapshot(), settings) > 0.0 && probe.renderer.atmosphere_stats().visible == 2,
          "Independent atmospheric bodies receive separate LUTs and composite passes");
    std::cout << "PASS: atmospheric LUTs, orbital limb, ground sky, planet shadow, RGB optics, depth, multiple stars, caching and precision.\n";
}

void ring_rendering_and_shadows(RenderProbe& probe) {
    probe.resize(EXTENT);
    auto loaded = fixture();
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{5.0, 2.0, 2.0}});

    ephemeris::BodyDefinition planet;
    planet.name = "RingedPlanet";
    planet.type = "Gas Giant";
    planet.radius_rsun = 0.1;
    planet.color_rgb = {0.85f, 0.80f, 0.70f};
    planet.pole_ecl = glm::normalize(glm::dvec3(0.0, -0.5, 0.8660254037844386));

    ephemeris::RingLayerDefinition ring{};
    ring.inner = 1.4;
    ring.outer = 2.4;
    ring.color_rgb = {0.9f, 0.85f, 0.7f};
    ring.opacity = 1.2f;
    ring.scatter = 0.5f;
    ring.asymmetry = 0.75f;
    ring.backscatter = -0.45f;
    ring.unlit_factor = 1.0f;
    ring.is_textured = false;
    ring.gradient = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    planet.rings.push_back(ring);

    loaded.definition.name_to_idx[planet.name] = 1;
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();

    ephemeris::Simulation sim;
    check(sim.replace_system(loaded).accepted, "Ring rendering fixture");

    double r_au = planet.radius_au();
    Camera cam = camera_at(r_au * 6.0);
    rendering::RenderSettings settings{};
    settings.exposure = 50.0f;
    settings.show_rings = false;
    settings.ring_shadows_enabled = false;
    double flux_no_rings = probe.render(cam, sim.snapshot(), settings);
    float ansa_no_rings = probe.red_at(184, 128);
    check(ansa_no_rings == 0.0f, "Off-limb ansa pixel is zero when show_rings is false");

    settings.show_rings = true;
    double flux_with_rings = probe.render(cam, sim.snapshot(), settings);
    float ansa_lit = probe.red_at(184, 128);
    float ansa_shadowed = probe.red_at(72, 128);
    check(flux_with_rings > flux_no_rings * 1.15, "Enabling show_rings increases total scene radiance");
    check(ansa_lit > 0.01f, "Sunlit ring ansa pixel has positive HDR radiance");

    // Place Sun directly to +X (with slight +Y elevation) so -X ring ansa lies in planet's shadow cylinder
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{5.0, 0.8, 0.0}});
    loaded.definition.bodies[1].pole_ecl = glm::normalize(glm::dvec3(0.0, -0.35, 0.9367496997597597));
    check(sim.replace_system(loaded).accepted, "Ring eclipse fixture");
    probe.render(cam, sim.snapshot(), settings);
    ansa_lit = probe.red_at(184, 128);
    ansa_shadowed = probe.red_at(72, 128);
    check(ansa_lit > 0.01f, "Sunlit +X ring ansa is bright");
    check(ansa_shadowed < ansa_lit * 0.05f, "Planet casts umbral shadow across -X ring ansa");

    // Place Sun in front/above so ring casts shadow across sunlit planet hemisphere
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{{0.0, 1.5, 5.0}});
    loaded.definition.bodies[1].pole_ecl = glm::normalize(glm::dvec3(0.0, 0.15, 0.9886859966642594));
    check(sim.replace_system(loaded).accepted, "Ring surface shadow fixture");

    settings.show_rings = true;
    settings.ring_shadows_enabled = false;
    double planet_unshadowed = probe.render(cam, sim.snapshot(), settings);

    settings.ring_shadows_enabled = true;
    double planet_shadowed = probe.render(cam, sim.snapshot(), settings);
    check(planet_shadowed < planet_unshadowed * 0.98, "Ring shadow LUT attenuates sunlight on planet surface");
    auto* shadow_pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> filtered_shadow(shadow_pixels, shadow_pixels + size_t(EXTENT.width) * EXTENT.height * 4);
    settings.ring_shadow_reference_all_hosts = true;
    probe.render(cam, sim.snapshot(), settings);
    check(std::equal(filtered_shadow.begin(), filtered_shadow.end(), shadow_pixels), "Own-ring planet shadows match the all-host reference");
    settings.ring_shadow_reference_all_hosts = false;

    cam = camera_at(r_au * 1.1);
    settings.ring_shadows_enabled = false;
    settings.show_rings = false;
    probe.render(cam, sim.snapshot(), settings);
    auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> opaque(pixels, pixels + size_t(EXTENT.width) * EXTENT.height * 4);
    settings.show_rings = true;
    probe.render(cam, sim.snapshot(), settings);
    check(std::equal(opaque.begin(), opaque.end(), pixels), "Opaque planet fully occludes rings at close range");

    cam = camera_at(r_au * 6.0);
    auto& layers = loaded.definition.bodies[1].rings;
    layers.front().opacity = 0.1f;
    check(sim.replace_system(loaded).accepted, "Single-layer reference fixture");
    probe.render(cam, sim.snapshot(), settings);
    std::vector<uint16_t> single_layer(pixels, pixels + size_t(EXTENT.width) * EXTENT.height * 4);
    auto active_layer = layers.front();
    layers.assign(8, active_layer);
    for (size_t i = 0; i < 7; ++i) layers[i].opacity = 0.0f;
    check(sim.replace_system(loaded).accepted, "Sparse eighth-layer fixture");
    probe.render(cam, sim.snapshot(), settings);
    check(std::equal(single_layer.begin(), single_layer.end(), pixels), "Only active eighth layer matches a single layer exactly");

    for (auto& layer : layers) layer.opacity = 0.1f;
    check(sim.replace_system(loaded).accepted, "Eight overlapping ring layers fixture");
    double eight_flux = probe.render(cam, sim.snapshot(), settings);
    float eight_ansa = probe.red_at(184, 128);
    layers.assign(1, active_layer);
    layers.front().opacity = 1.0f - std::pow(0.9f, 8.0f);
    check(sim.replace_system(loaded).accepted, "Combined optical-depth reference fixture");
    probe.render(cam, sim.snapshot(), settings);
    float combined_ansa = probe.red_at(184, 128);
    check(eight_ansa > 0.001f && std::abs(eight_ansa - combined_ansa) < combined_ansa * 0.01f,
          "Eight active ring layers preserve combined interior radiance");

    layers.assign(8, active_layer);
    auto second_star = loaded.definition.bodies[0];
    second_star.name = "Second Sun";
    loaded.definition.name_to_idx[second_star.name] = static_cast<int>(loaded.definition.bodies.size());
    loaded.definition.bodies.push_back(second_star);
    loaded.motions.push_back(loaded.motions[0]);
    check(sim.replace_system(loaded).accepted, "Multiple-star ring fixture");
    double two_star_flux = probe.render(cam, sim.snapshot(), settings);
    check(std::abs(two_star_flux / eight_flux - 2.0) < 0.01, "All active ring layers receive both stars' illumination");
    std::cout << "Ring rendering, planet-on-ring eclipse, and ring-on-planet shadow contracts passed.\n";
}

void ring_eclipse_candidate_filtering(RenderProbe& probe) {
    probe.resize(EXTENT);
    auto loaded = fixture();
    glm::dvec3 star_pos{0.0, 2.0, 5.0};
    loaded.motions[0].position = render_to_ecliptic(RenderPosition{star_pos});
    ephemeris::BodyDefinition planet;
    planet.name = "Ring receiver";
    planet.radius_rsun = 0.1;
    planet.pole_ecl = glm::normalize(glm::dvec3(0.0, -0.5, 0.8660254037844386));
    ephemeris::RingLayerDefinition ring;
    ring.inner = 1.4;
    ring.outer = 2.4;
    ring.opacity = 0.8f;
    planet.rings.push_back(ring);
    loaded.definition.name_to_idx[planet.name] = 1;
    loaded.definition.bodies.push_back(planet);
    loaded.motions.emplace_back();
    double radius = planet.radius_au();
    auto camera = camera_at(radius * 6.0);
    rendering::RenderSettings settings;
    settings.exposure = 50.0f;
    settings.ring_shadows_enabled = false;
    ephemeris::Simulation simulation;
    check(simulation.replace_system(loaded).accepted, "Ring candidate baseline fixture");
    probe.render(camera, simulation.snapshot(), settings);
    auto* pixels = static_cast<const uint16_t*>(probe.readback.mapped);
    std::vector<uint16_t> baseline(pixels, pixels + size_t(EXTENT.width) * EXTENT.height * 4);
    float lit = probe.red_at(184, 128);
    check(lit > 0.01f, "Ring candidate probe is sunlit");
    check(probe.renderer.ring_eclipse_candidate_counts().size() == 1 &&
          probe.renderer.ring_eclipse_candidate_counts()[0] == 1, "Host planet retained as a ring blocker");

    auto add_caster = [&](glm::dvec3 position, double caster_radius) {
        ephemeris::BodyDefinition caster;
        caster.name = "Blocker " + std::to_string(loaded.definition.bodies.size());
        caster.radius_rsun = caster_radius / SOLAR_RADII_TO_AU;
        loaded.definition.name_to_idx[caster.name] = static_cast<int>(loaded.definition.bodies.size());
        loaded.definition.bodies.push_back(caster);
        ephemeris::BodyMotion motion;
        motion.position = render_to_ecliptic(RenderPosition{position});
        loaded.motions.push_back(motion);
    };
    for (int i = 0; i < 24; ++i) add_caster({10.0 + i, 10.0, -10.0}, radius * 0.05);
    check(simulation.replace_system(loaded).accepted, "Irrelevant ring blocker fixture");
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.renderer.ring_eclipse_candidate_counts()[0] == 1, "All distant ring blockers rejected on CPU");
    check(std::equal(baseline.begin(), baseline.end(), pixels), "Rejected ring blockers leave HDR output unchanged");

    glm::dvec3 pole = glm::normalize(simulation.snapshot().body(1).render.pole);
    auto inverse_view_proj = glm::inverse(glm::dmat4(camera.proj_matrix(1.0f, camera.compute_near_plane_au()) *
                                                    camera.view_matrix_eye_relative()));
    auto near_point = inverse_view_proj * glm::dvec4(2.0 * 184.5 / 256.0 - 1.0, 2.0 * 128.5 / 256.0 - 1.0, 1.0, 1.0);
    auto ray = glm::normalize(glm::dvec3(near_point) / near_point.w);
    auto hit = camera.cam_pos_rel - ray * glm::dot(camera.cam_pos_rel, pole) / glm::dot(ray, pole);
    auto blocker_pos = hit + glm::normalize(star_pos - hit) * (30.0 * radius);
    add_caster(blocker_pos, radius * 0.2);
    check(simulation.replace_system(loaded).accepted, "Unrelated ring blocker fixture");
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.renderer.ring_eclipse_candidate_counts()[0] == 2, "Unrelated body beyond the old sixteen-caster cutoff retained");
    float shadowed = probe.red_at(184, 128);
    check(shadowed < lit * 0.05f, "Unrelated body casts the expected ring umbra");
    loaded.definition.bodies.back().parent_idx = 1;
    loaded.definition.bodies.back().parent_id = planet.name;
    loaded.definition.bodies.back().type = "Moon";
    check(simulation.replace_system(loaded).accepted, "Moon-to-ring eclipse fixture");
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.red_at(184, 128) == shadowed && probe.renderer.ring_eclipse_candidate_counts()[0] == 2,
          "Moon casts the same ring eclipse regardless of reference-parent classification");

    for (int i = 0; i < 17; ++i) add_caster(blocker_pos, radius * 0.01);
    check(simulation.replace_system(loaded).accepted, "Many aligned ring blockers fixture");
    probe.render(camera, simulation.snapshot(), settings);
    check(probe.renderer.ring_eclipse_candidate_counts()[0] == 19, "Ring candidate lists are not capped at sixteen");

    auto second_star = loaded.definition.bodies[0];
    second_star.name = "Second Sun";
    loaded.definition.name_to_idx[second_star.name] = static_cast<int>(loaded.definition.bodies.size());
    loaded.definition.bodies.push_back(second_star);
    ephemeris::BodyMotion second_motion;
    second_motion.position = render_to_ecliptic(RenderPosition{{5.0, 2.0, 0.0}});
    loaded.motions.push_back(second_motion);
    check(simulation.replace_system(loaded).accepted, "Per-star ring blocker fixture");
    probe.render(camera, simulation.snapshot(), settings);
    auto counts = probe.renderer.ring_eclipse_candidate_counts();
    check(counts.size() == 2 && counts[0] == 19 && counts[1] == 1, "Each star uses its own ring blocker list");
    check(probe.red_at(184, 128) > shadowed + 0.01f, "Unblocked second star illuminates the eclipsed ring");
    std::cout << "Ring eclipse candidate filtering passed: distant rejection, unrelated blockers, uncapped lists and per-star selection.\n";
}

}

#include "terrain_gpu_tests.hpp"
#include "ring_gpu_tests.hpp"
#include "eclipse_gpu_tests.hpp"
#include "atmosphere_gpu_tests.hpp"

int main(int argc, char** argv) {
    try {
        std::cout << std::unitbuf;
        RenderProbe probe;
        if (argc == 3 && std::string_view(argv[1]) == "--atmosphere-surface-preview") {
            atmosphere_surface_preview(probe, argv[2]);
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--atmosphere-benchmark") {
            atmosphere_benchmark(probe, argv[2]);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--eclipse-optics") {
            eclipse_optics_contracts(probe);
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--eclipse-benchmark") {
            eclipse_benchmark(probe, argv[2]);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--eclipse-lighting") {
            eclipse_optics_contracts(probe);
            eclipse_lighting(probe);
            occlusion(probe);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--surface-ring-shadows") {
            surface_ring_shadow_contracts(probe);
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--surface-ring-benchmark") {
            surface_ring_benchmark(probe, argv[2]);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--smooth-surfaces") {
            smooth_surface_contracts(probe);
            reflected_light(probe);
            return 0;
        }
        if (argc >= 3 && std::string_view(argv[1]) == "--ring-benchmark") {
            VkExtent2D extent{1920, 1080};
            if (argc >= 6) extent = {static_cast<uint32_t>(std::stoul(argv[4])), static_cast<uint32_t>(std::stoul(argv[5]))};
            ring_benchmark(probe, argv[2], argc >= 4 ? std::string_view(argv[3]) : std::string_view{}, extent,
                           argc >= 7 ? static_cast<uint32_t>(std::stoul(argv[6])) : 1u,
                           argc >= 8 ? std::string_view(argv[7]) : std::string_view{"optimized"});
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--ring-filtering-only") {
            ring_filtering_contracts(probe);
            return 0;
        }
        if (argc == 3 && std::string_view(argv[1]) == "--atmosphere-preview") {
            atmosphere_contracts(probe, argv[2]);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--atmospheres-only") {
            atmosphere_contracts(probe);
            atmosphere_floor_contracts(probe);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--rings-only") {
            surface_ring_shadow_contracts(probe);
            ring_rendering_and_shadows(probe);
            ring_eclipse_candidate_filtering(probe);
            ring_system_cache_contracts(probe);
            ring_filtering_contracts(probe);
            ring_specialization_contracts(probe);
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--bake-benchmark") {
            geometry_bake_benchmark(probe);
            return 0;
        }
        if (argc >= 4 && std::string_view(argv[1]) == "--terrain-source-benchmark") {
            terrain_source_benchmark(probe, argv[2], argv[3], argc > 4 && std::string_view(argv[4]) == "--paced");
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--surface-2km-benchmark") {
            surface_2km_benchmark(probe, argc >= 3 ? argv[2] : "scratch/surface_2km_bench");
            return 0;
        }
        if (argc >= 3 && std::string_view(argv[1]) == "--surface-contracts") {
            surface_contracts(probe, argv[2]);
            return 0;
        }
        if (argc >= 4 && std::string_view(argv[1]) == "--surface-preview") {
            surface_preview(probe, argv[2], argv[3]);
            return 0;
        }
        if (argc >= 3 && std::string_view(argv[1]) == "--terrain-imported") {
            streaming_uploads(probe);
            imported_terrain(probe, argv[2], argc >= 4 ? std::filesystem::path(argv[3]) : std::filesystem::path{});
            return 0;
        }
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
        atmosphere_contracts(probe);
        terrain_surfaces(probe);
        smooth_surface_contracts(probe);
        terrain_precision(probe);
        if (argc == 3 && std::string_view(argv[1]) == "--earth-closeups") {
            earth_closeups(probe, argv[2]);
            return 0;
        }
        earth_closeups(probe, {});
        eclipse_optics_contracts(probe);
        eclipse_lighting(probe);
        surface_ring_shadow_contracts(probe);
        ring_rendering_and_shadows(probe);
        ring_eclipse_candidate_filtering(probe);
        ring_system_cache_contracts(probe);
        ring_filtering_contracts(probe);
        ring_specialization_contracts(probe);
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
