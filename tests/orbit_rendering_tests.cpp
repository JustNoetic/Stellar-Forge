#include "ephemeris/simulation.hpp"
#include "rendering/orbit_renderer.hpp"
#include "rendering/transparency_renderer.hpp"
#include "rhi/shader_cache.hpp"
#include <GLFW/glfw3.h>
#include <glm/gtc/packing.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace sf;

namespace {

void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

ephemeris::LoadedSystem fixture(bool distant = false) {
    ephemeris::LoadedSystem loaded;
    ephemeris::BodyDefinition primary;
    primary.name = "Primary";
    primary.mass_msun = 1.0;
    loaded.definition.bodies.push_back(primary);
    loaded.motions.emplace_back();
    ephemeris::BodyDefinition companion;
    companion.name = "Companion";
    companion.parent_id = "Primary";
    companion.parent_idx = 0;
    companion.mass_msun = 0.01;
    loaded.definition.bodies.push_back(companion);
    loaded.motions.push_back({EclipticPosition{{1.0, 0.0, 0.0}},
                             EclipticVelocity{{0.0, 0.0, std::sqrt(G_CONST * 1.01 * 1.2)}}});
    if (distant) {
        primary.name = "Distant";
        primary.mass_msun = 1.01;
        loaded.definition.bodies.push_back(primary);
        loaded.motions.push_back({EclipticPosition{{2e8, 0.0, 0.0}}, EclipticVelocity{}});
    }
    return loaded;
}

struct Probe {
    static constexpr VkExtent2D extent{256, 256};
    GLFWwindow* window = nullptr;
    rhi::VkContext vk;
    rendering::OrbitRenderer renderer;
    rendering::TransparencyRenderer transparency;
    rhi::Image color, depth;
    rhi::Buffer readback;

    Probe() {
        check(glfwInit() != 0, "GLFW startup");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = glfwCreateWindow(256, 256, "Orbit rendering contracts", nullptr, nullptr);
        check(window != nullptr, "Hidden GPU test window");
        rhi::ShaderCache::init();
        vk.init(window, false);
        renderer.init(vk);
        transparency.init(vk);
        color = vk.create_image_2d(extent.width, extent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        depth = vk.create_image_2d(extent.width, extent.height, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        readback = vk.create_host_buffer(extent.width * extent.height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    ~Probe() {
        vk.wait_idle();
        renderer.shutdown(vk);
        transparency.shutdown(vk);
        vk.destroy_buffer(readback);
        vk.destroy_image(depth);
        vk.destroy_image(color);
        vk.shutdown();
        rhi::ShaderCache::shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    std::vector<float> render(const ephemeris::SystemSnapshot& snapshot, const Camera& camera,
                              const rendering::RenderSettings& settings, float scene_depth = 0.0f) {
        vk.immediate_submit([&](VkCommandBuffer cmd) {
            transparency.prepare(cmd, vk, extent, !settings.transparency_reference_legacy);
            vk.transition_image(cmd, color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vk.transition_image(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            auto color_attachment = rhi::RenderingAttachment::clear_color(color);
            auto depth_attachment = rhi::RenderingAttachment::clear_depth_reverse_z(depth, scene_depth);
            vk.begin_rendering(cmd, extent, {&color_attachment, 1}, &depth_attachment);
            vk.set_viewport_and_scissor(cmd, extent);
            auto writes = transparency.writes();
            renderer.render(cmd, vk, camera, snapshot, settings, extent,
                transparency.active() ? std::span{writes} : std::span<const rhi::PushDescriptorWrite>{});
            vk.end_rendering(cmd);
            transparency.resolve(cmd, vk, color);
            vk.transition_image(cmd, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            vkCmdCopyImageToBuffer(cmd, color.handle, color.layout, readback.handle, 1, &copy);
            vk.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        });
        vmaInvalidateAllocation(vk.allocator(), readback.allocation, 0, VK_WHOLE_SIZE);
        auto* pixels = static_cast<const uint16_t*>(readback.mapped);
        std::vector<float> values(extent.width * extent.height);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = glm::unpackHalf2x16(pixels[4*i]).x;
            check(std::isfinite(values[i]), "Finite orbit HDR output");
        }
        return values;
    }
};

size_t coverage(const std::vector<float>& pixels) {
    return std::count_if(pixels.begin(), pixels.end(), [](float v) { return v > 0.0f; });
}

}

int main() {
    try {
        Probe probe;
        ephemeris::Simulation simulation;
        check(simulation.replace_system(fixture()).accepted, "Load orbit fixture");
        Camera camera;
        camera.fov_deg = 60.0;
        camera.cam_pos_rel = {0.0, 0.0, 4.0};
        camera.cam_look_aim = false;
        camera.yaw_actual_deg = -90.0;
        camera.pitch_actual_deg = 0.0;
        camera.origin_render = simulation.snapshot().body(0).render.position.value;
        rendering::RenderSettings settings;
        auto visible = probe.render(simulation.snapshot(), camera, settings);
        check(coverage(visible) > 250, "Osculating curve and reflex are visible");
        auto statistics = probe.transparency.stats(probe.vk);
        check(!probe.vk.fragment_stores_enabled() || (statistics.fragments >= coverage(visible) && statistics.overflow == 0),
              "Real orbit shaders capture visible fragments without overflowing the shared resolver");
        settings.transparency_reference_legacy = true;
        auto legacy = probe.render(simulation.snapshot(), camera, settings);
        check(coverage(visible) == coverage(legacy), "Orbit coverage is preserved by fragment capture and resolve");
        settings.transparency_reference_legacy = false;
        check(coverage(probe.render(simulation.snapshot(), camera, settings, 1.0f)) == 0, "Opaque reverse-Z scene occludes orbit lines");
        settings.show_orbits = false;
        check(coverage(probe.render(simulation.snapshot(), camera, settings)) == 0, "Visibility toggle suppresses orbit lines");
        settings.show_orbits = true;
        auto far_camera = camera;
        far_camera.cam_pos_rel = {0.0, 0.0, 1e8};
        check(coverage(probe.render(simulation.snapshot(), far_camera, settings)) == 0, "Angular culling hides tiny orbits");
        check(coverage(probe.render(simulation.snapshot(), camera, settings)) > 250, "Paused camera movement restores culled orbits");
        check(simulation.replace_system(fixture(true)).accepted, "Load distant barycentric fixture");
        camera.origin_render = simulation.snapshot().body(0).render.position.value;
        auto distant = probe.render(simulation.snapshot(), camera, settings);
        check(coverage(distant) > 250, "Orbit remains visible at 1e8 AU world offset");
        size_t differences = 0;
        for (size_t i = 0; i < visible.size(); ++i) differences += (visible[i] > 0.0f) != (distant[i] > 0.0f);
        check(differences <= 4, "Camera subtraction retains orbital geometry before float conversion");
        std::cout << "PASS: orbit GPU visibility, reverse-Z occlusion, paused camera culling and 1e8 AU precision.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
