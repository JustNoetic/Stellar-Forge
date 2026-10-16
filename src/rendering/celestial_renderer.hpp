#pragma once

#include "core/camera.hpp"
#include "ephemeris/system_manager.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <glm/glm.hpp>

namespace sf::rendering {

inline constexpr uint32_t MAX_CELESTIAL_BODIES = 1024;
inline constexpr uint32_t MAX_SCENE_STARS = 16;

struct alignas(16) BodyInstanceGPU {
    glm::vec4 pos_radius{0.0f};
    glm::vec4 color_is_star{1.0f};
    glm::vec4 pole_oblateness{0.0f, 1.0f, 0.0f, 0.0f};
    glm::vec4 pole_color_lum{1.0f};
    glm::vec4 spin_flags{0.0f};
};

struct alignas(16) SceneCelestialUBO {
    glm::mat4 view_proj{1.0f};
    glm::mat4 view{1.0f};
    glm::vec4 stars_pos_radius[MAX_SCENE_STARS]{};
    glm::vec4 stars_color_lum[MAX_SCENE_STARS]{};
    glm::vec4 screen_params{1600.0f, 900.0f, 1.2f, 1.0f};
    uint32_t num_stars = 0;
    uint32_t num_bodies = 0;
    uint32_t hdr_enabled = 1;
    uint32_t pad0 = 0;
};

class CelestialRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);

    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                const ephemeris::SystemState& sys, VkExtent2D extent);

private:
    void build_sphere_mesh(rhi::VkContext& ctx, uint32_t sectors, uint32_t stacks);

    rhi::Buffer sphere_vbo_{};
    rhi::Buffer sphere_ibo_{};
    uint32_t sphere_index_count_ = 0;

    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> scene_ubos_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> instance_ssbos_{};

    rhi::Pipeline sphere_pipeline_{};
    rhi::Pipeline point_pipeline_{};
};

} // namespace sf::rendering
