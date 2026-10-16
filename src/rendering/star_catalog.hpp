#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <filesystem>
#include <glm/glm.hpp>

namespace sf::rendering {

struct alignas(32) StarGpuRecord {
    glm::dvec4 pos0{0.0};
    glm::vec4 pm_flux{0.0f};
    glm::vec4 color{1.0f};
    glm::vec4 reserved{0.0f};
};

class StarCatalog {
public:
    bool init(rhi::VkContext& ctx, const std::filesystem::path& bin_path);
    void shutdown(rhi::VkContext& ctx);

    void cull(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
              units::Years sim_time, const RenderSettings& settings, VkExtent2D extent);

    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                units::Years sim_time, const RenderSettings& settings, VkExtent2D extent) const;

    bool loaded() const noexcept { return loaded_; }
    uint32_t count() const noexcept { return count_; }
    const rhi::Buffer& indirect_buffer(uint32_t frame_index) const noexcept { return indirect_buffers_[frame_index]; }

private:
    bool loaded_ = false;
    uint32_t count_ = 0;
    rhi::Buffer star_ssbo_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> indirect_buffers_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> visible_index_buffers_{};
    rhi::Pipeline cull_pipeline_{};
    rhi::Pipeline pipeline_{};
};

} // namespace sf::rendering
