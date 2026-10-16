#pragma once

#include "core/camera.hpp"
#include "rhi/vk_context.hpp"
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

    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                double sim_time_yr, VkExtent2D extent) const;

    bool loaded() const noexcept { return loaded_; }
    uint32_t count() const noexcept { return count_; }

private:
    bool loaded_ = false;
    uint32_t count_ = 0;
    rhi::Buffer star_ssbo_{};
    rhi::Pipeline pipeline_{};
};

} // namespace sf::rendering
