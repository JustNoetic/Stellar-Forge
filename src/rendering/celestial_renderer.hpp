#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/ring_renderer.hpp"
#include "ephemeris/system_manager.hpp"
#include "rhi/vk_context.hpp"
#include "terrain/terrain_renderer.hpp"
#include "terrain/surface_textures.hpp"
#include <array>
#include <span>
#include <vector>
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
    glm::uvec4 surface_flags{0};
};

struct alignas(16) SceneCelestialUBO {
    glm::mat4 view_proj{1.0f};
    glm::mat4 view{1.0f};
    glm::vec4 stars_pos_radius[MAX_SCENE_STARS]{};
    glm::vec4 stars_color_lum[MAX_SCENE_STARS]{};
    glm::vec4 screen_params{1600.0f, 900.0f, 1.2f, 1.0f};
    glm::vec4 point_params{DEFAULT_POINT_DIAMETER_PX, 0.0f, 0.0f, 0.0f};
    uint32_t num_stars = 0;
    uint32_t num_bodies = 0;
    uint32_t hdr_enabled = 1;
    uint32_t num_ring_hosts = 0;
    glm::mat4 projection{1.0f};
};

class CelestialRenderer {
public:
    void init(rhi::VkContext& ctx, bool load_surface_assets = true);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                 const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent);

    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent);
    const terrain::TerrainStats& terrain_stats() const noexcept { return terrain_.stats(); }
    const terrain::TextureStats& texture_stats() const noexcept { return textures_.stats(); }
    std::span<const uint32_t> ring_eclipse_candidate_counts() const noexcept { return ring_eclipse_candidate_counts_; }
    void set_color_sources(std::vector<terrain::ColorBinding> bindings) { textures_.set_sources(std::move(bindings)); }
    void set_height_source(std::string body, std::shared_ptr<const terrain::HeightSource> source) {
        terrain_.set_height_source(std::move(body), std::move(source));
    }
    void invalidate_ring_textures() noexcept { rings_.invalidate_texture_cache(); }

private:
    void build_sphere_mesh(rhi::VkContext& ctx, uint32_t sectors, uint32_t stacks);

    rhi::Buffer sphere_vbo_{};
    rhi::Buffer sphere_ibo_{};
    uint32_t sphere_index_count_ = 0;

    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> scene_ubos_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> instance_ssbos_{};
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> eclipse_ssbos_{};
    std::vector<BodyInstanceGPU> body_instances_;
    std::vector<uint32_t> eclipse_words_;
    std::vector<uint32_t> ring_eclipse_candidate_counts_;

    rhi::Pipeline sphere_pipeline_{};
    rhi::Pipeline point_pipeline_{};
    terrain::TerrainRenderer terrain_;
    terrain::SurfaceTextures textures_;
    RingRenderer rings_;
};

} // namespace sf::rendering
