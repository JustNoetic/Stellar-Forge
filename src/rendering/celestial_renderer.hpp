#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rendering/planetshine.hpp"
#include "rendering/atmosphere_renderer.hpp"
#include "rendering/ring_renderer.hpp"
#include "rendering/refraction_renderer.hpp"
#include "rendering/eclipse_refraction_renderer.hpp"
#include "rendering/transparency_renderer.hpp"
#include "ephemeris/system_manager.hpp"
#include "rhi/vk_context.hpp"
#include "terrain/terrain_renderer.hpp"
#include "terrain/surface_textures.hpp"
#include <array>
#include <map>
#include <string>
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
    PlanetshineGPU planetshine{};
    physics::SurfaceMaterialGPU material{};
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
    glm::uvec4 ring_shadow_params{0};
    glm::uvec4 eclipse_params{0};
    glm::uvec4 ringshine_params{0};
    glm::mat4 projection{1.0f};
};

class CelestialRenderer {
public:
    void init(rhi::VkContext& ctx, bool load_surface_assets = true);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                 const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent,
                 bool shadow_diagnostics = false, std::map<std::string, double>* timings = nullptr);

    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& cam,
                const ephemeris::SystemSnapshot& sys, const RenderSettings& settings, VkExtent2D extent,
                std::map<std::string, double>* timings = nullptr);
    const terrain::TerrainStats& terrain_stats() const noexcept { return terrain_.stats(); }
    const terrain::TextureStats& texture_stats() const noexcept { return textures_.stats(); }
    const terrain::TextureStats& normal_stats() const noexcept { return normals_.stats(); }
    const AtmosphereStats& atmosphere_stats() const noexcept { return atmospheres_.stats(); }
    ShadowHistoryDiagnostics shadow_history_diagnostics(rhi::VkContext& ctx) const { return atmospheres_.shadow_history_diagnostics(ctx); }
    void resolve_transparency(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth) {
        transparency_.resolve(cmd, ctx, color, depth);
    }
    void composite_atmospheres(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth) {
        auto writes = transparency_.writes();
        atmospheres_.composite(cmd, ctx, color, depth, transparency_.active() ? std::span{writes} : std::span<const rhi::PushDescriptorWrite>{});
        transparency_.resolve(cmd, ctx, color, depth);
    }
    const TransparencyRenderer& transparency() const noexcept { return transparency_; }
    std::array<rhi::PushDescriptorWrite, 2> refraction_writes(rhi::VkContext& ctx) const { return refraction_.writes(ctx); }
    const RefractionHost& refraction_host() const noexcept { return refraction_.host(); }
    uint32_t refraction_observer_builds() const noexcept { return refraction_.observer_builds(); }
    const RefractionRenderer& refraction() const noexcept { return refraction_; }
    uint32_t eclipse_refraction_builds() const { return eclipse_refraction_.builds(); }
    double eclipse_refraction_bake_ms() const { return eclipse_refraction_.last_bake_ms(); }
    std::span<const uint32_t> ring_eclipse_candidate_counts() const noexcept { return ring_eclipse_candidate_counts_; }
     std::span<const uint32_t> surface_ring_candidate_masks() const noexcept { return surface_ring_candidate_masks_; }
    uint32_t ringshine_bake_count() const noexcept { return rings_.ringshine_bake_count(); }
    uint32_t ringshine_frame_bake_count() const noexcept { return rings_.ringshine_frame_bake_count(); }
    std::span<const uint32_t> ringshine_sample_counts() const noexcept { return rings_.ringshine_sample_counts(); }
    void set_color_sources(std::vector<terrain::ColorBinding> bindings) { textures_.set_sources(std::move(bindings)); }
    void set_height_source(std::string body, std::shared_ptr<const terrain::HeightSource> source) {
        normals_.set_height_source(body, source);
        terrain_.set_height_source(std::move(body), std::move(source));
    }
    void invalidate_ring_textures() noexcept { rings_.invalidate_texture_cache(); }
    void set_custom_ring_texture(std::string_view system_name, std::string_view body_name, std::vector<glm::vec4> texture_samples) {
        rings_.set_custom_ring_texture(system_name, body_name, std::move(texture_samples));
    }
    bool has_custom_ring_texture(std::string_view system_name, std::string_view body_name) const noexcept {
        return rings_.has_custom_ring_texture(system_name, body_name);
    }

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
    std::vector<uint32_t> surface_ring_candidate_masks_;

    rhi::Pipeline sphere_pipeline_{};
    rhi::Pipeline point_pipeline_{};
    terrain::TerrainRenderer terrain_;
    terrain::SurfaceTextures textures_;
    terrain::SurfaceTextures normals_;
    RingRenderer rings_;
    AtmosphereRenderer atmospheres_;
    TransparencyRenderer transparency_;
    RefractionRenderer refraction_;
    EclipseRefractionRenderer eclipse_refraction_;
    std::vector<float> atmosphere_minima_;
    std::vector<PlanetshineGPU> planetshine_;
    SurfaceMaterialCache materials_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> material_phases_{};
    std::array<uint64_t, rhi::FRAMES_IN_FLIGHT> material_uploads_{};
};

} // namespace sf::rendering
