#pragma once

#include "core/camera.hpp"
#include "rendering/render_settings.hpp"
#include "rhi/vk_context.hpp"
#include <array>
#include <memory>
#include <functional>
#include <vector>

namespace sf::rendering {

struct RingHostGPU;
class RingRenderer;
struct PlanetshineGPU;
struct RefractionHost;
struct AtmosphereFrame;
struct AtmosphereChanges;
struct LutRecorder;

struct alignas(16) AtmosphereShadowHeaderGPU {
    glm::vec4 eye_focal{};
    glm::uvec4 flags{};
    std::array<glm::vec4, 16> stars{};
    std::array<glm::uvec4, 16> ranges{};
    glm::uvec4 temporal{};
    glm::ivec4 shadow_bounds{};
    glm::ivec4 shadow_coarse_bounds{};
};

struct alignas(16) AtmosphereShadowCasterGPU {
    glm::vec4 position_radius{};
    glm::vec4 pole_inner{};
    glm::vec4 support{};
};

struct alignas(16) AtmosphereParametersGPU {
    glm::vec4 radii_heights{};
    glm::vec4 reference_shape{};
    glm::vec4 world_pole{};
    glm::vec4 rayleigh{};
    glm::vec4 mie_scattering{};
    glm::vec4 mie_extinction{};
    glm::vec4 ground_albedo{};
    glm::vec4 camera{};
    glm::vec4 axis_x{}, axis_y{}, axis_z{};
    std::array<glm::vec4, 16> sun_direction{};
    std::array<glm::vec4, 16> sun_irradiance{};
    glm::mat4 inverse_view_projection{1.0f};
    glm::vec4 depth_params{};
    glm::vec4 mixed_absorption{}, ozone_absorption{}, ozone_params{}, profile_domain{};
    std::array<glm::vec4, physics::ATMOSPHERE_PROFILE_SAMPLES> log_density{};
    glm::vec4 aerosol_count{};
    std::array<glm::vec4, ephemeris::MAX_AEROSOL_LAYERS> aerosol_shape{}, aerosol_scattering{}, aerosol_extinction{};
    std::array<glm::vec4, ephemeris::MAX_AEROSOL_LAYERS*physics::AEROSOL_PHASE_SAMPLES> aerosol_phase{};
    std::array<glm::vec4, 3> view_ray{};
    std::array<glm::vec4, 64> limb_bend{};
    std::array<glm::vec4, 16> star_radiance{};
    std::array<glm::vec4, 16> star_disc{};
    glm::vec4 limb_image{};
    glm::vec4 cloud_layer{}; // x: body index, y: 1 when the 2D cloud sheet is composited inline
};

AtmosphereParametersGPU make_atmosphere_parameters(const ephemeris::BodyDefinition& body,
    const std::shared_ptr<const physics::PreparedAtmosphere>& prepared, glm::dvec3 pole, double minimum_height = 0);

enum ShadowHistoryReset : uint32_t {
    ShadowHistoryInitial = 1u << 0,
    ShadowHistoryInterrupted = 1u << 1,
    ShadowHistoryCamera = 1u << 2,
    ShadowHistoryProjection = 1u << 3,
    ShadowHistoryAtmosphere = 1u << 4,
    ShadowHistoryLighting = 1u << 5,
    ShadowHistoryCasters = 1u << 6,
    ShadowHistoryRings = 1u << 7,
    ShadowHistorySampling = 1u << 8,
    ShadowHistoryViewport = 1u << 9,
    ShadowHistoryDefinition = 1u << 10,
    ShadowHistoryTransport = 1u << 11
};

struct ShadowHistoryDiagnostics {
    uint64_t serial = 0;
    uint32_t columns = 0, depth_resets = 0, refining = 0, complete = 0;
    uint32_t minimum_frames = 0, maximum_frames = 0, reset_reasons = 0;
    uint32_t reprojected = 0, reprojection_rejected = 0;
    bool valid = false;
};

struct AtmosphereStats {
    uint32_t visible = 0;
    uint32_t optical_lut_updates = 0;
    uint32_t sky_lut_updates = 0;
    uint32_t aerial_lut_updates = 0;
    uint32_t surface_lut_updates = 0;
    uint32_t point_lut_updates = 0;
    uint32_t stellar_lut_updates = 0;
    uint32_t secondary_lut_updates = 0;
    uint32_t secondary_source_lut_updates = 0;
    uint32_t shadow_updates = 0;
    uint32_t shadow_vrs_updates = 0;
    uint32_t shadow_history_resets = 0;
    uint32_t shadow_history_reset_reasons = 0;
    uint32_t shadow_rotation_updates = 0;
};

class AtmosphereRenderer {
public:
    void init(rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                 const ephemeris::SystemSnapshot& system, const RenderSettings& settings, VkExtent2D extent,
                 std::span<const float> minimum_heights_km = {}, std::span<const RingHostGPU> rings = {},
                 const rhi::Image* ring_shadow_lut = nullptr, uint64_t ring_shadow_revision = 0,
                 bool shadow_diagnostics = false, std::span<const PlanetshineGPU> planetshine = {},
                 const RingRenderer* ring_lighting = nullptr, const RefractionHost* refraction = nullptr);
    // cloud_writes bind the inline cloud layer (33: colour, 34: depth); empty binds cleared placeholders.
    void composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth,
                   std::span<const rhi::PushDescriptorWrite> transparency_writes = {},
                   const std::function<void(uint32_t)>& draw_point = {},
                   std::span<const rhi::PushDescriptorWrite> cloud_writes = {});
    // True when this frame's atmosphere composite of `body` blends its 2D cloud sheet from the cloud layer.
    bool composites_clouds(uint32_t body) const noexcept;
    const AtmosphereStats& stats() const noexcept { return stats_; }
    ShadowHistoryDiagnostics shadow_history_diagnostics(rhi::VkContext& ctx) const;
    const rhi::Buffer& point_responses(uint32_t frame) const noexcept { return point_responses_[frame]; }
    const rhi::Image& surface_lut() const noexcept { return surface_lut_; }
    const rhi::Buffer& surface_records(uint32_t frame) const noexcept { return surface_records_[frame]; }

private:
    struct Tables {
        rhi::Image transmittance, multiscatter, sky, throughput;
        rhi::Buffer point_partials, point_response;
        physics::SurfaceMaterialGPU point_material{};
        glm::vec4 point_color{};
        glm::uvec2 point_surface{};
        // Geometry the point response was last baked for: world-space view and pole directions,
        // and per star its world-space direction (xyz) and angular radius (w).
        glm::dvec3 point_view{}, point_pole{};
        std::array<glm::dvec4, 16> point_suns{};
        float point_stars = 0.0f;
        bool point_valid = false;
        rhi::Image stellar_source;
        rhi::Image aerial_radiance, aerial_optical_depth, aerial_range;
        rhi::Image secondary_sky, secondary_aerial, secondary_range;
        rhi::Image secondary_source;
        glm::vec4 secondary_direction{}, secondary_irradiance{};
        std::array<glm::vec4, 4> secondary_directions{}, secondary_weights{};
        std::array<glm::vec4, 17> secondary_ring_key{};
        uint64_t secondary_ring_revision = 0;
        bool secondary_approximate = false;
        float secondary_altitude = 0.0f;
        float secondary_source_radius = 0.0f;
        bool secondary_source_valid = false;
        bool secondary_active = false, secondary_valid = false, secondary_aerial_valid = false;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> uniforms;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> shadows;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> transparent_tiles;
        rhi::Image shadow_fraction, shadow_depth;
        rhi::Image shadow_energy, shadow_loss;
        AtmosphereShadowHeaderGPU shadow_header{};
        std::vector<AtmosphereShadowCasterGPU> shadow_casters;
        bool shadow_active = false;
        bool limb_active = false;
        bool shadow_history_valid = false, shadow_history_reusable = false;
        bool shadow_history_reproject = false, shadow_rotation_enabled = false;
        uint64_t shadow_recorded_prepare = 0, shadow_ring_revision = 0;
        VkImage shadow_ring_image = VK_NULL_HANDLE;
        uint32_t shadow_reset_reasons = 0;
        glm::mat4 shadow_view{1.0f}, shadow_projection{1.0f};
        glm::mat4 shadow_reprojection{1.0f}, shadow_previous_inverse{1.0f};
        glm::dvec3 shadow_eye_relative{};
        AtmosphereParametersGPU cached{};
        uint64_t identity = 0, definition_revision = 0, geometry_revision = 0, lighting_revision = 0;
        uint64_t last_used = 0;
        uint32_t body_index = 0;
        bool valid = false;
        bool aerial_valid = false;
        bool stellar_source_resolved = true;
    };
    struct Candidate { uint32_t index; double distance; glm::dmat3 apparent; };
    bool begin_prepare(rhi::VkContext& ctx, const ephemeris::SystemSnapshot& system, const RenderSettings& settings,
                       VkExtent2D extent, const rhi::Image* ring_shadow_lut, bool shadow_diagnostics);
    std::vector<Candidate> select_candidates(const Camera& camera, const ephemeris::SystemSnapshot& system, VkExtent2D extent,
                                             const RefractionHost* refraction) const;
    void assign_tables(std::span<const Candidate> candidates, const ephemeris::SystemSnapshot& system,
                       std::vector<size_t>& slots, std::vector<size_t>& surface_slots) const;
    void update_shadow_history(Tables& t, const AtmosphereFrame& frame, glm::dvec3 relative_eye, bool identity_dirty,
                               uint32_t shadow_changes, uint64_t ring_shadow_revision, const RenderSettings& settings);
    void bake_luts(const LutRecorder& lut, Tables& t, const AtmosphereFrame& frame, const AtmosphereChanges& changes,
                   const ephemeris::SystemSnapshot& system, uint32_t index, size_t surface_slot, const RenderSettings& settings);
    void bake_secondary(const LutRecorder& lut, Tables& t, const AtmosphereFrame& frame, const AtmosphereChanges& changes,
                        const ephemeris::SystemSnapshot& system, uint32_t index, const RenderSettings& settings,
                        std::span<const PlanetshineGPU> planetshine, const RingRenderer* ring_lighting);
    void resolve_shadows(VkCommandBuffer cmd, rhi::VkContext& ctx, Tables& table, rhi::Image& color, rhi::Image& depth,
                         std::span<const rhi::PushDescriptorWrite> transparency_writes,
                         std::span<const rhi::PushDescriptorWrite> cloud_writes);
    std::array<rhi::PushDescriptorWrite, 2> cloud_layer_writes(rhi::VkContext& ctx, std::span<const rhi::PushDescriptorWrite> cloud_writes) const;
    void retire_shadow_history(rhi::VkContext& ctx, Tables& table);
    static void begin_tiles(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Buffer& tiles, VkExtent2D extent);
    static void dispatch_tiles(VkCommandBuffer cmd, rhi::VkContext& ctx, const rhi::Buffer& tiles);
    std::vector<std::unique_ptr<Tables>> tables_;
    std::vector<Tables*> active_;
    std::array<Tables*, 32> surface_layers_{};
    std::array<glm::vec2, 32> surface_keys_{};
    uint64_t prepare_serial_ = 0;
    rhi::Pipeline transmittance_pipeline_, multiscatter_pipeline_, sky_pipeline_, composite_pipeline_;
    rhi::Pipeline aerial_pipeline_, point_pipeline_, point_reduce_pipeline_;
    rhi::Pipeline small_sky_pipeline_, small_aerial_pipeline_;
    rhi::Pipeline molecular_sky_pipeline_, molecular_aerial_pipeline_;
    rhi::Pipeline legacy_sky_pipeline_, legacy_aerial_pipeline_;
    rhi::Pipeline dual_source_pipeline_, surface_pipeline_;
    rhi::Image dummy_cloud_color_, dummy_cloud_depth_;
    rhi::Pipeline limb_dual_source_pipeline_, limb_transparent_pipeline_, limb_unshadowed_transparent_pipeline_;
    rhi::Pipeline limb_secondary_dual_source_pipeline_, limb_secondary_transparent_pipeline_, limb_secondary_unshadowed_transparent_pipeline_;
    // Lean transparent transport that reads only cached shadow endpoints; the shadowed pipelines above then run as
    // deferred passes over the tiles it lists: plain, limb, secondary, limb+secondary.
    std::array<rhi::Pipeline, 4> transparent_lean_pipelines_;
    rhi::Pipeline stellar_source_pipeline_;
    std::array<rhi::Pipeline, 6> secondary_bake_pipelines_;
    std::array<rhi::Pipeline, 24> secondary_approximate_pipelines_;
    rhi::Pipeline secondary_composite_pipeline_, secondary_dual_source_pipeline_;
    rhi::Pipeline secondary_source_pipeline_;
    rhi::Pipeline secondary_transparent_pipeline_, secondary_composite_transparency_pipeline_;
    rhi::Pipeline secondary_unshadowed_transparent_pipeline_, secondary_unshadowed_composite_transparency_pipeline_;
    void init_secondary(rhi::VkContext& ctx);
    rhi::Pipeline transparent_pipeline_, composite_transparency_pipeline_;
    rhi::Pipeline unshadowed_transparent_pipeline_, unshadowed_composite_transparency_pipeline_;
    rhi::Pipeline shadow_pipeline_, shadow_transparency_pipeline_;
    rhi::Pipeline shadow_accumulate_pipeline_, shadow_accumulate_transparency_pipeline_;
    rhi::Pipeline shadow_reproject_pipeline_, shadow_rotate_pipeline_, shadow_rotate_transparency_pipeline_;
    rhi::Pipeline shadow_coarse_pipeline_, shadow_coarse_transparency_pipeline_, shadow_vrs_pipeline_, shadow_vrs_transparency_pipeline_;
    // Closed-form shadow variants with the sampled paths compiled out: plain, +transparency, coarse,
    // coarse+transparency. The VRS resolve only upsamples the coarse pass, so it has no closed-form variant.
    std::array<rhi::Pipeline, 4> shadow_closed_pipelines_;
    // Ring penumbra table owned by RingRenderer; a 1x1 zero layer when there is none.
    const rhi::Image* ring_penumbra_ = nullptr;
    rhi::Image ring_penumbra_dummy_;
    struct RetiredImage { rhi::Image image; uint64_t serial; };
    std::vector<RetiredImage> retired_images_;
    rhi::Image shadow_coarse_fraction_, shadow_coarse_guide_;
    rhi::Image shadow_reproject_energy_, shadow_reproject_loss_;
    rhi::Image surface_lut_;
    rhi::Image stbn_tex_;
    rhi::Image dummy_stbn_;
    std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> surface_records_, point_responses_;
    struct ShadowDiagnosticFrame {
        rhi::Buffer counters;
        uint64_t serial = 0;
        uint32_t reset_reasons = 0;
    };
    std::array<ShadowDiagnosticFrame, rhi::FRAMES_IN_FLIGHT> shadow_diagnostics_;
    ShadowHistoryDiagnostics shadow_last_diagnostics_;
    bool shadow_diagnostics_enabled_ = false;
    bool dual_source_ = false;
    bool shadow_vrs_enabled_ = true;
    bool shadow_temporal_enabled_ = false;
    const rhi::Image* ring_shadow_lut_ = nullptr;
    AtmosphereStats stats_;
};

}
