#pragma once

#include "rendering/atmosphere_renderer.hpp"
#include <span>
#include <vector>

namespace sf::rendering {

struct RingHostGPU;
struct RingPenumbraSlots;

inline constexpr double LIMB_RESOLVED_PIXELS = 32.0;

inline constexpr rhi::DescriptorLayoutBinding source_bindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT}};

// Push constants of the composite variants: secondary irradiance, then the ring slot mode in ring.x.
struct CompositePush {
    glm::vec4 irradiance{0.0f};
    glm::uvec4 ring{0u};
};

inline constexpr rhi::DescriptorLayoutBinding composite_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, {17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {33, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}, {34, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT},
        {38, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT}};


struct AtmosphereFrame {
    AtmosphereParametersGPU p{};
    glm::dvec3 pole{}, x{}, y{}, z{};
    double polar_ratio = 1.0, distance = 0.0, point_radius = 0.0;
    float height = 0.0f, bottom = 0.0f, radius = 0.0f;
    glm::mat4 projection{1.0f}, view{1.0f};
    bool limb_active = false;

    glm::dvec3 atmosphere_space(glm::dvec3 v) const { return v+pole*glm::dot(v, pole)*(1.0/polar_ratio-1.0); }
};

struct AtmosphereChanges {
    bool optics = false, identity = false, lighting_mode = false, sky = false, aerial = false;
    bool small_stars = false, scattering = false;
    float finite_radius = 0.0f;
};

struct LutRecorder {
    VkCommandBuffer cmd;
    rhi::VkContext& ctx;

    rhi::Image image(uint32_t w, uint32_t h, VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT) const {
        return ctx.create_image_2d(w, h, format,
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    }
    void bake(const rhi::Pipeline& pipeline, std::span<const rhi::PushDescriptorWrite> writes, uint32_t x, uint32_t y, uint32_t z = 1) const {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        ctx.push_descriptors(cmd, pipeline, writes);
        vkCmdDispatch(cmd, x, y, z);
    }
    void writable(rhi::Image& img) const { ctx.transition_image(cmd, img, VK_IMAGE_LAYOUT_GENERAL); }
    void readable(rhi::Image& img) const {
        ctx.transition_image(cmd, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
};

bool stellar_source_resolved(const AtmosphereParametersGPU& p);
bool prepare_shadows(rhi::VkContext& ctx, rhi::Buffer& buffer, const ephemeris::SystemSnapshot& system,
                     uint32_t receiver, glm::dvec3 eye, float outer, float ground, float focal,
                     const Camera& camera, const glm::dmat3& apparent, VkExtent2D extent, double refraction_pad,
                     const RenderSettings& settings, std::span<const RingHostGPU> rings,
                     AtmosphereShadowHeaderGPU& previous, std::vector<AtmosphereShadowCasterGPU>& previous_casters,
                     uint32_t sequence, uint32_t& changes, const RingPenumbraSlots* penumbra_slots = nullptr);
AtmosphereFrame make_atmosphere_frame(const ephemeris::SystemSnapshot& system, uint32_t index, const Camera& camera,
                                      const RenderSettings& settings, VkExtent2D extent, const AtmosphereRefraction& refraction_warp,
                                      double candidate_distance, double focal, glm::dvec3 eye, std::span<const float> minimum_heights_km);

}
