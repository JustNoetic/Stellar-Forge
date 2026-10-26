#pragma once

#include "comparison/size_comparison.hpp"
#include "rendering/atmosphere_renderer.hpp"
#include "rendering/render_settings.hpp"
#include "terrain/surface_textures.hpp"
#include <map>

namespace sf::rendering {

class SizeComparatorRenderer {
public:
    void init(rhi::VkContext& ctx, bool reference_atmosphere = false);
    void prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const comparison::SizeComparison& scene,
                 const comparison::SizeView& view, const RenderSettings& settings, VkExtent2D extent);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx);
    void composite(VkCommandBuffer cmd, rhi::VkContext& ctx, rhi::Image& color, rhi::Image& depth);
    void shutdown(rhi::VkContext& ctx);
    bool initialized() const noexcept { return initialized_; }
    uint64_t transport_bakes() const noexcept { return transport_bakes_; }
    const terrain::TextureStats& texture_stats() const noexcept { return textures_.stats(); }
    void set_color_sources(std::vector<terrain::ColorBinding> bindings) { textures_.set_sources(std::move(bindings)); }

private:
    struct Resources {
        std::shared_ptr<const ephemeris::SystemDefinition> definition;
        size_t body_index = 0;
        rhi::Buffer profiles;
        std::array<rhi::Buffer, rhi::FRAMES_IN_FLIGHT> uniforms;
        rhi::Image transmittance, multiscatter;
        rhi::Image transport_radiance, transport_throughput;
        bool optics = false;
        uint64_t used = 0;
    };
    struct alignas(16) Push {
        glm::vec4 rect{}, shape{}, color{}, resources{}, display{};
    };
    struct Draw { Resources* resources; Push surface, overlay; };
    struct Retired { std::unique_ptr<Resources> resources; uint64_t serial; };

    void destroy(rhi::VkContext& ctx, Resources& resources);
    std::map<std::string, std::unique_ptr<Resources>> resources_;
    std::vector<Retired> retired_;
    std::vector<Draw> draws_;
    rhi::Pipeline surface_, overlay_, transmittance_, multiscatter_, transport_;
    rhi::Image dummy_image_;
    terrain::SurfaceTextures textures_, normals_;
    bool initialized_ = false;
    bool reference_atmosphere_ = false;
    uint64_t transport_bakes_ = 0;
    uint64_t serial_ = 0;
};

}
