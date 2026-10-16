#pragma once

#include "rhi/vk_context.hpp"
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sf::rhi {

class ShaderCache {
public:
    static void init();
    static void shutdown();

    static std::string load_and_preprocess(const std::filesystem::path& shader_path,
                                           std::span<const std::pair<std::string, std::string>> macros = {});

    static std::vector<uint32_t> compile_to_spirv(std::string_view relative_path,
                                                  VkShaderStageFlagBits stage,
                                                  std::span<const std::pair<std::string, std::string>> macros = {});

    static VkShaderModule create_shader_module(VkDevice device,
                                               std::string_view relative_path,
                                               VkShaderStageFlagBits stage,
                                               std::span<const std::pair<std::string, std::string>> macros = {});

    static Pipeline create_compute_pipeline(VkContext& ctx,
                                            std::string_view comp_path,
                                            std::span<const DescriptorLayoutBinding> bindings,
                                            uint32_t push_constant_size = 0,
                                            std::span<const std::pair<std::string, std::string>> macros = {});

    static Pipeline create_graphics_pipeline(VkContext& ctx, const GraphicsPipelineDesc& desc);

    static void destroy_pipeline(VkContext& ctx, Pipeline& pipeline);
};

} // namespace sf::rhi
