#pragma once

#include "core/log.hpp"
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>
#include <vk_mem_alloc.h>

#define VK_CHECK(expr)                                                                         \
    do {                                                                                       \
        VkResult _res = (expr);                                                                \
        if (_res != VK_SUCCESS) {                                                              \
            sf::log::fatal("Vulkan error {} at {}:{} in {}", static_cast<int>(_res), __FILE__, \
                           __LINE__, #expr);                                                   \
        }                                                                                      \
    } while (0)

namespace sf::rhi {

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* mapped = nullptr;

    explicit operator bool() const noexcept { return handle != VK_NULL_HANDLE; }
};

struct Image {
    VkImage handle = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{0, 0, 0};
    uint32_t mip_levels = 1;
    uint32_t layers = 1;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    explicit operator bool() const noexcept { return handle != VK_NULL_HANDLE; }
};

struct DescriptorLayoutBinding {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkShaderStageFlags stages = VK_SHADER_STAGE_ALL;
    uint32_t count = 1;
};

struct Pipeline {
    VkPipeline handle = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkShaderStageFlags push_constant_stages = VK_SHADER_STAGE_ALL;

    explicit operator bool() const noexcept { return handle != VK_NULL_HANDLE; }
};

struct PushDescriptorWrite {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorBufferInfo buffer_info{};
    VkDescriptorImageInfo image_info{};

    static PushDescriptorWrite uniform_buffer(uint32_t binding, const Buffer& buf, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) {
        PushDescriptorWrite w{};
        w.binding = binding;
        w.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.buffer_info = {buf.handle, offset, range};
        return w;
    }

    static PushDescriptorWrite storage_buffer(uint32_t binding, const Buffer& buf, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE) {
        PushDescriptorWrite w{};
        w.binding = binding;
        w.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.buffer_info = {buf.handle, offset, range};
        return w;
    }

    static PushDescriptorWrite sampled_image(uint32_t binding, const Image& img, VkSampler sampler, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        PushDescriptorWrite w{};
        w.binding = binding;
        w.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.image_info = {sampler, img.view, layout};
        return w;
    }

    static PushDescriptorWrite storage_image(uint32_t binding, const Image& img, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
        PushDescriptorWrite w{};
        w.binding = binding;
        w.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.image_info = {VK_NULL_HANDLE, img.view, layout};
        return w;
    }
};

struct RenderingAttachment {
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp store_op = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearValue clear{};

    static RenderingAttachment clear_color(const Image& img, float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f) {
        RenderingAttachment att{};
        att.view = img.view;
        att.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.store_op = VK_ATTACHMENT_STORE_OP_STORE;
        att.clear.color = {{r, g, b, a}};
        return att;
    }

    static RenderingAttachment load_color(const Image& img) {
        RenderingAttachment att{};
        att.view = img.view;
        att.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
        att.store_op = VK_ATTACHMENT_STORE_OP_STORE;
        return att;
    }

    static RenderingAttachment clear_depth_reverse_z(const Image& img, float depth = 0.0f) {
        RenderingAttachment att{};
        att.view = img.view;
        att.layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        att.load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.store_op = VK_ATTACHMENT_STORE_OP_STORE;
        att.clear.depthStencil = {depth, 0};
        return att;
    }

    static RenderingAttachment load_depth(const Image& img) {
        RenderingAttachment att{};
        att.view = img.view;
        att.layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        att.load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
        att.store_op = VK_ATTACHMENT_STORE_OP_STORE;
        return att;
    }
};

struct VertexAttributeDesc {
    uint32_t location = 0;
    uint32_t binding = 0;
    VkFormat format = VK_FORMAT_R32G32B32_SFLOAT;
    uint32_t offset = 0;
};

struct VertexBindingDesc {
    uint32_t binding = 0;
    uint32_t stride = 0;
    VkVertexInputRate input_rate = VK_VERTEX_INPUT_RATE_VERTEX;
};

struct GraphicsPipelineDesc {
    std::string vert_path;
    std::string frag_path;
    std::string geom_path;
    std::vector<std::pair<std::string, std::string>> macros;
    std::vector<DescriptorLayoutBinding> bindings;
    uint32_t push_constant_size = 0;
    std::vector<VertexBindingDesc> vertex_bindings;
    std::vector<VertexAttributeDesc> vertex_attributes;
    std::vector<VkFormat> color_formats = {VK_FORMAT_R16G16B16A16_SFLOAT};
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    bool depth_test = true;
    bool depth_write = true;
    VkCompareOp depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    bool alpha_blend = false;
    bool additive_blend = false;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
};

} // namespace sf::rhi
