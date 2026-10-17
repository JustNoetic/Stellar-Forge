#pragma once

#include "rhi/vk_context.hpp"
#include "rhi/upload_ring.hpp"

namespace sf::rhi {

class StagingUploads {
public:
    void init(VkContext& ctx, VkDeviceSize capacity);
    void shutdown(VkContext& ctx);
    bool buffer(VkCommandBuffer cmd, VkContext& ctx, const Buffer& destination, VkDeviceSize offset, std::span<const std::byte> bytes);
    bool image(VkCommandBuffer cmd, VkContext& ctx, const Image& destination, const VkBufferImageCopy& region,
               std::span<const std::byte> bytes);

private:
    std::optional<VkDeviceSize> stage(VkContext& ctx, std::span<const std::byte> bytes);
    Buffer staging_;
    UploadRing ring_;
};

}
