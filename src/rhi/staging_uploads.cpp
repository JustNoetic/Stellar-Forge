#include "rhi/staging_uploads.hpp"
#include <cstring>
#include <stdexcept>

namespace sf::rhi {

void StagingUploads::init(VkContext& ctx, VkDeviceSize capacity) {
    if (capacity == 0 || capacity % 16) throw std::invalid_argument("Staging capacity must be a nonzero multiple of 16");
    staging_ = ctx.create_host_buffer(capacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    ring_ = UploadRing{capacity};
}

void StagingUploads::shutdown(VkContext& ctx) {
    ctx.destroy_buffer(staging_);
    ring_ = UploadRing{};
}

std::optional<VkDeviceSize> StagingUploads::stage(VkContext& ctx, std::span<const std::byte> bytes) {
    auto offset = ring_.allocate(bytes.size(), ctx.recording_serial(), ctx.completed_serial());
    if (!offset) return {};
    std::memcpy(static_cast<std::byte*>(staging_.mapped) + *offset, bytes.data(), bytes.size());
    ctx.flush_buffer(staging_, *offset, bytes.size());
    return *offset;
}

bool StagingUploads::buffer(VkCommandBuffer cmd, VkContext& ctx, const Buffer& destination, VkDeviceSize offset,
                            std::span<const std::byte> bytes) {
    if (offset > destination.size || bytes.size() > destination.size-offset || offset % 4 || bytes.size() % 4) return false;
    auto staged = stage(ctx, bytes);
    if (!staged) return false;
    VkBufferCopy region{*staged, offset, bytes.size()};
    vkCmdCopyBuffer(cmd, staging_.handle, destination.handle, 1, &region);
    return true;
}

bool StagingUploads::image(VkCommandBuffer cmd, VkContext& ctx, const Image& destination,
                           const VkBufferImageCopy& region, std::span<const std::byte> bytes) {
    if (region.imageSubresource.mipLevel >= destination.mip_levels || region.imageSubresource.layerCount == 0 ||
        region.imageSubresource.baseArrayLayer >= destination.layers ||
        region.imageSubresource.layerCount > destination.layers-region.imageSubresource.baseArrayLayer) return false;
    uint64_t texel_bytes = 0;
    switch (destination.format) {
    case VK_FORMAT_R8_UNORM: texel_bytes = 1; break;
    case VK_FORMAT_R8G8_UNORM: texel_bytes = 2; break;
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: texel_bytes = 4; break;
    case VK_FORMAT_R16G16B16A16_SFLOAT: texel_bytes = 8; break;
    case VK_FORMAT_R32G32B32A32_SFLOAT: texel_bytes = 16; break;
    default: return false;
    }
    auto mip = region.imageSubresource.mipLevel;
    auto extent = region.imageExtent;
    auto position = region.imageOffset;
    if (position.x < 0 || position.y < 0 || position.z < 0 || extent.width == 0 || extent.height == 0 || extent.depth == 0 ||
        uint64_t(position.x)+extent.width > std::max(1u, destination.extent.width >> mip) ||
        uint64_t(position.y)+extent.height > std::max(1u, destination.extent.height >> mip) ||
        uint64_t(position.z)+extent.depth > std::max(1u, destination.extent.depth >> mip) ||
        region.imageSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT) return false;
    uint64_t row = region.bufferRowLength ? region.bufferRowLength : extent.width;
    uint64_t height = region.bufferImageHeight ? region.bufferImageHeight : extent.height;
    if (row < extent.width || height < extent.height) return false;
    uint64_t required = ((uint64_t(extent.depth)*region.imageSubresource.layerCount-1)*height*row+
                         (extent.height-1)*row+extent.width)*texel_bytes;
    if (bytes.size() < required) return false;
    auto offset = stage(ctx, bytes);
    if (!offset) return false;
    auto copy = region;
    copy.bufferOffset = *offset;
    vkCmdCopyBufferToImage(cmd, staging_.handle, destination.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    return true;
}

}
