#include "rendering/cloud_maps.hpp"
#include "core/path_utils.hpp"
#include "core/log.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>

namespace sf::rendering {

void CloudMaps::init(rhi::VkContext& ctx) {
    struct Pack { std::string name; std::vector<uint8_t> bytes; };
    std::vector<Pack> packs;
    uint32_t width = 1, height = 1, mip_count = 1;
    if (auto catalog = paths::read_json(paths::external_path("data", "clouds.json")); catalog && catalog->is_object()) {
        for (const auto& [name, value] : catalog->items()) {
            try {
                if (packs.size() >= std::min(15u, ctx.device_properties().limits.maxImageArrayLayers))
                    throw std::runtime_error("Cloud map atlas layer limit exceeded");
                auto directory = paths::external_path(value.get<std::string>());
                auto manifest = paths::read_json(directory / "manifest.json");
                if (!manifest || manifest->at("version") != 1 || manifest->at("encoding") != "r8-unorm" ||
                    manifest->at("projection") != "equirectangular" || manifest->at("orientation") != "north-top-east-right")
                    throw std::runtime_error("Unsupported cloud coverage pack; run scripts/bake_cloud_maps.py");
                uint32_t pack_width = manifest->at("levels").at(0).at("width");
                uint32_t pack_height = manifest->at("levels").at(0).at("height");
                if (pack_width < 2 || pack_width > 16384 || pack_width % 2 != 0 || pack_height != pack_width/2)
                    throw std::runtime_error("Cloud coverage must be an even-width 2:1 map up to 16384 pixels wide");
                if (pack_width > ctx.device_properties().limits.maxImageDimension2D)
                    throw std::runtime_error("Cloud coverage exceeds this GPU's maximum texture dimensions");
                uint32_t pack_mips = std::bit_width(pack_width);
                if (manifest->at("levels").size() != pack_mips ||
                    (!packs.empty() && (pack_width != width || pack_height != height)))
                    throw std::runtime_error("Cloud atlas maps must have matching dimensions and complete mip pyramids");
                uint64_t offset = 0;
                for (uint32_t mip = 0; mip < pack_mips; ++mip) {
                    const auto& level = manifest->at("levels").at(mip);
                    uint32_t w = std::max(pack_width >> mip, 1u), h = std::max(pack_height >> mip, 1u);
                    if (level.at("width") != w || level.at("height") != h || level.at("offset") != offset)
                        throw std::runtime_error("Invalid cloud coverage mip layout");
                    offset += uint64_t(w) * h;
                }
                if (std::filesystem::file_size(directory / "coverage.r8") != offset)
                    throw std::runtime_error("Incomplete cloud coverage pack");
                auto bytes = paths::read_binary_file(directory / "coverage.r8");
                if (!bytes || bytes->size() != offset) throw std::runtime_error("Cannot read cloud coverage pack");
                width = pack_width; height = pack_height; mip_count = pack_mips;
                packs.push_back({name, std::move(*bytes)});
            } catch (const std::exception& error) {
                log::warn("Cloud map '{}': {}", name, error.what());
            }
        }
    }
    uint32_t w = packs.empty() ? 1 : width, h = packs.empty() ? 1 : height;
    uint32_t levels = packs.empty() ? 1 : mip_count;
    uint32_t count = std::max(uint32_t(packs.size()), 1u);
    atlas_ = ctx.create_image_2d(w, h, VK_FORMAT_R8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, levels, count, true);
    std::vector<uint8_t> pixels;
    std::vector<VkBufferImageCopy> regions;
    for (uint32_t layer = 0; layer < count; ++layer) {
        uint64_t source_offset = 0;
        if (layer < packs.size()) layers_[packs[layer].name] = layer+1;
        for (uint32_t mip = 0; mip < levels; ++mip) {
            uint32_t mw = std::max(w >> mip, 1u), mh = std::max(h >> mip, 1u);
            pixels.resize((pixels.size() + 3) & ~size_t(3), 255);
            VkBufferImageCopy copy{};
            copy.bufferOffset = pixels.size();
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
            copy.imageExtent = {mw, mh, 1};
            regions.push_back(copy);
            size_t size = size_t(mw) * mh;
            if (layer < packs.size()) {
                const auto& bytes = packs[layer].bytes;
                pixels.insert(pixels.end(), bytes.begin() + source_offset, bytes.begin() + source_offset + size);
            } else pixels.insert(pixels.end(), size, 255);
            source_offset += size;
        }
    }
    ctx.upload_image_2d(atlas_, pixels.data(), pixels.size(), regions);
    log::info("Cloud coverage maps: {}, {}x{}, {} mips, {:.1f} MiB", packs.size(), w, h, levels, double(pixels.size())/(1024.0*1024.0));
}

void CloudMaps::shutdown(rhi::VkContext& ctx) {
    ctx.destroy_image(atlas_);
    layers_.clear();
}

uint32_t CloudMaps::layer(const std::string& name) const noexcept {
    auto found = layers_.find(name);
    return found == layers_.end() ? 0 : found->second;
}

}
