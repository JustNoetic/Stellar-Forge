#include "rendering/cloud_maps.hpp"
#include "rendering/cloud_volume.hpp"
#include "core/path_utils.hpp"
#include "core/log.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace sf::rendering {

CloudBoundPyramid build_cloud_bound_pyramid(std::span<const uint8_t> level0, uint32_t width, uint32_t height,
                                            uint32_t max_width, double dilation_texels) {
    CloudBoundPyramid result;
    result.width = std::max(std::min(width, max_width), 1u);
    result.height = std::max(uint32_t((uint64_t(result.width) * height + width - 1) / width), 1u);
    uint32_t cw = result.width, ch = result.height;
    std::vector<uint8_t> cells(size_t(cw) * ch, 0);
    std::vector<uint32_t> column(width);
    for (uint32_t x = 0; x < width; ++x) column[x] = uint32_t(uint64_t(x) * cw / width);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* row = cells.data() + size_t(uint64_t(y) * ch / height) * cw;
        const uint8_t* source = level0.data() + size_t(y) * width;
        for (uint32_t x = 0; x < width; ++x) row[column[x]] = std::max(row[column[x]], source[x]);
    }
    // A cell spans width/cw source texels, so this many neighbouring cells cover the dilation on each side; the
    // extra texel covers a source texel that straddles two cells but was counted in the first.
    int kx = int(std::ceil((dilation_texels + 1.0) * cw / double(width)));
    int ky = int(std::ceil((dilation_texels + 1.0) * ch / double(height)));
    std::vector<uint8_t> pass(cells.size(), 0);
    for (uint32_t y = 0; y < ch; ++y) for (uint32_t x = 0; x < cw; ++x) {
        uint8_t value = 0;
        for (int dx = -kx; dx <= kx; ++dx)
            value = std::max(value, cells[size_t(y) * cw + uint32_t((int64_t(x) + dx % int(cw) + cw) % cw)]);
        pass[size_t(y) * cw + x] = value;
    }
    for (uint32_t y = 0; y < ch; ++y) for (uint32_t x = 0; x < cw; ++x) {
        uint8_t value = 0;
        for (int dy = -ky; dy <= ky; ++dy)
            value = std::max(value, pass[size_t(std::clamp(int(y) + dy, 0, int(ch) - 1)) * cw + x]);
        cells[size_t(y) * cw + x] = value;
    }
    result.levels.push_back(std::move(cells));
    uint32_t pw = cw, ph = ch;
    while (pw > 1 || ph > 1) {
        uint32_t w = std::max(pw / 2, 1u), h = std::max(ph / 2, 1u);
        const auto& parent = result.levels.back();
        std::vector<uint8_t> level(size_t(w) * h, 0);
        for (uint32_t y = 0; y < h; ++y) for (uint32_t x = 0; x < w; ++x) {
            uint32_t x0 = uint32_t(uint64_t(x) * pw / w), x1 = uint32_t((uint64_t(x + 1) * pw + w - 1) / w);
            uint32_t y0 = uint32_t(uint64_t(y) * ph / h), y1 = uint32_t((uint64_t(y + 1) * ph + h - 1) / h);
            uint8_t value = 0;
            for (uint32_t py = y0; py < y1; ++py) for (uint32_t px = x0; px < x1; ++px)
                value = std::max(value, parent[size_t(py) * pw + px]);
            level[size_t(y) * w + x] = value;
        }
        result.levels.push_back(std::move(level));
        pw = w; ph = h;
    }
    return result;
}

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

    // A bilinear tap at mip L reads level-0 texels within 1.5*2^L of the point, and the polar clamp moves the point
    // by up to half a texel at that mip, so 2*2^L+1 texels covers every read the density can make.
    auto bound_start = std::chrono::steady_clock::now();
    double dilation = 2.0 * double(1u << CLOUD_VOLUME_MAX_COVERAGE_LOD) + 1.0;
    std::vector<CloudBoundPyramid> pyramids;
    for (uint32_t layer = 0; layer < count; ++layer) {
        if (layer < packs.size())
            pyramids.push_back(build_cloud_bound_pyramid({packs[layer].bytes.data(), size_t(w) * h}, w, h,
                                                         CLOUD_BOUND_MAX_WIDTH, dilation));
        else {
            std::vector<uint8_t> full(size_t(w) * h, 255);
            pyramids.push_back(build_cloud_bound_pyramid(full, w, h, CLOUD_BOUND_MAX_WIDTH, dilation));
        }
    }
    const auto& shape = pyramids.front();
    bounds_ = ctx.create_image_2d(shape.width, shape.height, VK_FORMAT_R8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, uint32_t(shape.levels.size()), count, true);
    pixels.clear();
    regions.clear();
    for (uint32_t layer = 0; layer < count; ++layer) for (uint32_t mip = 0; mip < shape.levels.size(); ++mip) {
        pixels.resize((pixels.size() + 3) & ~size_t(3), 0);
        VkBufferImageCopy copy{};
        copy.bufferOffset = pixels.size();
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
        copy.imageExtent = {std::max(shape.width >> mip, 1u), std::max(shape.height >> mip, 1u), 1};
        regions.push_back(copy);
        const auto& level = pyramids[layer].levels[mip];
        pixels.insert(pixels.end(), level.begin(), level.end());
    }
    ctx.upload_image_2d(bounds_, pixels.data(), pixels.size(), regions);
    log::info("Cloud coverage bounds: {}x{}, {} levels, {:.0f} ms", shape.width, shape.height, shape.levels.size(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - bound_start).count());
}

void CloudMaps::shutdown(rhi::VkContext& ctx) {
    ctx.destroy_image(atlas_);
    ctx.destroy_image(bounds_);
    layers_.clear();
}

uint32_t CloudMaps::layer(const std::string& name) const noexcept {
    auto found = layers_.find(name);
    return found == layers_.end() ? 0 : found->second;
}

}
