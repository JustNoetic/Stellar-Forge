#include "terrain/surface_textures.hpp"
#include "core/path_utils.hpp"
#include "core/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace sf::terrain {
namespace {

constexpr uint32_t PAGE_WORDS = COLOR_BODIES*4 + COLOR_SOURCES*8190;

uint32_t crc32(std::span<const uint8_t> bytes) {
    static const auto table = [] {
        std::array<uint32_t, 256> result{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) value = (value >> 1) ^ ((value & 1) ? 0xedb88320u : 0u);
            result[i] = value;
        }
        return result;
    }();
    uint32_t value = 0xffffffffu;
    for (uint8_t byte : bytes) value = table[(value ^ byte) & 255u] ^ (value >> 8);
    return value ^ 0xffffffffu;
}

class TextureError final : public TerrainSource {
public:
    SourceDomain domain() const noexcept override { return {}; }
    SurfaceTile describe(TileId tile, double polar, double amplitude) const override {
        return {bounds(tile, polar, amplitude), std::ldexp(4.0/COLOR_CORE, -int(tile.level))};
    }
};

}

std::shared_ptr<const ColorSource> ColorSource::open(const std::filesystem::path& directory) {
    if (std::filesystem::file_size(directory/"manifest.json") > 2*1024*1024) throw std::runtime_error("Albedo manifest exceeds its index budget");
    auto document = paths::read_json(directory/"manifest.json");
    if (!document) throw std::runtime_error("Cannot read albedo manifest");
    const auto& j = *document;
    if (j.at("version") != 1 || j.at("encoding") != "rgba8-srgb" || j.at("core") != COLOR_CORE ||
        j.at("gutter") != COLOR_GUTTER || j.at("projection") != "engine-oblate-cube-v1" ||
        j.at("orientation") != "x-east-y-north-z-prime") throw std::runtime_error("Unsupported albedo tile format");
    auto result = std::make_shared<ColorSource>();
    result->domain = {j.at("identity"), j.at("revision")};
    result->level = j.at("data_level");
    result->polar_ratio = j.at("polar_ratio");
    if (result->level > 5 || result->domain.identity == 0 || result->domain.revision == 0 ||
        !std::isfinite(result->polar_ratio) || result->polar_ratio < .05 || result->polar_ratio > 1)
        throw std::runtime_error("Invalid albedo domain or shape");
    result->checksums = j.at("checksums").get<std::vector<uint32_t>>();
    uint32_t expected = 2*((1u << (2*(result->level+1)))-1);
    result->payload = directory/"tiles.rgba";
    if (result->count() != expected || std::filesystem::file_size(result->payload) != uint64_t(expected)*COLOR_BYTES)
        throw std::runtime_error("Incomplete albedo tile pack");
    return result;
}

uint32_t ColorSource::index(TileId tile) const {
    if (tile.face >= 6 || tile.level > level || tile.x >= (1u << tile.level) || tile.y >= (1u << tile.level))
        throw std::runtime_error("Invalid albedo tile address");
    uint32_t n = 1u << tile.level;
    return 2*(n*n-1)+tile.face*n*n+tile.y*n+tile.x;
}

std::shared_ptr<const ColorTile> ColorSource::read(TileId tile) const {
    auto i = index(tile);
    auto bytes = std::make_shared<ColorTile>();
    std::ifstream stream(payload, std::ios::binary);
    stream.seekg(std::streamoff(uint64_t(i)*COLOR_BYTES));
    stream.read(reinterpret_cast<char*>(bytes->data()), bytes->size());
    if (!stream || crc32(*bytes) != checksums[i]) throw std::runtime_error("Albedo tile is truncated or corrupt");
    return bytes;
}

void SurfaceTextures::init(rhi::VkContext& ctx) {
    capacity_ = std::min(COLOR_SLOTS, ctx.device_properties().limits.maxImageArrayLayers);
    slots_ = GeometryResidency{capacity_};
    atlas_ = ctx.create_image_2d(COLOR_SIDE, COLOR_SIDE, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, capacity_);
    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        ctx.transition_image(cmd, atlas_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    });
    std::vector<uint32_t> empty(PAGE_WORDS);
    for (auto& page : pages_) page = ctx.create_host_buffer(empty.size()*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, empty.data());
    staging_.init(ctx, 4*1024*1024);
    words_.resize(COLOR_BODIES*4);
}

void SurfaceTextures::shutdown(rhi::VkContext& ctx) {
    worker_.request_stop();
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    staging_.shutdown(ctx);
    for (auto& page : pages_) ctx.destroy_buffer(page);
    ctx.destroy_image(atlas_);
}

void SurfaceTextures::set_sources(std::vector<ColorBinding> bindings) {
    if (bindings.size() > COLOR_SOURCES) throw std::runtime_error("Surface catalog exceeds 32 sources");
    std::unordered_set<std::string> names;
    for (const auto& binding : bindings)
        if (!binding.source || binding.body.empty() || !names.insert(binding.body).second)
            throw std::runtime_error("Surface bindings require unique body names and valid sources");
    worker_.request_stop();
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    sources_ = std::move(bindings);
    selections_.assign(sources_.size(), {});
    slots_ = GeometryResidency{capacity_};
    owners_.fill({});
    offsets_.clear();
    words_.assign(COLOR_BODIES*4, 0);
    for (const auto& binding : sources_) {
        offsets_.push_back(uint32_t(words_.size()));
        words_.resize(words_.size()+binding.source->count(), 0);
    }
    queue_.clear(); pending_.clear(); failed_.clear(); ready_.clear(); error_.clear(); reported_error_ = false; planning_ = false;
    if (!sources_.empty()) worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

std::vector<std::pair<std::string, std::filesystem::path>> SurfaceTextures::load_catalog(const std::filesystem::path& path) {
    std::vector<std::pair<std::string, std::filesystem::path>> heights;
    if (!std::filesystem::exists(path)) return heights;
    auto j = paths::read_json(path);
    if (!j || j->at("version") != 1 || j->at("bodies").size() > COLOR_SOURCES) throw std::runtime_error("Invalid surface asset catalog");
    std::vector<ColorBinding> bindings;
    uint32_t missing = 0;
    for (const auto& entry : j->at("bodies")) {
        std::string body = entry.at("body");
        auto albedo = paths::external_path(entry.at("albedo").get<std::string>());
        if (std::filesystem::exists(albedo/"manifest.json")) bindings.push_back({body, ColorSource::open(albedo)});
        else ++missing;
        if (entry.contains("height")) {
            auto height = paths::external_path(entry.at("height").get<std::string>());
            if (std::filesystem::exists(height/"manifest.json")) heights.emplace_back(body, height);
        }
    }
    set_sources(std::move(bindings));
    log::info("Surface assets: {} color sources, {} elevation sources, {} missing packs", sources_.size(), heights.size(), missing);
    return heights;
}

GeometryKey SurfaceTextures::key(uint32_t source, TileId tile) const {
    GeometryKey result;
    result.body = source;
    result.source = sources_[source].source->domain;
    result.tile = tile.key();
    return result;
}

void SurfaceTextures::request(const GeometryKey& requested) {
    if (slots_.find(requested) >= 0) return;
    std::lock_guard lock(mutex_);
    if (pending_.contains(requested) || failed_.contains(requested) || pending_.size() >= 128 || failed_.size() >= 128) return;
    if (std::any_of(ready_.begin(), ready_.end(), [&](const auto& entry) { return entry.key == requested; })) return;
    pending_.insert(requested);
    queue_.push_back(requested);
    wake_.notify_one();
}

void SurfaceTextures::run(std::stop_token stop) {
    while (!stop.stop_requested()) {
        GeometryKey requested;
        {
            std::unique_lock lock(mutex_);
            if (!wake_.wait(lock, stop, [&] { return !queue_.empty() && ready_.size() < 64 && !planning_; })) return;
            requested = queue_.front();
            queue_.pop_front();
        }
        std::shared_ptr<const ColorTile> data;
        std::string error;
        try { data = sources_[requested.body].source->read(TileId::from_key(requested.tile)); }
        catch (const std::exception& failure) { error = failure.what(); }
        {
            std::lock_guard lock(mutex_);
            pending_.erase(requested);
            if (data) ready_.push_back({requested, std::move(data)});
            else { if (failed_.size() < 128) failed_.insert(requested); if (error_.empty()) error_ = std::move(error); }
        }
    }
}

void SurfaceTextures::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                              const ephemeris::SystemSnapshot& system, VkExtent2D extent, bool enabled) {
    stats_ = {};
    stats_.sources = uint32_t(sources_.size());
    stats_.capacity = capacity_;
    std::fill(words_.begin(), words_.begin()+COLOR_BODIES*4, 0);
    std::vector<GeometryKey> protected_keys;
    std::unordered_set<GeometryKey, GeometryKeyHash> wanted;
    auto want = [&](uint32_t source, TileId tile) {
        auto requested = key(source, tile);
        wanted.insert(requested);
        if (slots_.find(requested) >= 0) protected_keys.push_back(requested);
        else request(requested);
    };
    if (enabled) {
        for (uint32_t s = 0; s < sources_.size(); ++s) for (uint32_t face = 0; face < 6; ++face) want(s, {face});
        double tan_y = std::tan(camera.fov_deg*DEG_TO_RAD*.5);
        double focal = std::max(1u, extent.height)/(2*tan_y);
        std::vector<std::pair<double, uint32_t>> visible_bodies;
        for (uint32_t i = 0; i < std::min<uint32_t>(system.size(), COLOR_BODIES); ++i) {
            auto body = system.body(i);
            double distance = glm::length(camera_relative(body.render.position, camera.eye_pos_render()));
            double pixels = body.definition.radius_au()*focal/std::max(distance, 1e-15);
            if (pixels >= 2) visible_bodies.emplace_back(-pixels, i);
        }
        std::sort(visible_bodies.begin(), visible_bodies.end());
        uint32_t remaining = capacity_-uint32_t(sources_.size())*6;
        for (auto [priority, body_index] : visible_bodies) {
            auto body = system.body(body_index);
            for (uint32_t s = 0; s < sources_.size(); ++s) {
                if (sources_[s].body != body.definition.name ||
                    std::abs(sources_[s].source->polar_ratio-std::clamp(1-body.definition.oblateness, .05, 1.0)) > 1e-10) continue;
                SelectionView view;
                auto rotation = body_rotation(body.render.pole, body.render.spin_angle_rad);
                auto center = camera_relative(body.render.position, camera.eye_pos_render());
                view.eye = glm::transpose(rotation)*(-center/body.definition.radius_au());
                view.view_from_local = camera.view_rotation_double()*rotation;
                view.tan_half_fov_y = tan_y;
                view.tan_half_fov_x = tan_y*std::max(1u, extent.width)/std::max(1u, extent.height);
                view.focal_pixels = focal;
                view.target_error_pixels = 1.0;
                view.polar_ratio = sources_[s].source->polar_ratio;
                view.height_amplitude = .01;
                view.max_level = sources_[s].source->level;
                view.max_patches = std::max(6u, std::min(remaining, 256u));
                auto& entry = selections_[s];
                if (!entry.valid || !(entry.view == view)) {
                    entry.selection = select(view, TextureError{});
                    entry.view = view;
                    entry.valid = true;
                    ++stats_.selection_rebuilds;
                    stats_.selection_nodes += entry.selection.visited;
                }
                const auto& selection = entry.selection;
                remaining -= std::min(remaining, uint32_t(selection.patches.size()));
                for (const auto& patch : selection.patches) {
                    auto tile = patch.tile;
                    bool fallback_pinned = false;
                    for (;;) {
                        auto requested = key(s, tile);
                        wanted.insert(requested);
                        if (!fallback_pinned && slots_.find(requested) >= 0) { protected_keys.push_back(requested); fallback_pinned = true; }
                        else if (slots_.find(requested) < 0) request(requested);
                        if (!tile.level) break;
                        tile = tile.parent();
                    }
                }
            }
        }
    }
    std::vector<Completed> ready;
    {
        std::lock_guard lock(mutex_);
        planning_ = true;
        for (auto& entry : ready_) if (wanted.contains(entry.key) && slots_.find(entry.key) < 0) ready.push_back(std::move(entry));
        ready_.clear();
    }
    std::sort(ready.begin(), ready.end(), [](const auto& a, const auto& b) {
        auto x = TileId::from_key(a.key.tile), y = TileId::from_key(b.key.tile);
        return x.level != y.level ? x.level < y.level : a.key.body != b.key.body ? a.key.body < b.key.body : x.key() < y.key();
    });
    auto keys = protected_keys;
    for (const auto& entry : ready) keys.push_back(entry.key);
    auto plan = slots_.request(keys, COLOR_UPLOADS);
    for (const auto& bake : plan.bakes) {
        const auto& entry = ready[bake.request-protected_keys.size()];
        uint32_t slot = bake.slot;
        if (owners_[slot]) {
            ++stats_.evictions;
            auto owner = *owners_[slot];
            words_[offsets_[owner.body]+sources_[owner.body].source->index(TileId::from_key(owner.tile))] = 0;
        }
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, slot, 1};
        ctx.transition_image_range(cmd, atlas_, range, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, slot, 1};
        copy.imageExtent = {COLOR_SIDE, COLOR_SIDE, 1};
        if (!staging_.image(cmd, ctx, atlas_, copy, std::as_bytes(std::span{*entry.payload}))) throw std::logic_error("Albedo staging budget exhausted");
        ctx.transition_image_range(cmd, atlas_, range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        owners_[slot] = entry.key;
        words_[offsets_[entry.key.body]+sources_[entry.key.body].source->index(TileId::from_key(entry.key.tile))] = slot+1;
        ++stats_.uploads;
    }
    {
        std::lock_guard lock(mutex_);
        for (size_t i = 0; i < ready.size(); ++i) if (plan.slots[protected_keys.size()+i] < 0) ready_.push_back(std::move(ready[i]));
        planning_ = false;
        stats_.pending = uint32_t(pending_.size()); stats_.ready = uint32_t(ready_.size()); stats_.failed = uint32_t(failed_.size());
        if (!error_.empty() && !reported_error_) { log::error("Surface texture streaming: {}", error_); reported_error_ = true; }
    }
    wake_.notify_one();
    stats_.resident = slots_.size();
    if (enabled) {
        for (uint32_t i = 0; i < std::min<uint32_t>(system.size(), COLOR_BODIES); ++i) {
            auto body = system.body(i);
            for (uint32_t s = 0; s < sources_.size(); ++s) {
                if (sources_[s].body != body.definition.name || std::abs(sources_[s].source->polar_ratio-std::clamp(1-body.definition.oblateness, .05, 1.0)) > 1e-10) continue;
                bool roots = true;
                for (uint32_t face = 0; face < 6; ++face) roots &= words_[offsets_[s]+face] != 0;
                if (roots) { words_[i*4] = offsets_[s]; words_[i*4+1] = sources_[s].source->level; }
            }
        }
    }
    auto& page = pages_[ctx.frame_index()];
    std::memcpy(page.mapped, words_.data(), words_.size()*sizeof(uint32_t));
    ctx.flush_buffer(page, 0, words_.size()*sizeof(uint32_t));
}

}
