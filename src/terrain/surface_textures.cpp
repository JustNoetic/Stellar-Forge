#include "terrain/surface_textures.hpp"
#include "core/path_utils.hpp"
#include "core/log.hpp"
#include "rhi/shader_cache.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace sf::terrain {
namespace {

constexpr uint32_t PAGE_WORDS = COLOR_BODIES*4 + COLOR_SOURCES*8190;

struct HeightMipPush {
    uint32_t mode = 0;
    uint32_t size = 0;
    float height_scale_m = 1.0f;
    float ocean_floor_m = -1e30f;
};

struct HorizonJobItem {
    uint32_t source = 0;
    uint32_t dst_slot = 0;
    TileId tile{};
    std::vector<float> apron_m;
};

rhi::PushDescriptorWrite sampled_view(uint32_t binding, VkImageView view, VkSampler sampler) {
    rhi::PushDescriptorWrite w{};
    w.binding = binding;
    w.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.image_info = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    return w;
}

rhi::PushDescriptorWrite storage_view(uint32_t binding, VkImageView view) {
    rhi::PushDescriptorWrite w{};
    w.binding = binding;
    w.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.image_info = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    return w;
}

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

template <typename GlobalsContainer>
void destroy_global_state(rhi::VkContext& ctx, GlobalsContainer& state) {
    for (VkImageView view : state.mip_views) {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(ctx.device(), view, nullptr);
    }
    state.mip_views.clear();
    ctx.destroy_image(state.pyramid);
    ctx.destroy_image(state.raw);
}

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
    result->has_specular_ = j.value("has_specular", false);
    if (result->level > 5 || result->domain.identity == 0 || result->domain.revision == 0 ||
        !std::isfinite(result->polar_ratio) || result->polar_ratio < .05 || result->polar_ratio > 1)
        throw std::runtime_error("Invalid albedo domain or shape");
    result->checksums = j.at("checksums").get<std::vector<uint32_t>>();
    uint32_t expected = 2*((1u << (2*(result->level+1)))-1);
    result->payload = directory/"tiles.rgba";
    if (result->count() != expected || std::filesystem::file_size(result->payload) != uint64_t(expected)*COLOR_BYTES)
        throw std::runtime_error("Incomplete albedo tile pack");
    glm::dvec3 sum_rgb(0.0);
    double sum_w = 0.0;
    auto srgb_to_linear = [](uint8_t byte) {
        double x = double(byte)/255.0;
        return x <= 0.04045 ? x/12.92 : std::pow((x+0.055)/1.055, 2.4);
    };
    for (uint32_t face = 0; face < 6; ++face) {
        auto tile = result->read({face, 0, 0, 0});
        for (uint32_t y = 0; y < COLOR_CORE; ++y) {
            double v = -1.0+2.0*(double(y)+0.5)/double(COLOR_CORE);
            for (uint32_t x = 0; x < COLOR_CORE; ++x) {
                double u = -1.0+2.0*(double(x)+0.5)/double(COLOR_CORE);
                double w = std::pow(1.0+u*u+v*v, -1.5);
                size_t offset = ((size_t(y)+COLOR_GUTTER)*COLOR_SIDE+(size_t(x)+COLOR_GUTTER))*4;
                sum_rgb.r += w*srgb_to_linear((*tile)[offset]);
                sum_rgb.g += w*srgb_to_linear((*tile)[offset+1]);
                sum_rgb.b += w*srgb_to_linear((*tile)[offset+2]);
                sum_w += w;
            }
        }
    }
    if (sum_w > 0.0) result->mean_albedo_ = glm::vec3(sum_rgb/sum_w);
    return result;
}

uint32_t SurfaceSource::index(TileId tile) const {
    if (tile.face >= 6 || tile.level > level || tile.x >= (1u << tile.level) || tile.y >= (1u << tile.level))
        throw std::runtime_error("Invalid surface tile address");
    uint32_t n = 1u << tile.level;
    return 2*(n*n-1)+tile.face*n*n+tile.y*n+tile.x;
}

bool SurfaceSource::compatible(double, double polar) const noexcept {
    return std::abs(polar_ratio-polar) <= 1e-10;
}

std::vector<std::byte> ColorSource::load(TileId tile) const {
    auto payload = read(tile);
    auto bytes = std::as_bytes(std::span{*payload});
    return {bytes.begin(), bytes.end()};
}

NormalSource::NormalSource(std::shared_ptr<const HeightSource> source) : height(std::move(source)) {
    domain = height->domain();
    level = height->info().data_level;
    polar_ratio = height->info().polar_ratio;
    height_scale_m = std::max({std::abs(height->info().minimum_m), std::abs(height->info().maximum_m), 1.0});
}

bool NormalSource::compatible(double radius_m, double polar) const noexcept {
    return height->compatible(radius_m, polar);
}

NormalSource::TileWithApron NormalSource::evaluate_tile(TileId tile, bool need_normals, bool need_apron) const {
    index(tile);
    std::unordered_map<uint64_t, HeightTile> payloads;
    auto scale = glm::dvec3(1, polar_ratio, 1);
    double radius_m = height->info().equatorial_radius_m;
    auto sample_height = [&](double u, double v, bool cubic, glm::dvec3* out_dir = nullptr) {
        auto direction = glm::normalize(cube_vector(tile.face, u, v));
        if (out_dir) *out_dir = direction;
        auto at = tile_at(direction, tile.level);
        auto axis = cube_vector(at.face, 0, 0);
        double divisor = glm::dot(direction, axis);
        double fu = glm::dot(direction, cube_vector(at.face, 1, 0)-axis)/divisor;
        double fv = glm::dot(direction, cube_vector(at.face, 0, 1)-axis)/divisor;
        auto found = payloads.find(at.key());
        if (found == payloads.end()) found = payloads.emplace(at.key(), height->read(at)).first;
        double n = double(1u << tile.level);
        double x = ((fu+1)*.5*n-at.x)*GRID_CELLS;
        double y = ((fv+1)*.5*n-at.y)*GRID_CELLS;
        return cubic ? height_sample_cubic(found->second, x, y) : height_sample(found->second, x, y);
    };
    auto point = [&](double u, double v) {
        glm::dvec3 direction{};
        double h = sample_height(u, v, true, &direction) / radius_m;
        return direction*scale + glm::normalize(direction/scale)*h;
    };
    TileWithApron result;
    std::vector<int16_t> values;
    if (need_normals) values.resize(COLOR_SIDE*COLOR_SIDE*4);
    if (need_apron) result.apron_m.resize(HORIZON_APRON_SIDE*HORIZON_APRON_SIDE);
    double width = std::ldexp(2.0, -int(tile.level));
    double step = width/height->info().grid_cells*.25;
    for (uint32_t y = 0; y < COLOR_SIDE; ++y) for (uint32_t x = 0; x < COLOR_SIDE; ++x) {
        double u = -1+(tile.x+(double(x)-COLOR_GUTTER+.5)/COLOR_CORE)*width;
        double v = -1+(tile.y+(double(y)-COLOR_GUTTER+.5)/COLOR_CORE)*width;
        double h_m = sample_height(u, v, true);
        if (need_apron)
            result.apron_m[(y+HORIZON_APRON_PAD)*HORIZON_APRON_SIDE + (x+HORIZON_APRON_PAD)] = float(h_m);
        if (need_normals) {
            auto du = point(u+step, v)-point(u-step, v);
            auto dv = point(u, v+step)-point(u, v-step);
            auto normal = glm::normalize(glm::cross(du, dv));
            for (uint32_t c = 0; c < 3; ++c)
                values[4*(y*COLOR_SIDE+x)+c] = int16_t(std::round(std::clamp(normal[c], -1.0, 1.0)*32767));
            values[4*(y*COLOR_SIDE+x)+3] = int16_t(std::round(std::clamp(h_m/height_scale_m, -1.0, 1.0)*32767));
        }
    }
    if (need_apron) {
        for (uint32_t ay = 0; ay < HORIZON_APRON_SIDE; ++ay) for (uint32_t ax = 0; ax < HORIZON_APRON_SIDE; ++ax) {
            if (ax >= HORIZON_APRON_PAD && ax < HORIZON_APRON_PAD+COLOR_SIDE &&
                ay >= HORIZON_APRON_PAD && ay < HORIZON_APRON_PAD+COLOR_SIDE) continue;
            double u = -1+(tile.x+(double(ax)-HORIZON_APRON_PAD-COLOR_GUTTER+.5)/COLOR_CORE)*width;
            double v = -1+(tile.y+(double(ay)-HORIZON_APRON_PAD-COLOR_GUTTER+.5)/COLOR_CORE)*width;
            result.apron_m[ay*HORIZON_APRON_SIDE+ax] = float(sample_height(u, v, false));
        }
    }
    if (need_normals) {
        auto bytes = std::as_bytes(std::span{values});
        result.normals.assign(bytes.begin(), bytes.end());
    }
    return result;
}

std::vector<std::byte> NormalSource::load(TileId tile) const {
    return evaluate_tile(tile, true, false).normals;
}

NormalSource::TileWithApron NormalSource::load_with_apron(TileId tile) const {
    return evaluate_tile(tile, true, true);
}

std::vector<float> NormalSource::load_apron(TileId tile) const {
    return evaluate_tile(tile, false, true).apron_m;
}

std::vector<float> NormalSource::load_global_face(uint32_t face) const {
    uint32_t glevel = global_level();
    uint32_t n = 1u << glevel;
    uint32_t size = global_face_size();
    std::vector<float> out(size_t(size)*size);
    for (uint32_t ty = 0; ty < n; ++ty) for (uint32_t tx = 0; tx < n; ++tx) {
        HeightTile ht = height->read({face, glevel, tx, ty});
        for (uint32_t ly = 0; ly < COLOR_CORE; ++ly) for (uint32_t lx = 0; lx < COLOR_CORE; ++lx) {
            double gx = (double(lx)+0.5)/double(COLOR_CORE)*GRID_CELLS;
            double gy = (double(ly)+0.5)/double(COLOR_CORE)*GRID_CELLS;
            out[(ty*COLOR_CORE+ly)*size + (tx*COLOR_CORE+lx)] = float(height_sample_cubic(ht, gx, gy));
        }
    }
    return out;
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

void SurfaceTextures::init(rhi::VkContext& ctx, bool normals, bool horizons) {
    normals_ = normals;
    horizons_ = normals && horizons;
    capacity_ = std::min(normals ? COLOR_SLOTS/2 : COLOR_SLOTS,
        horizons_ ? ctx.device_properties().limits.maxImageArrayLayers/HORIZON_LAYERS_PER_SLOT
                  : ctx.device_properties().limits.maxImageArrayLayers);
    slots_ = GeometryResidency{capacity_};
    atlas_ = ctx.create_image_2d(COLOR_SIDE, COLOR_SIDE, normals ? VK_FORMAT_R16G16B16A16_SNORM : VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, capacity_);
    horizon_bounds_ = ctx.create_device_buffer(std::max(capacity_, 1u) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (horizons_) {
        horizon_atlas_ = ctx.create_image_2d(COLOR_SIDE, COLOR_SIDE, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 1, capacity_*HORIZON_LAYERS_PER_SLOT, true);
        apron_scratch_ = ctx.create_image_2d(HORIZON_APRON_SIDE, HORIZON_APRON_SIDE, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, HORIZON_BAKES_PER_FRAME, true);
    }
    ctx.immediate_submit([&](VkCommandBuffer cmd) {
        ctx.transition_image(cmd, atlas_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        vkCmdFillBuffer(cmd, horizon_bounds_.handle, 0, std::max(capacity_, 1u) * sizeof(uint32_t), 255u);
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        if (horizons_) {
            ctx.transition_image(cmd, horizon_atlas_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            ctx.transition_image(cmd, apron_scratch_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    });
    if (horizons_) {
        for (auto& job_buf : horizon_jobs_)
            job_buf = ctx.create_host_buffer(HORIZON_BAKES_PER_FRAME*sizeof(HorizonBakeJobGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        rhi::DescriptorLayoutBinding mip_bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        };
        height_mip_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "terrain/terrain_height_mip.comp",
            mip_bindings, sizeof(HeightMipPush));
        rhi::DescriptorLayoutBinding horizon_bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1},
        };
        horizon_pipeline_ = rhi::ShaderCache::create_compute_pipeline(ctx, "terrain/terrain_horizon.comp",
            horizon_bindings, sizeof(uint32_t));
    }
    std::vector<uint32_t> empty(PAGE_WORDS);
    for (auto& page : pages_) page = ctx.create_host_buffer(empty.size()*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, empty.data());
    staging_.init(ctx, (horizons_ ? 16 : normals ? 8 : 4)*1024*1024);
    words_.resize(COLOR_BODIES*4);
}

void SurfaceTextures::set_color_source(std::string body, std::shared_ptr<const SurfaceSource> source) {
    if (body.empty() && !source) { set_sources({}); return; }
    auto bindings = sources_;
    auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto& binding) { return binding.body == body; });
    if (found != bindings.end() && source && found->source->domain == source->domain) return;
    if (found == bindings.end() && !source) return;
    std::erase_if(bindings, [&](const auto& binding) { return binding.body == body; });
    if (source) bindings.push_back({std::move(body), std::move(source)});
    set_sources(std::move(bindings));
}

void SurfaceTextures::set_height_source(std::string body, std::shared_ptr<const HeightSource> source) {
    if (body.empty() && !source) { set_sources({}); return; }
    auto bindings = sources_;
    auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto& binding) { return binding.body == body; });
    if (found != bindings.end() && source && found->source->domain == source->domain()) return;
    if (found == bindings.end() && !source) return;
    std::erase_if(bindings, [&](const auto& binding) { return binding.body == body; });
    if (source) bindings.push_back({std::move(body), std::make_shared<NormalSource>(std::move(source))});
    set_sources(std::move(bindings));
}

std::shared_ptr<const SurfaceSource> SurfaceTextures::color_source(std::string_view body) const noexcept {
    for (const auto& binding : sources_) {
        if (binding.body == body) return binding.source;
    }
    return nullptr;
}

void SurfaceTextures::shutdown(rhi::VkContext& ctx) {
    worker_.request_stop();
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (auto& state : globals_) destroy_global_state(ctx, state);
    globals_.clear();
    for (auto& retired : retired_globals_) destroy_global_state(ctx, retired);
    retired_globals_.clear();
    rhi::ShaderCache::destroy_pipeline(ctx, horizon_pipeline_);
    rhi::ShaderCache::destroy_pipeline(ctx, height_mip_pipeline_);
    for (auto& job_buf : horizon_jobs_) ctx.destroy_buffer(job_buf);
    ctx.destroy_image(apron_scratch_);
    ctx.destroy_image(horizon_atlas_);
    ctx.destroy_buffer(horizon_bounds_);
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
    for (auto& state : globals_) {
        if (state.raw || state.pyramid || !state.mip_views.empty())
            retired_globals_.push_back({state.raw, state.pyramid, std::move(state.mip_views), 0});
    }
    sources_ = std::move(bindings);
    selections_.assign(sources_.size(), {});
    globals_.assign(sources_.size(), {});
    slots_ = GeometryResidency{capacity_};
    owners_.fill({});
    horizon_valid_.fill(0);
    offsets_.clear();
    words_.assign(COLOR_BODIES*4, 0);
    for (const auto& binding : sources_) {
        offsets_.push_back(uint32_t(words_.size()));
        words_.resize(words_.size()+binding.source->count(), 0);
    }
    queue_.clear(); pending_.clear(); failed_.clear(); ready_.clear();
    rebake_queue_.clear(); rebake_pending_.clear(); apron_ready_.clear();
    global_queue_.clear(); global_pending_ = 0; global_ready_.clear();
    error_.clear(); reported_error_ = false; planning_ = false;
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
        if (entry.contains("albedo")) {
            auto albedo = paths::external_path(entry.at("albedo").get<std::string>());
            if (std::filesystem::exists(albedo/"manifest.json")) bindings.push_back({body, ColorSource::open(albedo)});
            else ++missing;
        }
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
    if (pending_.contains(requested) || failed_.contains(requested) ||
        pending_.size() + rebake_pending_.size() + global_pending_ >= 128 || failed_.size() >= 128) return;
    if (std::any_of(ready_.begin(), ready_.end(), [&](const auto& entry) { return entry.key == requested; })) return;
    pending_.insert(requested);
    queue_.push_back(requested);
    wake_.notify_one();
}

void SurfaceTextures::run(std::stop_token stop) {
    while (!stop.stop_requested()) {
        int work_kind = 0;
        GlobalFaceRequest global_req{};
        GeometryKey tile_req{};
        bool with_apron = false;
        {
            std::unique_lock lock(mutex_);
            if (!wake_.wait(lock, stop, [&] {
                return !planning_ && (global_ready_.size() + apron_ready_.size() + ready_.size() < 64) &&
                       (!global_queue_.empty() || !rebake_queue_.empty() || !queue_.empty());
            })) return;
            if (!global_queue_.empty()) {
                work_kind = 1;
                global_req = global_queue_.front();
                global_queue_.pop_front();
            } else if (!rebake_queue_.empty()) {
                work_kind = 2;
                tile_req = rebake_queue_.front();
                rebake_queue_.pop_front();
            } else {
                work_kind = 3;
                tile_req = queue_.front();
                queue_.pop_front();
                with_apron = horizons_ && horizons_active_.load(std::memory_order_relaxed);
            }
        }
        if (work_kind == 1) {
            std::vector<float> heights;
            std::string error;
            try {
                if (const auto* ns = dynamic_cast<const NormalSource*>(sources_[global_req.source].source.get()))
                    heights = ns->load_global_face(global_req.face);
            } catch (const std::exception& failure) { error = failure.what(); }
            {
                std::lock_guard lock(mutex_);
                if (global_pending_ > 0) --global_pending_;
                if (!heights.empty())
                    global_ready_.push_back({global_req.source, global_req.face, global_req.domain, std::move(heights)});
                else if (error_.empty() && !error.empty()) error_ = std::move(error);
            }
        } else if (work_kind == 2) {
            std::vector<float> apron;
            std::string error;
            try {
                if (const auto* ns = dynamic_cast<const NormalSource*>(sources_[tile_req.body].source.get()))
                    apron = ns->load_apron(TileId::from_key(tile_req.tile));
            } catch (const std::exception& failure) { error = failure.what(); }
            {
                std::lock_guard lock(mutex_);
                rebake_pending_.erase(tile_req);
                if (!apron.empty()) apron_ready_.push_back({tile_req, std::move(apron)});
                else {
                    if (failed_.size() < 128) failed_.insert(tile_req);
                    if (error_.empty() && !error.empty()) error_ = std::move(error);
                }
            }
        } else if (work_kind == 3) {
            NormalSource::TileWithApron result;
            std::string error;
            try {
                if (with_apron) {
                    if (const auto* ns = dynamic_cast<const NormalSource*>(sources_[tile_req.body].source.get()))
                        result = ns->load_with_apron(TileId::from_key(tile_req.tile));
                    else
                        result.normals = sources_[tile_req.body].source->load(TileId::from_key(tile_req.tile));
                } else {
                    result.normals = sources_[tile_req.body].source->load(TileId::from_key(tile_req.tile));
                }
            } catch (const std::exception& failure) { error = failure.what(); }
            {
                std::lock_guard lock(mutex_);
                pending_.erase(tile_req);
                if (!result.normals.empty())
                    ready_.push_back({tile_req, std::move(result.normals), std::move(result.apron_m)});
                else {
                    if (failed_.size() < 128) failed_.insert(tile_req);
                    if (error_.empty()) error_ = std::move(error);
                }
            }
        }
    }
}

void SurfaceTextures::prepare(VkCommandBuffer cmd, rhi::VkContext& ctx, const Camera& camera,
                              const ephemeris::SystemSnapshot& system, VkExtent2D extent, bool enabled,
                              bool horizons_enabled, bool oceans_enabled) {
    std::vector<SurfaceTextureView> views;
    double tan_y = std::tan(camera.fov_deg*DEG_TO_RAD*.5);
    double focal = std::max(1u, extent.height)/(2*tan_y);
    for (uint32_t i = 0; i < std::min<uint32_t>(system.size(), COLOR_BODIES); ++i) {
        auto body = system.body(i);
        auto center = camera_relative(body.render.position, camera.eye_pos_render());
        auto rotation = body_rotation(body.render.pole, body.render.spin_angle_rad);
        SurfaceTextureView input;
        input.definition = &body.definition;
        input.index = i;
        input.radius_pixels = body.definition.radius_au()*focal/std::max(glm::length(center), 1e-15);
        input.view.eye = glm::transpose(rotation)*(-center/body.definition.radius_au());
        input.view.view_from_local = camera.view_rotation_double()*rotation;
        input.view.tan_half_fov_y = tan_y;
        input.view.tan_half_fov_x = tan_y*std::max(1u, extent.width)/std::max(1u, extent.height);
        input.view.focal_pixels = focal;
        input.ocean_floor_m = oceans_enabled && body.definition.ocean.enabled
            ? float(body.definition.ocean.sea_level_km * 1000.0) : -1e30f;
        views.push_back(input);
    }
    prepare_views(cmd, ctx, views, enabled, horizons_enabled);
}

int SurfaceTextures::source_index(const ephemeris::BodyDefinition& body) const {
    for (uint32_t i = 0; i < sources_.size(); ++i)
        if (sources_[i].body == body.name && sources_[i].source->compatible(
            body.radius_au()*AU_TO_KM*1000, std::clamp(1-body.oblateness, .05, 1.0))) return int(i);
    return -1;
}

std::optional<glm::vec3> SurfaceTextures::mean_albedo(const ephemeris::BodyDefinition& body) const noexcept {
    int idx = source_index(body);
    if (idx < 0) return std::nullopt;
    return sources_[idx].source->mean_albedo();
}

void SurfaceTextures::prepare_views(VkCommandBuffer cmd, rhi::VkContext& ctx,
                                    std::span<const SurfaceTextureView> views, bool enabled,
                                    bool horizons_enabled) {
    stats_ = {};
    stats_.sources = uint32_t(sources_.size());
    stats_.capacity = capacity_;
    horizons_active_.store(horizons_ && horizons_enabled, std::memory_order_relaxed);
    std::fill(words_.begin(), words_.begin()+COLOR_BODIES*4, 0);

    for (auto& retired : retired_globals_) {
        if (retired.serial == 0) retired.serial = ctx.recording_serial();
    }
    std::erase_if(retired_globals_, [&](auto& retired) {
        if (retired.serial == 0 || retired.serial > ctx.completed_serial()) return false;
        destroy_global_state(ctx, retired);
        return true;
    });

    bool any_enabled = enabled || (horizons_ && horizons_enabled);
    std::vector<bool> source_has_view(sources_.size(), false);
    std::vector<float> source_ocean_floor(sources_.size(), -1e30f);
    for (const auto& input : views) {
        const auto& body = *input.definition;
        for (uint32_t s = 0; s < sources_.size(); ++s) {
            if (sources_[s].body != body.name || !sources_[s].source->compatible(
                body.radius_au()*AU_TO_KM*1000, std::clamp(1-body.oblateness, .05, 1.0))) continue;
            source_has_view[s] = true;
            source_ocean_floor[s] = input.ocean_floor_m;
        }
    }

    if (horizons_ && horizons_enabled) {
        for (uint32_t s = 0; s < sources_.size(); ++s) {
            if (!source_has_view[s] || globals_[s].enqueued) continue;
            if (dynamic_cast<const NormalSource*>(sources_[s].source.get()) == nullptr) continue;
            std::lock_guard lock(mutex_);
            if (pending_.size() + rebake_pending_.size() + global_pending_ + 6 <= 128) {
                globals_[s].enqueued = true;
                for (uint32_t face = 0; face < 6; ++face)
                    global_queue_.push_back({s, face, sources_[s].source->domain});
                global_pending_ += 6;
                wake_.notify_one();
            }
        }
    }

    std::vector<GeometryKey> protected_keys;
    std::unordered_set<GeometryKey, GeometryKeyHash> wanted;
    auto want = [&](uint32_t source, TileId tile) {
        auto requested = key(source, tile);
        wanted.insert(requested);
        if (slots_.find(requested) >= 0) protected_keys.push_back(requested);
        else request(requested);
    };
    if (any_enabled) {
        for (uint32_t s = 0; s < sources_.size(); ++s) for (uint32_t face = 0; face < 6; ++face) want(s, {face});
        std::vector<std::pair<double, size_t>> visible_bodies;
        for (size_t i = 0; i < views.size(); ++i)
            if (views[i].radius_pixels >= 2) visible_bodies.emplace_back(-views[i].radius_pixels, i);
        std::sort(visible_bodies.begin(), visible_bodies.end());
        uint32_t remaining = capacity_-std::min(capacity_, uint32_t(sources_.size())*6);
        for (auto [priority, body_index] : visible_bodies) {
            const auto& input = views[body_index];
            const auto& body = *input.definition;
            for (uint32_t s = 0; s < sources_.size(); ++s) {
                if (sources_[s].body != body.name || !sources_[s].source->compatible(
                    body.radius_au()*AU_TO_KM*1000, std::clamp(1-body.oblateness, .05, 1.0))) continue;
                SelectionView view = input.view;
                view.target_error_pixels = 1.0;
                view.polar_ratio = sources_[s].source->polar_ratio;
                view.height_amplitude = .0015;
                view.max_level = sources_[s].source->level;
                view.max_patches = std::max(6u, std::min(normals_ ? remaining/2 : remaining, 256u));
                auto& entry = selections_[s];
                if (!entry.valid || !(entry.view == view)) {
                    entry.selection = select(view, TextureError{}, {}, &selection_scratch_);
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
    std::optional<GlobalFaceReady> global_upload;
    std::vector<ApronReady> ready_aprons;
    {
        std::lock_guard lock(mutex_);
        planning_ = true;
        for (auto& entry : ready_) if (wanted.contains(entry.key) && slots_.find(entry.key) < 0) ready.push_back(std::move(entry));
        ready_.clear();
        if (horizons_ && horizons_enabled && !global_ready_.empty()) {
            global_upload = std::move(global_ready_.front());
            global_ready_.erase(global_ready_.begin());
        }
        if (horizons_ && horizons_enabled && !apron_ready_.empty()) {
            ready_aprons = std::move(apron_ready_);
            apron_ready_.clear();
        }
    }

    if (global_upload && global_upload->source < sources_.size() &&
        sources_[global_upload->source].source->domain == global_upload->domain) {
        if (const auto* ns = dynamic_cast<const NormalSource*>(sources_[global_upload->source].source.get())) {
            auto& state = globals_[global_upload->source];
            uint32_t face_size = ns->global_face_size();
            uint32_t mip_levels = std::bit_width(face_size);
            if (!state.raw) {
                state.raw = ctx.create_image_2d(face_size, face_size, VK_FORMAT_R32_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, 6, true);
                state.pyramid = ctx.create_image_2d(face_size, face_size, VK_FORMAT_R16G16B16A16_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, mip_levels, 6, true);
                state.mip_views.resize(mip_levels);
                for (uint32_t m = 0; m < mip_levels; ++m) {
                    VkImageViewCreateInfo iv_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                    iv_ci.image = state.pyramid.handle;
                    iv_ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                    iv_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
                    iv_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 6};
                    VK_CHECK(vkCreateImageView(ctx.device(), &iv_ci, nullptr, &state.mip_views[m]));
                }
                ctx.transition_image_range(cmd, state.raw, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6},
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
            VkImageSubresourceRange face_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, global_upload->face, 1};
            ctx.transition_image_range(cmd, state.raw, face_range,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, global_upload->face, 1};
            copy.imageExtent = {face_size, face_size, 1};
            if (!staging_.image(cmd, ctx, state.raw, copy, std::as_bytes(std::span{global_upload->heights_m})))
                throw std::logic_error("Global height staging budget exhausted");
            ctx.transition_image_range(cmd, state.raw, face_range,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            state.uploaded_mask |= 1u << global_upload->face;
            ++stats_.global_height_uploads;
        }
    }

    uint32_t horizon_timer = UINT32_MAX;
    auto ensure_horizon_timer = [&] {
        if (horizon_timer == UINT32_MAX) horizon_timer = ctx.begin_gpu_timer(cmd, "gpu_terrain_horizons");
    };

    if (horizons_ && horizons_enabled) {
        for (uint32_t s = 0; s < sources_.size(); ++s) {
            auto& state = globals_[s];
            if (state.uploaded_mask != 0x3Fu) continue;
            const auto* ns = dynamic_cast<const NormalSource*>(sources_[s].source.get());
            if (!ns) continue;
            float target_floor = source_has_view[s] ? source_ocean_floor[s]
                : (state.pyramid_built ? state.built_ocean_floor_m : -1e30f);
            if (!state.pyramid_built || state.built_ocean_floor_m != target_floor) {
                ensure_horizon_timer();
                uint32_t face_size = ns->global_face_size();
                uint32_t mip_levels = uint32_t(state.mip_views.size());
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, height_mip_pipeline_.handle);
                for (uint32_t m = 0; m < mip_levels; ++m) {
                    uint32_t mip_size = std::max(1u, face_size >> m);
                    VkImageSubresourceRange mip_range{VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 6};
                    ctx.transition_image_range(cmd, state.pyramid, mip_range,
                        state.pyramid_initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_GENERAL,
                        state.pyramid_initialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
                        state.pyramid_initialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                    HeightMipPush push{m == 0 ? 0u : 1u, mip_size, float(ns->height_scale_m), target_floor};
                    VkImageView src = m == 0 ? state.raw.view : state.mip_views[m-1];
                    ctx.push_descriptors(cmd, height_mip_pipeline_, {
                        sampled_view(0, src, ctx.sampler_nearest_clamp()),
                        storage_view(1, state.mip_views[m]),
                    });
                    ctx.push_constants(cmd, height_mip_pipeline_, push);
                    vkCmdDispatch(cmd, (mip_size + 7) / 8, (mip_size + 7) / 8, 6);
                    ctx.transition_image_range(cmd, state.pyramid, mip_range,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                }
                if (state.pyramid_built) {
                    for (uint32_t slot = 0; slot < capacity_; ++slot) {
                        if (owners_[slot] && owners_[slot]->body == s) horizon_valid_[slot] = 0;
                    }
                    std::erase_if(ready_aprons, [&](const ApronReady& apron) { return apron.key.body == s; });
                }
                state.pyramid_initialized = true;
                state.pyramid_built = true;
                state.built_ocean_floor_m = target_floor;
            }
        }
    }

    std::sort(ready.begin(), ready.end(), [](const auto& a, const auto& b) {
        auto x = TileId::from_key(a.key.tile), y = TileId::from_key(b.key.tile);
        return x.level != y.level ? x.level < y.level : a.key.body != b.key.body ? a.key.body < b.key.body : x.key() < y.key();
    });
    auto keys = protected_keys;
    for (const auto& entry : ready) keys.push_back(entry.key);
    auto plan = slots_.request(keys, COLOR_UPLOADS);
    std::vector<HorizonJobItem> bake_batch;
    for (const auto& bake : plan.bakes) {
        auto& entry = ready[bake.request-protected_keys.size()];
        uint32_t slot = bake.slot;
        if (owners_[slot]) {
            ++stats_.evictions;
            auto owner = *owners_[slot];
            words_[offsets_[owner.body]+sources_[owner.body].source->index(TileId::from_key(owner.tile))] = 0;
        }
        horizon_valid_[slot] = 0;
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, slot, 1};
        ctx.transition_image_range(cmd, atlas_, range, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, slot, 1};
        copy.imageExtent = {COLOR_SIDE, COLOR_SIDE, 1};
        if (!staging_.image(cmd, ctx, atlas_, copy, entry.payload)) throw std::logic_error("Surface staging budget exhausted");
        ctx.transition_image_range(cmd, atlas_, range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        owners_[slot] = entry.key;
        TileId tile = TileId::from_key(entry.key.tile);
        words_[offsets_[entry.key.body]+sources_[entry.key.body].source->index(tile)] = slot+1;
        if (horizons_ && horizons_enabled && globals_[entry.key.body].pyramid_built &&
            !entry.apron_m.empty() && bake_batch.size() < HORIZON_BAKES_PER_FRAME) {
            bake_batch.push_back({entry.key.body, slot, tile, std::move(entry.apron_m)});
        }
        ++stats_.uploads;
    }

    std::vector<ApronReady> deferred_aprons;
    if (horizons_ && horizons_enabled) {
        for (auto& apron : ready_aprons) {
            int32_t slot = slots_.find(apron.key);
            if (slot < 0 || owners_[slot] != apron.key || horizon_valid_[slot] != 0) continue;
            if (globals_[apron.key.body].pyramid_built && bake_batch.size() < HORIZON_BAKES_PER_FRAME) {
                bake_batch.push_back({apron.key.body, uint32_t(slot), TileId::from_key(apron.key.tile), std::move(apron.apron_m)});
            } else {
                deferred_aprons.push_back(std::move(apron));
            }
        }
        for (uint32_t slot = 0; slot < capacity_; ++slot) {
            if (!owners_[slot] || horizon_valid_[slot] != 0) continue;
            const auto& k = *owners_[slot];
            if (!globals_[k.body].pyramid_built) continue;
            if (std::any_of(bake_batch.begin(), bake_batch.end(), [&](const HorizonJobItem& item) { return item.dst_slot == slot; }))
                continue;
            if (std::any_of(deferred_aprons.begin(), deferred_aprons.end(), [&](const ApronReady& item) { return item.key == k; }))
                continue;
            std::lock_guard lock(mutex_);
            if (rebake_pending_.contains(k) || failed_.contains(k)) continue;
            if (pending_.size() + rebake_pending_.size() + global_pending_ >= 128) continue;
            rebake_pending_.insert(k);
            rebake_queue_.push_back(k);
        }
    }

    if (!bake_batch.empty()) {
        ensure_horizon_timer();
        std::sort(bake_batch.begin(), bake_batch.end(), [](const auto& a, const auto& b) {
            return a.source != b.source ? a.source < b.source : a.dst_slot < b.dst_slot;
        });
        uint32_t count = uint32_t(bake_batch.size());
        ctx.transition_image_range(cmd, apron_scratch_, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, count},
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        auto* gpu_jobs = static_cast<HorizonBakeJobGPU*>(horizon_jobs_[ctx.frame_index()].mapped);
        for (uint32_t j = 0; j < count; ++j) {
            const auto& item = bake_batch[j];
            const auto* ns = static_cast<const NormalSource*>(sources_[item.source].source.get());
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, j, 1};
            copy.imageExtent = {HORIZON_APRON_SIDE, HORIZON_APRON_SIDE, 1};
            if (!staging_.image(cmd, ctx, apron_scratch_, copy, std::as_bytes(std::span{item.apron_m})))
                throw std::logic_error("Horizon apron staging budget exhausted");
            HorizonBakeJobGPU job;
            job.address = {item.tile.face, item.tile.level, item.tile.x, item.tile.y};
            job.target = {item.dst_slot, j, ns->global_face_size(), 0u};
            job.metrics = {float(ns->height->info().equatorial_radius_m), float(ns->polar_ratio),
                float(ns->height_scale_m), globals_[item.source].built_ocean_floor_m};
            job.bounds = {float(ns->height->info().minimum_m), float(ns->height->info().maximum_m),
                float(std::bit_width(ns->global_face_size()) - 1u), 0.0f};
            gpu_jobs[j] = job;
        }
        ctx.flush_buffer(horizon_jobs_[ctx.frame_index()], 0, count * sizeof(HorizonBakeJobGPU));
        ctx.transition_image_range(cmd, apron_scratch_, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, count},
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        for (const auto& item : bake_batch) {
            ctx.transition_image_range(cmd, horizon_atlas_,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, item.dst_slot * HORIZON_LAYERS_PER_SLOT, HORIZON_LAYERS_PER_SLOT},
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        for (const auto& item : bake_batch) {
            vkCmdFillBuffer(cmd, horizon_bounds_.handle, item.dst_slot * sizeof(uint32_t), sizeof(uint32_t), 0u);
        }
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, horizon_pipeline_.handle);
        for (uint32_t offset = 0; offset < count;) {
            uint32_t src = bake_batch[offset].source;
            uint32_t end = offset + 1;
            while (end < count && bake_batch[end].source == src) ++end;
            uint32_t slice = end - offset;
            ctx.push_descriptors(cmd, horizon_pipeline_, {
                rhi::PushDescriptorWrite::storage_buffer(0, horizon_jobs_[ctx.frame_index()]),
                rhi::PushDescriptorWrite::sampled_image(1, apron_scratch_, ctx.sampler_nearest_clamp()),
                rhi::PushDescriptorWrite::sampled_image(2, globals_[src].pyramid, ctx.sampler_linear_clamp()),
                rhi::PushDescriptorWrite::storage_image(3, horizon_atlas_),
                rhi::PushDescriptorWrite::storage_buffer(4, horizon_bounds_),
            });
            ctx.push_constants(cmd, horizon_pipeline_, offset);
            vkCmdDispatch(cmd, (COLOR_SIDE + 7) / 8, (COLOR_SIDE + 7) / 8, slice);
            offset = end;
        }
        for (const auto& item : bake_batch) {
            ctx.transition_image_range(cmd, horizon_atlas_,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, item.dst_slot * HORIZON_LAYERS_PER_SLOT, HORIZON_LAYERS_PER_SLOT},
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            horizon_valid_[item.dst_slot] = 1;
        }
        ctx.memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        stats_.horizon_bakes = count;
    }

    if (horizon_timer != UINT32_MAX) ctx.end_gpu_timer(cmd, horizon_timer);

    {
        std::lock_guard lock(mutex_);
        for (size_t i = 0; i < ready.size(); ++i) if (plan.slots[protected_keys.size()+i] < 0) ready_.push_back(std::move(ready[i]));
        for (auto& apron : deferred_aprons) apron_ready_.push_back(std::move(apron));
        planning_ = false;
        stats_.pending = uint32_t(pending_.size() + rebake_pending_.size() + global_pending_);
        stats_.ready = uint32_t(ready_.size() + apron_ready_.size() + global_ready_.size());
        stats_.failed = uint32_t(failed_.size());
        if (!error_.empty() && !reported_error_) { log::error("Surface texture streaming: {}", error_); reported_error_ = true; }
    }
    wake_.notify_one();
    stats_.resident = slots_.size();
    if (any_enabled) {
        for (const auto& input : views) {
            uint32_t i = input.index;
            if (i >= COLOR_BODIES) continue;
            const auto& body = *input.definition;
            for (uint32_t s = 0; s < sources_.size(); ++s) {
                if (sources_[s].body != body.name || !sources_[s].source->compatible(
                    body.radius_au()*AU_TO_KM*1000, std::clamp(1-body.oblateness, .05, 1.0))) continue;
                bool roots = true;
                for (uint32_t face = 0; face < 6; ++face) roots &= words_[offsets_[s]+face] != 0;
                bool all_horizons_ready = horizons_ && horizons_enabled && globals_[s].pyramid_built && roots;
                if (all_horizons_ready) {
                    for (uint32_t slot = 0; slot < capacity_; ++slot) {
                        if (owners_[slot] && owners_[slot]->body == s && !horizon_valid_[slot]) {
                            all_horizons_ready = false;
                            break;
                        }
                    }
                }
                uint32_t flags = (enabled ? 1u : 0u)
                    | ((all_horizons_ready || (!normals_ && enabled && sources_[s].source->has_specular_mask())) ? 2u : 0u);
                if (roots && flags != 0u) {
                    words_[i*4] = offsets_[s];
                    words_[i*4+1] = sources_[s].source->level;
                    words_[i*4+2] = flags;
                }
            }
        }
    }
    auto& page = pages_[ctx.frame_index()];
    std::memcpy(page.mapped, words_.data(), words_.size()*sizeof(uint32_t));
    ctx.flush_buffer(page, 0, words_.size()*sizeof(uint32_t));
}

}
