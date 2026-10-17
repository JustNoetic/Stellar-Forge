#include "terrain/terrain_source.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace sf::terrain {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double curvature(double polar, double height, double gradient, double q2 = 1, double normal_q2 = 1) {
    return .5/q2 + .5*height/(polar*polar*normal_q2) + gradient/(polar*std::sqrt(normal_q2));
}

}

SurfaceTile ProceduralSource::describe(TileId tile, double polar, double amplitude) const {
    TileMetadata local;
    if (amplitude > 0.0) local = cache_ ? cache_->get(tile) : tile_metadata(tile);
    double step = std::ldexp(2.0 / GRID_CELLS, -int(tile.level));
    return {amplitude > 0.0 ? surface_bounds(tile, polar, amplitude, local) : bounds(tile, polar, 0.0),
            surface_error_coefficient(polar, amplitude, local) * step * step};
}

std::shared_ptr<HeightSource> HeightSource::open(const std::filesystem::path& directory) {
    require(std::filesystem::file_size(directory / "manifest.json") <= 16 * 1024 * 1024, "Terrain manifest exceeds the bounded index size");
    auto json = paths::read_json(directory / "manifest.json");
    require(json.has_value(), "Cannot read terrain manifest");
    const auto& j = *json;
    require(j.at("version") == 1 && j.at("encoding") == "float32-le" &&
            j.at("reconstruction") == "nested-triangles" && j.at("grid_cells") == GRID_CELLS &&
            j.at("gutter") == 1 && j.at("height_units") == "m" &&
            j.at("scale") == 1 && j.at("offset") == 0 && j.at("coverage") == "global-filled" &&
            j.at("vertical_datum") == "ellipsoid" && j.at("projection") == "engine-oblate-cube-v1" &&
            j.at("orientation") == "x-east-y-north-z-prime" && j.at("latitude") == "geodetic" &&
            j.at("missing_policy") == "fill", "Unsupported terrain format or geographic convention");
    require(std::endian::native == std::endian::little, "Terrain FP32 payload requires a little-endian host");
    auto source = std::make_shared<HeightSource>();
    auto& info = source->info_;
    info.domain = {j.at("identity").get<uint64_t>(), j.at("revision").get<uint64_t>()};
    info.name = j.at("name").get<std::string>();
    info.equatorial_radius_m = j.at("equatorial_radius_m");
    info.polar_ratio = j.at("polar_ratio");
    info.prime_meridian_rad = j.at("prime_meridian_rad");
    info.encoding_error_m = j.at("encoding_error_m");
    info.data_level = j.at("data_level");
    require(info.domain.identity != 0 && info.domain.revision != 0 && info.data_level <= 5 &&
            std::isfinite(info.equatorial_radius_m) && info.equatorial_radius_m > 0 &&
            std::isfinite(info.polar_ratio) && info.polar_ratio >= .05 && info.polar_ratio <= 1 &&
            std::isfinite(info.prime_meridian_rad) && std::isfinite(info.encoding_error_m) && info.encoding_error_m >= 0,
            "Invalid terrain source domain, shape or precision");
    size_t expected = 0;
    for (uint32_t level = 0; level <= info.data_level; ++level) {
        source->level_offsets_[level] = expected;
        expected += size_t(6) << (2 * level);
    }
    const auto& records = j.at("tiles");
    require(expected <= MAX_SOURCE_RECORDS && records.size() == expected, "Terrain index is incomplete or exceeds its memory limit");
    source->records_.resize(expected);
    std::vector<bool> seen(expected);
    constexpr uint64_t block = sizeof(HeightTile);
    source->payload_ = directory / "heights.bin";
    require(std::filesystem::file_size(source->payload_) == expected * block, "Terrain payload size does not match index");
    for (const auto& entry : records) {
        HeightRecord r;
        r.tile = {entry.at(0), entry.at(1), entry.at(2), entry.at(3)};
        require(r.tile.face < 6 && r.tile.level <= info.data_level &&
                r.tile.x < (1u << r.tile.level) && r.tile.y < (1u << r.tile.level), "Invalid terrain tile address");
        size_t n = size_t(1) << r.tile.level;
        size_t index = source->level_offsets_[r.tile.level] + r.tile.face * n * n + r.tile.y * n + r.tile.x;
        require(!seen[index], "Duplicate terrain tile address");
        seen[index] = true;
        r.offset = entry.at(4);
        r.checksum = entry.at(5);
        r.minimum_m = entry.at(6);
        r.maximum_m = entry.at(7);
        r.height_error_m = entry.at(8);
        r.gradient_m = entry.at(9);
        require(r.offset == index * block && std::isfinite(r.minimum_m) && std::isfinite(r.maximum_m) &&
                r.minimum_m <= r.maximum_m && std::isfinite(r.height_error_m) && r.height_error_m >= 0 &&
                std::isfinite(r.gradient_m) && r.gradient_m >= 0 &&
                std::max(std::abs(r.minimum_m), std::abs(r.maximum_m)) < info.equatorial_radius_m * .01,
                "Invalid terrain tile metadata");
        source->records_[index] = r;
        info.shared_height_error_m[r.tile.level] = std::max(info.shared_height_error_m[r.tile.level], r.height_error_m);
        info.maximum_gradient_m = std::max(info.maximum_gradient_m, r.gradient_m);
    }
    info.minimum_m = source->records_[0].minimum_m;
    info.maximum_m = source->records_[0].maximum_m;
    for (const auto& r : source->records_) {
        if (r.tile.level == 0) {
            info.minimum_m = std::min(info.minimum_m, r.minimum_m);
            info.maximum_m = std::max(info.maximum_m, r.maximum_m);
        } else {
            const auto& parent = source->record(r.tile.parent());
            require(parent.minimum_m <= r.minimum_m && parent.maximum_m >= r.maximum_m &&
                    parent.height_error_m >= r.height_error_m && parent.gradient_m >= r.gradient_m,
                    "Terrain metadata does not conservatively contain descendants");
        }
    }
    return source;
}

const HeightRecord& HeightSource::record(TileId tile) const {
    require(tile.face < 6 && tile.level <= MAX_LEVEL && tile.x < (1u << tile.level) && tile.y < (1u << tile.level),
            "Invalid terrain query address");
    while (tile.level > info_.data_level) tile = tile.parent();
    size_t n = size_t(1) << tile.level;
    return records_[level_offsets_[tile.level] + tile.face * n * n + tile.y * n + tile.x];
}

HeightTile HeightSource::read(TileId tile) const {
    const auto& r = record(tile);
    HeightTile result;
    std::ifstream input(payload_, std::ios::binary);
    input.seekg(std::streamoff(r.offset));
    input.read(reinterpret_cast<char*>(&result), sizeof(result));
    require(bool(input) && height_checksum(std::as_bytes(std::span{result.heights_m})) == r.checksum,
            "Terrain payload is truncated or corrupt");
    for (float height : result.heights_m) require(std::isfinite(height), "Terrain payload contains a nonfinite height");
    if (r.tile.level == info_.data_level) retain_envelope(r.tile, result);
    return result;
}

void HeightSource::retain_envelope(TileId tile, const HeightTile& payload) const {
    std::lock_guard lock(envelope_mutex_);
    for (auto i = envelopes_.begin(); i != envelopes_.end(); ++i) {
        if (i->first == tile.key()) {
            envelopes_.splice(envelopes_.begin(), envelopes_, i);
            return;
        }
    }
    if (envelopes_.size() == 64) envelopes_.pop_back();
    envelopes_.emplace_front(tile.key(), std::make_shared<HeightTile>(payload));
    bounds_revision_.fetch_add(1, std::memory_order_relaxed);
}

SurfaceTile HeightSource::describe(TileId tile, double polar, double) const {
    const auto& r = record(tile);
    double radius = info_.equatorial_radius_m;
    TileMetadata local{r.minimum_m / radius, r.maximum_m / radius, 0, 0};
    if (tile.level > info_.data_level) {
        std::shared_ptr<const HeightTile> payload;
        {
            std::lock_guard lock(envelope_mutex_);
            for (auto i = envelopes_.begin(); i != envelopes_.end(); ++i) {
                if (i->first == r.tile.key()) {
                    payload = i->second;
                    envelopes_.splice(envelopes_.begin(), envelopes_, i);
                    break;
                }
            }
        }
        if (payload) {
            double subdivision = std::ldexp(1.0, int(tile.level-info_.data_level));
            double x0 = (tile.x/subdivision-r.tile.x)*GRID_CELLS;
            double y0 = (tile.y/subdivision-r.tile.y)*GRID_CELLS;
            double side = GRID_CELLS/subdivision;
            double low = height_sample(*payload, x0, y0), high = low;
            auto include = [&](double x, double y) {
                double h = height_sample(*payload, x, y);
                low = std::min(low, h); high = std::max(high, h);
            };
            include(x0+side, y0); include(x0, y0+side); include(x0+side, y0+side);
            for (int y = int(std::ceil(y0)); y <= int(std::floor(y0+side)); ++y)
                for (int x = int(std::ceil(x0)); x <= int(std::floor(x0+side)); ++x) include(x, y);
            local.minimum_height = (low-info_.encoding_error_m)/radius;
            local.maximum_height = (high+info_.encoding_error_m)/radius;
        }
    }
    return {surface_bounds(tile, polar, 1.0, local), approximation_error(tile, tile.level ? 2 : 1)};
}

double HeightSource::approximation_error(TileId tile, uint32_t stride) const {
    require((stride == 1 || stride == 2 || stride == 4) && stride <= (1u << std::min(tile.level, 2u)),
            "Invalid terrain approximation stride");
    const auto& r = record(tile);
    auto coarse = tile;
    for (uint32_t s = stride; s > 1; s /= 2) coarse = coarse.parent();
    double width = std::ldexp(2.0, -int(tile.level));
    double u0 = -1+tile.x*width, v0 = -1+tile.y*width;
    double u = std::max({u0, 0.0, -u0-width}), v = std::max({v0, 0.0, -v0-width});
    auto q = cube_vector(tile.face, u, v);
    auto normal_q = q/glm::dvec3(1, info_.polar_ratio, 1);
    double radius = info_.equatorial_radius_m;
    double height = std::max(std::abs(r.minimum_m), std::abs(r.maximum_m))/radius;
    double height_error = coarse.level <= info_.data_level ? record(coarse).height_error_m : 0.0;
    height_error = std::min(height_error, r.maximum_m-r.minimum_m);
    double step = width*stride/GRID_CELLS;
    return (height_error+info_.encoding_error_m)/radius +
        curvature(info_.polar_ratio, height, r.gradient_m/radius, glm::dot(q,q), glm::dot(normal_q,normal_q))*step*step +
        32*std::numeric_limits<double>::epsilon();
}

double HeightSource::morph_error(uint32_t level) const noexcept {
    level = std::min(level, MAX_LEVEL);
    double radius = info_.equatorial_radius_m;
    double step = std::ldexp(2.0 / GRID_CELLS, -int(level));
    double height = std::max(std::abs(info_.minimum_m), std::abs(info_.maximum_m)) / radius;
    return std::min(info_.shared_height_error_m[level], info_.maximum_m-info_.minimum_m) / radius +
        curvature(info_.polar_ratio, height, info_.maximum_gradient_m / radius) * step * step + info_.encoding_error_m / radius +
        32*std::numeric_limits<double>::epsilon();
}

bool HeightSource::compatible(double radius, double polar) const noexcept {
    return std::abs(radius - info_.equatorial_radius_m) <= info_.equatorial_radius_m * 1e-10 &&
           std::abs(polar - info_.polar_ratio) <= 1e-12;
}

double height_sample(const HeightTile& payload, double x, double y) noexcept {
    x = std::clamp(x, -1.0, double(GRID_CELLS + 1));
    y = std::clamp(y, -1.0, double(GRID_CELLS + 1));
    int ix = std::min(int(std::floor(x)), int(GRID_CELLS)), iy = std::min(int(std::floor(y)), int(GRID_CELLS));
    double u = x - ix, v = y - iy;
    auto at = [&](int dx, int dy) { return double(payload.heights_m[(iy + dy + 1) * HEIGHT_SIDE + ix + dx + 1]); };
    double b = at(1, 0), c = at(0, 1);
    return u + v <= 1 ? at(0, 0) * (1 - u - v) + b * u + c * v
                      : at(1, 1) * (u + v - 1) + b * (1 - v) + c * (1 - u);
}

uint32_t height_checksum(std::span<const std::byte> bytes) noexcept {
    uint32_t hash = 2166136261u;
    for (auto byte : bytes) hash = (hash ^ std::to_integer<uint8_t>(byte)) * 16777619u;
    return hash;
}

}
