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
    std::string reconstruction = j.at("reconstruction").get<std::string>();
    uint32_t grid_cells = j.at("grid_cells").get<uint32_t>();
    uint32_t gutter = j.at("gutter").get<uint32_t>();
    require(j.at("version") == 1 && j.at("encoding") == "float32-le" &&
            (reconstruction == "nested-triangles" || reconstruction == "cubic-bspline") &&
            (grid_cells == GRID_CELLS || grid_cells == DENSE_HEIGHT_CELLS) &&
            (gutter == 1 || gutter == 2) && (reconstruction != "cubic-bspline" || gutter >= 2) &&
            j.at("height_units") == "m" &&
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
    info.grid_cells = grid_cells;
    info.gutter = gutter;
    info.cubic = (reconstruction == "cubic-bspline");
    uint32_t subdivisions = std::countr_zero(grid_cells / GRID_CELLS);
    info.finest_level = std::min(MAX_LEVEL, info.data_level + subdivisions);
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
    uint32_t side = grid_cells + 1 + 2 * gutter;
    uint64_t block = uint64_t(side) * side * sizeof(float);
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
    for (uint32_t level = info.data_level + 1; level < info.finest_level; ++level)
        info.shared_height_error_m[level] = info.shared_height_error_m[info.data_level];
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
    if (r.tile.level == info_.data_level) {
        std::lock_guard lock(envelope_mutex_);
        for (auto i = envelopes_.begin(); i != envelopes_.end(); ++i) {
            if (i->first == r.tile.key()) {
                envelopes_.splice(envelopes_.begin(), envelopes_, i);
                return *i->second.payload;
            }
        }
    }
    HeightTile result;
    result.cells = info_.grid_cells;
    result.gutter = info_.gutter;
    result.side = info_.grid_cells + 1 + 2 * info_.gutter;
    result.cubic = info_.cubic;
    result.heights_m.resize(size_t(result.side) * result.side);
    std::ifstream input(payload_, std::ios::binary);
    input.seekg(std::streamoff(r.offset));
    input.read(reinterpret_cast<char*>(result.heights_m.data()), std::streamsize(result.heights_m.size() * sizeof(float)));
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
    Envelope env;
    env.payload = std::make_shared<HeightTile>(payload);
    uint32_t step = payload.cells / GRID_CELLS;
    uint32_t g = payload.gutter, s = payload.side;
    double grad_scale = double(payload.cells) / std::ldexp(2.0, -int(tile.level));
    auto at = [&](uint32_t gx, uint32_t gy) {
        return double(payload.heights_m[(gy + g) * s + gx + g]);
    };
    auto tri_err = [&](uint32_t x0, uint32_t y0, uint32_t sz) {
        double a = at(x0, y0), b = at(x0 + sz, y0), c = at(x0, y0 + sz), d = at(x0 + sz, y0 + sz);
        double inv = 1.0 / double(sz);
        double err = 0.0;
        for (uint32_t dy = 0; dy <= sz; ++dy) {
            double v = dy * inv;
            for (uint32_t dx = 0; dx <= sz; ++dx) {
                double u = dx * inv;
                double ch = dx + dy <= sz ? a * (1.0 - u - v) + b * u + c * v
                                          : d * (u + v - 1.0) + b * (1.0 - v) + c * (1.0 - u);
                err = std::max(err, std::abs(at(x0 + dx, y0 + dy) - ch));
            }
        }
        return float(err);
    };
    for (uint32_t my = 0; my < GRID_CELLS; ++my) {
        for (uint32_t mx = 0; mx < GRID_CELLS; ++mx) {
            uint32_t idx = my * GRID_CELLS + mx;
            uint32_t gx0 = mx * step, gy0 = my * step;
            float low = std::numeric_limits<float>::infinity();
            float high = -std::numeric_limits<float>::infinity();
            for (uint32_t dy = 0; dy <= step; ++dy) {
                uint32_t row = (gy0 + dy + g) * s + gx0 + g;
                for (uint32_t dx = 0; dx <= step; ++dx) {
                    float h = payload.heights_m[row + dx];
                    low = std::min(low, h);
                    high = std::max(high, h);
                }
            }
            double grad = 0.0;
            for (uint32_t dy = 0; dy < step; ++dy) {
                for (uint32_t dx = 0; dx < step; ++dx) {
                    double a = at(gx0 + dx, gy0 + dy), b = at(gx0 + dx + 1, gy0 + dy);
                    double c = at(gx0 + dx, gy0 + dy + 1), d = at(gx0 + dx + 1, gy0 + dy + 1);
                    grad = std::max({grad, std::hypot(b - a, c - a), std::hypot(d - c, d - b)});
                }
            }
            env.mip_min[idx] = low;
            env.mip_max[idx] = high;
            env.mip_grad[idx] = float(grad * grad_scale);
            if (payload.cells == DENSE_HEIGHT_CELLS) {
                float e2 = 0.0f;
                for (uint32_t sy = 0; sy < 4; ++sy)
                    for (uint32_t sx = 0; sx < 4; ++sx)
                        e2 = std::max(e2, tri_err(gx0 + sx * 2, gy0 + sy * 2, 2));
                float e4 = e2;
                for (uint32_t sy = 0; sy < 2; ++sy)
                    for (uint32_t sx = 0; sx < 2; ++sx)
                        e4 = std::max(e4, tri_err(gx0 + sx * 4, gy0 + sy * 4, 4));
                float e8 = std::max(e4, tri_err(gx0, gy0, 8));
                env.mip_err2[idx] = e2;
                env.mip_err4[idx] = e4;
                env.mip_err8[idx] = e8;
            }
        }
    }
    if (envelopes_.size() == 512) envelopes_.pop_back();
    envelopes_.emplace_front(tile.key(), std::move(env));
    bounds_revision_.fetch_add(1, std::memory_order_relaxed);
}

SurfaceTile HeightSource::describe(TileId tile, double polar, double) const {
    const auto& r = record(tile);
    double radius = info_.equatorial_radius_m;
    TileMetadata local{r.minimum_m / radius, r.maximum_m / radius, 0, 0};
    uint32_t stride = tile.level ? 2u : 1u;
    auto coarse = tile;
    for (uint32_t s = stride; s > 1; s /= 2) coarse = coarse.parent();
    double height_error = coarse.level < info_.finest_level ? record(coarse).height_error_m : 0.0;
    double low = r.minimum_m, high = r.maximum_m, grad = r.gradient_m;
    if (tile.level > info_.data_level) {
        uint32_t rel_level = tile.level - info_.data_level;
        uint32_t cell_shifts = std::countr_zero(info_.grid_cells);
        bool has_envelope = false;
        {
            std::lock_guard lock(envelope_mutex_);
            for (const auto& entry : envelopes_) {
                if (entry.first == r.tile.key()) {
                    const auto& env = entry.second;
                    has_envelope = true;
                    if (rel_level <= 4) {
                        uint32_t block = GRID_CELLS >> rel_level;
                        uint32_t mask = (1u << rel_level) - 1u;
                        uint32_t mx0 = (tile.x & mask) * block;
                        uint32_t my0 = (tile.y & mask) * block;
                        float l = std::numeric_limits<float>::infinity();
                        float h = -std::numeric_limits<float>::infinity();
                        float g = 0.0f;
                        float err = 0.0f;
                        uint32_t crel = coarse.level >= info_.data_level ? coarse.level - info_.data_level : 99u;
                        for (uint32_t my = my0; my < my0 + block; ++my) {
                            for (uint32_t mx = mx0; mx < mx0 + block; ++mx) {
                                uint32_t idx = my * GRID_CELLS + mx;
                                l = std::min(l, env.mip_min[idx]);
                                h = std::max(h, env.mip_max[idx]);
                                g = std::max(g, env.mip_grad[idx]);
                                if (info_.grid_cells == DENSE_HEIGHT_CELLS) {
                                    if (crel == 0) err = std::max(err, env.mip_err8[idx]);
                                    else if (crel == 1) err = std::max(err, env.mip_err4[idx]);
                                    else if (crel == 2) err = std::max(err, env.mip_err2[idx]);
                                }
                            }
                        }
                        low = double(l);
                        high = double(h);
                        grad = double(g);
                        if (info_.grid_cells == DENSE_HEIGHT_CELLS && crel < 3) height_error = double(err);
                    } else if (rel_level < cell_shifts) {
                        const auto& p = *env.payload;
                        uint32_t step = p.cells >> rel_level;
                        uint32_t mask = (1u << rel_level) - 1u;
                        uint32_t gx0 = (tile.x & mask) * step;
                        uint32_t gy0 = (tile.y & mask) * step;
                        float l = std::numeric_limits<float>::infinity();
                        float h = -std::numeric_limits<float>::infinity();
                        for (uint32_t dy = 0; dy <= step; ++dy) {
                            uint32_t row = (gy0 + dy + p.gutter) * p.side + gx0 + p.gutter;
                            for (uint32_t dx = 0; dx <= step; ++dx) {
                                float v = p.heights_m[row + dx];
                                l = std::min(l, v);
                                h = std::max(h, v);
                            }
                        }
                        uint32_t mx = (tile.x >> (rel_level - 4)) & 15u;
                        uint32_t my = (tile.y >> (rel_level - 4)) & 15u;
                        low = double(l);
                        high = double(h);
                        grad = double(env.mip_grad[my * GRID_CELLS + mx]);
                    } else {
                        const auto& p = *env.payload;
                        double subdivision = std::ldexp(1.0, int(rel_level));
                        double x0 = (tile.x / subdivision - r.tile.x) * GRID_CELLS;
                        double y0 = (tile.y / subdivision - r.tile.y) * GRID_CELLS;
                        double side = GRID_CELLS / subdivision;
                        double c00 = height_sample(p, x0, y0);
                        double c10 = height_sample(p, x0 + side, y0);
                        double c01 = height_sample(p, x0, y0 + side);
                        double c11 = height_sample(p, x0 + side, y0 + side);
                        uint32_t mx = (tile.x >> (rel_level - 4)) & 15u;
                        uint32_t my = (tile.y >> (rel_level - 4)) & 15u;
                        low = std::min({c00, c10, c01, c11});
                        high = std::max({c00, c10, c01, c11});
                        grad = double(env.mip_grad[my * GRID_CELLS + mx]);
                    }
                    break;
                }
            }
        }
        if (has_envelope) {
            local.minimum_height = (low - info_.encoding_error_m) / radius;
            local.maximum_height = (high + info_.encoding_error_m) / radius;
        }
    }
    return {surface_bounds(tile, polar, 1.0, local), approximation_error_with_height(tile, stride, low, high, grad, height_error)};
}

double HeightSource::approximation_error_with_height(TileId tile, uint32_t stride, double low_m, double high_m, double grad_m, double height_error) const noexcept {
    double width = std::ldexp(2.0, -int(tile.level));
    double u0 = -1+tile.x*width, v0 = -1+tile.y*width;
    double u = std::max({u0, 0.0, -u0-width}), v = std::max({v0, 0.0, -v0-width});
    auto q = cube_vector(tile.face, u, v);
    auto normal_q = q/glm::dvec3(1, info_.polar_ratio, 1);
    double radius = info_.equatorial_radius_m;
    double height = std::max(std::abs(low_m), std::abs(high_m))/radius;
    height_error = std::min(height_error, high_m-low_m);
    double step = width*stride/GRID_CELLS;
    return (height_error+info_.encoding_error_m)/radius +
        curvature(info_.polar_ratio, height, grad_m/radius, glm::dot(q,q), glm::dot(normal_q,normal_q))*step*step +
        32*std::numeric_limits<double>::epsilon();
}

double HeightSource::approximation_error(TileId tile, uint32_t stride) const {
    require((stride == 1 || stride == 2 || stride == 4) && stride <= (1u << std::min(tile.level, 2u)),
            "Invalid terrain approximation stride");
    const auto& r = record(tile);
    auto coarse = tile;
    for (uint32_t s = stride; s > 1; s /= 2) coarse = coarse.parent();
    double height_error = coarse.level < info_.finest_level ? record(coarse).height_error_m : 0.0;
    double low = r.minimum_m, high = r.maximum_m, grad = r.gradient_m;
    if (tile.level > info_.data_level) {
        uint32_t rel_level = tile.level - info_.data_level;
        uint32_t cell_shifts = std::countr_zero(info_.grid_cells);
        std::lock_guard lock(envelope_mutex_);
        for (const auto& entry : envelopes_) {
            if (entry.first == r.tile.key()) {
                const auto& env = entry.second;
                if (rel_level <= 4) {
                    uint32_t block = GRID_CELLS >> rel_level;
                    uint32_t mask = (1u << rel_level) - 1u;
                    uint32_t mx0 = (tile.x & mask) * block;
                    uint32_t my0 = (tile.y & mask) * block;
                    float l = std::numeric_limits<float>::infinity();
                    float h = -std::numeric_limits<float>::infinity();
                    float g = 0.0f;
                    float err = 0.0f;
                    uint32_t crel = coarse.level >= info_.data_level ? coarse.level - info_.data_level : 99u;
                    for (uint32_t my = my0; my < my0 + block; ++my) {
                        for (uint32_t mx = mx0; mx < mx0 + block; ++mx) {
                            uint32_t idx = my * GRID_CELLS + mx;
                            l = std::min(l, env.mip_min[idx]);
                            h = std::max(h, env.mip_max[idx]);
                            g = std::max(g, env.mip_grad[idx]);
                            if (info_.grid_cells == DENSE_HEIGHT_CELLS) {
                                if (crel == 0) err = std::max(err, env.mip_err8[idx]);
                                else if (crel == 1) err = std::max(err, env.mip_err4[idx]);
                                else if (crel == 2) err = std::max(err, env.mip_err2[idx]);
                            }
                        }
                    }
                    low = double(l);
                    high = double(h);
                    grad = double(g);
                    if (info_.grid_cells == DENSE_HEIGHT_CELLS && crel < 3) height_error = double(err);
                } else if (rel_level < cell_shifts) {
                    const auto& p = *env.payload;
                    uint32_t step = p.cells >> rel_level;
                    uint32_t mask = (1u << rel_level) - 1u;
                    uint32_t gx0 = (tile.x & mask) * step;
                    uint32_t gy0 = (tile.y & mask) * step;
                    float l = std::numeric_limits<float>::infinity();
                    float h = -std::numeric_limits<float>::infinity();
                    for (uint32_t dy = 0; dy <= step; ++dy) {
                        uint32_t row = (gy0 + dy + p.gutter) * p.side + gx0 + p.gutter;
                        for (uint32_t dx = 0; dx <= step; ++dx) {
                            float v = p.heights_m[row + dx];
                            l = std::min(l, v);
                            h = std::max(h, v);
                        }
                    }
                    uint32_t mx = (tile.x >> (rel_level - 4)) & 15u;
                    uint32_t my = (tile.y >> (rel_level - 4)) & 15u;
                    low = double(l);
                    high = double(h);
                    grad = double(env.mip_grad[my * GRID_CELLS + mx]);
                } else {
                    const auto& p = *env.payload;
                    double subdivision = std::ldexp(1.0, int(rel_level));
                    double x0 = (tile.x / subdivision - r.tile.x) * GRID_CELLS;
                    double y0 = (tile.y / subdivision - r.tile.y) * GRID_CELLS;
                    double side = GRID_CELLS / subdivision;
                    double c00 = height_sample(p, x0, y0);
                    double c10 = height_sample(p, x0 + side, y0);
                    double c01 = height_sample(p, x0, y0 + side);
                    double c11 = height_sample(p, x0 + side, y0 + side);
                    uint32_t mx = (tile.x >> (rel_level - 4)) & 15u;
                    uint32_t my = (tile.y >> (rel_level - 4)) & 15u;
                    low = std::min({c00, c10, c01, c11});
                    high = std::max({c00, c10, c01, c11});
                    grad = double(env.mip_grad[my * GRID_CELLS + mx]);
                }
                break;
            }
        }
    }
    return approximation_error_with_height(tile, stride, low, high, grad, height_error);
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
    double scale = double(payload.cells) / double(GRID_CELLS);
    double gx = std::clamp(x * scale, -1.0, double(payload.cells + 1));
    double gy = std::clamp(y * scale, -1.0, double(payload.cells + 1));
    int cells = int(payload.cells);
    int g = int(payload.gutter);
    int side = int(payload.side);
    int ix = std::min(int(std::floor(gx)), cells), iy = std::min(int(std::floor(gy)), cells);
    double u = gx - ix, v = gy - iy;
    auto at = [&](int dx, int dy) { return double(payload.heights_m[(iy + dy + g) * side + ix + dx + g]); };
    double b = at(1, 0), c = at(0, 1);
    return u + v <= 1 ? at(0, 0) * (1 - u - v) + b * u + c * v
                      : at(1, 1) * (u + v - 1) + b * (1 - v) + c * (1 - u);
}

double height_sample_cubic(const HeightTile& payload, double x, double y) noexcept {
    if (!payload.cubic || payload.gutter < 2) return height_sample(payload, x, y);
    double scale = double(payload.cells) / double(GRID_CELLS);
    double gx = std::clamp(x * scale, -1.0, double(payload.cells + 1));
    double gy = std::clamp(y * scale, -1.0, double(payload.cells + 1));
    int cells = int(payload.cells);
    int g = int(payload.gutter);
    int side = int(payload.side);
    int ix = std::clamp(int(std::floor(gx)), -1, cells);
    int iy = std::clamp(int(std::floor(gy)), -1, cells);
    double u = gx - ix, v = gy - iy;
    double wx[4] = {
        (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        (u * u * u) / 6.0
    };
    double wy[4] = {
        (1.0 - v) * (1.0 - v) * (1.0 - v) / 6.0,
        (3.0 * v * v * v - 6.0 * v * v + 4.0) / 6.0,
        (-3.0 * v * v * v + 3.0 * v * v + 3.0 * v + 1.0) / 6.0,
        (v * v * v) / 6.0
    };
    double sum = 0.0;
    for (int dy = -1; dy <= 2; ++dy) {
        int row = std::clamp(iy + dy + g, 0, side - 1) * side;
        double row_sum = 0.0;
        for (int dx = -1; dx <= 2; ++dx) {
            int col = std::clamp(ix + dx + g, 0, side - 1);
            row_sum += double(payload.heights_m[row + col]) * wx[dx + 1];
        }
        sum += row_sum * wy[dy + 1];
    }
    return sum;
}

uint32_t height_checksum(std::span<const std::byte> bytes) noexcept {
    uint32_t hash = 2166136261u;
    for (auto byte : bytes) hash = (hash ^ std::to_integer<uint8_t>(byte)) * 16777619u;
    return hash;
}

}
