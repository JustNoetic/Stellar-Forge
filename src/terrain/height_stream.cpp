#include "terrain/height_stream.hpp"
#include <algorithm>
#include <cmath>

namespace sf::terrain {
namespace {

struct FaceUV { uint32_t face; double u, v; };

FaceUV canonical(glm::dvec3 p) {
    auto a = glm::abs(p);
    if (a.x >= a.y && a.x >= a.z) return p.x >= 0 ? FaceUV{0, -p.z/a.x, p.y/a.x} : FaceUV{1, p.z/a.x, p.y/a.x};
    if (a.y >= a.z) return p.y >= 0 ? FaceUV{2, p.x/a.y, -p.z/a.y} : FaceUV{3, p.x/a.y, p.z/a.y};
    return p.z >= 0 ? FaceUV{4, p.x/a.z, p.y/a.z} : FaceUV{5, -p.x/a.z, p.y/a.z};
}

class Baker {
public:
    explicit Baker(const HeightSource& source) : source_(source) {}
    uint64_t reads = 0;

    HeightGeometry bake(TileId tile) {
        touched_.clear();
        const auto& info = source_.info();
        double width = std::ldexp(2.0, -int(tile.level));
        auto anchor = ellipsoid_point(tile.face, -1+(tile.x+.5)*width, -1+(tile.y+.5)*width, info.polar_ratio);
        double spacing = std::ldexp(2.0 / GRID_CELLS, -int(info.data_level));
        HeightGeometry result;
        for (uint32_t y = 0; y <= GRID_CELLS; ++y) {
            for (uint32_t x = 0; x <= GRID_CELLS; ++x) {
                double u = -1+(tile.x+double(x)/GRID_CELLS)*width, v = -1+(tile.y+double(y)/GRID_CELLS)*width;
                auto coordinates = canonical(cube_vector(tile.face, u, v));
                auto p = point(coordinates.face, coordinates.u, coordinates.v);
                auto du = point(coordinates.face, coordinates.u+spacing, coordinates.v) - point(coordinates.face, coordinates.u-spacing, coordinates.v);
                auto dv = point(coordinates.face, coordinates.u, coordinates.v+spacing) - point(coordinates.face, coordinates.u, coordinates.v-spacing);
                result[y*(GRID_CELLS+1)+x] = {glm::vec4(glm::vec3(p-anchor), 0),
                    glm::vec4(glm::vec3(glm::normalize(glm::cross(du, dv))), 0),
                    glm::vec4(glm::vec3(glm::normalize(cube_vector(tile.face, u, v))), 0)};
            }
        }
        return result;
    }

private:
    glm::dvec3 point(uint32_t face, double u, double v) {
        const auto& info = source_.info();
        auto direction = glm::normalize(cube_vector(face, u, v));
        auto at = canonical(direction);
        auto tile = tile_at(direction, info.data_level);
        const auto& payload = get(tile);
        double n = double(1u << info.data_level);
        double x = ((at.u+1)*.5*n-tile.x)*GRID_CELLS;
        double y = ((at.v+1)*.5*n-tile.y)*GRID_CELLS;
        double height = height_sample(payload, x, y) / info.equatorial_radius_m;
        auto scale = glm::dvec3(1, info.polar_ratio, 1);
        return direction*scale + glm::normalize(direction/scale)*height;
    }

    const HeightTile& get(TileId tile) {
        for (auto i = payloads_.begin(); i != payloads_.end(); ++i) {
            if (i->first == tile) {
                payloads_.splice(payloads_.begin(), payloads_, i);
                if (touched_.insert(tile.key()).second) source_.retain_envelope(tile, payloads_.front().second);
                return payloads_.front().second;
            }
        }
        auto payload = source_.read(tile);
        touched_.insert(tile.key());
        ++reads;
        if (payloads_.size() == 64) payloads_.pop_back();
        payloads_.emplace_front(tile, std::move(payload));
        return payloads_.front().second;
    }

    const HeightSource& source_;
    std::list<std::pair<TileId, HeightTile>> payloads_;
    std::unordered_set<uint64_t> touched_;
};

}

HeightStream::HeightStream(std::shared_ptr<const HeightSource> source, uint32_t capacity, uint32_t queue_limit)
    : source_(std::move(source)), capacity_(capacity), queue_limit_(queue_limit),
      worker_([this](std::stop_token stop) { run(stop); }) {}

HeightStream::~HeightStream() {
    worker_.request_stop();
    wake_.notify_all();
}

void HeightStream::request(TileId tile) {
    std::lock_guard lock(mutex_);
    if (capacity_ == 0 || pending_.size() >= queue_limit_ || pending_.contains(tile.key()) || failed_.contains(tile.key()) || failed_.size() >= capacity_) return;
    for (auto i = ready_.begin(); i != ready_.end(); ++i) {
        if (i->tile == tile) {
            ready_.splice(ready_.begin(), ready_, i);
            return;
        }
    }
    pending_.insert(tile.key());
    queue_.push_back(tile);
    wake_.notify_one();
}

std::vector<std::pair<TileId, std::shared_ptr<const HeightGeometry>>> HeightStream::completed() {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<TileId, std::shared_ptr<const HeightGeometry>>> result;
    result.reserve(ready_.size());
    for (const auto& entry : ready_) result.emplace_back(entry.tile, entry.geometry);
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.first.level != b.first.level ? a.first.level < b.first.level : a.first.key() < b.first.key();
    });
    return result;
}

void HeightStream::consumed(TileId tile) {
    std::lock_guard lock(mutex_);
    ready_.remove_if([&](const Ready& entry) { return entry.tile == tile; });
}

StreamStats HeightStream::stats() const {
    std::lock_guard lock(mutex_);
    auto result = stats_;
    result.pending = uint32_t(pending_.size());
    result.ready = uint32_t(ready_.size());
    result.failed = uint32_t(failed_.size());
    return result;
}

void HeightStream::run(std::stop_token stop) {
    Baker baker(*source_);
    while (!stop.stop_requested()) {
        TileId tile;
        {
            std::unique_lock lock(mutex_);
            if (!wake_.wait(lock, stop, [&] { return !queue_.empty(); })) return;
            tile = queue_.front();
            queue_.pop_front();
        }
        std::shared_ptr<HeightGeometry> geometry;
        std::string error;
        try { geometry = std::make_shared<HeightGeometry>(baker.bake(tile)); }
        catch (const std::exception& failure) { error = failure.what(); }
        {
            std::lock_guard lock(mutex_);
            pending_.erase(tile.key());
            stats_.payload_reads = baker.reads;
            if (geometry) {
                if (ready_.size() == capacity_) ready_.pop_back();
                ready_.push_front({tile, std::move(geometry)});
                ++stats_.builds;
            } else if (failed_.size() < capacity_) {
                failed_.insert(tile.key());
                if (stats_.error.empty()) stats_.error = std::move(error);
            }
        }
    }
}

}
