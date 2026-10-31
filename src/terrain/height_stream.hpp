#pragma once
#include <atomic>
#include <limits>

#include "terrain/terrain_source.hpp"
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace sf::terrain {

struct GeometryVertex {
    glm::vec4 position;
    glm::vec4 normal;
    glm::vec4 direction;
};

using HeightGeometry = std::array<GeometryVertex, (GRID_CELLS + 1) * (GRID_CELLS + 1)>;

struct StreamStats {
    uint32_t pending = 0;
    uint32_t ready = 0;
    uint32_t failed = 0;
    uint64_t builds = 0;
    uint64_t payload_reads = 0;
    std::string error;
};

class HeightStream {
public:
    explicit HeightStream(std::shared_ptr<const HeightSource> source, uint32_t capacity = 256, uint32_t queue_limit = 128);
    ~HeightStream();
    HeightStream(const HeightStream&) = delete;
    HeightStream& operator=(const HeightStream&) = delete;
    void request(TileId tile);
    std::vector<std::pair<TileId, std::shared_ptr<const HeightGeometry>>> completed();
    void consumed(TileId tile);
    StreamStats stats() const;
    const HeightSource& source() const noexcept { return *source_; }
    void set_floor_m(double floor) noexcept { floor_m_.store(floor); }
    double floor_m() const noexcept { return floor_m_.load(); }
    std::shared_ptr<const HeightSource> share_source() const noexcept { return source_; }

private:
    struct Ready {
        TileId tile;
        std::shared_ptr<const HeightGeometry> geometry;
    };
    void run(std::stop_token stop);
    std::shared_ptr<const HeightSource> source_;
    std::atomic<double> floor_m_{-std::numeric_limits<double>::infinity()};
    uint32_t capacity_;
    uint32_t queue_limit_;
    mutable std::mutex mutex_;
    std::condition_variable_any wake_;
    std::deque<TileId> queue_;
    std::unordered_set<uint64_t> pending_;
    std::unordered_set<uint64_t> failed_;
    std::list<Ready> ready_;
    StreamStats stats_;
    std::jthread worker_;
};

}
