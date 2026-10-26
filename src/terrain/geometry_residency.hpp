#pragma once

#include <cstdint>
#include <list>
#include <span>
#include <unordered_map>
#include <vector>
#include "terrain/terrain_source.hpp"

namespace sf::terrain {

struct GeometryKey {
    uint64_t identity = 0;
    uint64_t geometry_revision = 0;
    uint64_t tile = 0;
    uint32_t body = 0;
    double polar_ratio = 1.0;
    double height_amplitude = 0.0;
    SourceDomain source;
    bool operator==(const GeometryKey&) const = default;
};

struct GeometryKeyHash {
    size_t operator()(const GeometryKey& key) const noexcept;
};

struct CacheBake {
    uint32_t request = 0;
    uint32_t slot = 0;
};

struct CachePlan {
    std::vector<int32_t> slots;
    std::vector<CacheBake> bakes;
    uint32_t hits = 0;
    uint32_t misses = 0;
};

class GeometryResidency {
public:
    explicit GeometryResidency(uint32_t capacity) : capacity_(capacity) {}
    GeometryResidency(const GeometryResidency&) = delete;
    GeometryResidency& operator=(const GeometryResidency&) = delete;
    GeometryResidency(GeometryResidency&&) = default;
    GeometryResidency& operator=(GeometryResidency&&) = default;
    CachePlan request(std::span<const GeometryKey> keys, uint32_t bake_budget);
    int32_t find(const GeometryKey& key) const;
    uint32_t size() const noexcept { return uint32_t(entries_.size()); }
    uint64_t revision() const noexcept { return revision_; }

private:
    struct Entry {
        uint32_t slot;
        uint64_t protected_frame;
        std::list<GeometryKey>::iterator age;
    };
    uint32_t capacity_;
    uint64_t frame_ = 0;
    uint64_t revision_ = 0;
    std::list<GeometryKey> ages_;
    std::unordered_map<GeometryKey, Entry, GeometryKeyHash> entries_;
};

}
