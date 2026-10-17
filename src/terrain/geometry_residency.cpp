#include "terrain/geometry_residency.hpp"
#include <algorithm>
#include <bit>

namespace sf::terrain {

size_t GeometryKeyHash::operator()(const GeometryKey& key) const noexcept {
    uint64_t hash = key.tile;
    auto combine = [&](uint64_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    };
    combine(key.generation);
    combine(key.geometry_revision);
    combine(key.body);
    combine(std::bit_cast<uint64_t>(key.polar_ratio));
    combine(std::bit_cast<uint64_t>(key.height_amplitude == 0.0 ? 0.0 : key.height_amplitude));
    return size_t(hash);
}

CachePlan GeometryResidency::request(std::span<const GeometryKey> keys, uint32_t bake_budget) {
    ++frame_;
    CachePlan plan;
    plan.slots.assign(keys.size(), -1);
    plan.bakes.reserve(std::min(bake_budget, uint32_t(keys.size())));
    for (uint32_t i = 0; i < keys.size(); ++i) {
        if (auto found = entries_.find(keys[i]); found != entries_.end()) {
            found->second.protected_frame = frame_;
            ages_.splice(ages_.begin(), ages_, found->second.age);
            plan.slots[i] = int32_t(found->second.slot);
            ++plan.hits;
        }
    }
    plan.misses = uint32_t(keys.size()) - plan.hits;
    for (uint32_t i = 0; i < keys.size(); ++i) {
        if (plan.slots[i] >= 0) continue;
        if (auto duplicate = entries_.find(keys[i]); duplicate != entries_.end()) {
            plan.slots[i] = int32_t(duplicate->second.slot);
            continue;
        }
        if (plan.bakes.size() >= bake_budget || capacity_ == 0) continue;
        uint32_t slot = uint32_t(entries_.size());
        if (entries_.size() == capacity_) {
            auto oldest = entries_.find(ages_.back());
            if (oldest->second.protected_frame == frame_) continue;
            slot = oldest->second.slot;
            entries_.erase(oldest);
            ages_.pop_back();
        }
        ages_.push_front(keys[i]);
        entries_.emplace(keys[i], Entry{slot, frame_, ages_.begin()});
        plan.slots[i] = int32_t(slot);
        plan.bakes.push_back({i, slot});
    }
    return plan;
}

}
