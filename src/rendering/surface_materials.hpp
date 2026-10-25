#pragma once

#include "ephemeris/system_manager.hpp"
#include <map>
#include <vector>

namespace sf::rendering {

class SurfaceMaterialCache {
public:
    void update(const ephemeris::SystemSnapshot& system, uint32_t limit = 1024) {
        size_t count = std::min(system.size(), size_t(limit));
        std::vector<Key> keys;
        keys.reserve(count);
        std::vector<physics::SurfaceMaterialGPU> materials;
        for (size_t i = 0; i < count; ++i) {
            const auto& b = system.body(i).definition;
            auto m = physics::pack_material(physics::resolve_material(b.material, b.name, b.is_star));
            keys.push_back(key(m));
            materials.push_back(m);
        }
        if (keys == keys_) return;
        std::map<Key, CachedPhase> next;
        atlas_.clear();
        tables_.clear();
        for (size_t i = 0; i < count; ++i) {
            auto [it, inserted] = next.try_emplace(keys[i]);
            if (inserted) {
                auto old = phases_.find(keys[i]);
                it->second.table = old == phases_.end() ? physics::build_phase_table(materials[i]) : old->second.table;
                it->second.offset = uint32_t(atlas_.size());
                atlas_.insert(atlas_.end(), it->second.table.begin(), it->second.table.end());
            }
            materials[i].opposition.w = float(it->second.offset);
            tables_.push_back(it->second.table);
        }
        if (atlas_.empty()) atlas_.resize(physics::MATERIAL_PHASE_SAMPLES);
        materials_ = std::move(materials);
        phases_ = std::move(next);
        keys_ = std::move(keys);
        ++revision_;
    }
    const auto& material(size_t i) const { return materials_[i]; }
    const auto& phase(size_t i) const { return tables_[i]; }
    const auto& atlas() const { return atlas_; }
    uint64_t revision() const { return revision_; }

private:
    using Key = std::array<float, 11>;
    static Key key(const physics::SurfaceMaterialGPU& m) {
        return {m.grain.x,m.grain.y,m.grain.z,m.grain.w,m.surface.x,m.surface.y,m.surface.z,m.surface.w,
            m.opposition.x,m.opposition.y,m.opposition.z};
    }
    struct CachedPhase { physics::MaterialPhaseTable table{}; uint32_t offset = 0; };
    std::map<Key, CachedPhase> phases_;
    std::vector<Key> keys_;
    std::vector<physics::SurfaceMaterialGPU> materials_;
    std::vector<physics::MaterialPhaseTable> tables_;
    std::vector<float> atlas_;
    uint64_t revision_ = 0;
};

}
