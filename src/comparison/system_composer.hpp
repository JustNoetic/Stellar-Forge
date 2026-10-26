#pragma once

#include "ephemeris/jacobi.hpp"
#include "ephemeris/system_manager.hpp"
#include <memory>
#include <span>
#include <vector>
#include <glm/glm.hpp>

namespace sf::comparison {

struct CompositeSource {
    uint32_t slot = 0;
    uint32_t local = 0;
};

class CompositeMap {
public:
    size_t slots() const noexcept { return begins_.size(); }
    uint32_t begin(size_t slot) const { return begins_.at(slot); }
    uint32_t count(size_t slot) const { return counts_.at(slot); }
    size_t size() const noexcept { return begins_.empty() ? 0 : begins_.back() + counts_.back(); }
    CompositeSource source(size_t index) const;
    int composite(size_t slot, size_t local) const noexcept;
    uint32_t light_group(size_t index) const { return source(index).slot; }
    bool primary(size_t index) const noexcept { return !counts_.empty() && index < counts_.front(); }

private:
    friend class SystemComposer;
    std::vector<uint32_t> begins_, counts_;
};

struct ComposedMember {
    const ephemeris::SystemSnapshot* snapshot = nullptr;
    glm::dvec3 offset_au{0.0};
    uint32_t lit_by = 0;
};

class SystemComposer {
public:
    const ephemeris::SystemSnapshot& compose(std::span<const ComposedMember> members);
    const ephemeris::SystemSnapshot& scene() const noexcept { return scene_; }
    const CompositeMap& map() const noexcept { return map_; }
    uint64_t layout_revision() const noexcept { return scene_.generation_; }
    uint32_t definition_rebuilds() const noexcept { return definition_rebuilds_; }
    uint32_t hierarchy_rebuilds() const noexcept { return hierarchy_rebuilds_; }
    void reset();

private:
    struct MemberKey {
        std::shared_ptr<const ephemeris::SystemDefinition> definition;
        std::shared_ptr<const ephemeris::JacobiHierarchy> jacobi, orbit;
        std::shared_ptr<const ephemeris::ReferenceHierarchy> reference;
        uint64_t state_revision = 0;
        glm::dvec3 offset_au{0.0};
        uint32_t lit_by = 0;
    };

    ephemeris::SystemSnapshot scene_;
    CompositeMap map_;
    std::vector<MemberKey> keys_;
    std::vector<uint64_t> identities_;
    std::vector<uint64_t> lighting_epochs_;
    uint32_t definition_rebuilds_ = 0, hierarchy_rebuilds_ = 0;
};

}
