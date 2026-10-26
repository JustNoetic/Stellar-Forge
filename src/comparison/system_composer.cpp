#include "comparison/system_composer.hpp"
#include <algorithm>
#include <stdexcept>

namespace sf::comparison {

using namespace ephemeris;

namespace {

int shifted(int index, uint32_t offset) noexcept {
    return index < 0 ? index : index + int(offset);
}

std::vector<int> shifted(const std::vector<int>& indices, uint32_t offset) {
    std::vector<int> result(indices.size());
    std::ranges::transform(indices, result.begin(), [&](int index) { return shifted(index, offset); });
    return result;
}

void append(JacobiHierarchy& target, const JacobiHierarchy& source, uint32_t offset) {
    for (int parent : source.parents) target.parents.push_back(shifted(parent, offset));
    for (int body : source.traversal) target.traversal.push_back(shifted(body, offset));
    for (const auto& children : source.children) target.children.push_back(shifted(children, offset));
    target.subsystem_mass.insert(target.subsystem_mass.end(), source.subsystem_mass.begin(), source.subsystem_mass.end());
    target.inner_mass.insert(target.inner_mass.end(), source.inner_mass.begin(), source.inner_mass.end());
}

void append(ReferenceHierarchy& target, const ReferenceHierarchy& source, uint32_t offset) {
    for (int parent : source.parents) target.parents.push_back(shifted(parent, offset));
    for (const auto& children : source.children) target.children.push_back(shifted(children, offset));
    for (int root : source.roots) target.roots.push_back(shifted(root, offset));
}

}

CompositeSource CompositeMap::source(size_t index) const {
    if (index >= size()) throw std::out_of_range("Composite body index is out of range.");
    auto slot = size_t(std::ranges::upper_bound(begins_, uint32_t(index)) - begins_.begin()) - 1;
    return {uint32_t(slot), uint32_t(index - begins_[slot])};
}

int CompositeMap::composite(size_t slot, size_t local) const noexcept {
    return slot < begins_.size() && local < counts_[slot] ? int(begins_[slot] + local) : -1;
}

void SystemComposer::reset() {
    *this = {};
}

const SystemSnapshot& SystemComposer::compose(std::span<const ComposedMember> members) {
    bool definitions = keys_.size() != members.size(), hierarchies = definitions, state = definitions;
    for (size_t i = 0; i < members.size() && !definitions; ++i) {
        const auto& snapshot = *members[i].snapshot;
        const auto& key = keys_[i];
        auto orbit = snapshot.reference_jacobi_ ? snapshot.reference_jacobi_ : snapshot.jacobi_;
        definitions = key.definition != snapshot.definition_ || key.lit_by != members[i].lit_by;
        hierarchies = hierarchies || key.jacobi != snapshot.jacobi_ || key.orbit != orbit || key.reference != snapshot.reference_;
        state = state || key.state_revision != snapshot.state_revision_ || key.offset_au != members[i].offset_au;
    }
    hierarchies = hierarchies || definitions;
    state = state || definitions;

    lighting_epochs_.resize(members.size(), 0);
    for (size_t i = 0; i < members.size(); ++i)
        if (i >= keys_.size() || keys_[i].lit_by != members[i].lit_by || keys_[i].definition != members[i].snapshot->definition_)
            ++lighting_epochs_[i];

    if (state) {
        keys_.resize(members.size());
        for (size_t i = 0; i < members.size(); ++i) {
            const auto& snapshot = *members[i].snapshot;
            keys_[i] = {snapshot.definition_, snapshot.jacobi_, snapshot.reference_jacobi_ ? snapshot.reference_jacobi_ : snapshot.jacobi_,
                        snapshot.reference_, snapshot.state_revision_, members[i].offset_au, members[i].lit_by};
        }
    }

    map_.begins_.resize(members.size());
    map_.counts_.resize(members.size());
    uint32_t total = 0;
    for (size_t i = 0; i < members.size(); ++i) {
        map_.begins_[i] = total;
        map_.counts_[i] = uint32_t(members[i].snapshot->size());
        total += map_.counts_[i];
    }

    if (definitions) {
        auto definition = std::make_shared<SystemDefinition>();
        if (!members.empty()) {
            const auto& primary = members.front().snapshot->definition();
            definition->name = primary.name;
            definition->motion_model = primary.motion_model;
            definition->epoch_utc = primary.epoch_utc;
            definition->custom_bsp_path = primary.custom_bsp_path;
            definition->star_idx = primary.star_idx;
        }
        definition->bodies.reserve(total);
        definition->source_systems.reserve(total);
        definition->light_groups.reserve(total);
        definition->lit_by.resize(members.size());
        for (size_t slot = 0; slot < members.size(); ++slot) definition->lit_by[slot] = members[slot].lit_by;
        for (size_t slot = 0; slot < members.size(); ++slot) {
            const auto& source = members[slot].snapshot->definition();
            for (const auto& body : source.bodies) {
                int index = int(definition->bodies.size());
                definition->bodies.push_back(body);
                definition->bodies.back().parent_idx = shifted(body.parent_idx, map_.begins_[slot]);
                definition->source_systems.push_back(source.name);
                definition->light_groups.push_back(uint32_t(slot));
                definition->name_to_idx.emplace(body.name, index);
            }
        }
        scene_.definition_ = std::move(definition);
        ++scene_.definition_revision_;
        ++definition_rebuilds_;
    }

    if (hierarchies) {
        JacobiHierarchy jacobi, orbit;
        ReferenceHierarchy reference;
        for (size_t slot = 0; slot < members.size(); ++slot) {
            const auto& snapshot = *members[slot].snapshot;
            append(jacobi, snapshot.jacobi_hierarchy(), map_.begins_[slot]);
            append(orbit, snapshot.orbit_hierarchy(), map_.begins_[slot]);
            append(reference, snapshot.reference_hierarchy(), map_.begins_[slot]);
        }
        scene_.jacobi_ = std::make_shared<const JacobiHierarchy>(std::move(jacobi));
        scene_.reference_jacobi_ = std::make_shared<const JacobiHierarchy>(std::move(orbit));
        scene_.reference_ = std::make_shared<const ReferenceHierarchy>(std::move(reference));
        ++scene_.hierarchy_revision_;
        ++hierarchy_rebuilds_;
    }

    scene_.motions_.resize(total);
    scene_.render_.resize(total);
    scene_.revisions_.resize(total);
    bool layout = identities_.size() != total;
    identities_.resize(total);
    for (size_t slot = 0; slot < members.size(); ++slot) {
        const auto& snapshot = *members[slot].snapshot;
        glm::dvec3 offset = members[slot].offset_au;
        glm::dvec3 render_offset = ecliptic_to_render(EclipticPosition{offset}).value;
        for (size_t local = 0; local < snapshot.size(); ++local) {
            size_t index = map_.begins_[slot] + local;
            auto& motion = scene_.motions_[index] = snapshot.motions_[local];
            motion.position.value += offset;
            auto& render = scene_.render_[index] = snapshot.render_[local];
            render.position.value += render_offset;
            scene_.revisions_[index] = snapshot.revisions_[local];
            scene_.revisions_[index].lighting += lighting_epochs_[slot] << 32;
            layout = layout || identities_[index] != snapshot.revisions_[local].identity;
            identities_[index] = snapshot.revisions_[local].identity;
        }
    }
    scene_.time_ = members.empty() ? units::Years{} : members.front().snapshot->time_;
    if (state) ++scene_.state_revision_;
    if (layout) ++scene_.generation_;
    return scene_;
}

}
