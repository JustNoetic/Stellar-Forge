#pragma once

#include "ephemeris/simulation.hpp"

namespace sf::ui {

struct BodyEditDraft {
    int body_index = -1;
    uint64_t system_generation = 0;
    uint64_t body_revision = 0;
    ephemeris::BodyPhysicalProperties values{};
    bool dirty = false;
    std::string error;

    void reload(const ephemeris::SystemSnapshot& snapshot, int index) {
        body_index = index;
        system_generation = snapshot.generation();
        body_revision = snapshot.body(index).revisions.definition;
        values = ephemeris::BodyPhysicalProperties::from(snapshot.body(index).definition);
        dirty = false;
        error.clear();
    }

    void synchronize(const ephemeris::SystemSnapshot& snapshot, int index) {
        if (system_generation != snapshot.generation() || body_index != index ||
            (!dirty && body_revision != snapshot.body(index).revisions.definition)) reload(snapshot, index);
    }

    ephemeris::ApplyBodyPhysicalProperties command() const {
        return {system_generation, body_index, body_revision, values};
    }
};

struct WorkspaceState {
    int inspected_body = -1;
    BodyEditDraft body_draft{};
    std::string settings_error;

    bool has_selection() const noexcept { return inspected_body >= 0; }
    bool has_selection(const ephemeris::SystemSnapshot& snapshot) const noexcept {
        return inspected_body >= 0 && static_cast<size_t>(inspected_body) < snapshot.size();
    }
};

}
