#pragma once

#include "comparison/size_comparison.hpp"
#include "ui/workspace.hpp"

namespace sf::ui {

void size_comparison_labels(const comparison::SizeComparison& scene, const comparison::SizeView& view,
                            glm::dvec4 bounds, glm::dvec2 viewport, float scale);
bool size_comparison_hud(const WorkspaceLayout& layout, size_t systems, size_t bodies);

}
