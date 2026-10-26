#include "ui/size_comparison.hpp"
#include <algorithm>
#include <format>
#include <map>
#include <numeric>

namespace sf::ui {

void size_comparison_labels(const comparison::SizeComparison& scene, const comparison::SizeView& view,
    glm::dvec4 bounds, glm::dvec2 viewport, float scale) {
    auto* draw = ImGui::GetBackgroundDrawList();
    auto white = ImGui::GetColorU32(ImVec4(.87f,.91f,1,1));
    auto muted = ImGui::GetColorU32(ImVec4(.48f,.56f,.68f,1));
    std::map<std::string,size_t> names;
    for (const auto& b : scene.bodies) ++names[b.definition().name];
    std::vector<size_t> order(scene.bodies.size());
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b) { return scene.bodies[a].radius > scene.bodies[b].radius; });
    std::vector<glm::dvec4> occupied;
    double pixels = viewport.y/view.height;
    for (size_t index : order) {
        const auto& b = scene.bodies[index];
        const auto& def = b.definition();
        double r = b.radius*pixels;
        if (r < 2) continue;
        auto p = view.world_to_screen(b.position,viewport);
        if (p.x+r < 0 || p.x-r > viewport.x || p.y+r < 0 || p.y-r > viewport.y) continue;
        std::string name = def.name;
        if (names[name] > 1) name += " ("+b.system->name+")";
        auto radius = std::format("{:.0f} km radius",b.radius/comparison::SIZE_SCALE);
        auto digits = radius.find(' ');
        for (int separator = int(digits)-3; separator > 0; separator -= 3) radius.insert(size_t(separator),",");
        double nw = ImGui::CalcTextSize(name.c_str()).x, rw = ImGui::CalcTextSize(radius.c_str()).x;
        bool star = def.is_star || def.is_black_hole;
        bool moon = b.parent >= 0 && size_t(b.parent) < scene.bodies.size() &&
            !scene.bodies[b.parent].definition().is_star && !scene.bodies[b.parent].definition().is_black_hole;
        double tx = star ? p.x-r-16*scale-nw : moon ? p.x+r+8*scale : p.x-nw*.5;
        double rx = star ? p.x-r-16*scale-rw : moon ? p.x+r+8*scale : p.x-rw*.5;
        double ty = star || moon ? p.y-16*scale : p.y-std::max(b.radius,b.outer*.52)*pixels-35*scale;
        if (star && std::min(tx,rx) < bounds.x) { tx = p.x-nw*.5; rx = p.x-rw*.5; ty = p.y-r-35*scale; }
        glm::dvec4 box{std::min(tx,rx)-6*scale,ty-5*scale,std::max(tx+nw,rx+rw)+6*scale,ty+34*scale};
        if (box.x < bounds.x || box.z > bounds.z || box.y < bounds.y || box.w > bounds.w) continue;
        if (std::any_of(occupied.begin(),occupied.end(),[&](auto other) {
            return box.x < other.z && box.z > other.x && box.y < other.w && box.w > other.y;
        })) continue;
        occupied.push_back(box);
        draw->AddText(ImVec2(float(tx),float(ty)),white,name.c_str());
        draw->AddText(ImVec2(float(rx),float(ty+16*scale)),muted,radius.c_str());
    }
}

bool size_comparison_hud(const WorkspaceLayout& layout,size_t systems,size_t bodies) {
    ImGui::SetNextWindowPos(ImVec2((layout.left+layout.right)*.5f,layout.top),ImGuiCond_Always,ImVec2(.5f,0));
    ImGui::SetNextWindowBgAlpha(.85f);
    ImGui::Begin("Size Comparison###size_comparison_hud",nullptr,ImGuiWindowFlags_NoDecoration|
        ImGuiWindowFlags_AlwaysAutoResize|ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoMove);
    ImGui::TextColored(ACCENT,"Size Comparator | Relative Scale");
    ImGui::SameLine();
    bool fit = ImGui::SmallButton("Fit All (F)");
    if (ImGui::IsWindowHovered()) ImGui::SetTooltip("%zu systems, %zu bodies\nDrag to pan; scroll to zoom; double-click to focus.",systems,bodies);
    ImGui::End();
    return fit;
}

}
