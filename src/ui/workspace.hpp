#pragma once

#include "rhi/vk_context.hpp"
#include <imgui.h>
#include <string_view>

struct GLFWwindow;

namespace sf::ui {

inline constexpr ImVec4 ACCENT{0.38f, 0.78f, 0.86f, 1.0f};
inline constexpr ImVec4 AMBER{0.94f, 0.70f, 0.38f, 1.0f};

struct WorkspaceSettings {
    bool ui_visible = true;
    bool aligned_panels = true;
    bool show_outliner = true;
    bool show_inspector = true;
    bool show_time_hud = true;
    bool has_inspected_body = true;
    float outliner_width = 260.0f;
    float inspector_width = 420.0f;
    float scale = 1.0f;
    uint32_t layout_revision = 0;
};

struct WorkspaceLayout {
    float width = 0.0f;
    float height = 0.0f;
    float gap = 8.0f;
    float top = 31.0f;
    float bottom = 0.0f;
    float outliner_width = 260.0f;
    float inspector_width = 420.0f;
    float left = 8.0f;
    float right = 0.0f;
    float transport_height = 76.0f;
    float scale = 1.0f;

    float panel_height() const noexcept {
        return (bottom - top) > 40.0f ? (bottom - top) : 40.0f;
    }
};

enum class PanelSide { Left, Right };

class UiSystem {
public:
    void init(GLFWwindow* window, rhi::VkContext& ctx);
    void shutdown(rhi::VkContext& ctx);

    void begin_frame(const WorkspaceSettings& settings);
    void render(VkCommandBuffer cmd, rhi::VkContext& ctx, uint32_t swapchain_idx);

    WorkspaceLayout compute_layout(const WorkspaceSettings& settings, float fb_width, float fb_height) const;
    ImGuiWindowFlags prepare_side_panel(const WorkspaceSettings& settings, PanelSide side, WorkspaceLayout& out_layout);

private:
    void apply_theme(float scale);

    float applied_scale_ = -1.0f;
};

void heading(std::string_view title, std::string_view subtitle = {});
void section(std::string_view title, std::string_view note = {});
void value_row(std::string_view label, std::string_view value);
void metric_text(std::string_view text);

template <typename Fn>
auto field(std::string_view label, Fn&& widget_fn) {
    auto hash_pos = label.find("##");
    std::string_view visible = hash_pos != std::string_view::npos ? label.substr(0, hash_pos) : label;
    ImGui::TextWrapped("%.*s", static_cast<int>(visible.size()), visible.data());
    ImGui::PushItemWidth(-1.0f);
    std::string hidden_id = "##";
    hidden_id.append(label);
    if constexpr (std::is_void_v<decltype(widget_fn(hidden_id.c_str()))>) {
        widget_fn(hidden_id.c_str());
        ImGui::PopItemWidth();
    } else {
        auto result = widget_fn(hidden_id.c_str());
        ImGui::PopItemWidth();
        return result;
    }
}

} // namespace sf::ui
