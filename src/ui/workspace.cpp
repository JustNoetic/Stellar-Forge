#include "ui/workspace.hpp"
#include "core/path_utils.hpp"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>

namespace sf::ui {

static UiSystem* s_active_ui_system = nullptr;

void UiSystem::init(GLFWwindow* window, rhi::VkContext& ctx) {
    window_ = window;
    s_active_ui_system = this;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    std::filesystem::create_directories(paths::external_path("cache"));
    static std::string ini_path = paths::external_path("cache", "imgui.ini").string();
    io.IniFilename = ini_path.c_str();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    io.Fonts->AddFontDefault();

    ImFontConfig fallback_cfg;
    fallback_cfg.MergeMode = true;
    fallback_cfg.PixelSnapH = true;
    fallback_cfg.OversampleH = 1;
    fallback_cfg.OversampleV = 1;
    fallback_cfg.RasterizerMultiply = 1.5f;

    static const ImWchar symbol_ranges[] = {
        0x00A0, 0x00FF, // Latin-1 Supplement (°, ², ³, etc.)
        0x0370, 0x03FF, // Greek and Coptic (Ω, ω, ν, etc.)
        0x2070, 0x209F, // Superscripts and Subscripts
        0x2200, 0x22FF, // Mathematical Operators (⊕, etc.)
        0x25A0, 0x25FF, // Geometric Shapes
        0x2600, 0x26FF, // Miscellaneous Symbols (☉, ☽, ♃, ♄, etc.)
        0x2700, 0x27BF, // Dingbats
        0,
    };

    const char* win_dir = std::getenv("WINDIR");
    std::filesystem::path fonts_dir = win_dir ? (std::filesystem::path(win_dir) / "Fonts") : "C:\\Windows\\Fonts";

    const std::filesystem::path candidate_fonts[] = {
        fonts_dir / "seguisym.ttf",
        fonts_dir / "seguiemj.ttf",
        fonts_dir / "arial.ttf",
    };

    for (const auto& font_path : candidate_fonts) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(font_path, ec)) {
            io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), 13.0f, &fallback_cfg, symbol_ranges);
            break;
        }
    }

    ImGui_ImplGlfw_InitForVulkan(window, true);

    VkFormat color_fmt = ctx.swapchain_format();
    VkPipelineRenderingCreateInfo dyn_rendering_ci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    dyn_rendering_ci.colorAttachmentCount = 1;
    dyn_rendering_ci.pColorAttachmentFormats = &color_fmt;

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = ctx.instance();
    init_info.PhysicalDevice = ctx.physical_device();
    init_info.Device = ctx.device();
    init_info.QueueFamily = ctx.graphics_queue_family();
    init_info.Queue = ctx.graphics_queue();
    init_info.DescriptorPool = ctx.imgui_descriptor_pool();
    init_info.MinImageCount = 2;
    init_info.ImageCount = std::max(2u, ctx.swapchain_image_count());
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;
    init_info.PipelineRenderingCreateInfo = dyn_rendering_ci;

    ImGui_ImplVulkan_Init(&init_info);
    apply_theme(1.0f);
}

void UiSystem::shutdown(rhi::VkContext& ctx) {
    if (is_dragging_ && window_) {
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        is_dragging_ = false;
    }
    s_active_ui_system = nullptr;
    window_ = nullptr;
    ctx.wait_idle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UiSystem::begin_frame(const WorkspaceSettings& settings) {
    apply_theme(settings.scale);
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    update_drag_interaction();
}

void UiSystem::render(VkCommandBuffer cmd, rhi::VkContext& ctx, uint32_t swapchain_idx) {
    update_drag_interaction();
    ImGui::Render();

    rhi::RenderingAttachment att{};
    att.view = ctx.swapchain_view(swapchain_idx);
    att.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.store_op = VK_ATTACHMENT_STORE_OP_STORE;

    ctx.begin_rendering(cmd, ctx.swapchain_extent(), std::span{&att, 1}, nullptr);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    ctx.end_rendering(cmd);
}

void UiSystem::notify_drag_active(ImGuiID id) {
    active_drag_id_ = id;
}

void UiSystem::update_drag_interaction() {
    if (!window_) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    ImGuiContext* g = ImGui::GetCurrentContext();
    if (!g) {
        return;
    }

    bool mouse_down = io.MouseDown[0];
    bool is_focused = glfwGetWindowAttrib(window_, GLFW_FOCUSED) != 0;

    if (is_dragging_) {
        bool should_end = !mouse_down || !is_focused || (g->ActiveId == 0) || (active_drag_id_ != 0 && g->ActiveId != active_drag_id_);
        if (should_end) {
            glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwSetCursorPos(window_, static_cast<double>(drag_origin_.x), static_cast<double>(drag_origin_.y));
            io.MousePos = drag_origin_;
            io.MousePosPrev = drag_origin_;
            io.MouseDelta = ImVec2(0.0f, 0.0f);
            io.WantSetMousePos = false;
            is_dragging_ = false;
            active_drag_id_ = 0;
        } else {
            ImGui::SetMouseCursor(ImGuiMouseCursor_None);
        }
    } else {
        if (mouse_down && is_focused && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0f)) {
            bool is_drag_widget = false;
            if (active_drag_id_ != 0 && g->ActiveId == active_drag_id_) {
                is_drag_widget = true;
            } else if (g->ActiveId != 0 && g->ActiveIdSource == ImGuiInputSource_Mouse &&
                       g->MovingWindow == nullptr && !ImGui::TempInputIsActive(g->ActiveId) &&
                       g->InputTextState.ID != g->ActiveId &&
                       (g->ActiveIdUsingNavDirMask & ((1 << ImGuiDir_Left) | (1 << ImGuiDir_Right))) != 0) {
                is_drag_widget = true;
                active_drag_id_ = g->ActiveId;
            }

            if (is_drag_widget) {
                is_dragging_ = true;
                drag_origin_ = io.MouseClickedPos[0];
                glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                ImGui::SetMouseCursor(ImGuiMouseCursor_None);
            }
        }
    }
}

WorkspaceLayout UiSystem::compute_layout(const WorkspaceSettings& settings, const WorkspaceState& state, float fb_width, float fb_height) const {
    ImGuiIO& io = ImGui::GetIO();
    float width = io.DisplaySize.x > 0.0f ? io.DisplaySize.x : fb_width;
    float height = io.DisplaySize.y > 0.0f ? io.DisplaySize.y : fb_height;
    float scale = std::clamp(settings.scale, 0.9f, 1.5f);
    float gap = 8.0f * scale;
    float menu_height = ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2.0f;
    float top = (menu_height > 0.0f ? menu_height : 23.0f * scale) + gap;
    float transport_height = 76.0f * scale;
    float bottom = height - gap;
    if (settings.show_time_hud) {
        bottom -= transport_height + gap;
    }
    float outliner_w = std::min(settings.outliner_width * scale, width * 0.25f);
    float inspector_w = std::min(settings.inspector_width * scale, width * 0.40f);
    float left = settings.show_outliner ? (outliner_w + gap * 2.0f) : gap;
    bool show_insp = settings.show_inspector && state.has_selection();
    float right = width - (show_insp ? (inspector_w + gap * 2.0f) : gap);

    return WorkspaceLayout{
        width, height, gap, top, bottom,
        outliner_w, inspector_w, left, right,
        transport_height, scale
    };
}

ImGuiWindowFlags UiSystem::prepare_side_panel(const WorkspaceSettings& settings, const WorkspaceState& state, PanelSide side, WorkspaceLayout& out_layout) {
    out_layout = compute_layout(settings, state, 0.0f, 0.0f);
    float width = (side == PanelSide::Left) ? out_layout.outliner_width : out_layout.inspector_width;
    float x = (side == PanelSide::Left) ? out_layout.gap : (out_layout.width - width - out_layout.gap);

    ImGuiCond cond = settings.aligned_panels ? ImGuiCond_Always : ImGuiCond_FirstUseEver;
    ImGui::SetNextWindowPos(ImVec2(x, out_layout.top), cond);
    ImGui::SetNextWindowSize(ImVec2(width, out_layout.panel_height()), cond);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;
    if (settings.aligned_panels) {
        flags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
    }
    return flags;
}

void UiSystem::apply_theme(float raw_scale) {
    float scale = std::clamp(raw_scale, 0.9f, 1.5f);
    if (applied_scale_ == scale) {
        return;
    }
    applied_scale_ = scale;

    ImGui::GetIO().FontGlobalScale = scale;
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(12.0f * scale, 10.0f * scale);
    style.FramePadding = ImVec2(7.0f * scale, 5.0f * scale);
    style.ItemSpacing = ImVec2(8.0f * scale, 7.0f * scale);
    style.ItemInnerSpacing = ImVec2(6.0f * scale, 4.0f * scale);
    style.IndentSpacing = 14.0f * scale;
    style.ScrollbarSize = 10.0f * scale;
    style.WindowRounding = 5.0f * scale;
    style.ChildRounding = 4.0f * scale;
    style.FrameRounding = 3.0f * scale;
    style.PopupRounding = 5.0f * scale;
    style.GrabRounding = 3.0f * scale;
    style.TabRounding = 3.0f * scale;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = ImVec4(0.87f, 0.91f, 0.95f, 1.0f);
    c[ImGuiCol_TextDisabled] = ImVec4(0.48f, 0.56f, 0.64f, 1.0f);
    c[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.075f, 0.105f, 0.97f);
    c[ImGuiCol_ChildBg] = ImVec4(0.045f, 0.062f, 0.088f, 0.65f);
    c[ImGuiCol_PopupBg] = ImVec4(0.075f, 0.10f, 0.14f, 0.99f);
    c[ImGuiCol_Border] = ImVec4(0.18f, 0.24f, 0.30f, 0.65f);
    c[ImGuiCol_FrameBg] = ImVec4(0.11f, 0.15f, 0.20f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.15f, 0.25f, 0.31f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.18f, 0.31f, 0.38f, 1.0f);
    c[ImGuiCol_TitleBg] = ImVec4(0.07f, 0.10f, 0.14f, 1.0f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.10f, 0.17f, 0.22f, 1.0f);
    c[ImGuiCol_MenuBarBg] = ImVec4(0.06f, 0.085f, 0.12f, 1.0f);
    c[ImGuiCol_Button] = ImVec4(0.13f, 0.20f, 0.26f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.19f, 0.34f, 0.40f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.23f, 0.43f, 0.50f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.13f, 0.27f, 0.33f, 0.80f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.37f, 0.44f, 0.85f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.23f, 0.43f, 0.50f, 1.0f);
    c[ImGuiCol_Tab] = ImVec4(0.10f, 0.15f, 0.20f, 1.0f);
    c[ImGuiCol_TabHovered] = ImVec4(0.20f, 0.37f, 0.44f, 1.0f);
    c[ImGuiCol_TabActive] = ImVec4(0.15f, 0.29f, 0.35f, 1.0f);
    c[ImGuiCol_CheckMark] = ACCENT;
    c[ImGuiCol_SliderGrab] = ImVec4(ACCENT.x, ACCENT.y, ACCENT.z, 0.85f);
    c[ImGuiCol_SliderGrabActive] = ACCENT;
    c[ImGuiCol_Separator] = ImVec4(0.18f, 0.24f, 0.30f, 0.70f);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.04f, 0.06f, 0.08f, 0.60f);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.28f, 0.34f, 1.0f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.28f, 0.40f, 0.46f, 1.0f);
    c[ImGuiCol_ScrollbarGrabActive] = ACCENT;
    c[ImGuiCol_PlotHistogram] = AMBER;
}

void heading(std::string_view title, std::string_view subtitle) {
    ImGui::TextColored(ACCENT, "%.*s", static_cast<int>(title.size()), title.data());
    if (!subtitle.empty()) {
        ImGui::TextWrapped("%.*s", static_cast<int>(subtitle.size()), subtitle.data());
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
}

void section(std::string_view title, std::string_view note) {
    ImGui::Spacing();
    ImGui::TextColored(AMBER, "%.*s", static_cast<int>(title.size()), title.data());
    ImGui::Separator();
    if (!note.empty()) {
        ImGui::TextWrapped("%.*s", static_cast<int>(note.size()), note.data());
    }
}

void value_row(std::string_view label, std::string_view value) {
    std::string l_str(label);
    std::string v_str(value);
    float start = ImGui::GetCursorPosX();
    float available = ImGui::GetContentRegionAvail().x;
    float label_w = ImGui::CalcTextSize(l_str.c_str()).x;
    float value_w = ImGui::CalcTextSize(v_str.c_str()).x;
    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;

    ImGui::TextDisabled("%s", l_str.c_str());
    if (label_w + value_w + spacing * 2.0f <= available) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(std::max(start + label_w + spacing, start + available - value_w));
        ImGui::TextUnformatted(v_str.c_str());
    } else {
        ImGui::TextWrapped("%s", v_str.c_str());
    }
}

void metric_text(std::string_view text) {
    auto pos = text.find(':');
    if (pos != std::string_view::npos && pos + 1 < text.size()) {
        std::string_view lhs = text.substr(0, pos);
        std::string_view rhs = text.substr(pos + 1);
        while (! rhs.empty() && rhs.front() == ' ') rhs.remove_prefix(1);
        value_row(lhs, rhs);
    } else {
        ImGui::TextWrapped("%.*s", static_cast<int>(text.size()), text.data());
    }
}

bool drag_float(const char* label, float* v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) {
    bool changed = ImGui::DragFloat(label, v, v_speed, v_min, v_max, format, flags);
    if (ImGui::IsItemActive() && s_active_ui_system) {
        s_active_ui_system->notify_drag_active(ImGui::GetItemID());
    }
    return changed;
}

bool drag_float2(const char* label, float v[2], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) {
    bool changed = ImGui::DragFloat2(label, v, v_speed, v_min, v_max, format, flags);
    if (ImGui::IsItemActive() && s_active_ui_system) {
        s_active_ui_system->notify_drag_active(ImGui::GetItemID());
    }
    return changed;
}

bool drag_float3(const char* label, float v[3], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) {
    bool changed = ImGui::DragFloat3(label, v, v_speed, v_min, v_max, format, flags);
    if (ImGui::IsItemActive() && s_active_ui_system) {
        s_active_ui_system->notify_drag_active(ImGui::GetItemID());
    }
    return changed;
}

bool drag_float4(const char* label, float v[4], float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags) {
    bool changed = ImGui::DragFloat4(label, v, v_speed, v_min, v_max, format, flags);
    if (ImGui::IsItemActive() && s_active_ui_system) {
        s_active_ui_system->notify_drag_active(ImGui::GetItemID());
    }
    return changed;
}

bool render_star_creator_widgets(StarCreatorDraft& draft) {
    bool changed = false;

    const char* modes[] = {"Evolution Track", "Surface Physics"};
    int mode_idx = (draft.params.mode == physics::StarCreationMode::Evolution) ? 0 : 1;
    if (ImGui::Combo("Creation Mode##star_mode", &mode_idx, modes, 2)) {
        draft.params.mode = (mode_idx == 0) ? physics::StarCreationMode::Evolution : physics::StarCreationMode::Surface;
        changed = true;
    }

    ImGui::Spacing();

    if (draft.params.mode == physics::StarCreationMode::Evolution) {
        float mass_f = static_cast<float>(draft.params.mass_msun);
        if (drag_float("Mass (M\u2609)##star_mass", &mass_f, 0.05f, 0.01f, 300.0f, "%.4f")) {
            draft.params.mass_msun = std::max(0.01, static_cast<double>(mass_f));
            changed = true;
        }

        float met_f = static_cast<float>(draft.params.metallicity_dex);
        if (drag_float("[Fe/H] (dex)##star_met", &met_f, 0.01f, -4.0f, 1.0f, "%.3f")) {
            draft.params.metallicity_dex = static_cast<double>(met_f);
            changed = true;
        }

        float age_pct_100 = static_cast<float>(draft.params.age_pct * 100.0);
        if (drag_float("Life Cycle (%)##star_age", &age_pct_100, 0.1f, -5.0f, 120.0f, "%.1f%%")) {
            draft.params.age_pct = static_cast<double>(age_pct_100) / 100.0;
            changed = true;
        }

        if (draft.params.mass_msun >= 45.0) {
            ImGui::TextColored(AMBER, "Humphreys-Davidson Limit (Forced Stripping)");
            if (draft.params.evo_path != physics::EvolutionPath::Stripping) {
                draft.params.evo_path = physics::EvolutionPath::Stripping;
                changed = true;
            }
        } else if (draft.params.mass_msun >= 25.0) {
            const char* branches[] = {"Red Supergiant (Standard)", "LBV -> Wolf-Rayet (Stripping)"};
            int branch_idx = (draft.params.evo_path == physics::EvolutionPath::Standard) ? 0 : 1;
            if (ImGui::Combo("Evolution Branch##star_branch", &branch_idx, branches, 2)) {
                draft.params.evo_path = (branch_idx == 0) ? physics::EvolutionPath::Standard : physics::EvolutionPath::Stripping;
                changed = true;
            }
        } else {
            draft.params.evo_path = physics::EvolutionPath::Standard;
        }

        float rot_f = static_cast<float>(draft.params.rot_frac);
        if (drag_float("Rotation (\u03C9/\u03C9_crit)##star_rot", &rot_f, 0.01f, 0.0f, 1.0f, "%.2f")) {
            draft.params.rot_frac = std::clamp(static_cast<double>(rot_f), 0.0, 1.0);
            changed = true;
        }

        float inc_f = static_cast<float>(draft.params.inclination_deg);
        if (drag_float("Inclination (\u00B0)##star_inc", &inc_f, 0.5f, 0.0f, 90.0f, "%.1f\u00B0")) {
            draft.params.inclination_deg = std::clamp(static_cast<double>(inc_f), 0.0, 90.0);
            changed = true;
        }
    } else {
        ImGui::TextDisabled("Lock exactly 2 properties (3rd is solved via Stefan-Boltzmann):");

        int num_locked = (draft.params.locked_rad ? 1 : 0) + (draft.params.locked_temp ? 1 : 0) + (draft.params.locked_lum ? 1 : 0);

        bool c1 = ImGui::Checkbox("##lock_r", &draft.params.locked_rad);
        ImGui::SameLine();
        float rad_f = static_cast<float>(draft.params.radius_rsun);
        if (drag_float("Radius (R\u2609)##s_rad", &rad_f, 0.05f, 0.01f, 2500.0f, "%.4f")) {
            draft.params.radius_rsun = std::max(0.001, static_cast<double>(rad_f));
            changed = true;
        }

        bool c2 = ImGui::Checkbox("##lock_t", &draft.params.locked_temp);
        ImGui::SameLine();
        float temp_f = static_cast<float>(draft.params.temp_k);
        if (drag_float("Temp (K)##s_temp", &temp_f, 10.0f, 500.0f, 100000.0f, "%.0f K")) {
            draft.params.temp_k = std::max(100.0, static_cast<double>(temp_f));
            changed = true;
        }

        bool c3 = ImGui::Checkbox("##lock_l", &draft.params.locked_lum);
        ImGui::SameLine();
        float lum_f = static_cast<float>(draft.params.lum_lsun);
        if (drag_float("Luminosity (L\u2609)##s_lum", &lum_f, 0.05f, 0.0001f, 2000000.0f, "%.4f")) {
            draft.params.lum_lsun = std::max(1e-6, static_cast<double>(lum_f));
            changed = true;
        }

        if (c1 || c2 || c3) {
            num_locked = (draft.params.locked_rad ? 1 : 0) + (draft.params.locked_temp ? 1 : 0) + (draft.params.locked_lum ? 1 : 0);
            if (num_locked > 2) {
                if (c1) { if (draft.params.locked_temp) draft.params.locked_temp = false; else draft.params.locked_lum = false; }
                else if (c2) { if (draft.params.locked_rad) draft.params.locked_rad = false; else draft.params.locked_lum = false; }
                else if (c3) { if (draft.params.locked_rad) draft.params.locked_rad = false; else draft.params.locked_temp = false; }
            }
            changed = true;
        }

        if (num_locked != 2) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Select exactly 2 locked properties!");
        }

        float rot_f = static_cast<float>(draft.params.rot_frac);
        if (drag_float("Rotation (\u03C9/\u03C9_crit)##star_rot_s", &rot_f, 0.01f, 0.0f, 1.0f, "%.2f")) {
            draft.params.rot_frac = std::clamp(static_cast<double>(rot_f), 0.0, 1.0);
            changed = true;
        }

        float inc_f = static_cast<float>(draft.params.inclination_deg);
        if (drag_float("Inclination (\u00B0)##star_inc_s", &inc_f, 0.5f, 0.0f, 90.0f, "%.1f\u00B0")) {
            draft.params.inclination_deg = std::clamp(static_cast<double>(inc_f), 0.0, 90.0);
            changed = true;
        }
    }

    if (changed || draft.preview.mass_msun <= 0.0) {
        draft.recompute();
    }

    const auto& pr = draft.preview;
    ImGui::Separator();
    section("Stellar Properties", "Live astrophysical model preview");

    ImGui::ColorButton("##star_swatch",
                       ImVec4(pr.color_rgb.r, pr.color_rgb.g, pr.color_rgb.b, 1.0f),
                       ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
                       ImVec2(34.0f, 34.0f));
    ImGui::SameLine(0.0f, 10.0f);
    ImGui::BeginGroup();
    ImGui::TextColored(ACCENT, "%s", pr.spectral_class.c_str());
    ImGui::TextDisabled("%s", pr.stage_name.c_str());
    ImGui::EndGroup();

    value_row("Temperature", std::format("{:.0f} K", pr.temp_k));
    value_row("Luminosity", std::format("{:.4f} L\u2609", pr.lum_lsun));
    value_row("Radius", std::format("{:.4f} R\u2609 ({:.0f} km)", pr.radius_rsun, pr.radius_rsun * SOLAR_RADIUS_KM));
    value_row("Mass", std::format("{:.4f} M\u2609", pr.mass_msun));
    value_row("Surface Gravity", std::format("log g = {:.2f} ({:.1f} m/s\u00B2)", pr.log_g, std::pow(10.0, pr.log_g) / 100.0));
    value_row("Mean Density", std::format("{:.3f} g/cm\u00B3", pr.density_g_cm3));
    value_row("Escape Speed", std::format("{:.1f} km/s", pr.escape_vel_km_s));

    if (draft.params.mode == physics::StarCreationMode::Evolution) {
        if (pr.age_gyr < 0.1) {
            value_row("Age", std::format("{:.1f} Myr / {:.2f} Gyr ({:.1f}%)", pr.age_gyr * 1000.0, pr.lifespan_gyr, pr.age_pct));
        } else {
            value_row("Age", std::format("{:.3f} Gyr / {:.2f} Gyr ({:.1f}%)", pr.age_gyr, pr.lifespan_gyr, pr.age_pct));
        }
    }

    if (draft.params.rot_frac > 0.01) {
        value_row("Oblateness (f)", std::format("{:.4f}", pr.oblateness));
        value_row("Equatorial Speed", std::format("{:.1f} km/s", pr.v_eq_km_s));
        value_row("Poles / Equator", std::format("{:.0f} K / {:.0f} K", pr.t_pole_k, pr.t_eq_k));
    }

    if (pr.is_pulsating) {
        ImGui::TextColored(AMBER, "Pulsating Star: %s (%s)", pr.pulsation_type.c_str(), pr.pulsation_period_str.c_str());
    }

    if (pr.hz_inner_au > 0.0) {
        value_row("Habitable Zone", std::format("{:.2f} AU - {:.2f} AU", pr.hz_inner_au, pr.hz_outer_au));
    }

    return changed;
}

} // namespace sf::ui
