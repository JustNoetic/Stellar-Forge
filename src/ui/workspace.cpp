#include "ui/workspace.hpp"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <algorithm>
#include <string>

namespace sf::ui {

void UiSystem::init(GLFWwindow* window, rhi::VkContext& ctx) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

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
}

void UiSystem::render(VkCommandBuffer cmd, rhi::VkContext& ctx, uint32_t swapchain_idx) {
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

} // namespace sf::ui
