#include "app.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <format>

namespace sf {

void App::render_settings_window(rendering::RenderSettings& render_draft) {
    if (!ui_.begin_settings(ui_settings_, ui_state_)) return;
    const auto page = static_cast<ui::SettingsSection>(ui_settings_.settings_section);
    if (page == ui::SettingsSection::Display) {
        ui::section("Presentation");
        ui::field("Exposure##optics_exp", [&](const char* id) {
            float speed = std::max(0.005f, render_draft.exposure * 0.02f);
            const char* fmt = (render_draft.exposure >= 10000.0f || (render_draft.exposure > 0.0f && render_draft.exposure < 0.005f)) ? "%.4g" : "%.3f";
            if (ui::drag_float(id, &render_draft.exposure, speed, 0.0f, 0.0f, fmt)) {
                render_draft.exposure = std::max(0.0f, render_draft.exposure);
            }
        });
        ImGui::Checkbox("HDR Rendering", &render_draft.hdr_enabled);
        ImGui::Checkbox("VSync", &render_draft.vsync);
        ui::section("Star appearance");
        ui::field("Point Diameter##star_pt", [&](const char* id) {
            ImGui::SliderFloat(id, &render_draft.star_point_size, 1.5f, 6.0f, "%.1f px");
        });
        ui::field("Star Brightness##star_br", [&](const char* id) {
            float star_speed = std::max(0.01f, render_draft.star_intensity * 0.02f);
            if (ui::drag_float(id, &render_draft.star_intensity, star_speed, 0.0f, 0.0f, "%.2f")) {
                render_draft.star_intensity = std::max(0.0f, render_draft.star_intensity);
            }
        });
    } else if (page == ui::SettingsSection::Scene) {
        ui::section("Celestial overlays");
        ImGui::Checkbox("Gaia DR3 Starfield", &render_draft.show_gaia_stars);
        ImGui::Checkbox("Orbit Lines", &render_draft.show_orbits);
        ImGui::BeginDisabled(!render_draft.show_orbits);
        ui::field("Orbit Opacity##orb_op", [&](const char* id) {
            ImGui::SliderFloat(id, &render_draft.orbit_opacity, 0.0f, 1.0f, "%.2f");
        });

        ui::field("Orbit Resolution", [&](const char* id) {
            int segments = static_cast<int>(render_draft.orbit_segments);
            if (ImGui::SliderInt(id, &segments, 128, 8192, "%d segments", ImGuiSliderFlags_Logarithmic))
                render_draft.orbit_segments = static_cast<uint32_t>(segments);
        });
        ImGui::EndDisabled();
    } else if (page == ui::SettingsSection::Atmosphere) {
        ui::section("Atmospheric effects");
        ImGui::Checkbox("Atmospheres", &render_draft.show_atmospheres);
        ImGui::BeginDisabled(!render_draft.show_atmospheres);
        ImGui::Checkbox("Atmospheric surface lighting", &render_draft.atmosphere_surface_lighting);
        ImGui::Checkbox("Aerial perspective", &render_draft.aerial_perspective_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds atmospheric scattered light over surfaces. Atmospheric extinction remains active when disabled.");
        ImGui::Checkbox("Atmospheric refraction", &render_draft.refraction_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Bends apparent positions of stars, bodies, terrain, rings and orbits through the nearest physical atmosphere.");

        ui::section("Atmospheric shadows");
        ImGui::Checkbox("Atmospheric shadows", &render_draft.atmosphere_shadows_enabled);
        ImGui::BeginDisabled(!render_draft.atmosphere_shadows_enabled);
        ImGui::Checkbox("Progressive atmosphere shadows", &render_draft.atmosphere_shadow_temporal);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Refines stationary views over 64 frames. Moving the eye, lighting or shadows resets immediately. Disable to see single-frame shadow sampling without history.");
        ImGui::BeginDisabled(!render_draft.atmosphere_shadow_temporal);
        ImGui::Checkbox("Reuse history while turning", &render_draft.atmosphere_shadow_rotation);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reprojects smooth sky shadows while rotating in place. Depth edges, narrow shadows and transparent endpoints use fresh integration while turning.");
        ImGui::EndDisabled();
        ui::field(render_draft.atmosphere_shadow_adaptive ? "Maximum shadow samples" : "Shadow samples", [&](const char* id) {
            const char* samples[] = {"4", "8", "16"};
            int quality = render_draft.atmosphere_shadow_samples == 4 ? 0 : render_draft.atmosphere_shadow_samples == 8 ? 1 : 2;
            if (ImGui::Combo(id, &quality, samples, 3)) render_draft.atmosphere_shadow_samples = 4u << quality;
        });
        if (self_test_) ImGui::SetNextItemOpen(true, ImGuiCond_Always);
        if (ImGui::CollapsingHeader("Shadow sampling & performance")) {
            ImGui::BeginDisabled(render_draft.atmosphere_shadow_temporal);
            ImGui::Checkbox("Animate shadow jitter", &render_draft.atmosphere_shadow_animate_jitter);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("With progressive shadows off, changes the raymarch sample positions every frame. Disable to freeze the noise pattern.");
            ImGui::EndDisabled();
            ui::field("Shadow noise", [&](const char* id) {
                const char* types[] = {"Interleaved Gradient Noise (IGN)", "Spatiotemporal Blue Noise (STBN)"};
                int noise = static_cast<int>(render_draft.atmosphere_shadow_noise_type);
                if (ImGui::Combo(id, &noise, types, 2)) render_draft.atmosphere_shadow_noise_type = static_cast<uint32_t>(noise);
            });
            ImGui::Checkbox("Adaptive shadow samples", &render_draft.atmosphere_shadow_adaptive);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uses fewer samples for short or vertical atmospheric columns. Grazing rays and sharp shadows retain more samples.");
            ImGui::Checkbox("Cached shadow weights", &render_draft.atmosphere_shadow_cached_weights);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reuses aerial-perspective light contributions for single-star ring shadows. Unsupported endpoints and atmosphere profiles retain native integration.");
            ImGui::BeginDisabled(render_draft.atmosphere_shadow_temporal);
            ImGui::Checkbox("Deterministic shadow strata", &render_draft.atmosphere_shadow_analytic);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Integrates single-ring shadows over each stratum without jitter and samples smooth eclipse penumbrae at stratum midpoints. Progressive shadows keep jittered accumulation.");
            ImGui::EndDisabled();
            ImGui::BeginDisabled(render_draft.atmosphere_shadow_temporal);
            ImGui::Checkbox("Atmospheric shadow VRS", &render_draft.atmosphere_shadow_vrs);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reuses smooth shadow fractions between pixels. Depth edges, narrow shadow detail and transparent endpoints use native resolution.");
            ImGui::BeginDisabled(!render_draft.atmosphere_shadow_vrs);
            ui::field("Shadow VRS rate", [&](const char* id) {
                const char* rates[] = {"1/2 (25% of pixels)", "1/4 (6.25% of pixels)", "1/8 (1.6% of pixels)", "1/16 (0.39% of pixels)"};
                int rate = std::bit_width(std::clamp(render_draft.atmosphere_shadow_vrs_scale, 2u, 16u))-2;
                if (ImGui::Combo(id, &rate, rates, 4)) render_draft.atmosphere_shadow_vrs_scale = 2u << rate;
            });
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Coarse shadow-fraction grid spacing per axis. Pixels failing the reuse guards still integrate natively.");
            ImGui::EndDisabled();
            ImGui::EndDisabled();
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
    } else if (page == ui::SettingsSection::Lighting) {
        ui::section("Illumination");
        ImGui::Checkbox("Direct sunlight only", &render_draft.direct_sunlight_only);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Disables atmospheric multiple scattering, ground skylight, planetshine and ringshine. Keeps direct sunlight, single atmospheric scattering and extinction.");
        if (render_draft.direct_sunlight_only) ImGui::TextWrapped("Turn off direct sunlight only to adjust reflected lighting.");
        ImGui::Checkbox("Eclipse refraction", &render_draft.eclipse_refraction_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds approximate atmospheric colouring to eclipse shadows using cached textures.");
        ImGui::BeginDisabled(render_draft.direct_sunlight_only);
        ImGui::Checkbox("Planetshine", &render_draft.planetshine_enabled);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(render_draft.direct_sunlight_only || !render_draft.planetshine_enabled || !render_draft.show_rings);
        ui::field("Planetshine on rings", [&](const char* id) {
            const char* modes[] = {"Analytical", "Numerical (64 samples)"};
            int mode = int(render_draft.ring_planetshine_mode);
            if (ImGui::Combo(id, &mode, modes, 2)) render_draft.ring_planetshine_mode = uint32_t(mode);
        });
        ImGui::EndDisabled();
        ui::section("Planetary rings");
        ImGui::Checkbox("Planetary Rings", &render_draft.show_rings);
        ImGui::BeginDisabled(!render_draft.show_rings);
        ImGui::Checkbox("Ring Shadows", &render_draft.ring_shadows_enabled);
        ImGui::BeginDisabled(!render_draft.ring_shadows_enabled);
        ui::field("Shadow Softness##ring_shadow_softness", [&](const char* id) {
            ImGui::SliderFloat(id, &render_draft.ring_shadow_softness, 0.25f, 4.0f, "%.2fx");
        });
        ImGui::EndDisabled();
        ImGui::BeginDisabled(render_draft.direct_sunlight_only);
        ImGui::Checkbox("Ringshine", &render_draft.ringshine_enabled);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(render_draft.direct_sunlight_only || !render_draft.ringshine_enabled);
        ui::field("Ringshine Bands##ringshine_bands", [&](const char* id) {
            int bands = static_cast<int>(render_draft.ringshine_band_count);
            if (ImGui::SliderInt(id, &bands, 4, 1024, "%d", ImGuiSliderFlags_Logarithmic))
                render_draft.ringshine_band_count = static_cast<uint32_t>(bands);
        });
        ui::value_row("Requested Radial Samples", std::format("{} per host", 4 * render_draft.ringshine_band_count));
        auto samples = celestial_renderer_.ringshine_sample_counts();
        if (!samples.empty()) {
            auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
            ui::value_row("Active Radial Samples", *minimum == *maximum
                ? std::format("{} per host", *minimum)
                : std::format("{} to {} per host", *minimum, *maximum));
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
    } else if (page == ui::SettingsSection::Surfaces) {
        ui::section("Surface appearance");
        ImGui::Checkbox("Surface Textures", &render_draft.surface_textures_enabled);
        ImGui::Checkbox("Imported Elevation", &render_draft.terrain_imported_enabled);
        ImGui::Checkbox("Detailed Terrain Shading", &render_draft.terrain_normal_maps_enabled);
        ImGui::Checkbox("Synthetic Terrain Preview", &render_draft.terrain_enabled);
        ui::section("Detail & performance", "Lower pixel error increases surface detail and GPU work.");
        ui::field("Terrain Error##terrain_error", [&](const char* id) {
            ImGui::SliderFloat(id, &render_draft.terrain_error_pixels, 0.5f, 16.0f, "%.1f px");
        });
        ImGui::BeginDisabled(!render_draft.terrain_enabled);
        ui::field("Synthetic Relief##terrain_height", [&](const char* id) {
            ImGui::SliderFloat(id, &render_draft.terrain_height_km, 0.0f, 20.0f, "%.2f km");
        });
        ImGui::EndDisabled();
        ui::field("Patches per Body##terrain_patches", [&](const char* id) {
            int limit = static_cast<int>(render_draft.terrain_patch_limit);
            if (ImGui::SliderInt(id, &limit, 64, 4096)) render_draft.terrain_patch_limit = static_cast<uint32_t>(limit);
        });
    } else if (page == ui::SettingsSection::Navigation) {
        ui::section("Camera");
        ui::field("Movement Mode", [&](const char* id) {
            const char* modes[] = {"Free Flight", "Simple Orbit"};
            ImGui::Combo(id, &camera_.movement_mode, modes, 2);
        });
        ui::field("Field of View", [&](const char* id) {
            const double minimum = 0.001, maximum = 120.0;
            ImGui::SliderScalar(id, ImGuiDataType_Double, &camera_.fov_deg, &minimum, &maximum, "%.3f deg", ImGuiSliderFlags_Logarithmic);
        });
        ui::field("Flight Speed", [&](const char* id) {
            const double minimum = 1e-12, maximum = MAX_FLIGHT_SPEED_AU_S;
            ImGui::SliderScalar(id, ImGuiDataType_Double, &camera_.flight_speed_au_s, &minimum, &maximum, "%.4g AU/s", ImGuiSliderFlags_Logarithmic);
        });
        ImGui::Checkbox("Auto Horizon Level (< 200 km)", &camera_.horizon_align);
    } else if (page == ui::SettingsSection::Workspace) {
        ui::section("Interface");
        ui::field("Interface Scale", [&](const char* id) {
            ImGui::SliderFloat(id, &ui_settings_.scale, 0.9f, 1.5f, "%.2fx");
        });
        ImGui::Checkbox("Aligned Panels", &ui_settings_.aligned_panels);
        ImGui::Checkbox("Show System Outliner", &ui_settings_.show_outliner);
        ImGui::Checkbox("Show Body Inspector", &ui_settings_.show_inspector);
        ImGui::Checkbox("Show Time Transport HUD", &ui_settings_.show_time_hud);
        ImGui::Checkbox("Show Engine Telemetry Window", &ui_settings_.show_telemetry);
        ImGui::Separator();
        ImGui::Checkbox("Confirm Object Deletion", &ui_settings_.confirm_delete);
        ImGui::Checkbox("Keplerian Motion for Compared Systems", &ui_settings_.comparison_force_keplerian);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Compared systems switch to analytical Keplerian motion when added, so they can follow jumps.");
        if (ImGui::Checkbox("Auto-Space Compared Systems", &ui_settings_.comparison_auto_spacing)) respace_system_comparisons();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Place compared systems along +X from each system's outermost orbit.");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Require confirmation dialog when deleting celestial bodies. Disable to quickly delete objects.");
        }
        ImGui::Separator();
        ui::field("Outliner Width", [&](const char* id) {
            ImGui::SliderFloat(id, &ui_settings_.outliner_width, 180.0f, 500.0f, "%.0f px");
        });
        ui::field("Inspector Width", [&](const char* id) {
            ImGui::SliderFloat(id, &ui_settings_.inspector_width, 300.0f, 700.0f, "%.0f px");
        });
        ImGui::Spacing();
        if (ImGui::Button("Reset Workspace Layout")) {
            ui_settings_.reset_layout();
        }

        ui::section("Distance display");
        ui::field("Light-Year Display Threshold (AU)", [&](const char* id) {
            ImGui::InputDouble(id, &ui_settings_.ly_threshold_au, 100.0, 1000.0, "%.1f");
            if (!std::isfinite(ui_settings_.ly_threshold_au)) ui_settings_.ly_threshold_au = DEFAULT_LY_THRESHOLD_AU;
            ui_settings_.ly_threshold_au = std::max(1.0, ui_settings_.ly_threshold_au);
        });
    } else if (page == ui::SettingsSection::Advanced) {
        ui::section("Experimental rendering");
        ImGui::Checkbox("Dual-source atmosphere blending", &render_draft.atmosphere_dual_source);
        ImGui::Checkbox("Fast ring shadow geometry (experimental)", &render_draft.atmosphere_shadow_fast_rings);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Simplifies projection math for single-ring atmospheric shadows while retaining the current ring filter. Grazing stellar horizons use the original calculation.");

        ui::section("Terrain diagnostics");
        ImGui::Checkbox("Terrain Tile Colors", &render_draft.terrain_debug_tiles);
        ImGui::Checkbox("FP64 Terrain Reference", &render_draft.terrain_reference_fp64);
        ImGui::Checkbox("FP64 Cached Terrain", &render_draft.terrain_cached_fp64);
        ImGui::Checkbox("Show Engine Telemetry Window", &ui_settings_.show_telemetry);
        if (ImGui::CollapsingHeader("Live rendering statistics")) {
            const auto& terrain = celestial_renderer_.terrain_stats();
            ui::value_row("Terrain Patches", std::format("{} / {}", terrain.patches, terrain::MAX_PATCHES));
            ui::value_row("Smooth Patches", std::to_string(terrain.smooth_patches));
            ui::value_row("Deepest Terrain Level", std::to_string(terrain.deepest_level));
            ui::value_row("Geometry Cache", std::format("{} cached / {} reference", terrain.cached_patches, terrain.reference_patches));
            ui::value_row("Geometry Bakes", std::format("{} / {} per frame", terrain.cache_bakes, terrain::MAX_GEOMETRY_BAKES));
            if (terrain.budget_limited) ImGui::TextDisabled("Patch budget reached; using coarser terrain.");
            const auto& textures = celestial_renderer_.texture_stats();
            ui::value_row("Surface Maps", std::format("{} bodies / {} resident tiles", textures.sources, textures.resident));
            ui::value_row("Texture Streaming", std::format("{} uploads / {} waiting", textures.uploads, textures.pending));
        }
    }

    auto current = capture_preferences();
    current.graphics = render_draft;
    std::string status = current == saved_preferences_ ? "Changes save automatically" : "Saving changes...";
    bool needs_attention = false;
    for (const auto* error : {&preferences_notice_, &preferences_save_error_, &ui_state_.settings_error}) {
        if (!error->empty()) {
            status = *error;
            needs_attention = true;
        }
    }
    if (ui_.end_settings(ui_settings_, status, needs_attention)) restore_preferences(render_draft);
}

}
