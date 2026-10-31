#include "core/user_settings.hpp"
#include "core/path_utils.hpp"
#include <cmath>
#include <limits>
#include <type_traits>

namespace sf {
namespace {

template <typename Settings, typename Visitor>
void graphics_fields(Settings& settings, Visitor&& visit) {
#define SF_SETTING(name) visit(#name, settings.name)
    SF_SETTING(exposure);
    SF_SETTING(hdr_enabled);
    SF_SETTING(bloom_mode);
    SF_SETTING(bloom_strength);
    SF_SETTING(bloom_highlight_limit);
    SF_SETTING(bloom_radius);
    SF_SETTING(bloom_diffraction);
    SF_SETTING(bloom_rotation_deg);
    SF_SETTING(bloom_spike_threshold);
    SF_SETTING(show_gaia_stars);
    SF_SETTING(show_orbits);
    SF_SETTING(orbit_opacity);
    SF_SETTING(orbit_segments);
    SF_SETTING(star_point_size);
    SF_SETTING(star_intensity);
    SF_SETTING(vsync);
    SF_SETTING(terrain_enabled);
    SF_SETTING(terrain_imported_enabled);
    SF_SETTING(terrain_normal_maps_enabled);
    SF_SETTING(surface_textures_enabled);
    SF_SETTING(terrain_debug_tiles);
    SF_SETTING(terrain_reference_fp64);
    SF_SETTING(terrain_cached_fp64);
    SF_SETTING(terrain_error_pixels);
    SF_SETTING(terrain_height_km);
    SF_SETTING(terrain_patch_limit);
    SF_SETTING(show_rings);
    SF_SETTING(transparency_reference_legacy);
    SF_SETTING(show_atmospheres);
    SF_SETTING(direct_sunlight_only);
    SF_SETTING(aerial_perspective_enabled);
    SF_SETTING(refraction_enabled);
    SF_SETTING(eclipse_refraction_enabled);
    SF_SETTING(eclipse_refraction_probe);
    SF_SETTING(atmosphere_dual_source);
    SF_SETTING(atmosphere_surface_lighting);
    SF_SETTING(atmosphere_shadows_enabled);
    SF_SETTING(atmosphere_shadow_reference);
    SF_SETTING(atmosphere_shadow_cached_weights);
    SF_SETTING(atmosphere_shadow_fast_rings);
    SF_SETTING(atmosphere_shadow_analytic);
    SF_SETTING(atmosphere_shadow_temporal);
    SF_SETTING(atmosphere_shadow_animate_jitter);
    SF_SETTING(atmosphere_shadow_rotation);
    SF_SETTING(atmosphere_shadow_adaptive);
    SF_SETTING(atmosphere_shadow_vrs);
    SF_SETTING(atmosphere_shadow_vrs_scale);
    SF_SETTING(atmosphere_shadow_noise_type);
    SF_SETTING(atmosphere_shadow_samples);
    SF_SETTING(ring_shadows_enabled);
    SF_SETTING(ringshine_enabled);
    SF_SETTING(planetshine_enabled);
    SF_SETTING(oceans_enabled);
    SF_SETTING(clouds_enabled);
    SF_SETTING(cloud_mode);
    SF_SETTING(cloud_shadows_enabled);
    SF_SETTING(ring_planetshine_mode);
    SF_SETTING(ringshine_band_count);
    SF_SETTING(eclipse_reference_smoothstep);
    SF_SETTING(ring_shadow_reference_all_hosts);
    SF_SETTING(ring_shadow_softness);
    SF_SETTING(ring_shadow_single_box);
    SF_SETTING(ring_shadow_reference_linear);
    SF_SETTING(ring_reference_full_annulus);
    SF_SETTING(ring_reference_finite_source);
    SF_SETTING(ring_reference_base_level);
#undef SF_SETTING
}

template <typename Settings, typename Visitor>
void workspace_fields(Settings& settings, Visitor&& visit) {
#define SF_SETTING(name) visit(#name, settings.name)
    SF_SETTING(aligned_panels);
    SF_SETTING(show_outliner);
    SF_SETTING(show_inspector);
    SF_SETTING(show_time_hud);
    SF_SETTING(show_telemetry);
    SF_SETTING(ly_threshold_au);
    SF_SETTING(outliner_width);
    SF_SETTING(inspector_width);
    SF_SETTING(scale);
    SF_SETTING(confirm_delete);
    SF_SETTING(comparison_force_keplerian);
    SF_SETTING(comparison_auto_spacing);
#undef SF_SETTING
}

template <typename Settings, typename Visitor>
void navigation_fields(Settings& settings, Visitor&& visit) {
    visit("fov_deg", settings.fov_deg);
    visit("flight_speed_au_s", settings.flight_speed_au_s);
    visit("movement_mode", settings.movement_mode);
    visit("horizon_align", settings.horizon_align);
}

bool valid(const rendering::RenderSettings& settings) {
    rendering::RenderSettingsStore store;
    return static_cast<bool>(store.apply(settings));
}

bool valid(const ui::WorkspaceSettings& settings) {
    return std::isfinite(settings.ly_threshold_au) && settings.ly_threshold_au >= 1.0 &&
        std::isfinite(settings.outliner_width) && settings.outliner_width >= 180.0f && settings.outliner_width <= 500.0f &&
        std::isfinite(settings.inspector_width) && settings.inspector_width >= 300.0f && settings.inspector_width <= 700.0f &&
        std::isfinite(settings.scale) && settings.scale >= 0.9f && settings.scale <= 1.5f;
}

bool valid(const NavigationSettings& settings) {
    return std::isfinite(settings.fov_deg) && settings.fov_deg >= 0.001 && settings.fov_deg <= 120.0 &&
        std::isfinite(settings.flight_speed_au_s) && settings.flight_speed_au_s >= 1e-12 && settings.flight_speed_au_s <= MAX_FLIGHT_SPEED_AU_S &&
        settings.movement_mode >= 0 && settings.movement_mode <= 1;
}

template <typename T>
bool read_value(const nlohmann::json& value, T& destination) {
    if constexpr (std::is_same_v<T, bool>) {
        if (!value.is_boolean()) return false;
    } else {
        if constexpr (std::is_integral_v<T>) {
            if (!value.is_number_integer()) return false;
        } else {
            if (!value.is_number()) return false;
        }
        const auto number = value.get<long double>();
        if (!std::isfinite(number) || number < std::numeric_limits<T>::lowest() || number > std::numeric_limits<T>::max()) return false;
    }
    destination = value.get<T>();
    return true;
}

template <typename Settings, typename Fields>
bool read_section(const nlohmann::json& document, const char* key, Settings& settings, Fields fields) {
    const auto section = document.find(key);
    if (section == document.end()) return true;
    if (!section->is_object()) return false;
    bool accepted = true;
    fields(settings, [&](const char* name, auto& field) {
        const auto entry = section->find(name);
        if (entry == section->end()) return;
        const auto previous = field;
        if (!read_value(*entry, field) || !valid(settings)) {
            field = previous;
            accepted = false;
        }
    });
    return accepted;
}

}

std::filesystem::path user_settings_path() {
    return paths::external_path("data", "settings.json");
}

UserSettingsLoad load_user_settings(const std::filesystem::path& path) {
    UserSettingsLoad result;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) && !ec) return result;
    const auto document = paths::read_json(path);
    if (!document || !document->is_object()) {
        result.warning = "Settings could not be read. Using defaults until you change a setting.";
        return result;
    }
    int version = 1;
    const auto schema = document->find("version");
    if (schema != document->end() && (!read_value(*schema, version) || version != 1)) {
        result.warning = "Unsupported settings version. Using defaults; the saved file will be preserved.";
        result.writable = false;
        return result;
    }
    const bool graphics_ok = read_section(*document, "graphics", result.values.graphics,
        [](auto& settings, auto visit) { graphics_fields(settings, visit); });
    const bool workspace_ok = read_section(*document, "workspace", result.values.workspace,
        [](auto& settings, auto visit) { workspace_fields(settings, visit); });
    const bool navigation_ok = read_section(*document, "navigation", result.values.navigation,
        [](auto& settings, auto visit) { navigation_fields(settings, visit); });
    if (!graphics_ok || !workspace_ok || !navigation_ok)
        result.warning = "Some saved settings were invalid and used their defaults.";
    return result;
}

CommandResult save_user_settings(const std::filesystem::path& path, const UserSettings& values) {
    if (!valid(values.graphics) || !valid(values.workspace) || !valid(values.navigation))
        return {false, false, "Settings contain an invalid value and could not be saved."};
    try {
        nlohmann::json document = {{"version", 1}};
        auto write_section = [&](const char* name) {
            document[name] = nlohmann::json::object();
            return [&, name](const char* key, const auto& value) { document[name][key] = value; };
        };
        graphics_fields(values.graphics, write_section("graphics"));
        workspace_fields(values.workspace, write_section("workspace"));
        navigation_fields(values.navigation, write_section("navigation"));
        if (paths::write_json_atomic(path, document)) return {true, true, {}};
    } catch (const std::exception&) {
        return {false, false, "Settings could not be saved. Check the settings file and available disk space."};
    }
    return {false, false, "Settings could not be saved. Check the settings folder permissions and available disk space."};
}

}
