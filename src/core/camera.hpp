#pragma once

#include "core/constants.hpp"
#include "ephemeris/system_manager.hpp"
#include <glm/glm.hpp>

namespace sf {

inline constexpr double ROT_FOLLOW_ALT_AU = 200.0 / AU_TO_KM;

struct Camera {
    glm::dvec3 target_au{0.0, 0.0, 0.0};
    glm::dvec3 cam_pos_rel{0.0, 19.02, 40.78};
    glm::dvec3 cam_vel{0.0, 0.0, 0.0};
    glm::dvec3 up{0.0, 1.0, 0.0};

    double yaw_deg = -90.0;
    double pitch_deg = -25.0;
    double yaw_actual_deg = -90.0;
    double pitch_actual_deg = -25.0;
    double roll_deg = 0.0;
    double fov_deg = FOV_DEG;
    double flight_speed_au_s = 0.1;
    double approach_delta = 0.0;

    float exposure = 1.0f;
    bool hdr_enabled = true;
    bool show_gaia_stars = true;
    float star_point_size = 2.0f;
    float star_intensity = 1.0f;

    int movement_mode = 0;
    bool cam_look_aim = true;
    bool horizon_align = true;
    bool is_near_surface = false;

    int tracking_idx = 0;
    int inspected_idx = 0;

    bool left_dragging = false;
    bool right_dragging = false;
    double last_mouse_x = 0.0;
    double last_mouse_y = 0.0;

    bool key_w = false;
    bool key_a = false;
    bool key_s = false;
    bool key_d = false;
    bool key_q = false;
    bool key_e = false;

    glm::dvec3 origin_render{0.0};
    double reference_altitude_au = 0.0;

    int prev_spin_track_idx = -1;
    double prev_spin_angle = 0.0;
    bool has_prev_spin = false;

    void update(double dt_sec, const ephemeris::SystemState& sys);
    void on_mouse_button(int button, int action, double xpos, double ypos);
    void on_cursor_pos(double xpos, double ypos, const ephemeris::SystemState& sys);
    void on_scroll(double yoffset, bool shift_held);
    void snap_to_tracked(const ephemeris::SystemState& sys);
    void focus_body(int body_idx, const ephemeris::SystemState& sys, bool fly_close = false);

    glm::dvec3 forward_dir() const noexcept;
    glm::dvec3 up_dir() const noexcept;
    glm::dvec3 right_dir() const noexcept;
    glm::dvec3 eye_pos_render() const noexcept { return origin_render + cam_pos_rel; }

    glm::mat4 view_matrix_origin_relative() const noexcept;
    glm::mat4 view_matrix_eye_relative() const noexcept;
    glm::mat4 proj_matrix(float aspect, float z_near_au) const noexcept;
    float compute_near_plane_au() const noexcept;
};

double ellipsoid_surface_radius(double r_eq, double f, const glm::dvec3& pole, const glm::dvec3& dir) noexcept;
glm::dvec3 oblate_surface_normal(const glm::dvec3& rel, double f, const glm::dvec3& pole) noexcept;
glm::dmat3 rodrigues_rotation(const glm::dvec3& axis, double angle_rad) noexcept;

} // namespace sf
