#include "core/camera.hpp"
#include "core/math_utils.hpp"
#include "ephemeris/jacobi.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace sf {

namespace {

glm::dvec3 angles_to_forward(double yaw_deg, double pitch_deg) noexcept {
    double yaw = yaw_deg * DEG_TO_RAD;
    double pitch = pitch_deg * DEG_TO_RAD;
    return {
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch),
    };
}

std::pair<double, double> forward_to_angles(const glm::dvec3& fwd) noexcept {
    double n = glm::length(fwd);
    if (n < 1e-30) {
        return {0.0, 0.0};
    }
    glm::dvec3 u = fwd / n;
    double yaw = std::atan2(u.z, u.x) * RAD_TO_DEG;
    double pitch = std::asin(std::clamp(u.y, -1.0, 1.0)) * RAD_TO_DEG;
    return {yaw, pitch};
}

glm::dvec3 orthogonalize_up(const glm::dvec3& raw_up, const glm::dvec3& fwd) noexcept {
    double nf = glm::length(fwd);
    if (nf < 1e-12) {
        return {0.0, 1.0, 0.0};
    }
    glm::dvec3 f = fwd / nf;
    glm::dvec3 u = glm::length(raw_up) > 1e-6 ? raw_up : glm::dvec3{0.0, 1.0, 0.0};
    glm::dvec3 proj = u - glm::dot(u, f) * f;
    double np = glm::length(proj);
    if (np > 1e-6) {
        return proj / np;
    }
    glm::dvec3 alt1{1.0, 0.0, 0.0};
    proj = alt1 - glm::dot(alt1, f) * f;
    np = glm::length(proj);
    if (np > 1e-6) {
        return proj / np;
    }
    glm::dvec3 alt2{0.0, 0.0, 1.0};
    proj = alt2 - glm::dot(alt2, f) * f;
    return glm::normalize(proj);
}

void apply_pivot_rotation(Camera& cam, double dx, double dy, double sensitivity,
                          double track_f, const glm::dvec3& track_pole) noexcept {
    glm::dvec3 fwd = cam.forward_dir();
    glm::dvec3 up_v = cam.up_dir();
    glm::dvec3 right_v = cam.right_dir();

    glm::dvec3 h_axis = (cam.horizon_align && cam.is_near_surface)
        ? oblate_surface_normal(cam.cam_pos_rel, track_f, track_pole)
        : up_v;

    glm::dmat3 R_vert = rodrigues_rotation(right_v, -dy * sensitivity * DEG_TO_RAD);
    glm::dvec3 fwd_test = glm::normalize(R_vert * fwd);
    if (cam.horizon_align && cam.is_near_surface) {
        double d = glm::dot(fwd_test, h_axis);
        if (std::abs(d) > 0.999 && std::abs(d) > std::abs(glm::dot(fwd, h_axis))) {
            R_vert = glm::dmat3(1.0);
        }
    }

    glm::dmat3 R = rodrigues_rotation(h_axis, -dx * sensitivity * DEG_TO_RAD) * R_vert;
    glm::dvec3 fwd_new = glm::normalize(R * fwd);
    glm::dvec3 up_new = glm::normalize(R * up_v);

    auto [y_new, p_new] = forward_to_angles(fwd_new);
    cam.yaw_deg = cam.yaw_actual_deg = y_new;
    cam.pitch_deg = cam.pitch_actual_deg = p_new;
    cam.up = up_new;
    cam.cam_look_aim = false;
}

void apply_orbit_rotation(Camera& cam, double dx, double dy, double sensitivity) noexcept {
    double r = glm::length(cam.cam_pos_rel);
    if (r < 1e-30) {
        return;
    }
    glm::dvec3 fwd = cam.forward_dir();
    glm::dvec3 up_v = cam.up_dir();
    glm::dvec3 right_v = cam.right_dir();

    glm::dmat3 R_h = rodrigues_rotation(up_v, dx * sensitivity * DEG_TO_RAD);
    glm::dmat3 R_v = rodrigues_rotation(right_v, dy * sensitivity * DEG_TO_RAD);
    glm::dmat3 R = R_h * R_v;

    glm::dvec3 rel_new = R * cam.cam_pos_rel;
    double rn = glm::length(rel_new);
    if (rn > 1e-30) {
        rel_new = (rel_new / rn) * r;
    }
    cam.cam_pos_rel = rel_new;
    cam.up = glm::normalize(R * up_v);

    if (!cam.cam_look_aim) {
        glm::dvec3 fwd_new = glm::normalize(R * fwd);
        auto [y_new, p_new] = forward_to_angles(fwd_new);
        cam.yaw_deg = cam.yaw_actual_deg = y_new;
        cam.pitch_deg = cam.pitch_actual_deg = p_new;
    }
}

} // namespace

double ellipsoid_surface_radius(double r_eq, double f, const glm::dvec3& pole, const glm::dvec3& dir) noexcept {
    if (f <= 0.0 || r_eq <= 0.0) {
        return r_eq;
    }
    double np = glm::length(pole);
    double nd = glm::length(dir);
    if (np < 1e-12) return r_eq;
    if (nd < 1e-12) return r_eq * (1.0 - f);
    glm::dvec3 p = pole / np;
    glm::dvec3 u = dir / nd;
    double b = r_eq * (1.0 - f);
    double cos_t = glm::dot(u, p);
    double sin_t2 = std::max(0.0, 1.0 - cos_t * cos_t);
    double denom = b * b * sin_t2 + r_eq * r_eq * cos_t * cos_t;
    if (denom < 1e-30) {
        return b;
    }
    return std::sqrt((r_eq * r_eq * b * b) / denom);
}

glm::dvec3 oblate_surface_normal(const glm::dvec3& rel, double f, const glm::dvec3& pole) noexcept {
    double r_norm = glm::length(rel);
    if (r_norm < 1e-30) {
        return {0.0, 1.0, 0.0};
    }
    glm::dvec3 radial = rel / r_norm;
    double np = glm::length(pole);
    if (f <= 0.0 || np < 1e-12) {
        return radial;
    }
    glm::dvec3 p = pole / np;
    double fc = std::clamp(f, 0.0, 0.9);
    double scale_para = 1.0 / ((1.0 - fc) * (1.0 - fc));
    double para_len = glm::dot(rel, p);
    glm::dvec3 N = rel + (scale_para - 1.0) * para_len * p;
    double nl = glm::length(N);
    return nl > 1e-12 ? (N / nl) : radial;
}

glm::dmat3 rodrigues_rotation(const glm::dvec3& axis, double angle_rad) noexcept {
    double n = glm::length(axis);
    if (n < 1e-12 || std::abs(angle_rad) < 1e-15) {
        return glm::dmat3(1.0);
    }
    glm::dvec3 a = axis / n;
    glm::dmat3 K(
        glm::dvec3(0.0, a.z, -a.y),
        glm::dvec3(-a.z, 0.0, a.x),
        glm::dvec3(a.y, -a.x, 0.0)
    );
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    return glm::dmat3(1.0) + s * K + (1.0 - c) * (K * K);
}

glm::dvec3 Camera::forward_dir() const noexcept {
    if (cam_look_aim) {
        double r = glm::length(cam_pos_rel);
        if (r > 1e-30) {
            return -cam_pos_rel / r;
        }
    }
    return glm::normalize(angles_to_forward(yaw_actual_deg, pitch_actual_deg));
}

glm::dvec3 Camera::up_dir() const noexcept {
    return orthogonalize_up(up, forward_dir());
}

glm::dvec3 Camera::right_dir() const noexcept {
    glm::dvec3 f = forward_dir();
    glm::dvec3 u = up_dir();
    glm::dvec3 r = glm::cross(f, u);
    double nr = glm::length(r);
    return nr > 1e-9 ? (r / nr) : glm::dvec3{1.0, 0.0, 0.0};
}

void Camera::update(units::Seconds elapsed, const ephemeris::SystemSnapshot& sys) {
    double dt_sec = elapsed.value;
    glm::dvec3 rel_prev = cam_pos_rel;

    double lerp_factor = 1.0 - std::exp(-15.0 * dt_sec);
    yaw_actual_deg += (yaw_deg - yaw_actual_deg) * lerp_factor;
    pitch_actual_deg += (pitch_deg - pitch_actual_deg) * lerp_factor;

    double track_r = 0.0;
    double track_f = 0.0;
    glm::dvec3 track_pole{0.0, 1.0, 0.0};

    if (tracking_idx >= 0 && tracking_idx < static_cast<int>(sys.size())) {
        if (tracking_barycenter) {
            auto coords = ephemeris::extract_jacobi(sys.definition(), sys.motions(), sys.orbit_hierarchy());
            origin_render = ecliptic_to_render(EclipticPosition{coords.subsystems[tracking_idx].position}).value;
            track_r = 0.0;
            track_f = 0.0;
        } else {
            const auto& b = sys.body(tracking_idx);
            origin_render = b.render.position.value;
            track_r = b.opaque_radius_au();
            track_f = b.definition.oblateness;
            track_pole = b.render.pole;
        }
    } else {
        origin_render = target_au;
    }

    if (centered_idx >= 0 && centered_idx < static_cast<int>(sys.size())) {
        glm::dvec3 target_pos{0.0};
        if (centered_barycenter) {
            auto coords = ephemeris::extract_jacobi(sys.definition(), sys.motions(), sys.orbit_hierarchy());
            target_pos = ecliptic_to_render(EclipticPosition{coords.subsystems[centered_idx].position}).value;
        } else {
            target_pos = sys.body(centered_idx).render.position.value;
        }
        glm::dvec3 eye = origin_render + cam_pos_rel;
        glm::dvec3 to_target = target_pos - eye;
        double dist = glm::length(to_target);
        if (dist > 1e-12) {
            glm::dvec3 fwd = to_target / dist;
            auto [yn, pn] = forward_to_angles(fwd);
            yaw_deg = yaw_actual_deg = yn;
            pitch_deg = pitch_actual_deg = pn;
            up = orthogonalize_up(up, fwd);
            cam_look_aim = true;
        }
    }

    double ref_r = track_r > 0.0 ? ellipsoid_surface_radius(track_r, track_f, track_pole, cam_pos_rel) : 0.0;
    reference_altitude_au = std::max(0.0, glm::length(cam_pos_rel) - ref_r);
    is_near_surface = (track_r > 0.0 && reference_altitude_au < ROT_FOLLOW_ALT_AU);

    if (approach_delta != 0.0) {
        double r_ap = glm::length(cam_pos_rel);
        if (r_ap > 1e-30 && track_r > 0.0) {
            double alt_new = (r_ap - ref_r) * std::exp(-approach_delta);
            double r_new = std::max(1e-12, ref_r + alt_new);
            cam_pos_rel = (cam_pos_rel / r_ap) * r_new;
        } else {
            double dolly = approach_delta * std::max(1e-4, r_ap);
            cam_pos_rel += forward_dir() * dolly;
        }
        approach_delta = 0.0;
        ref_r = track_r > 0.0 ? ellipsoid_surface_radius(track_r, track_f, track_pole, cam_pos_rel) : 0.0;
        reference_altitude_au = std::max(0.0, glm::length(cam_pos_rel) - ref_r);
    }

    if (track_r > 0.0 && tracking_idx >= 0 && tracking_idx < static_cast<int>(sys.size()) &&
        reference_altitude_au < ROT_FOLLOW_ALT_AU) {
        double alt_km = reference_altitude_au * AU_TO_KM;
        double rot_weight = 1.0;
        if (alt_km > 50.0) {
            double t = (alt_km - 50.0) / 150.0;
            rot_weight = 1.0 - (3.0 * t * t - 2.0 * t * t * t);
        }
        double spin_angle = sys.body(tracking_idx).render.spin_angle_rad;
        if (has_prev_spin && prev_spin_track_idx == tracking_idx) {
            double d_spin = std::fmod(spin_angle - prev_spin_angle + PI, TWO_PI);
            if (d_spin < 0.0) d_spin += TWO_PI;
            d_spin -= PI;
            if (std::abs(d_spin) > 1e-12) {
                glm::dmat3 R_spin = rodrigues_rotation(track_pole, d_spin * rot_weight);
                glm::dvec3 cur_fwd = forward_dir();
                glm::dvec3 cur_up = up_dir();
                cam_pos_rel = R_spin * cam_pos_rel;
                up = glm::normalize(R_spin * cur_up);
                if (!cam_look_aim) {
                    glm::dvec3 fwd_new = glm::normalize(R_spin * cur_fwd);
                    auto [yn, pn] = forward_to_angles(fwd_new);
                    yaw_deg = yaw_actual_deg = yn;
                    pitch_deg = pitch_actual_deg = pn;
                }
            }
        }
        prev_spin_angle = spin_angle;
        prev_spin_track_idx = tracking_idx;
        has_prev_spin = true;
    } else {
        has_prev_spin = false;
        prev_spin_track_idx = -1;
    }

    if (horizon_align && is_near_surface) {
        double align_strength = 1.0 - (reference_altitude_au / ROT_FOLLOW_ALT_AU);
        glm::dvec3 fwd_al = forward_dir();
        glm::dvec3 local_up = oblate_surface_normal(cam_pos_rel, track_f, track_pole);
        glm::dvec3 target_up = local_up - glm::dot(local_up, fwd_al) * fwd_al;
        double nt = glm::length(target_up);
        if (nt > 1e-6) {
            target_up /= nt;
            glm::dvec3 cur_up = up_dir();
            double c_dot = std::clamp(glm::dot(cur_up, target_up), -1.0, 1.0);
            double angle = std::acos(c_dot);
            if (angle > 1e-6) {
                glm::dvec3 rot_axis = glm::cross(cur_up, target_up);
                if (glm::dot(rot_axis, fwd_al) < 0.0) {
                    angle = -angle;
                }
                double d_ang = angle * std::min(1.0, 1.5 * align_strength * dt_sec);
                up = glm::normalize(rodrigues_rotation(fwd_al, d_ang) * cur_up);
            }
        }
    }

    if (key_q || key_e) {
        double roll_speed = 1.2 * dt_sec * ((key_q ? -1.0 : 0.0) + (key_e ? 1.0 : 0.0));
        glm::dvec3 fwd = forward_dir();
        up = glm::normalize(rodrigues_rotation(fwd, roll_speed) * up_dir());
    }

    glm::dvec3 front = forward_dir();
    glm::dvec3 right_v = right_dir();
    glm::dvec3 move{0.0};
    if (key_w) move += front;
    if (key_s) move -= front;
    if (key_d) move += right_v;
    if (key_a) move -= right_v;

    double nm = glm::length(move);
    if (nm > 1e-12) {
        if (cam_look_aim) {
            auto [yn, pn] = forward_to_angles(front);
            yaw_deg = yaw_actual_deg = yn;
            pitch_deg = pitch_actual_deg = pn;
            cam_look_aim = false;
        }
        cam_pos_rel += (move / nm) * flight_speed_au_s * dt_sec;
    }

    cam_vel = (cam_pos_rel - rel_prev) / std::max(dt_sec, 1e-9);
}

void Camera::on_mouse_button(int button, int action, double xpos, double ypos) {
    if (action == GLFW_RELEASE) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) left_dragging = false;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) right_dragging = false;
        return;
    }
    if (action == GLFW_PRESS) {
        last_mouse_x = xpos;
        last_mouse_y = ypos;
        if (button == GLFW_MOUSE_BUTTON_LEFT) left_dragging = true;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) right_dragging = true;
    }
}

void Camera::on_cursor_pos(double xpos, double ypos, const ephemeris::SystemSnapshot& sys) {
    double dx = xpos - last_mouse_x;
    double dy = ypos - last_mouse_y;
    last_mouse_x = xpos;
    last_mouse_y = ypos;

    if (!left_dragging && !right_dragging) {
        return;
    }

    if (std::abs(dx) > 0.0 || std::abs(dy) > 0.0) {
        centered_idx = -1;
    }

    double fov_ratio = std::clamp(fov_deg / 45.0, 0.0001, 1.0);
    double sensitivity = 0.25 * fov_ratio;

    double track_f = 0.0;
    glm::dvec3 track_pole{0.0, 1.0, 0.0};
    if (tracking_idx >= 0 && tracking_idx < static_cast<int>(sys.size())) {
        track_f = sys.body(tracking_idx).definition.oblateness;
        track_pole = sys.body(tracking_idx).render.pole;
    }

    if (left_dragging && right_dragging) {
        approach_delta += dy * 0.0125;
    } else if (left_dragging) {
        if (movement_mode == 1) {
            cam_look_aim = true;
            apply_orbit_rotation(*this, dx, dy, sensitivity);
        } else {
            apply_pivot_rotation(*this, dx, dy, sensitivity, track_f, track_pole);
        }
    } else if (right_dragging) {
        if (movement_mode == 1) {
            apply_pivot_rotation(*this, dx, dy, sensitivity, track_f, track_pole);
        } else {
            apply_orbit_rotation(*this, dx, dy, sensitivity);
        }
    }
}

void Camera::on_scroll(double yoffset, bool shift_held) {
    if (shift_held) {
        if (yoffset > 0.0) {
            fov_deg *= std::pow(0.85, yoffset);
        } else if (yoffset < 0.0) {
            fov_deg /= std::pow(0.85, std::abs(yoffset));
        }
        fov_deg = std::clamp(fov_deg, 0.001, 120.0);
    } else if (movement_mode == 1) {
        approach_delta += yoffset * 0.15;
        centered_idx = -1;
    } else {
        flight_speed_au_s = std::clamp(flight_speed_au_s * std::pow(2.0, yoffset), 1e-12, MAX_FLIGHT_SPEED_AU_S);
    }
}

void Camera::snap_to_tracked(const ephemeris::SystemSnapshot& sys) {
    double r = glm::length(cam_pos_rel);
    if (r < 1e-30) {
        return;
    }
    if (is_near_surface && tracking_idx >= 0 && tracking_idx < static_cast<int>(sys.size())) {
        const auto& b = sys.body(tracking_idx);
        glm::dvec3 local_up = oblate_surface_normal(cam_pos_rel, b.definition.oblateness, b.render.pole);
        glm::dvec3 fwd = forward_dir();
        glm::dvec3 horiz = fwd - glm::dot(fwd, local_up) * local_up;
        if (glm::length(horiz) > 1e-6) {
            horiz = glm::normalize(horiz);
            auto [yn, pn] = forward_to_angles(horiz);
            yaw_deg = yaw_actual_deg = yn;
            pitch_deg = pitch_actual_deg = pn;
            up = local_up;
            cam_look_aim = false;
            return;
        }
    }
    glm::dvec3 fwd = -cam_pos_rel / r;
    auto [yn, pn] = forward_to_angles(fwd);
    yaw_deg = yaw_actual_deg = yn;
    pitch_deg = pitch_actual_deg = pn;
    cam_look_aim = true;
}

void Camera::focus_body(int body_idx, const ephemeris::SystemSnapshot& sys, bool fly_close, bool is_barycenter) {
    if (body_idx < 0 || body_idx >= static_cast<int>(sys.size())) {
        return;
    }
    centered_idx = -1;
    tracking_barycenter = is_barycenter;

    if (!fly_close && (tracking_idx != body_idx || tracking_barycenter != is_barycenter)) {
        glm::dvec3 old_eye = eye_pos_render().value;
        tracking_idx = body_idx;
        if (is_barycenter) {
            auto coords = ephemeris::extract_jacobi(sys.definition(), sys.motions(), sys.orbit_hierarchy());
            origin_render = ecliptic_to_render(EclipticPosition{coords.subsystems[body_idx].position}).value;
        } else {
            origin_render = sys.body(body_idx).render.position.value;
        }
        cam_pos_rel = old_eye - origin_render;
        snap_to_tracked(sys);
        return;
    }

    tracking_idx = body_idx;
    double dist = 0.01;
    if (is_barycenter) {
        auto coords = ephemeris::extract_jacobi(sys.definition(), sys.motions(), sys.orbit_hierarchy());
        origin_render = ecliptic_to_render(EclipticPosition{coords.subsystems[body_idx].position}).value;
        dist = std::max(sys.body(body_idx).definition.semi_major_au * 2.5, 0.01);
    } else {
        const auto& b = sys.body(body_idx);
        origin_render = b.render.position.value;
        dist = std::max(b.definition.radius_au() * 4.5, 1e-6);
    }

    glm::dvec3 dir = glm::normalize(glm::dvec3{0.0, 0.38, 0.925});
    cam_pos_rel = dir * dist;
    up = {0.0, 1.0, 0.0};
    cam_look_aim = true;
    auto [yn, pn] = forward_to_angles(-dir);
    yaw_deg = yaw_actual_deg = yn;
    pitch_deg = pitch_actual_deg = pn;
    flight_speed_au_s = std::clamp(dist * 0.5, 1e-9, MAX_FLIGHT_SPEED_AU_S);
}

void Camera::center_body(int body_idx, const ephemeris::SystemSnapshot& sys, bool is_barycenter) {
    if (body_idx < 0 || body_idx >= static_cast<int>(sys.size())) {
        centered_idx = -1;
        return;
    }
    centered_idx = body_idx;
    centered_barycenter = is_barycenter;

    glm::dvec3 target_pos{0.0};
    if (is_barycenter) {
        auto coords = ephemeris::extract_jacobi(sys.definition(), sys.motions(), sys.orbit_hierarchy());
        target_pos = ecliptic_to_render(EclipticPosition{coords.subsystems[body_idx].position}).value;
    } else {
        target_pos = sys.body(body_idx).render.position.value;
    }
    glm::dvec3 eye = eye_pos_render().value;
    glm::dvec3 to_target = target_pos - eye;
    double dist = glm::length(to_target);
    if (dist > 1e-12) {
        glm::dvec3 fwd = to_target / dist;
        auto [yn, pn] = forward_to_angles(fwd);
        yaw_deg = yaw_actual_deg = yn;
        pitch_deg = pitch_actual_deg = pn;
        up = orthogonalize_up(up, fwd);
        cam_look_aim = true;
        approach_delta = 0.0;
    }
}

glm::mat4 Camera::view_matrix_origin_relative() const noexcept {
    glm::dvec3 eye = cam_pos_rel;
    glm::dvec3 f = forward_dir();
    glm::dvec3 u = up_dir();
    return glm::mat4(glm::lookAt(eye, eye + f, u));
}

glm::mat4 Camera::view_matrix_eye_relative() const noexcept {
    glm::dvec3 f = forward_dir();
    glm::dvec3 u = up_dir();
    return glm::mat4(glm::lookAt(glm::dvec3(0.0), f, u));
}

glm::dmat3 Camera::view_rotation_double() const noexcept {
    return glm::dmat3(glm::lookAt(glm::dvec3(0.0), forward_dir(), up_dir()));
}

float Camera::compute_near_plane_au() const noexcept {
    if (is_near_surface) {
        return static_cast<float>(0.0001 / AU_TO_KM);
    }
    double r = glm::length(cam_pos_rel);
    double near_au = std::clamp(r * 1e-5, 0.0001 / AU_TO_KM, 0.05);
    return static_cast<float>(near_au);
}

glm::mat4 Camera::proj_matrix(float aspect, float z_near_au) const noexcept {
    return sf::math::reverse_z_perspective(static_cast<float>(fov_deg * DEG_TO_RAD), aspect, z_near_au);
}

} // namespace sf
