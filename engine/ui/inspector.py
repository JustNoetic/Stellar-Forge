import math
import os
import json
import numpy as np
import imgui
import glfw
from engine.core.constants import (
    DEFAULT_LY_THRESHOLD_AU,
    G,
    C_AU_YR,
    SOLAR_RADII_TO_AU,
    AU_TO_KM
)
from engine.core.math_utils import (
    pole_to_ecliptic,
    rotate_ecliptic_to_equatorial,
    rotate_equatorial_to_ecliptic,
    get_cartesian_from_keplerian
)
from engine.core.input_handler import _camera_yaw_pitch_from, _camera_get_up
from engine.physics.refraction import get_apparent_look_direction
from engine.physics.physics_core import compute_keplerian_elements
from engine.physics.star_calc import StarCalculator
from engine.physics.atmosphere_physics import compute_mie_coefficients, GAS_PROPERTIES
from engine.rendering.render_utils import (
    compute_surface_albedo,
    format_distance_au,
    format_altitude,
    rebuild_ring_render_group,
    generate_ring_shadow_grad
)
from engine.rendering.planetshine import get_cached_atmosphere_properties
from engine.rendering.texture_baker import bake_and_export_ring_textures
from engine.ephemeris.system_manager import SystemManager

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def _ellipsoid_surface_radius(r_eq, f, pole, u):
    """Compute local radius on an oblate spheroid (semi-major r_eq, flattening f)
    in direction vector u (pole = unit spin axis)."""
    r_eq = float(r_eq)
    f = float(f)
    if f <= 0.0 or r_eq <= 0.0:
        return r_eq
    if f >= 1.0:
        f = 0.9999
    pole = np.asarray(pole, dtype='f8')
    p_len = np.linalg.norm(pole)
    if p_len < 1e-12:
        return r_eq
    pole = pole / p_len

    u = np.asarray(u, dtype='f8')
    u_len = np.linalg.norm(u)
    if u_len < 1e-12:
        return r_eq * (1.0 - f)
    u = u / u_len

    b = r_eq * (1.0 - f)
    cos_phi = float(np.dot(u, pole))
    sin_phi_sq = max(0.0, 1.0 - cos_phi * cos_phi)
    denom = b * b * sin_phi_sq + r_eq * r_eq * (cos_phi * cos_phi)
    if denom < 1e-30:
        return b
    return (r_eq * b) / math.sqrt(denom)

def compute_body_albedos(app, body_info, insp_idx, insp_is_cmp, visual_arr, atmo_bodies, cur_mass_snap):
    """Compute Geometric Albedo (A_g), Bond Albedo (A_b), Phase Integral (q), and Top-of-Atmosphere RGB reflectance."""
    if insp_is_cmp and hasattr(app, 'visual_arr_cmp') and app.visual_arr_cmp is not None and len(app.visual_arr_cmp) > insp_idx:
        v_color = app.visual_arr_cmp[insp_idx, 0:3]
    elif visual_arr is not None and len(visual_arr) > insp_idx:
        v_color = visual_arr[insp_idx, 0:3]
    else:
        v_color = None

    p_surf, _, _ = compute_surface_albedo(body_info, getattr(app, 'texture_mean_colors', None), v_color)

    atmo_bodies_list = app.atmo_bodies_cmp if insp_is_cmp else atmo_bodies
    atmo_item = next((a for a in atmo_bodies_list if a.get('body_idx') == insp_idx), None)

    if atmo_item:
        mass_snap_buf = app.mass_snap_cmp if insp_is_cmp else cur_mass_snap
        mass_val = mass_snap_buf[insp_idx] if (mass_snap_buf is not None and len(mass_snap_buf) > insp_idx) else 3.0e-6
        props, _, _ = get_cached_atmosphere_properties(atmo_item, mass_val)

        H_r_m = max(0.0, float(props['scale_height_km'])) * 1000.0
        H_m_m = max(0.0, float(atmo_item.get('h_mie', 1.2))) * 1000.0

        beta_R = np.nan_to_num(props['beta_rayleigh'], nan=0.0, posinf=0.0, neginf=0.0)
        beta_M = np.nan_to_num(compute_mie_coefficients(atmo_item.get('beta_mie', 2.0e-6), atmo_item.get('mie_angstrom', None)), nan=0.0, posinf=0.0, neginf=0.0)
        mie_alb = np.nan_to_num(np.asarray(atmo_item.get('mie_albedo', np.array([1.0, 1.0, 1.0], dtype=np.float32)), dtype=np.float64), nan=1.0)
        beta_O3 = np.nan_to_num(props['beta_abs_layered'], nan=0.0, posinf=0.0, neginf=0.0)
        beta_mixed = np.nan_to_num(props['beta_abs_mixed'], nan=0.0, posinf=0.0, neginf=0.0)

        tau_R = beta_R * H_r_m
        tau_M_ext = beta_M * H_m_m
        tau_M_sca = tau_M_ext * mie_alb
        tau_O3 = beta_O3 * 14179.6
        tau_mixed = beta_mixed * H_r_m

        tau_total = tau_R + tau_M_ext + tau_O3 + tau_mixed
        T_surf = np.exp(-tau_total)
        p_surf_contrib = p_surf * (T_surf ** 2)

        p_Rayleigh = 0.69 * (1.0 - np.exp(-tau_R)) * np.exp(-2.0 * tau_O3)
        mie_g = float(atmo_item.get('mie_g', 0.76))
        p_mie_peak = (1.0 + mie_g) / max(1e-4, 1.0 - mie_g)
        p_Mie = 0.75 * np.minimum(p_mie_peak, 2.5) * (1.0 - np.exp(-tau_M_sca)) * np.exp(-2.0 * tau_O3)

        p_rgb = p_surf_contrib + (1.0 - p_surf_contrib) * (p_Rayleigh + p_Mie)
        p_rgb = np.clip(p_rgb, 0.0, 1.0)
        A_g = float(0.2126 * p_rgb[0] + 0.7152 * p_rgb[1] + 0.0722 * p_rgb[2])

        tau_scat = tau_R + tau_M_sca
        tau_scat_scalar = float(0.2126 * tau_scat[0] + 0.7152 * tau_scat[1] + 0.0722 * tau_scat[2])
        tau_R_scalar = float(0.2126 * tau_R[0] + 0.7152 * tau_R[1] + 0.0722 * tau_R[2])
        tau_M_scalar = float(0.2126 * tau_M_sca[0] + 0.7152 * tau_M_sca[1] + 0.0722 * tau_M_sca[2])

        w_R = tau_R_scalar / max(1e-6, tau_scat_scalar)
        w_M = tau_M_scalar / max(1e-6, tau_scat_scalar)
        q_R = 1.35
        q_M = 1.50 * (1.0 - 0.6 * (mie_g ** 1.2))
        q_atmo_scat = w_R * q_R + w_M * q_M

        f_atmo = 1.0 - math.exp(-tau_scat_scalar)
        b_type = body_info.get('type', 'Terrestrial')
        if b_type == 'Gas Giant':
            base_q_surf = 1.50
        else:
            atmo_weight = min(1.0, tau_scat_scalar * 2.0)
            base_q_surf = 0.38 * (1.0 - atmo_weight) + 1.25 * atmo_weight
        q = (1.0 - f_atmo) * base_q_surf + f_atmo * q_atmo_scat
        A_b = float(np.clip(q * A_g, 0.0, 1.0))
    else:
        p_rgb = np.clip(p_surf, 0.0, 1.0)
        A_g = float(0.2126 * p_rgb[0] + 0.7152 * p_rgb[1] + 0.0722 * p_rgb[2])
        b_type = body_info.get('type', 'Terrestrial')
        q = 1.50 if b_type == 'Gas Giant' else 0.38
        A_b = float(np.clip(q * A_g, 0.0, 1.0))

    return A_g, A_b, q, p_rgb

def _apply_body_edits(app, insp_idx, body_info, parent_idx, cur_mass_snap, cur_pos_snap_render, cur_vel_snap_render, visual_arr):
    """Package and dispatch edited physical and orbital properties to physics thread and system JSON."""
    ed = app.camera["edit_data"]
    rot_hours = float(ed.get("rotation_period", 0.0))
    m_val = float(ed.get("mass", cur_mass_snap[insp_idx]))
    r_km = float(ed.get("radius", body_info.get('r', 0.0) * 696340.0))
    b_type = ed.get("type", body_info.get("type", "Terrestrial"))
    body_name = body_info['name']

    f = 0.0
    j2 = 0.0
    j4 = 0.0
    if rot_hours > 0.0 and m_val > 0.0 and r_km > 0.0:
        omega = 2.0 * math.pi / (rot_hours * 3600.0)
        r_eq_m = r_km * 1000.0
        mass_kg = m_val * 1.98847e30
        G_SI = 6.6743e-11
        m_param = (omega**2 * r_eq_m**3) / (G_SI * mass_kg)
        if b_type in ("Moon", "Dwarf Planet"):
            chi = 1.25
        elif b_type == "Terrestrial":
            chi = 0.95
        else:
            chi = 0.65
        f = chi * m_param
        j2 = m_param * (chi - 0.5)
        j4 = -0.15 * (f**2)

    pole_render = visual_arr[insp_idx, 5:8]
    pole_ecl = [float(pole_render[0]), float(-pole_render[2]), float(pole_render[1])]

    payload = {
        "action": "UPDATE",
        "idx": insp_idx,
        "name": body_name,
        "mass": m_val,
        "radius": r_km,
        "type": b_type,
        "rotation_period": rot_hours,
        "oblateness": f,
        "J2": j2,
        "j4": j4,
        "pole_ecl": pole_ecl
    }

    if b_type == "Star" and ed.get("_preview"):
        pr = ed["_preview"]
        hx = pr["visual"]["colorHex"].lstrip('#')
        payload["color"] = [int(hx[0:2], 16)/255.0, int(hx[2:4], 16)/255.0, int(hx[4:6], 16)/255.0]
        payload["star_props"] = {
            "mode": ed.get("star_mode", "evolution"),
            "mass": ed["mass"] if ed.get("star_mode") == "evolution" else pr["physical"]["mass_msun"],
            "metallicity": float(ed.get("metallicity", 0.0)),
            "age_pct": float(ed.get("age_pct", 46.0)) / 100.0,
            "evo_path": ed.get("evo_path", "standard"),
            "radius": pr["physical"]["radius_rsun"],
            "temp": pr["physical"]["temp_k"],
            "lum": pr["physical"]["lum_lsun"],
            "locked_rad": ed.get("locked_rad", True),
            "locked_temp": ed.get("locked_temp", True),
            "locked_lum": ed.get("locked_lum", False),
            "class": pr["classification"]["fullDesignation"],
            "stage": pr["evolution"]["phase"]
        }

    if parent_idx >= 0:
        p_pos = cur_pos_snap_render[parent_idx]
        p_vel = cur_vel_snap_render[parent_idx]
        ppx, ppy, ppz = p_pos[0], -p_pos[2], p_pos[1]
        pvx, pvy, pvz = p_vel[0], -p_vel[2], p_vel[1]
        parent_m = cur_mass_snap[parent_idx]

        a_val = float(ed.get("a", body_info.get("a", 1.0)))
        e_val = float(ed.get("e", body_info.get("e", 0.0)))
        inc_val = float(ed.get("inc", body_info.get("inc", 0.0)))
        Omega_val = float(ed.get("Omega", body_info.get("Omega", 0.0)))
        omega_val = float(ed.get("omega", body_info.get("omega", 0.0)))
        M_val = float(ed.get("M", body_info.get("M", 0.0)))

        c_pos, c_vel = get_cartesian_from_keplerian(
            parent_m, m_val, a_val, e_val,
            inc_val, Omega_val, omega_val, M_val
        )

        if app.camera.get("inspector_frame", 0) == 1:
            pole_render_p = visual_arr[parent_idx, 5:8]
            pole_ecl_p = np.array([pole_render_p[0], -pole_render_p[2], pole_render_p[1]])
            c_pos, c_vel = rotate_equatorial_to_ecliptic(c_pos, c_vel, pole_ecl_p)

        new_ecl_x = ppx + c_pos[0]
        new_ecl_y = ppy + c_pos[1]
        new_ecl_z = ppz + c_pos[2]

        new_ecl_vx = pvx + c_vel[0]
        new_ecl_vy = pvy + c_vel[1]
        new_ecl_vz = pvz + c_vel[2]

        payload["pos"] = [new_ecl_x, new_ecl_y, new_ecl_z]
        payload["vel"] = [new_ecl_vx, new_ecl_vy, new_ecl_vz]
        payload["rel_pos"] = [float(c_pos[0]), float(c_pos[1]), float(c_pos[2])]
        payload["rel_vel"] = [float(c_vel[0]), float(c_vel[1]), float(c_vel[2])]
        payload["keplerian"] = {
            "a": a_val, "e": e_val, "inc": inc_val,
            "Omega": Omega_val, "omega": omega_val, "M": M_val
        }

    with app.shared_state["lock"]:
        app.shared_state["crud_queue"].append(payload)

    body_info['r'] = r_km / 696340.0
    visual_arr[insp_idx, 3] = body_info['r'] * SOLAR_RADII_TO_AU

    app.camera["edit_mode"] = False

def render_body_inspector(app, ctx, bodies_data, num_bodies, parent_snap, mass_snap, subsys_mass_buf, subsys_pos_buf, subsys_vel_buf, pos_snap_render, vel_snap_render, visual_arr, cam_world_pos_f8, active_system_name, atmo_bodies, ring_bodies, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex):
    """Render the right-side tabbed Body Inspector panel."""
    if not getattr(app, "show_inspector", True):
        return

    insp_is_cmp = app.camera.get("inspected_is_cmp", False)
    insp_idx = app.camera.get("inspected_idx", None)
    limit_bodies = app.num_bodies_cmp if insp_is_cmp else num_bodies

    if insp_idx is None or insp_idx < 0 or insp_idx >= limit_bodies:
        return

    inspect_bary = app.camera.get("inspect_bary", False)
    cur_bodies_data = app.bodies_data_cmp if insp_is_cmp else bodies_data
    cur_parent_snap = app.parent_snap_cmp if insp_is_cmp else parent_snap
    cur_mass_snap = app.mass_snap_cmp if insp_is_cmp else mass_snap
    cur_subsys_mass_buf = app.subsys_mass_buf_cmp if insp_is_cmp else subsys_mass_buf
    cur_subsys_pos_buf = app.subsys_pos_buf_cmp if insp_is_cmp else subsys_pos_buf
    cur_subsys_vel_buf = app.subsys_vel_buf_cmp if insp_is_cmp else subsys_vel_buf
    cur_pos_snap_render = app.pos_snap_cmp if insp_is_cmp else pos_snap_render
    cur_vel_snap_render = app.vel_snap_cmp if insp_is_cmp else vel_snap_render
    cur_visual_arr = app.visual_data_cmp if insp_is_cmp else visual_arr

    body_info = cur_bodies_data[insp_idx]
    body_name = body_info['name']
    parent_idx = int(cur_parent_snap[insp_idx])

    if inspect_bary:
        win_title = f"{body_name} System Barycenter###inspector"
    else:
        win_title = f"{body_name}###inspector"

    insp_w = 350
    insp_x = app.fb_width - insp_w - 10
    insp_y = 32
    bottom_space = 70 if getattr(app, "show_time_hud", True) else 15
    insp_h = max(200, app.fb_height - insp_y - bottom_space)

    force_layout = getattr(app, "_last_fb_width", 0) != app.fb_width or getattr(app, "_last_fb_height", 0) != app.fb_height
    cond = imgui.ALWAYS if force_layout else imgui.ONCE

    # Reset edit mode if inspected body changed
    last_insp = app.camera.get("_last_inspected_idx")
    last_cmp = app.camera.get("_last_inspected_is_cmp")
    if last_insp != insp_idx or last_cmp != insp_is_cmp:
        app.camera["edit_mode"] = False
        app.camera["_last_inspected_idx"] = insp_idx
        app.camera["_last_inspected_is_cmp"] = insp_is_cmp

    imgui.set_next_window_position(insp_x, insp_y, cond)
    imgui.set_next_window_size(insp_w, insp_h, cond)

    expanded, opened = imgui.begin(win_title, True)
    if not opened:
        app.camera["inspected_idx"] = None
        app.camera["inspect_bary"] = False
        app.camera["edit_mode"] = False
        imgui.end()
        return

    if not expanded:
        imgui.end()
        return

    # ── Top Bar: Type Badge, Edit Mode, & Tracking ──
    if inspect_bary:
        imgui.text_colored("Mode: Barycenter", 0.6, 0.9, 1.0)
    else:
        obj_type = body_info.get('type', 'Unknown')
        imgui.text_colored(f"Type: {obj_type}", 0.7, 0.8, 1.0)
        if not insp_is_cmp:
            imgui.same_line(spacing=15)
            changed_edit, app.camera["edit_mode"] = imgui.checkbox("Edit Mode", app.camera.get("edit_mode", False))
            if changed_edit:
                if app.camera["edit_mode"]:
                    app.time_ctrl["multiplier"] = 0.0
                    app.time_ctrl["paused"] = True
                    sp = body_info.get("star_props", {})
                    app.camera["edit_data"] = {
                        "mass": float(cur_mass_snap[insp_idx]),
                        "radius": float(body_info.get('r', 0.0) * 696340.0),
                        "rotation_period": float(body_info.get("rotation_period", 0.0)),
                        "type": body_info.get("type", "Moon"),
                        "a": float(body_info.get("a", 1.0)),
                        "e": float(body_info.get("e", 0.0)),
                        "inc": float(body_info.get("inc", 0.0)),
                        "Omega": float(body_info.get("Omega", 0.0)),
                        "omega": float(body_info.get("omega", 0.0)),
                        "M": float(body_info.get("M", 0.0)),
                        "init_orbit": True,
                        "star_mode": sp.get("mode", "evolution"),
                        "metallicity": float(sp.get("metallicity", 0.0)),
                        "age_pct": float(sp.get("age_pct", 0.46) * 100.0),
                        "s_rad": float(sp.get("radius", float(body_info.get('r', 1.0)))),
                        "s_temp": float(sp.get("temp", 5778.0)),
                        "s_lum": float(sp.get("lum", 1.0)),
                        "locked_rad": sp.get("locked_rad", True),
                        "locked_temp": sp.get("locked_temp", True),
                        "locked_lum": sp.get("locked_lum", False)
                    }

    # Target position and radius calculation
    target_pos = cur_subsys_pos_buf[insp_idx].copy() if inspect_bary else cur_pos_snap_render[insp_idx].copy()
    if insp_is_cmp:
        target_pos += np.array([app.comparison_offset_au, 0.0, 0.0], dtype='f8')
    cam_rel = cam_world_pos_f8 - target_pos
    dist_to_center_km = np.linalg.norm(cam_rel) * 149597870.7
    dist_to_center_au = dist_to_center_km / 149597870.7
    body_r_km = body_info.get('r', 0.0) * 696340.0
    body_mass = cur_mass_snap[insp_idx]

    # Target approach distance
    if inspect_bary:
        oe_a_val = float(body_info.get('a', 1.0))
        target_d = max(oe_a_val * 2.5, 0.01)
    else:
        r_au = (body_r_km / 149597870.7)
        target_d = max(r_au * 3.5, 1e-6) if r_au > 0 else 0.01

    # ── Action Buttons: Track, Center, Go To, Barycenter ──
    is_tracked = (app.camera["tracking_idx"] == insp_idx and
                  app.camera.get("tracking_is_cmp", False) == insp_is_cmp and
                  app.camera.get("tracking_bary", False) == inspect_bary)

    # 1. Track Button
    track_btn_label = "Tracking" if is_tracked else "Track"
    if is_tracked:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.3)
    if imgui.button(f"{track_btn_label}##track_btn"):
        if is_tracked:
            app.camera["target"] = cam_world_pos_f8.copy()
            app.camera["cam_pos_rel"] = np.zeros(3, dtype='f8')
            app.camera["cam_pos_rel_prev"] = np.zeros(3, dtype='f8')
            app.camera["tracking_idx"] = None
            app.camera["cam_look"] = "free"
            app.camera["centered_idx"] = None
        else:
            app.camera["tracking_idx"] = insp_idx
            app.camera["tracking_is_cmp"] = insp_is_cmp
            app.camera["tracking_bary"] = inspect_bary
            app.camera["tracking_mode"] = "bary" if inspect_bary else "body"
            app.camera["cam_pos_rel"] = (cam_world_pos_f8 - target_pos).copy()
            app.camera["cam_look"] = "aim"
            app.camera["approach_delta"] = 0.0
            app.camera["centered_idx"] = None
    if is_tracked:
        imgui.pop_style_color()

    # 2. Center Object Button
    is_centered = (app.camera.get("centered_idx") == insp_idx and
                   app.camera.get("centered_is_cmp", False) == insp_is_cmp and
                   app.camera.get("centered_bary", False) == inspect_bary)
    imgui.same_line(spacing=6)
    center_btn_label = "Centered" if is_centered else "Center"
    if is_centered:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.3)
    if imgui.button(f"{center_btn_label}##center_btn"):
        if is_centered:
            app.camera["centered_idx"] = None
        else:
            app.camera["centered_idx"] = insp_idx
            app.camera["centered_is_cmp"] = insp_is_cmp
            app.camera["centered_bary"] = inspect_bary

            # Immediately align view to center the object
            refract_params, grav_lens_params = app._get_active_refraction_and_lens_params(
                cam_world_pos_f8,
                atmo_bodies=atmo_bodies,
                pos_snap_render=pos_snap_render,
                mass_snap=mass_snap,
                bodies_data=bodies_data,
                num_bodies=num_bodies,
            )
            fwd_t = get_apparent_look_direction(
                cam_world_pos_f8,
                target_pos,
                refract_params=refract_params,
                grav_lens_params=grav_lens_params,
                target_body_idx=insp_idx,
                target_is_cmp=insp_is_cmp,
            )
            y_t, p_t = _camera_yaw_pitch_from(fwd_t)
            app.camera["yaw"] = app.camera["yaw_actual"] = y_t
            app.camera["pitch"] = app.camera["pitch_actual"] = p_t
            cur_up = _camera_get_up(app.camera, fwd_t)
            app.camera["up"] = cur_up.tolist()
            app.camera["cam_look"] = "free"
            app.camera["approach_delta"] = 0.0
    if is_centered:
        imgui.pop_style_color()

    # 3. Go To Planet / Object Button
    imgui.same_line(spacing=6)
    if imgui.button("Go To##goto_btn"):
        app.camera["centered_idx"] = None
        app.camera["tracking_idx"] = insp_idx
        app.camera["tracking_is_cmp"] = insp_is_cmp
        app.camera["tracking_bary"] = inspect_bary
        app.camera["tracking_mode"] = "bary" if inspect_bary else "body"
        app.camera["cam_look"] = "aim"

        delta = cam_world_pos_f8 - target_pos
        d = np.linalg.norm(delta)
        if d > 1e-12:
            app.camera["cam_pos_rel"] = (delta / d) * target_d
        else:
            app.camera["cam_pos_rel"] = np.array([0.0, 0.0, target_d], dtype='f8')
        app.camera["approach_delta"] = 0.0

        if app.camera.get("movement_mode", 0) == 0:
            app.camera["flight_speed"] = max(target_d * 0.05, 1e-10)

    # 4. Barycenter / Body Mode Toggle
    imgui.same_line(spacing=6)
    bary_btn_label = "Barycenter" if not inspect_bary else "Body"
    if imgui.button(f"{bary_btn_label}##toggle_bary_btn"):
        app.camera["inspect_bary"] = not inspect_bary
        if app.camera["tracking_idx"] == insp_idx and app.camera.get("tracking_is_cmp", False) == insp_is_cmp:
            app.camera["tracking_bary"] = app.camera["inspect_bary"]
            app.camera["tracking_mode"] = "bary" if app.camera["inspect_bary"] else "body"
        if app.camera.get("centered_idx") == insp_idx and app.camera.get("centered_is_cmp", False) == insp_is_cmp:
            app.camera["centered_bary"] = app.camera["inspect_bary"]

    # Altitude & Distance Readout
    thresh_au = app.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
    if dist_to_center_km > 1.49597e7:
        imgui.text(f"Distance: {format_distance_au(dist_to_center_au, threshold_au=thresh_au)}")
    else:
        imgui.text(f"Distance: {dist_to_center_km:,.0f} km")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Distance from the camera to the center of the body.")

    if not inspect_bary and body_r_km > 0.0:
        f_oblate = float(body_info.get("oblateness", 0.0))
        if not insp_is_cmp and app.camera.get("edit_mode", False) and "edit_data" in app.camera:
            ed = app.camera["edit_data"]
            if "oblateness" in ed:
                f_oblate = float(ed["oblateness"])
        if f_oblate <= 0.0 and cur_visual_arr is not None and len(cur_visual_arr) > insp_idx:
            f_oblate = float(cur_visual_arr[insp_idx, 8])
        if f_oblate <= 0.0:
            sp = body_info.get('star_props', {})
            if sp and sp.get('r_eq') and sp.get('r_pole') and float(sp['r_eq']) > 0:
                f_oblate = max(0.0, 1.0 - float(sp['r_pole']) / float(sp['r_eq']))

        if cur_visual_arr is not None and len(cur_visual_arr) > insp_idx:
            body_pole = cur_visual_arr[insp_idx, 5:8]
        elif hasattr(app, "pole_n_arr") and app.pole_n_arr is not None and len(app.pole_n_arr) > insp_idx:
            body_pole = app.pole_n_arr_cmp[insp_idx] if insp_is_cmp else app.pole_n_arr[insp_idx]
        else:
            pole_ra = body_info.get('pole_ra')
            pole_dec = body_info.get('pole_dec')
            if pole_ra is not None and pole_dec is not None:
                p_ecl = pole_to_ecliptic(float(pole_ra), float(pole_dec))
                body_pole = np.array([p_ecl[0], p_ecl[2], -p_ecl[1]], dtype='f8')
            else:
                body_pole = np.array([0.0, 1.0, 0.0], dtype='f8')

        active_r_km = float(app.camera["edit_data"].get("radius", body_r_km)) if (not insp_is_cmp and app.camera.get("edit_mode", False) and "edit_data" in app.camera) else body_r_km
        surf_r_km = _ellipsoid_surface_radius(active_r_km, f_oblate, body_pole, cam_rel)
        alt_km = dist_to_center_km - surf_r_km
        alt_au = alt_km / 149597870.7
        imgui.text(f"Altitude: {format_altitude(alt_km, alt_au, threshold_au=thresh_au)}")
        if imgui.is_item_hovered():
            if f_oblate > 0.0:
                r_pole_km = active_r_km * (1.0 - f_oblate)
                imgui.set_tooltip(
                    f"Camera altitude above the oblate spheroid surface along line of sight.\n"
                    f"Local surface radius: {surf_r_km:,.1f} km (Flattening f = {f_oblate:.4f})\n"
                    f"Equatorial: {active_r_km:,.1f} km | Polar: {r_pole_km:,.1f} km"
                )
            else:
                imgui.set_tooltip(
                    f"Camera altitude above the surface along line of sight.\n"
                    f"Surface radius: {surf_r_km:,.1f} km"
                )


    imgui.separator()

    # ── Real-Time Relative Vectors & Osculating Keplerian Elements ──
    has_parent = (parent_idx >= 0 and parent_idx < limit_bodies)
    oe_a = float(body_info.get('a', 1.0))
    oe_e = float(body_info.get('e', 0.0))
    oe_inc = float(body_info.get('inc', 0.0))
    oe_Omega = float(body_info.get('Omega', 0.0))
    oe_omega = float(body_info.get('omega', 0.0))
    oe_nu = 0.0
    oe_M = float(body_info.get('M', 0.0))
    oe_P = 0.0
    rel_r = np.zeros(3, dtype='f8')
    rel_v = np.zeros(3, dtype='f8')
    axial_tilt_deg = None

    if has_parent:
        if inspect_bary:
            bp = cur_subsys_pos_buf[insp_idx]
            bv = cur_subsys_vel_buf[insp_idx]
            pp = cur_pos_snap_render[parent_idx]
            pv = cur_vel_snap_render[parent_idx]
            rel_r = bp - pp
            rel_v = bv - pv
        else:
            rel_r = cur_pos_snap_render[insp_idx] - cur_pos_snap_render[parent_idx]
            rel_v = cur_vel_snap_render[insp_idx] - cur_vel_snap_render[parent_idx]
            orb_h = np.cross(rel_r, rel_v)
            h_norm = np.linalg.norm(orb_h)
            if h_norm > 1e-12:
                orb_normal = orb_h / h_norm
                pole_ra = body_info.get('pole_ra')
                pole_dec = body_info.get('pole_dec')
                if pole_ra is not None and pole_dec is not None:
                    body_pole = pole_to_ecliptic(pole_ra, pole_dec)
                    body_pole_engine = np.array([body_pole[0], body_pole[2], -body_pole[1]])
                    tilt_rad = math.acos(np.clip(np.dot(body_pole_engine, orb_normal), -1.0, 1.0))
                    axial_tilt_deg = math.degrees(tilt_rad)

        ecl_rx, ecl_ry, ecl_rz = rel_r[0], -rel_r[2], rel_r[1]
        ecl_vx, ecl_vy, ecl_vz = rel_v[0], -rel_v[2], rel_v[1]

        if app.camera.get("inspector_frame", 0) == 1:
            pole_render_p = cur_visual_arr[parent_idx, 5:8]
            pole_ecl_p = np.array([pole_render_p[0], -pole_render_p[2], pole_render_p[1]])
            c_pos, c_vel = rotate_ecliptic_to_equatorial(
                np.array([ecl_rx, ecl_ry, ecl_rz]), 
                np.array([ecl_vx, ecl_vy, ecl_vz]), 
                pole_ecl_p
            )
            ecl_rx, ecl_ry, ecl_rz = c_pos
            ecl_vx, ecl_vy, ecl_vz = c_vel

        orb_mass = body_mass if not inspect_bary else 0.0
        mu = G * (cur_mass_snap[parent_idx] + orb_mass)
        oe_a, oe_e, oe_inc, oe_Omega, oe_omega, oe_nu, oe_M, oe_P = \
            compute_keplerian_elements(ecl_rx, ecl_ry, ecl_rz,
                                       ecl_vx, ecl_vy, ecl_vz, mu)

    # Initialize edit_data orbit if needed
    if not insp_is_cmp and app.camera.get("edit_mode", False) and not inspect_bary:
        ed = app.camera["edit_data"]
        if ed.get("init_orbit"):
            if has_parent:
                ed["a"] = float(oe_a)
                ed["e"] = float(oe_e)
                ed["inc"] = float(oe_inc)
                ed["Omega"] = float(oe_Omega)
                ed["omega"] = float(oe_omega)
                ed["M"] = float(oe_M)
            ed["init_orbit"] = False

    # ── Tabbed View ──
    has_atmo = (body_info.get('atmosphere') is not None)
    has_rings = (body_info.get('rings') is not None and len(body_info.get('rings', [])) > 0)

    if imgui.begin_tab_bar("InspectorTabBar"):
        # ── Tab 1: Overview & Physical ──
        if imgui.begin_tab_item("Overview")[0]:
            imgui.text_colored("Physical Properties", 1.0, 0.85, 0.4)
            if inspect_bary:
                bary_mass = cur_subsys_mass_buf[insp_idx]
                if bary_mass > 1e-4:
                    imgui.text(f"  System Mass: {bary_mass:.6e} M\u2609")
                else:
                    m_earth = bary_mass / 3.003e-6
                    imgui.text(f"  System Mass: {m_earth:.4f} M\u2295")
            elif not insp_is_cmp and app.camera.get("edit_mode", False):
                ed = app.camera["edit_data"]
                types_list = ["Star", "Terrestrial", "Gas Giant", "Ice Giant", "Dwarf Planet", "Moon"]
                if "type" not in ed: ed["type"] = body_info.get("type", "Moon")
                type_idx = types_list.index(ed["type"]) if ed["type"] in types_list else 1
                changed_t, type_idx = imgui.combo("Type", type_idx, types_list)
                if changed_t: ed["type"] = types_list[type_idx]

                if ed["type"] == "Star":
                    changed_m, mode_idx = imgui.combo("Star Mode", 0 if ed.get("star_mode", "evolution")=="evolution" else 1, ["Evolution Track", "Surface Physics"])
                    if changed_m: ed["star_mode"] = ["evolution", "surface"][mode_idx]

                    if ed["star_mode"] == "evolution":
                        _, ed["mass"] = imgui.drag_float(u"Mass (M\u2609)", ed["mass"], 0.01, 0.01, 300.0, format="%.4f")
                        _, ed["metallicity"] = imgui.drag_float("[Fe/H] (dex)", ed.get("metallicity", 0.0), 0.01, -4.0, 1.0, format="%.3f")
                        _, ed["age_pct"] = imgui.drag_float("Life Cycle (%)", ed.get("age_pct", 46.0), 0.1, -5.0, 120.0, format="%.1f%%")

                        if ed["mass"] >= 45.0:
                            imgui.text_colored("Humphreys-Davidson Limit Reached", 1.0, 0.5, 0.2)
                            ed["evo_path"] = "stripping"
                        elif ed["mass"] >= 25.0:
                            if "evo_path" not in ed: ed["evo_path"] = "standard"
                            changed_p, p_idx = imgui.combo("Evolution Branch", 0 if ed["evo_path"]=="standard" else 1, ["Red Supergiant", "LBV -> Wolf-Rayet"])
                            if changed_p: ed["evo_path"] = ["standard", "stripping"][p_idx]
                        else:
                            ed["evo_path"] = "standard"

                        try:
                            age_pct = ed["age_pct"] / 100.0
                            preview = StarCalculator.forge(mode="evolution", evo_path=ed.get("evo_path", "standard"), mass=ed["mass"], metallicity=ed["metallicity"], age_pct=age_pct)
                            ed["radius"] = preview["physical"]["radius_rsun"] * 696340.0
                            ed["_preview"] = preview
                        except Exception:
                            ed["_preview"] = None
                    else:
                        if "locked_rad" not in ed: ed["locked_rad"] = True
                        if "locked_temp" not in ed: ed["locked_temp"] = True
                        if "locked_lum" not in ed: ed["locked_lum"] = False
                        num_locked = ed["locked_rad"] + ed["locked_temp"] + ed["locked_lum"]

                        c1, ed["locked_rad"] = imgui.checkbox("##lr", ed["locked_rad"]); imgui.same_line(); _, ed["s_rad"] = imgui.drag_float(u"Radius (R\u2609)", ed.get("s_rad", 1.0), 0.05, 0.01, 2500.0, format="%.4f")
                        c2, ed["locked_temp"] = imgui.checkbox("##lt", ed["locked_temp"]); imgui.same_line(); _, ed["s_temp"] = imgui.drag_float("Temp (K)", ed.get("s_temp", 5778.0), 10.0, 1000.0, 100000.0, format="%.0f")
                        c3, ed["locked_lum"] = imgui.checkbox("##ll", ed["locked_lum"]); imgui.same_line(); _, ed["s_lum"] = imgui.drag_float(u"Luminosity (L\u2609)", ed.get("s_lum", 1.0), 0.01, 0.0001, 2000000.0, format="%.4f")

                        if c1 and ed["locked_rad"] and num_locked == 2:
                            if ed["locked_temp"] and not c2: ed["locked_temp"] = False
                            elif ed["locked_lum"] and not c3: ed["locked_lum"] = False
                        elif c2 and ed["locked_temp"] and num_locked == 2:
                            if ed["locked_rad"] and not c1: ed["locked_rad"] = False
                            elif ed["locked_lum"] and not c3: ed["locked_lum"] = False
                        elif c3 and ed["locked_lum"] and num_locked == 2:
                            if ed["locked_rad"] and not c1: ed["locked_rad"] = False
                            elif ed["locked_temp"] and not c2: ed["locked_temp"] = False

                        if ed["locked_rad"] + ed["locked_temp"] + ed["locked_lum"] != 2:
                            imgui.text_colored("Must lock exactly 2 properties!", 1.0, 0.3, 0.3)
                            ed["_preview"] = None
                        else:
                            try:
                                preview = StarCalculator.forge(mode="surface", mass=1.0, metallicity=0.0,
                                    radius=ed["s_rad"] if ed["locked_rad"] else None,
                                    temp=ed["s_temp"] if ed["locked_temp"] else None,
                                    lum=ed["s_lum"] if ed["locked_lum"] else None)
                                ed["radius"] = preview["physical"]["radius_rsun"] * 696340.0
                                ed["mass"] = preview["physical"]["mass_msun"]
                                ed["_preview"] = preview
                            except Exception:
                                ed["_preview"] = None

                    edit_mass = ed["mass"]
                    edit_r_km = ed["radius"]
                else:
                    _, ed["mass"] = imgui.input_double(u"Mass (M\u2609)", ed["mass"], format="%e")
                    _, ed["radius"] = imgui.input_double("Radius (km)", ed["radius"], format="%.1f")

                    is_locked = False
                    if has_parent:
                        parent_m = cur_mass_snap[parent_idx]
                        body_m = ed["mass"]
                        r_km = ed["radius"]
                        if body_m > 0 and r_km > 0:
                            r_tid = 0.00084 * ((parent_m**2 / body_m)**(1.0/6.0)) * math.sqrt(r_km)
                            a_val = ed.get("a", oe_a)
                            if a_val < r_tid:
                                is_locked = True
                                total_m = parent_m + body_m
                                p_years = math.sqrt(a_val**3 / total_m) if total_m > 0 else 0.0
                                ed["rotation_period"] = p_years * 365.25 * 24.0

                    if is_locked:
                        imgui.text(f"  Rot Period: {ed['rotation_period']:.4f} hours [Tidally Locked]")
                    else:
                        _, ed["rotation_period"] = imgui.input_double("Rot Period (hours)", ed["rotation_period"], format="%.4f")
                    edit_mass = ed["mass"]
                    edit_r_km = ed["radius"]

                if edit_r_km > 0.0:
                    g_m_s2 = (1.32712440018e14 * edit_mass) / (edit_r_km ** 2)
                    g_earth = g_m_s2 / 9.80665
                    if g_m_s2 >= 1e-4:
                        imgui.text(f"  Surface G: {g_m_s2:.3f} m/s² ({g_earth:.3f} g)")
                    else:
                        imgui.text(f"  Surface G: {g_m_s2:.3e} m/s² ({g_earth:.3e} g)")
            else:
                if body_mass > 1e-4:
                    imgui.text(f"  Mass:    {body_mass:.6e} M\u2609")
                elif body_mass > 1e-10:
                    m_earth = body_mass / 3.003e-6
                    imgui.text(f"  Mass:    {m_earth:.6f} M\u2295")
                else:
                    m_lunar = body_mass / 3.694e-8
                    imgui.text(f"  Mass:    {m_lunar:.4e} M\u263E")

                if body_r_km > 100:
                    imgui.text(f"  Radius:  {body_r_km:,.0f} km")
                elif body_r_km > 0.1:
                    imgui.text(f"  Radius:  {body_r_km:.1f} km")

                if body_r_km > 0.0:
                    g_m_s2 = (1.32712440018e14 * body_mass) / (body_r_km ** 2)
                    g_earth = g_m_s2 / 9.80665
                    if g_m_s2 >= 1e-4:
                        imgui.text(f"  Surface G: {g_m_s2:.3f} m/s² ({g_earth:.3f} g)")
                    else:
                        imgui.text(f"  Surface G: {g_m_s2:.3e} m/s² ({g_earth:.3e} g)")

                rot_p = float(body_info.get("rotation_period", 0.0))
                if rot_p != 0.0:
                    imgui.text(f"  Rot Period: {rot_p:,.2f} hours")

                f_oblate = float(body_info.get("oblateness", 0.0))
                if f_oblate > 0.0:
                    imgui.text(f"  Oblateness (f): {f_oblate:.4f}")
                    if imgui.is_item_hovered():
                        r_pole_km = body_r_km * (1.0 - f_oblate)
                        imgui.set_tooltip(
                            f"Equatorial Radius: {body_r_km:,.1f} km\n"
                            f"Polar Radius:      {r_pole_km:,.1f} km\n"
                            f"Polar Flattening:  {body_r_km - r_pole_km:,.1f} km"
                        )

            # Stellar details
            if not inspect_bary and ('star_props' in body_info or (not insp_is_cmp and app.camera.get("edit_mode", False) and app.camera.get("edit_data", {}).get("type") == "Star")):
                ed_pr = app.camera.get("edit_data", {}).get("_preview") if (not insp_is_cmp and app.camera.get("edit_mode", False)) else None
                sp = body_info.get('star_props', {})
                imgui.separator()
                imgui.text_colored("Stellar Properties", 1.0, 0.85, 0.4)
                if ed_pr:
                    imgui.text(f"  Temperature: {ed_pr['physical']['temp_k']:,.0f} K")
                    imgui.text(f"  Luminosity:  {ed_pr['physical']['lum_lsun']:.4f} L\u2609")
                    imgui.text(f"  Spectral Cl: {ed_pr['classification']['fullDesignation']}")
                    imgui.text(f"  Stage:       {ed_pr['evolution']['phase']}")
                else:
                    imgui.text(f"  Temperature: {sp.get('temp', 0.0):,.0f} K")
                    imgui.text(f"  Luminosity:  {sp.get('lum', 0.0):.4f} L\u2609")
                    imgui.text(f"  Spectral Cl: {sp.get('class', 'Unknown')}")
                    imgui.text(f"  Stage:       {sp.get('stage', 'Unknown')}")

                hz_lum = ed_pr['physical']['lum_lsun'] if ed_pr else sp.get('lum', 0.0)
                if math.isfinite(hz_lum) and hz_lum > 0:
                    imgui.text("  Habitable Zone (optimistic):")
                    imgui.text(f"    Inner: {format_distance_au(math.sqrt(hz_lum / 1.78), threshold_au=thresh_au, precision=5)}")
                    imgui.text(f"    Outer: {format_distance_au(math.sqrt(hz_lum / 0.32), threshold_au=thresh_au, precision=5)}")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Luminosity-based boundaries matching the habitable-zone overlay.\nNot a prediction of an individual planet's climate.")

                mode_val = app.camera.get("edit_data", {}).get("star_mode", sp.get('mode', 'evolution')) if (not insp_is_cmp and app.camera.get("edit_mode", False)) else sp.get('mode', 'evolution')
                if mode_val == 'evolution':
                    met_val = app.camera.get("edit_data", {}).get("metallicity", sp.get('metallicity', 0.0)) if (not insp_is_cmp and app.camera.get("edit_mode", False)) else sp.get('metallicity', 0.0)
                    imgui.text(f"  Metallicity: {met_val:.3f}")
                    if ed_pr:
                        age_gyr = ed_pr['evolution']['age_gyr']
                    else:
                        age_gyr = sp.get('age', 0.0)
                    if age_gyr < 0.1:
                        imgui.text(f"  Age:         {age_gyr * 1000.0:,.1f} Myr")
                    else:
                        imgui.text(f"  Age:         {age_gyr:,.3f} Gyr")
            elif not inspect_bary:
                # Albedos & Observational
                A_g, A_b, q_val, _ = compute_body_albedos(app, body_info, insp_idx, insp_is_cmp, visual_arr, atmo_bodies, cur_mass_snap)
                imgui.separator()
                imgui.text_colored("Observational Dynamics", 1.0, 0.85, 0.4)
                imgui.text(f"  Geom Albedo: {A_g:.3f}")
                imgui.text(f"  Bond Albedo: {A_b:.3f} (q = {q_val:.2f})")

            imgui.end_tab_item()

        # ── Tab 2: Orbit & Dynamics ──
        if imgui.begin_tab_item("Orbit")[0]:
            if not has_parent:
                imgui.text_colored("Primary Central Body (No Parent Orbit)", 0.6, 0.9, 1.0)
            else:
                parent_name = cur_bodies_data[parent_idx]['name'] if 0 <= parent_idx < len(cur_bodies_data) else "Barycenter"
                imgui.text_colored(f"Parent: {parent_name}", 0.6, 0.9, 1.0)
                imgui.separator()

                if axial_tilt_deg is not None:
                    if axial_tilt_deg > 90.0:
                        imgui.text(f"  Axial Tilt: {axial_tilt_deg:.2f}° (Retrograde)")
                    else:
                        imgui.text(f"  Axial Tilt: {axial_tilt_deg:.2f}°")

                app.camera.setdefault("inspector_frame", 0)
                changed_frame, app.camera["inspector_frame"] = imgui.combo("Reference Frame", app.camera["inspector_frame"], ["Ecliptic", "Equatorial"])
                if changed_frame:
                    app.save_settings()
                    if not insp_is_cmp and app.camera.get("edit_mode", False):
                        app.camera["edit_data"]["init_orbit"] = True

                if not insp_is_cmp and app.camera.get("edit_mode", False) and not inspect_bary:
                    ed = app.camera["edit_data"]
                    imgui.separator()
                    imgui.text_colored("Edit Orbit", 0.5, 0.8, 1.0)
                    _, ed["a"] = imgui.input_double("Semi-Major Axis (AU)", ed.get("a", oe_a), format="%.6f")
                    _, ed["e"] = imgui.input_double("Eccentricity", ed.get("e", oe_e), format="%.6f")
                    _, ed["inc"] = imgui.input_double("Inclination (deg)", ed.get("inc", oe_inc), format="%.3f")
                    _, ed["Omega"] = imgui.input_double(u"\u03A9 (Long Asc Node)", ed.get("Omega", oe_Omega), format="%.3f")
                    _, ed["omega"] = imgui.input_double(u"\u03C9 (Arg Periapsis)", ed.get("omega", oe_omega), format="%.3f")
                    _, ed["M"] = imgui.input_double("Mean Anomaly (deg)", ed.get("M", oe_M), format="%.3f")

                else:
                    use_km = oe_a < 0.01
                    if use_km:
                        imgui.text(f"  Semi-major (a): {oe_a * AU_TO_KM:,.0f} km")
                    else:
                        imgui.text(f"  Semi-major (a): {format_distance_au(oe_a, threshold_au=thresh_au, precision=6)}")
                    imgui.text(f"  Eccentricity (e): {oe_e:.6f}")
                    imgui.text(f"  Inclination (i):  {oe_inc:.3f}°")
                    imgui.text(f"  Long Asc Node (Ω):{oe_Omega:.3f}°")
                    imgui.text(f"  Arg Periapsis (ω):{oe_omega:.3f}°")
                    imgui.text(f"  True Anomaly (ν): {oe_nu:.3f}°")
                    imgui.text(f"  Mean Anomaly (M): {oe_M:.3f}°")

                imgui.separator()
                imgui.text_colored("Derived Quantities", 1.0, 0.85, 0.4)
                if oe_P > 0:
                    if oe_P < 730:
                        imgui.text(f"  Period:      {oe_P:.3f} days")
                    else:
                        imgui.text(f"  Period:      {oe_P/365.25:.4f} years")

                active_a = ed["a"] if (not insp_is_cmp and app.camera.get("edit_mode", False) and not inspect_bary) else oe_a
                active_e = ed["e"] if (not insp_is_cmp and app.camera.get("edit_mode", False) and not inspect_bary) else oe_e
                periapsis = active_a * (1.0 - active_e)
                apoapsis = active_a * (1.0 + active_e)
                dist_au = math.sqrt(rel_r[0]**2 + rel_r[1]**2 + rel_r[2]**2)
                vel_au_yr = math.sqrt(rel_v[0]**2 + rel_v[1]**2 + rel_v[2]**2)
                vel_km_s = vel_au_yr * 4.7405

                use_km = active_a < 0.01
                if use_km:
                    imgui.text(f"  Periapsis:   {periapsis * AU_TO_KM:,.0f} km")
                    imgui.text(f"  Apoapsis:    {apoapsis * AU_TO_KM:,.0f} km")
                    imgui.text(f"  Distance:    {dist_au * AU_TO_KM:,.0f} km")
                else:
                    imgui.text(f"  Periapsis:   {format_distance_au(periapsis, threshold_au=thresh_au, precision=6)}")
                    imgui.text(f"  Apoapsis:    {format_distance_au(apoapsis, threshold_au=thresh_au, precision=6)}")
                    imgui.text(f"  Distance:    {format_distance_au(dist_au, threshold_au=thresh_au, precision=6)}")
                imgui.text(f"  Velocity:    {vel_km_s:.3f} km/s")

                if not inspect_bary:
                    imgui.separator()
                    imgui.text_colored("Gravitational Limits", 1.0, 0.85, 0.4)
                    parent_mass = cur_mass_snap[parent_idx]
                    if parent_mass > 0 and body_mass > 0 and dist_au > 0:
                        # Match the instantaneous radius used by hierarchy selection.
                        hill_au = dist_au * (body_mass / (3.0 * parent_mass)) ** (1.0 / 3.0)
                        imgui.text(f"  Hill Radius (now): {format_distance_au(hill_au, threshold_au=thresh_au, precision=5)}")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Approximate sphere of influence around this body, at its current distance from its parent.\nUses the live state, not unapplied edits; not a guaranteed stable satellite orbit.\nThe approximation assumes this body is much less massive than its parent.")
                    else:
                        imgui.text_disabled("  Hill Radius: N/A")

                    editing = not insp_is_cmp and app.camera.get("edit_mode", False)
                    limit_mass = float(ed.get("mass", body_mass)) if editing else body_mass
                    limit_radius = float(ed.get("radius", body_r_km)) if editing else body_r_km
                    if parent_mass > 0 and limit_mass > 0 and limit_radius > 0:
                        roche_au = 2.44 * limit_radius / AU_TO_KM * (parent_mass / limit_mass) ** (1.0 / 3.0)
                        imgui.text(f"  Roche Limit (fluid): {format_distance_au(roche_au, threshold_au=thresh_au, precision=5)}")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Distance from the parent's center below which this body may be tidally disrupted.\nFluid, strengthless-body approximation using this body's mass and radius.\nUses proposed mass, radius and periapsis in Edit Mode.")
                        if periapsis < roche_au:
                            imgui.text_colored("  Periapsis is inside Roche limit", 1.0, 0.3, 0.3)
                    else:
                        imgui.text_disabled("  Roche Limit: N/A")

                if oe_a > 0 and oe_e < 1.0 and not inspect_bary:
                    imgui.separator()
                    imgui.text_colored("Precession (arcsec/cy)", 1.0, 0.85, 0.4)
                    c2 = C_AU_YR * C_AU_YR
                    p_param = oe_a * (1.0 - oe_e * oe_e)
                    if p_param > 0:
                        gr_rad_per_orbit = 3.0 * mu / (oe_a * c2 * (1.0 - oe_e*oe_e))
                        orbits_per_century = 36525.0 / oe_P if oe_P > 0 else 0
                        gr_arcsec_cy = gr_rad_per_orbit * orbits_per_century * (180.0/math.pi) * 3600.0
                        imgui.text(f"  GR (apsidal):  {gr_arcsec_cy:.2f}")
                    else:
                        imgui.text("  GR (apsidal):  0.00")

                    j2_apsidal = 0.0
                    j2_nodal = 0.0
                    parent_j2 = cur_bodies_data[parent_idx].get('J2', 0.0)
                    if parent_j2 > 0 and oe_P > 0:
                        parent_r_au = cur_bodies_data[parent_idx].get('r', 0.0) * SOLAR_RADII_TO_AU
                        n_mean = 2.0 * math.pi / (oe_P / 365.25)
                        if p_param > 0:
                            ratio2 = (parent_r_au / p_param) ** 2
                            j2_apsidal_rad_yr = 1.5 * n_mean * parent_j2 * ratio2
                            j2_nodal_rad_yr = -j2_apsidal_rad_yr * math.cos(math.radians(oe_inc))
                            j2_apsidal = j2_apsidal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0
                            j2_nodal = j2_nodal_rad_yr * 100.0 * (180.0/math.pi) * 3600.0

                    imgui.text(f"  J2 (apsidal):  {j2_apsidal:.2f}")
                    imgui.text(f"  J2 (nodal):    {j2_nodal:.2f}")

            imgui.end_tab_item()

        # ── Tab 3: Atmosphere ──
        if not inspect_bary and imgui.begin_tab_item("Atmosphere")[0]:
            cur_atmo_bodies = app.atmo_bodies_cmp if insp_is_cmp else atmo_bodies
            atmo_item = next((a for a in cur_atmo_bodies if a['body_idx'] == insp_idx), None)
            mass_val = cur_mass_snap[insp_idx] if len(cur_mass_snap) > insp_idx else 3.0e-6
            R_km = atmo_item.get('planet_radius_km', body_r_km) if atmo_item else body_r_km

            if atmo_item:
                # Compute T_eq for the planet
                star_idx_local = -1
                for k, b in enumerate(cur_bodies_data):
                    if b.get('type') == 'Star':
                        star_idx_local = k
                        break
                if star_idx_local != -1:
                    star_body = cur_bodies_data[star_idx_local]
                    star_lum = star_body.get('star_props', {}).get('lum', 1.0)
                    star_pos = cur_pos_snap_render[star_idx_local]
                    planet_pos = cur_pos_snap_render[insp_idx]
                    to_planet = planet_pos - star_pos
                    dist_to_star_au = np.linalg.norm(to_planet)
                    if dist_to_star_au > 1e-12:
                        L_dir = to_planet / dist_to_star_au
                    else:
                        L_dir = np.array([0.0, 1.0, 0.0])

                    star_pole = app.visual_arr_cmp[star_idx_local, 5:8] if insp_is_cmp else visual_arr[star_idx_local, 5:8]
                    sin_lat = abs(np.dot(L_dir, star_pole))

                    star_sp = star_body.get('star_props', {})
                    if star_sp.get('rot_frac', 0.0) > 0.0:
                        lum_eq = star_sp.get('lum_eq', star_lum)
                        lum_pole = star_sp.get('lum_pole', star_lum)
                        star_lum_dir = lum_eq * (1.0 - sin_lat) + lum_pole * sin_lat
                    else:
                        star_lum_dir = star_lum
                else:
                    star_lum_dir = 1.0

                eff_a = 1.0
                curr = insp_idx
                while curr >= 0:
                    p_id = cur_parent_snap[curr]
                    if p_id == -1:
                        break
                    p_body = cur_bodies_data[p_id]
                    if p_body.get('type') == 'Star':
                        eff_a = cur_bodies_data[curr].get('a', 1.0)
                        break
                    curr = p_id

                _, A_b, q_val, _ = compute_body_albedos(app, body_info, insp_idx, insp_is_cmp, visual_arr, atmo_bodies, cur_mass_snap)
                albedo = A_b

                eff_a_safe = max(1e-4, eff_a) if not (math.isnan(eff_a) or math.isinf(eff_a)) else 1.0
                star_lum_safe = max(0.0, star_lum_dir) if not (math.isnan(star_lum_dir) or math.isinf(star_lum_dir)) else 1.0
                albedo_safe = min(0.9999, max(0.0, float(albedo))) if not (math.isnan(albedo) or math.isinf(albedo)) else 0.3
                t_eq_atmo = 278.5 * ((star_lum_safe / (eff_a_safe**2))**0.25) * ((1.0 - albedo_safe)**0.25)
                v_c_atmo = app.visual_arr_cmp[insp_idx, 0:3] if insp_is_cmp else visual_arr[insp_idx, 0:3]
                _, A_surf_atmo, surf_src_atmo = compute_surface_albedo(body_info, getattr(app, 'texture_mean_colors', None), v_c_atmo)
                src_label_atmo = "texture map" if surf_src_atmo == 'texture' else ("albedo scale" if surf_src_atmo == 'albedo_scale' else "base color")
                imgui.text(f"Surface Albedo: {A_surf_atmo:.3f} ({src_label_atmo})")
                imgui.text(f"Bond Albedo (A_b): {albedo:.3f} (q = {q_val:.2f})")
                imgui.text(f"Equilibrium Temperature: {t_eq_atmo:.1f} K")
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Before greenhouse warming; uses the existing atmospheric model's orbital distance and Bond albedo.")

                # Calculate temperature dynamically from greenhouse effect
                comp = atmo_item.get('composition', {})
                potencies = {'CO2': 1.0, 'H2O': 1.5, 'CH4': 25.0, 'SO2': 5.0, 'Tholin': -15.0}
                f_gh = sum(comp.get(gas, 0.0) * factor for gas, factor in potencies.items())
                f_gh = max(0.0, f_gh)

                press = max(0.0, float(atmo_item.get('surface_pressure', 1.0)))
                tau = (press ** 0.63) * (0.84 + 2.51 * f_gh)
                t_calc = t_eq_atmo * ((1.0 + 0.75 * tau) ** 0.25)
                if math.isnan(t_calc) or math.isinf(t_calc) or t_calc <= 0.0:
                    t_calc = 288.15
                if abs(atmo_item.get('temperature', 0.0) - t_calc) > 1e-4:
                    atmo_item['temperature'] = t_calc
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

                if 'atmosphere' not in body_info:
                    body_info['atmosphere'] = {}
                body_info['atmosphere']['temperature'] = atmo_item['temperature']
                body_info['atmosphere']['surface_pressure'] = atmo_item['surface_pressure']
                body_info['atmosphere']['composition'] = atmo_item['composition']

                props, _, _ = get_cached_atmosphere_properties(atmo_item, mass_val)
                atmo_h_km = float(atmo_item.get('height', props['atmo_height_km']))
                atmo_item['atmo_radius_km'] = R_km + atmo_h_km
                atmo_item['atmo_radius_au'] = atmo_item['atmo_radius_km'] / 149597870.7
                body_info['atmosphere']['height'] = atmo_h_km

                current_height = atmo_item['atmo_radius_km'] - R_km
                imgui.text(f"Atmosphere Height: {current_height:,.1f} km")
                imgui.text(f"Gas Scale Height: {props['scale_height_km']:.2f} km")
                imgui.text(f"Mean Molar Mass: {props['molar_mass'] * 1000.0:.2f} g/mol")

                changed_p, new_p = imgui.drag_float("Surface Pressure (atm)", atmo_item.get('surface_pressure', 1.0), 0.01, 0.0, 100.0)
                if changed_p:
                    atmo_item['surface_pressure'] = new_p
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

                imgui.text("Temperature: {:.1f} K ({:+.1f} °C) [Calculated]".format(atmo_item['temperature'], atmo_item['temperature'] - 273.15))

                comp = atmo_item.get('composition', {"N2": 0.78, "O2": 0.21})
                if imgui.tree_node("Composition"):
                    comp_keys = list(comp.keys())
                    gas_changed = False
                    for gas in comp_keys:
                        changed, new_pct = imgui.slider_float(f"{gas}##slider", comp[gas] * 100.0, 0.0, 100.0, "%.1f%%")
                        if changed:
                            comp[gas] = new_pct / 100.0
                            gas_changed = True

                        imgui.same_line()
                        if imgui.button(f"X##{gas}"):
                            del comp[gas]
                            gas_changed = True

                    total_pct = sum(comp.values()) * 100.0
                    imgui.text_disabled(f"Total: {total_pct:.1f}%  (auto-normalized)")

                    available_gases = [g for g in GAS_PROPERTIES.keys() if g not in comp]
                    if available_gases:
                        if 'selected_gas' not in atmo_item:
                            atmo_item['selected_gas'] = 0

                        atmo_item['selected_gas'] = min(atmo_item['selected_gas'], len(available_gases) - 1)
                        changed, new_idx = imgui.combo("##AddGasCombo", atmo_item['selected_gas'], available_gases)
                        if changed:
                            atmo_item['selected_gas'] = new_idx

                        imgui.same_line()
                        if imgui.button("Add Gas"):
                            comp[available_gases[atmo_item['selected_gas']]] = 0.10
                            gas_changed = True

                    if gas_changed:
                        atmo_item['composition'] = comp
                        atmo_item['_dirty'] = True
                        if 'lut_tex' in atmo_item:
                            atmo_item['lut_tex'].release()
                            del atmo_item['lut_tex']
                    imgui.tree_pop()

                changed_m, b_mie = imgui.drag_float("Aerosol Beta (x10^-6)", atmo_item.get('beta_mie', 2.0e-6)*1e6, 0.1)
                if changed_m:
                    atmo_item['beta_mie'] = b_mie * 1e-6
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']
                changed_hm, new_hm = imgui.drag_float("Aerosol Scale (km)", atmo_item.get('h_mie', 1.2), 0.1, 0.1, 1000.0)
                if changed_hm:
                    atmo_item['h_mie'] = new_hm
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']
                changed_mg, new_mg = imgui.slider_float("Aerosol Asymmetry", atmo_item.get('mie_g', 0.758), 0.0, 0.999)
                if changed_mg:
                    atmo_item['mie_g'] = new_mg
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

                cur_alb = atmo_item.get('mie_albedo', None)
                cur_alb_val = 1.0 if cur_alb is None else float(np.mean(cur_alb))
                changed_alb, new_alb = imgui.slider_float("Aerosol Albedo (w0)", cur_alb_val, 0.0, 1.0)
                if changed_alb:
                    atmo_item['mie_albedo'] = np.array([new_alb, new_alb, new_alb], dtype=np.float32)
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

                is_manual = ('mie_angstrom' in atmo_item and atmo_item['mie_angstrom'] is not None)
                changed_mode, use_manual = imgui.checkbox("Manual Angstrom Exponent", is_manual)
                if changed_mode:
                    atmo_item['mie_angstrom'] = 1.2 if use_manual else None
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

                if 'mie_angstrom' in atmo_item and atmo_item['mie_angstrom'] is not None:
                    changed_ang, new_ang = imgui.slider_float("Aerosol Angstrom", atmo_item['mie_angstrom'], 0.0, 5.0)
                    if changed_ang:
                        atmo_item['mie_angstrom'] = new_ang
                        atmo_item['_dirty'] = True
                        if 'lut_tex' in atmo_item:
                            atmo_item['lut_tex'].release()
                            del atmo_item['lut_tex']
                else:
                    auto_val = 1.2 * math.exp(-atmo_item.get('beta_mie', 2.0e-6) / 5.0e-6)
                    imgui.text(f"  Auto Angstrom Exponent: {auto_val:.3f} (based on Beta)")

                _, atmo_item['intensity'] = imgui.drag_float("Intensity", atmo_item.get('intensity', 1.0), 0.5, 0.0, 1000.0)

                imgui.separator()
                if imgui.button("Remove Atmosphere", width=-1):
                    cur_atmo_bodies.remove(atmo_item)
                    if 'atmosphere' in body_info:
                        del body_info['atmosphere']
            else:
                imgui.text_colored("No Atmosphere Present", 0.6, 0.6, 0.6)
                if imgui.button("Add Atmosphere", width=-1):
                    new_atmo = {
                        'body_idx': insp_idx,
                        'planet_radius_km': body_r_km,
                        'surface_radius_au': body_r_km / 149597870.7,
                        'atmo_radius_km': body_r_km * 1.025,
                        'atmo_radius_au': (body_r_km * 1.025) / 149597870.7,
                        'surface_pressure': 1.0,
                        'temperature': 288.15,
                        'composition': {"N2": 0.78, "O2": 0.21},
                        'beta_mie': 2.0e-6,
                        'h_mie': 1.2,
                        'mie_g': 0.758,
                        'mie_albedo': np.array([1.0, 1.0, 1.0], dtype=np.float32),
                        'intensity': 1.0
                    }
                    cur_atmo_bodies.append(new_atmo)
                    body_info['atmosphere'] = {
                        'surface_pressure': 1.0,
                        'temperature': 288.15,
                        'composition': {"N2": 0.78, "O2": 0.21}
                    }
            imgui.end_tab_item()

        # ── Tab 4: Rings ──
        if not inspect_bary and imgui.begin_tab_item("Rings")[0]:
            rings = [r for r in ring_precomputed if r['body_idx'] == insp_idx]

            for i, ring_item in enumerate(rings):
                is_tex_layer = ring_item.get('is_textured', False)
                node_title = f"Textured Ring Layer {i}" if is_tex_layer else f"Ring Layer {i}"
                if imgui.tree_node(node_title):
                    changed_in, new_in = imgui.drag_float(f"Inner Radius (km)##{i}", ring_item['inner_r'] * 149597870.7, 10.0, body_r_km, ring_item['outer_r'] * 149597870.7 - 10)
                    changed_out, new_out = imgui.drag_float(f"Outer Radius (km)##{i}", ring_item['outer_r'] * 149597870.7, 10.0, ring_item['inner_r'] * 149597870.7 + 10, body_r_km * 50.0)
                    changed_col, new_col = imgui.color_edit3(f"Color##{i}", *ring_item['raw_color'])
                    changed_op, new_op = imgui.drag_float(f"Opacity##{i}", ring_item['opacity'], 0.005, 0.0, 2.0, "%.4f")

                    changed_scat = False
                    if not is_tex_layer:
                        changed_scat, new_scat = imgui.drag_float(f"Phase Balance (Back <-> Fwd)##{i}", ring_item.get('scatter', 1.0), 0.005, 0.0, 1.0, "%.4f")

                    changed_asym, new_asym = imgui.drag_float(f"Forward Scatter Asym##{i}", ring_item.get('asymmetry', 0.7), 0.005, -0.999, 0.999, "%.4f")
                    changed_bks, new_bks = imgui.drag_float(f"Backscatter##{i}", ring_item.get('backscatter', -0.3), 0.005, -0.999, 0.999, "%.4f")

                    changed_unlit = False
                    if not is_tex_layer:
                        changed_unlit, new_unlit = imgui.drag_float(f"Unlit Side Multiplier##{i}", ring_item.get('unlit_factor', 1.0), 0.01, 0.0, 2.0, "%.3f")

                    changed_sat = False
                    changed_hue = False
                    changed_bri = False
                    changed_boost = False
                    reset_edits = False

                    if is_tex_layer:
                        if imgui.tree_node(f"Texture Editor##{i}"):
                            changed_unlit, new_unlit = imgui.drag_float(f"Unlit Side Multiplier##{i}", ring_item.get('unlit_factor', 1.0), 0.01, 0.0, 2.0, "%.3f")
                            changed_sat, new_sat = imgui.drag_float(f"Saturation##{i}", ring_item.get('saturation', 1.0), 0.01, 0.0, 3.0, "%.2f")
                            changed_hue, new_hue = imgui.drag_float(f"Hue Shift##{i}", ring_item.get('hue_shift', 0.0), 0.005, -1.0, 1.0, "%.3f")
                            changed_bri, new_bri = imgui.drag_float(f"Brightness##{i}", ring_item.get('brightness', 1.0), 0.01, 0.0, 3.0, "%.2f")
                            changed_boost, new_boost = imgui.drag_float(f"Alpha Boost##{i}", ring_item.get('alpha_boost', 1.0), 0.01, 0.1, 5.0, "%.2f")
                            if imgui.button(f"Reset Edits##{i}"):
                                ring_item['unlit_factor'] = 1.0
                                ring_item['saturation'] = 1.0
                                ring_item['hue_shift'] = 0.0
                                ring_item['brightness'] = 1.0
                                ring_item['alpha_boost'] = 1.0
                                reset_edits = True
                            imgui.same_line()
                            if imgui.button(f"Bake & Save Texture##{i}"):
                                b_name = cur_bodies_data[insp_idx]['name']
                                bake_and_export_ring_textures(app, b_name, ring_item, ring_precomputed, ring_render_groups, ring_gradient_tex)
                            imgui.tree_pop()

                    grad_changed = False
                    grad_sort_needed = False
                    if imgui.tree_node(f"Alpha Gradient##{i}"):
                        grad = ring_item['gradient']
                        if imgui.button(f"Add Stop##{i}"):
                            grad.append({'p': 1.0, 'a': 1.0})
                            grad_sort_needed = True

                        stops_to_remove = []
                        for j, stop in enumerate(grad):
                            imgui.push_item_width(100)
                            changed_p, n_p = imgui.drag_float(f"Pos##{i}_{id(stop)}", stop['p'], 0.005, 0.0, 1.0, "%.4f")
                            if imgui.is_item_deactivated_after_edit():
                                grad_sort_needed = True
                            imgui.same_line()
                            changed_a, n_a = imgui.drag_float(f"Alpha##{i}_{id(stop)}", stop['a'], 0.005, 0.0, 1.0, "%.4f")
                            imgui.same_line()
                            if imgui.button(f"X##{i}_{id(stop)}"):
                                stops_to_remove.append(j)
                                grad_sort_needed = True
                            imgui.pop_item_width()

                            if changed_p or changed_a:
                                stop['p'] = n_p
                                stop['a'] = n_a
                                grad_changed = True

                        for j in reversed(stops_to_remove):
                            grad.pop(j)
                            grad_changed = True

                        if grad_sort_needed:
                            grad.sort(key=lambda x: x['p'])
                            grad_changed = True

                        imgui.tree_pop()

                    if changed_in or changed_out or changed_col or changed_op or changed_scat or changed_asym or changed_bks or changed_unlit or changed_sat or changed_hue or changed_bri or changed_boost or grad_changed or reset_edits:
                        if changed_in: ring_item['inner_r'] = new_in / 149597870.7
                        if changed_out: ring_item['outer_r'] = new_out / 149597870.7
                        if changed_col: ring_item['raw_color'] = new_col
                        if changed_op: ring_item['opacity'] = new_op
                        if changed_scat: ring_item['scatter'] = new_scat
                        if changed_asym: ring_item['asymmetry'] = new_asym
                        if changed_bks: ring_item['backscatter'] = new_bks
                        if not reset_edits:
                            if changed_unlit: ring_item['unlit_factor'] = new_unlit
                            if changed_sat: ring_item['saturation'] = new_sat
                            if changed_hue: ring_item['hue_shift'] = new_hue
                            if changed_bri: ring_item['brightness'] = new_bri
                            if changed_boost: ring_item['alpha_boost'] = new_boost

                        shadow_grad = generate_ring_shadow_grad(
                            ring_item['gradient'],
                            tex_sampled=ring_item.get('tex_sampled'),
                            raw_color=ring_item.get('raw_color', (1.0, 1.0, 1.0)),
                            hue_shift=ring_item.get('hue_shift', 0.0),
                            saturation=ring_item.get('saturation', 1.0),
                            brightness=ring_item.get('brightness', 1.0),
                            alpha_boost=ring_item.get('alpha_boost', 1.0)
                        )
                        ring_item['shadow_grad'] = shadow_grad
                        app.body_ring_indices = rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)

                    if imgui.button(f"Remove Layer##{i}"):
                        ring_precomputed[:] = [r for r in ring_precomputed if r is not ring_item]
                        app.body_ring_indices = rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)

                    imgui.tree_pop()

            if imgui.button("Add Ring Layer", width=-1):
                if len(rings) < 16:
                    r_in = body_r_km * 1.2 / 149597870.7
                    if rings:
                        r_in = rings[-1]['outer_r'] + (100 / 149597870.7)
                    r_out = r_in + (body_r_km * 0.5 / 149597870.7)
                    pole_render = cur_visual_arr[insp_idx, 5:8]
                    color = cur_visual_arr[insp_idx, 0:3]
                    grad = [{'p': 0.0, 'a': 0.0}, {'p': 0.5, 'a': 1.0}, {'p': 1.0, 'a': 0.0}]
                    pole_n = pole_render / np.linalg.norm(pole_render)
                    shadow_grad = generate_ring_shadow_grad(grad, raw_color=color)
                    ring_precomputed.append({
                        'body_idx': insp_idx,
                        'pole': pole_n.astype('f4'),
                        'inner_r': r_in, 'outer_r': r_out, 'opacity': 0.8,
                        'scatter': 0.0, 'asymmetry': 0.8, 'backscatter': -0.3, 'shadow_grad': shadow_grad,
                        'raw_color': color, 'gradient': grad,
                        'unlit_factor': 1.0, 'saturation': 1.0, 'hue_shift': 0.0,
                        'brightness': 1.0, 'alpha_boost': 1.0,
                        'row_idx': len(ring_precomputed)
                    })
                    app.body_ring_indices = rebuild_ring_render_group(insp_idx, ctx, prog_rings, ring_precomputed, ring_render_groups, ring_gradient_tex)
            imgui.end_tab_item()

        # ── Tab 5: Cosmetics ──
        if not inspect_bary and imgui.begin_tab_item("Cosmetics")[0]:
            imgui.text_colored("Cosmetics & Surface Color", 1.0, 0.85, 0.4)
            imgui.separator()

            # Convert linear albedo to sRGB for the color picker
            if insp_is_cmp:
                srgb_c = [pow(c, 1.0/2.2) if c > 0 else 0.0 for c in app.visual_arr_cmp[insp_idx, 0:3]]
            else:
                srgb_c = [pow(c, 1.0/2.2) if c > 0 else 0.0 for c in visual_arr[insp_idx, 0:3]]
            changed_c, new_c = imgui.color_edit3("Base Color (sRGB)", *srgb_c)
            if changed_c:
                linear_c = [pow(c, 2.2) if c > 0 else 0.0 for c in new_c]
                if insp_is_cmp:
                    app.visual_arr_cmp[insp_idx, 0:3] = linear_c
                    app.visual_data_cmp[insp_idx][0:3] = linear_c
                    atmo_item = next((a for a in app.atmo_bodies_cmp if a['body_idx'] == insp_idx), None)
                else:
                    visual_arr[insp_idx, 0:3] = linear_c
                    if hasattr(app, "visual_data") and len(app.visual_data) > insp_idx:
                        app.visual_data[insp_idx][0:3] = linear_c
                    atmo_item = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)

                if atmo_item:
                    atmo_item['_dirty'] = True
                    if 'lut_tex' in atmo_item:
                        atmo_item['lut_tex'].release()
                        del atmo_item['lut_tex']

            v_c_cosmetic = app.visual_arr_cmp[insp_idx, 0:3] if insp_is_cmp else visual_arr[insp_idx, 0:3]
            _, A_surf_c, surf_src_c = compute_surface_albedo(body_info, getattr(app, 'texture_mean_colors', None), v_c_cosmetic)
            src_lbl = "texture map" if surf_src_c == 'texture' else ("albedo scale" if surf_src_c == 'albedo_scale' else "base color")
            imgui.text(f"Surface Albedo: {A_surf_c:.3f} ({src_lbl})")

            imgui.separator()
            if not insp_is_cmp:
                if imgui.button("Export Body Cosmetics", width=-1):
                    is_hidden_combo = (hasattr(app, 'window') and glfw.get_key(app.window, glfw.KEY_BACKSLASH) == glfw.PRESS)
                    def fmt(val, dec=5): return round(float(val), dec)

                    if is_hidden_combo:
                        if active_system_name == SystemManager.SOLAR_SYSTEM_NAME:
                            system_file = "data/system.json"
                        else:
                            system_file = app.sys_mgr._system_json_path(active_system_name)
                        try:
                            with open(system_file, "r") as f:
                                sys_data = json.load(f)

                            for s_body in sys_data:
                                name = s_body.get("name")
                                b_idx = None
                                for i_b, b in enumerate(bodies_data):
                                    if b.get("name") == name:
                                        b_idx = i_b
                                        break

                                if b_idx is not None:
                                    c = visual_arr[b_idx, 0:3]
                                    s_body["color"] = '#%02x%02x%02x' % (min(255, max(0, int(c[0]*255))), min(255, max(0, int(c[1]*255))), min(255, max(0, int(c[2]*255))))

                                    atmo_it = next((a for a in atmo_bodies if a['body_idx'] == b_idx), None)
                                    if atmo_it:
                                        s_body["atmosphere"] = {
                                            "surface_pressure": float(fmt(atmo_it.get("surface_pressure", 1.0), 3)),
                                            "temperature": float(fmt(atmo_it.get("temperature", 288.15), 2)),
                                            "composition": atmo_it.get("composition", {"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
                                            "height": float(fmt(atmo_it["atmo_radius_km"] - atmo_it["planet_radius_km"], 2))
                                        }

                                    ring_segs = [r for r in ring_precomputed if r['body_idx'] == b_idx]
                                    if ring_segs:
                                        s_body["rings"] = []
                                        body_r_au = (bodies_data[b_idx].get('r', 0.0) * 696340.0) / 1.495978707e8
                                        for r in ring_segs:
                                            rc = r.get('raw_color', [1.0, 1.0, 1.0])
                                            hex_col = '#%02x%02x%02x' % (min(255, max(0, int(rc[0]*255))), min(255, max(0, int(rc[1]*255))), min(255, max(0, int(rc[2]*255))))
                                            s_body["rings"].append({
                                                "inner": fmt(r['inner_r'] / body_r_au) if body_r_au > 0 else 1.0,
                                                "outer": fmt(r['outer_r'] / body_r_au) if body_r_au > 0 else 2.0,
                                                "color": hex_col,
                                                "opacity": fmt(r['opacity']),
                                                "scatter": fmt(r['scatter']),
                                                "asymmetry": fmt(r['asymmetry']),
                                                "backscatter": fmt(r['backscatter']),
                                                "gradient": [{"p": fmt(g['p']), "a": fmt(g['a'])} for g in r.get('gradient', [])]
                                            })

                            with open(system_file, "w") as f:
                                json.dump(sys_data, f, indent=2, cls=NumpyEncoder)
                            print(f"[Cosmetics] Secret Combo: Exported entire system cosmetics directly to {system_file}!")
                        except Exception as e:
                            print(f"Error updating system.json: {e}")
                    else:
                        exp = {}
                        c = visual_arr[insp_idx, 0:3]
                        exp["color"] = '#%02x%02x%02x' % (min(255, max(0, int(c[0]*255))), min(255, max(0, int(c[1]*255))), min(255, max(0, int(c[2]*255))))

                        atmo_it = next((a for a in atmo_bodies if a['body_idx'] == insp_idx), None)
                        if atmo_it:
                            exp["atmosphere"] = {
                                "surface_pressure": float(fmt(atmo_it.get("surface_pressure", 1.0), 3)),
                                "temperature": float(fmt(atmo_it.get("temperature", 288.15), 2)),
                                "composition": atmo_it.get("composition", {"N2": 0.78, "O2": 0.21, "Ar": 0.01}),
                                "height": float(fmt(atmo_it["atmo_radius_km"] - atmo_it["planet_radius_km"], 2))
                            }

                        ring_segs = [r for r in ring_precomputed if r['body_idx'] == insp_idx]
                        if ring_segs:
                            exp["rings"] = []
                            body_r_au = (body_info.get('r', 0.0) * 696340.0) / 1.495978707e8
                            for r in ring_segs:
                                rc = r.get('raw_color', [1.0, 1.0, 1.0])
                                hex_col = '#%02x%02x%02x' % (min(255, max(0, int(rc[0]*255))), min(255, max(0, int(rc[1]*255))), min(255, max(0, int(rc[2]*255))))
                                exp["rings"].append({
                                    "inner": fmt(r['inner_r'] / body_r_au) if body_r_au > 0 else 1.0,
                                    "outer": fmt(r['outer_r'] / body_r_au) if body_r_au > 0 else 2.0,
                                    "color": hex_col,
                                    "opacity": fmt(r['opacity']),
                                    "scatter": fmt(r['scatter']),
                                    "asymmetry": fmt(r['asymmetry']),
                                    "backscatter": fmt(r['backscatter']),
                                    "gradient": [{"p": fmt(g['p']), "a": fmt(g['a'])} for g in r.get('gradient', [])]
                                })

                        os.makedirs("exports", exist_ok=True)
                        filename = os.path.join("exports", f"{body_info['name'].replace(' ', '_').lower()}_cosmetics.json")
                        with open(filename, 'w') as f:
                            json.dump(exp, f, indent=4, cls=NumpyEncoder)
                        print(f"[Cosmetics] Exported cosmetics to {filename}")
            imgui.end_tab_item()

        imgui.end_tab_bar()

        # ── Edit Mode Action Bar ──
        if not insp_is_cmp and app.camera.get("edit_mode", False) and not inspect_bary:
            imgui.separator()
            ed = app.camera.get("edit_data", {})
            roche_violates = False
            if has_parent:
                parent_m = cur_mass_snap[parent_idx]
                body_m = float(ed.get("mass", body_mass))
                r_km = float(ed.get("radius", body_r_km))
                a_val = float(ed.get("a", oe_a))
                e_val = float(ed.get("e", oe_e))
                if body_m > 0 and r_km > 0:
                    d_roche_km = 2.44 * r_km * ((parent_m / body_m)**(1.0/3.0))
                    d_roche_au = d_roche_km / AU_TO_KM
                    periapsis_au = a_val * (1.0 - e_val)
                    if periapsis_au < d_roche_au:
                        roche_violates = True

            if roche_violates:
                imgui.text_colored("Warning: Orbit violates Roche limit!", 1.0, 0.3, 0.3)
                imgui.text_colored("[Apply Changes Disabled]", 0.5, 0.5, 0.5)
                if imgui.button("Cancel##edit_cancel_disabled", width=-1):
                    app.camera["edit_mode"] = False
            else:
                half_w = (insp_w - 30) // 2
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.2, 0.6, 0.3)
                if imgui.button("Apply Changes##apply_edits", width=half_w):
                    _apply_body_edits(app, insp_idx, body_info, parent_idx, cur_mass_snap, cur_pos_snap_render, cur_vel_snap_render, cur_visual_arr)
                imgui.pop_style_color()
                imgui.same_line(spacing=10)
                if imgui.button("Cancel##edit_cancel", width=half_w):
                    app.camera["edit_mode"] = False

    imgui.end()

