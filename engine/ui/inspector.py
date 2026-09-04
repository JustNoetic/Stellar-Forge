import math
import os
import json
import numpy as np
import imgui
import glfw
from engine.core.constants import DEFAULT_LY_THRESHOLD_AU
from engine.physics.atmosphere_physics import compute_mie_coefficients, GAS_PROPERTIES
from engine.rendering.render_utils import (
    compute_surface_albedo,
    format_distance_au,
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
    """Compute local radius on an oblate spheroid in direction unit vector u."""
    if f <= 0.0 or r_eq <= 0.0:
        return r_eq
    u = np.asarray(u, dtype='f8')
    pole = np.asarray(pole, dtype='f8')
    cos_phi = float(np.dot(u, pole))
    sin_phi_sq = max(0.0, 1.0 - cos_phi * cos_phi)
    one_minus_f_sq = (1.0 - f) * (1.0 - f)
    denom = sin_phi_sq + (cos_phi * cos_phi) / one_minus_f_sq
    if denom <= 0.0:
        return r_eq
    return r_eq / math.sqrt(denom)

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

    imgui.set_next_window_position(insp_x, insp_y, cond)
    imgui.set_next_window_size(insp_w, insp_h, cond)

    expanded, opened = imgui.begin(win_title, True)
    if not opened:
        app.camera["inspected_idx"] = None
        app.camera["inspect_bary"] = False
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
            changed_edit, app.camera["edit_mode"] = imgui.checkbox("Edit Mode", app.camera["edit_mode"])
            if changed_edit and app.camera["edit_mode"]:
                app.time_ctrl["multiplier"] = 0.0
                sp = body_info.get("star_props", {})
                app.camera["edit_data"] = {
                    "mass": float(cur_mass_snap[insp_idx]),
                    "radius": float(body_info.get('r', 0.0) * 696340.0),
                    "rotation_period": float(body_info.get("rotation_period", 0.0)),
                    "type": body_info.get("type", "Moon"),
                    "a": 0.0, "e": 0.0, "inc": 0.0, "Omega": 0.0, "omega": 0.0, "M": 0.0,
                    "init_orbit": True,
                    "star_mode": sp.get("mode", "evolution"),
                    "metallicity": sp.get("metallicity", 0.0),
                    "age_pct": sp.get("age_pct", 0.46) * 100.0,
                    "s_rad": sp.get("radius", float(body_info.get('r', 1.0))),
                    "s_temp": sp.get("temp", 5778.0),
                    "s_lum": sp.get("lum", 1.0),
                    "locked_rad": sp.get("locked_rad", True),
                    "locked_temp": sp.get("locked_temp", True),
                    "locked_lum": sp.get("locked_lum", False)
                }

    # Target position and radius calculation
    target_pos = cur_subsys_pos_buf[insp_idx].copy() if inspect_bary else cur_pos_snap_render[insp_idx].copy()
    if insp_is_cmp:
        target_pos += np.array([app.comparison_offset_au, 0.0, 0.0], dtype='f8')
    dist_to_center_km = np.linalg.norm(target_pos - cam_world_pos_f8) * 149597870.7
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
            app.camera["tracking_idx"] = None
            app.camera["cam_look"] = "free"
        else:
            app.camera["tracking_idx"] = insp_idx
            app.camera["tracking_is_cmp"] = insp_is_cmp
            app.camera["tracking_bary"] = inspect_bary
            app.camera["tracking_mode"] = "bary" if inspect_bary else "body"
            app.camera["cam_pos_rel"] = (cam_world_pos_f8 - target_pos).copy()
            app.camera["cam_look"] = "aim"
            app.camera["approach_delta"] = 0.0
    if is_tracked:
        imgui.pop_style_color()

    # 2. Center Object Button
    imgui.same_line(spacing=6)
    if imgui.button("Center##center_btn"):
        app.camera["tracking_idx"] = insp_idx
        app.camera["tracking_is_cmp"] = insp_is_cmp
        app.camera["tracking_bary"] = inspect_bary
        app.camera["tracking_mode"] = "bary" if inspect_bary else "body"
        app.camera["cam_pos_rel"] = (cam_world_pos_f8 - target_pos).copy()
        app.camera["cam_look"] = "aim"
        app.camera["approach_delta"] = 0.0

    # 3. Go To Planet / Object Button
    imgui.same_line(spacing=6)
    if imgui.button("Go To##goto_btn"):
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

    # Altitude & Distance Readout
    thresh_au = app.camera.get("ly_threshold_au", DEFAULT_LY_THRESHOLD_AU)
    if dist_to_center_km > 1.49597e7:
        imgui.text(f"Distance: {format_distance_au(dist_to_center_au, threshold_au=thresh_au)}")
    else:
        imgui.text(f"Distance: {dist_to_center_km:,.0f} km")

    imgui.separator()

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

            # Stellar details
            if not inspect_bary and 'star_props' in body_info:
                sp = body_info['star_props']
                imgui.separator()
                imgui.text_colored("Stellar Properties", 1.0, 0.85, 0.4)
                imgui.text(f"  Temperature: {sp.get('temp', 0.0):,.0f} K")
                imgui.text(f"  Luminosity:  {sp.get('lum', 0.0):.4f} L\u2609")
                imgui.text(f"  Spectral Cl: {sp.get('class', 'Unknown')}")
                imgui.text(f"  Stage:       {sp.get('stage', 'Unknown')}")

                if sp.get('mode') == 'evolution':
                    imgui.text(f"  Metallicity: {sp.get('metallicity', 0.0):.3f}")
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
            parent_name = cur_bodies_data[parent_idx]['name'] if 0 <= parent_idx < len(cur_bodies_data) else "Barycenter"
            imgui.text_colored(f"Parent: {parent_name}", 0.6, 0.9, 1.0)
            imgui.separator()

            oe_a = float(body_info.get('a', 1.0))
            oe_e = float(body_info.get('e', 0.0))
            oe_inc = float(body_info.get('inc', 0.0))
            oe_Omega = float(body_info.get('Omega', 0.0))
            oe_omega = float(body_info.get('omega', 0.0))
            oe_M = float(body_info.get('M', 0.0))

            imgui.text(f"  Semi-major (a): {format_distance_au(oe_a, threshold_au=thresh_au, precision=6)}")
            imgui.text(f"  Eccentricity (e): {oe_e:.6f}")
            imgui.text(f"  Inclination (i):  {oe_inc:.3f}°")
            imgui.text(f"  Long Asc Node (Ω):{oe_Omega:.3f}°")
            imgui.text(f"  Arg Periapsis (ω):{oe_omega:.3f}°")
            imgui.text(f"  Mean Anomaly (M): {oe_M:.3f}°")

            periapsis = oe_a * (1.0 - oe_e)
            apoapsis = oe_a * (1.0 + oe_e)
            imgui.separator()
            imgui.text(f"  Periapsis: {format_distance_au(periapsis, threshold_au=thresh_au, precision=5)}")
            imgui.text(f"  Apoapsis:  {format_distance_au(apoapsis, threshold_au=thresh_au, precision=5)}")

            # Orbital period
            if parent_idx >= 0:
                parent_m = cur_mass_snap[parent_idx]
                tot_m = parent_m + body_mass
                if tot_m > 0 and oe_a > 0:
                    p_yr = math.sqrt(oe_a**3 / tot_m)
                    if p_yr >= 1.0:
                        imgui.text(f"  Period:    {p_yr:,.2f} years")
                    else:
                        imgui.text(f"  Period:    {p_yr * 365.25:,.2f} days")

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
                    changed_sat = False
                    changed_hue = False
                    changed_bri = False
                    changed_boost = False

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
                                changed_unlit = True
                            imgui.same_line()
                            if imgui.button(f"Bake & Save Textures##{i}"):
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

                    if changed_in or changed_out or changed_col or changed_op or changed_scat or changed_asym or changed_bks or changed_unlit or changed_sat or changed_hue or changed_bri or changed_boost or grad_changed:
                        if changed_in: ring_item['inner_r'] = new_in / 149597870.7
                        if changed_out: ring_item['outer_r'] = new_out / 149597870.7
                        if changed_col: ring_item['raw_color'] = new_col
                        if changed_op: ring_item['opacity'] = new_op
                        if changed_scat: ring_item['scatter'] = new_scat
                        if changed_asym: ring_item['asymmetry'] = new_asym
                        if changed_bks: ring_item['backscatter'] = new_bks
                        if changed_unlit: ring_item['unlit_factor'] = new_unlit
                        if changed_sat: ring_item['saturation'] = new_sat
                        if changed_hue: ring_item['hue_shift'] = new_hue
                        if changed_bri: ring_item['brightness'] = new_bri
                        if changed_boost: ring_item['alpha_boost'] = new_boost

                        shadow_grad = generate_ring_shadow_grad(ring_item['gradient'], tex_sampled=ring_item.get('tex_sampled'))
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
                    shadow_grad = generate_ring_shadow_grad(grad)
                    ring_precomputed.append({
                        'body_idx': insp_idx,
                        'pole': pole_n.astype('f4'),
                        'inner_r': r_in, 'outer_r': r_out, 'opacity': 0.8,
                        'scatter': 2.5, 'asymmetry': 0.8, 'backscatter': -0.3, 'shadow_grad': shadow_grad,
                        'raw_color': color, 'gradient': grad,
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

    imgui.end()
