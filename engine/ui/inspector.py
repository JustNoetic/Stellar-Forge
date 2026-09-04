import math
import numpy as np
import imgui
from engine.core.constants import DEFAULT_LY_THRESHOLD_AU
from engine.physics.atmosphere_physics import compute_mie_coefficients
from engine.rendering.render_utils import (
    compute_surface_albedo,
    format_distance_au,
    rebuild_ring_render_group
)
from engine.rendering.planetshine import get_cached_atmosphere_properties
from engine.ephemeris.system_manager import SystemManager

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
        if (has_atmo or app.camera["edit_mode"]) and imgui.begin_tab_item("Atmosphere")[0]:
            atmo_item = body_info.get('atmosphere')
            if atmo_item:
                p_bar = atmo_item.get('pressure_bar', atmo_item.get('pressure_atm', 1.0))
                imgui.text(f"Surface Pressure: {p_bar:.3f} bar")
                imgui.text(f"Rayleigh Scale Height: {atmo_item.get('scale_height_km', 8.5):.1f} km")

                comp = atmo_item.get('composition', {})
                if comp:
                    imgui.separator()
                    imgui.text_colored("Gas Composition:", 0.6, 0.9, 1.0)
                    for gas, frac in comp.items():
                        imgui.text(f"  {gas}: {frac * 100.0:.1f}%")
            else:
                imgui.text_colored("No Atmosphere Present", 0.6, 0.6, 0.6)
            imgui.end_tab_item()

        # ── Tab 4: Rings ──
        if (has_rings or app.camera["edit_mode"]) and imgui.begin_tab_item("Rings")[0]:
            rings_list = body_info.get('rings', [])
            if rings_list:
                for ri, r in enumerate(rings_list):
                    imgui.text_colored(f"Ring #{ri+1}:", 0.6, 0.9, 1.0)
                    r_in_km = r.get('inner_r', 1.5) * body_r_km
                    r_out_km = r.get('outer_r', 2.5) * body_r_km
                    imgui.text(f"  Inner: {r.get('inner_r', 1.5):.2f} R ({r_in_km:,.0f} km)")
                    imgui.text(f"  Outer: {r.get('outer_r', 2.5):.2f} R ({r_out_km:,.0f} km)")
                    imgui.text(f"  Opacity: {r.get('opacity', 0.8):.2f}")
            else:
                imgui.text_colored("No Planetary Rings", 0.6, 0.6, 0.6)
            imgui.end_tab_item()

        # ── Tab 5: Cosmetics ──
        if imgui.begin_tab_item("Cosmetics")[0]:
            imgui.text_colored("Cosmetics & Export", 1.0, 0.85, 0.4)
            imgui.separator()

            if not insp_is_cmp:
                if imgui.button("Export Body Cosmetics", width=-1):
                    # Export cosmetic properties for this body
                    import json, os
                    os.makedirs("exports", exist_ok=True)
                    fname = f"exports/{body_name}_cosmetics.json"
                    export_dict = {
                        "name": body_name,
                        "color": list(cur_visual_arr[insp_idx, 0:3]),
                        "rings": body_info.get("rings", []),
                        "atmosphere": body_info.get("atmosphere", {})
                    }
                    with open(fname, 'w') as f:
                        json.dump(export_dict, f, indent=4)
                    print(f"[Cosmetics] Exported {body_name} cosmetics to {fname}")
            imgui.end_tab_item()

        imgui.end_tab_bar()

    imgui.end()
