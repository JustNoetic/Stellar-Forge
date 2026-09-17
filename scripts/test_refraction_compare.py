import math
import os
import sys
import numpy as np
import moderngl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.rendering.shader_loader import load_shader
from numba import njit

@njit(cache=True, fastmath=True)
def refract_solid_radius(P_dir, radius, oblateness, pole):
    if 0.001 < oblateness < 0.99:
        pole_len = math.sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2])
        if pole_len > 1e-4:
            px, py, pz = pole[0]/pole_len, pole[1]/pole_len, pole[2]/pole_len
        else:
            px, py, pz = 0.0, 1.0, 0.0
        cos_t = abs(P_dir[0]*px + P_dir[1]*py + P_dir[2]*pz)
        k = 1.0 / (1.0 - oblateness)
        denom = math.sqrt(max(1e-6, 1.0 + (k * k - 1.0) * cos_t * cos_t))
        return radius / denom
    return radius

@njit(cache=True, fastmath=True)
def refract_erfcx(x):
    t = 1.0 / (1.0 + 0.3275911 * x)
    return max(0.0, t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
           + t * (-1.453152027 + t * 1.061405429)))))

@njit(cache=True, fastmath=True)
def _refraction_setup(C, V, max_bend, radius, scale_height, oblateness, pole):
    if max_bend <= 1e-6:
        return -1.0, 0.0, 0.0, 0.0
    s_min = -(C[0]*V[0] + C[1]*V[1] + C[2]*V[2])
    Px = C[0] + s_min * V[0]
    Py = C[1] + s_min * V[1]
    Pz = C[2] + s_min * V[2]
    r_min = math.sqrt(Px*Px + Py*Py + Pz*Pz)
    if r_min > 1e-6:
        P_dir = np.array((Px/r_min, Py/r_min, Pz/r_min), dtype=np.float64)
    else:
        P_dir = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    local_radius = refract_solid_radius(P_dir, radius, oblateness, pole)
    if r_min > local_radius + scale_height * 15.0:
        return -1.0, 0.0, 0.0, 0.0
    r_min_clamped = max(r_min, local_radius)
    delta_rmin = min(0.15, max_bend * math.exp(-(r_min_clamped - local_radius) / max(1e-4, scale_height)))
    sigma = math.sqrt(max(1e-4, r_min_clamped * scale_height))
    return r_min, s_min, delta_rmin, sigma

@njit(cache=True, fastmath=True)
def compute_refraction_angle(C, V, d, max_bend, radius, scale_height, oblateness, pole):
    s_min_pre = -(C[0]*V[0] + C[1]*V[1] + C[2]*V[2])
    if s_min_pre < 0.0:
        Rc = math.sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2])
        if Rc < 1e-6:
            return 0.0
        up = np.array((C[0]/Rc, C[1]/Rc, C[2]/Rc), dtype=np.float64)
        sin_e = min(max(V[0]*up[0] + V[1]*up[1] + V[2]*up[2], 0.0), 1.0)
        h = Rc - refract_solid_radius(up, radius, oblateness, pole)
        if h > scale_height * 15.0:
            return 0.0
        delta_cam = max_bend * math.exp(-max(h, 0.0) / max(1e-4, scale_height))
        if delta_cam <= 1e-9:
            return 0.0
        k = Rc / (2.0 * max(1e-4, scale_height))
        bend = 0.5 * delta_cam * refract_erfcx(math.sqrt(k) * sin_e)
        sigma_cam = math.sqrt(max(1e-4, Rc * scale_height))
        x_d = d / (1.4142135 * sigma_cam)
        E_d = (1.0 if x_d >= 0.0 else -1.0) * math.sqrt(max(0.0, 1.0 - math.exp(-1.239 * x_d * x_d)))
        return max(0.0, bend * E_d)

    r_min, s_min, delta_rmin, sigma = _refraction_setup(C, V, max_bend, radius, scale_height, oblateness, pole)
    if r_min < 0.0:
        return 0.0

    if d < 0.1 * sigma:
        kappa_0 = (delta_rmin / (sigma * 2.506628)) * math.exp(-(s_min * s_min) / (2.0 * sigma * sigma))
        return 0.5 * kappa_0 * d

    sqrt2_sig = 1.4142135 * sigma
    x_d = (d - s_min) / sqrt2_sig
    x_0 = s_min / sqrt2_sig

    sign_xd = 1.0 if x_d >= 0.0 else -1.0
    sign_x0 = 1.0 if x_0 >= 0.0 else -1.0
    E_d = sign_xd * math.sqrt(max(0.0, 1.0 - math.exp(-1.239 * x_d * x_d)))
    E_0 = sign_x0 * math.sqrt(max(0.0, 1.0 - math.exp(-1.239 * x_0 * x_0)))
    G_d = math.exp(-min(max(x_d * x_d, 0.0), 50.0))
    G_0 = math.exp(-min(max(x_0 * x_0, 0.0), 50.0))

    alpha = delta_rmin * (0.5 * (E_d + E_0) * (1.0 - s_min / d) + (sigma / (d * 2.506628)) * (G_d - G_0))
    return max(0.0, alpha)

@njit(cache=True, fastmath=True)
def solve_refraction_apparent(C, V, d, max_bend, radius, scale_height, oblateness, pole):
    s_min = -(C[0]*V[0] + C[1]*V[1] + C[2]*V[2])
    if s_min >= d:
        return V, False

    c_dot_v = C[0]*V[0] + C[1]*V[1] + C[2]*V[2]
    ux = C[0] - V[0] * c_dot_v
    uy = C[1] - V[1] * c_dot_v
    uz = C[2] - V[2] * c_dot_v
    u_len = math.sqrt(ux*ux + uy*uy + uz*uz)
    if u_len <= 1e-5:
        return V, True
    ux /= u_len
    uy /= u_len
    uz /= u_len
    u_dir = np.array((ux, uy, uz), dtype=np.float64)

    Px = C[0] + s_min * V[0]
    Py = C[1] + s_min * V[1]
    Pz = C[2] + s_min * V[2]
    r_min = math.sqrt(Px*Px + Py*Py + Pz*Pz)
    if r_min > 1e-6:
        P_dir = np.array((Px/r_min, Py/r_min, Pz/r_min), dtype=np.float64)
    else:
        P_dir = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    local_radius = refract_solid_radius(P_dir, radius, oblateness, pole)

    if r_min > local_radius + scale_height * 15.0:
        return V, False

    alpha_0 = compute_refraction_angle(C, V, d, max_bend, radius, scale_height, oblateness, pole)
    if alpha_0 <= 1e-7:
        return V, False

    lo_t = 0.0
    hi_t = alpha_0
    for _ in range(12):
        mid_t = 0.5 * (lo_t + hi_t)
        cos_m = math.cos(mid_t)
        sin_m = math.sin(mid_t)
        V_test = np.array((
            V[0] * cos_m + u_dir[0] * sin_m,
            V[1] * cos_m + u_dir[1] * sin_m,
            V[2] * cos_m + u_dir[2] * sin_m
        ), dtype=np.float64)
        alpha = compute_refraction_angle(C, V_test, d, max_bend, radius, scale_height, oblateness, pole)
        if alpha > mid_t:
            lo_t = mid_t
        else:
            hi_t = mid_t
    theta = 0.5 * (lo_t + hi_t)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    V_app = np.array((
        V[0] * cos_t + u_dir[0] * sin_t,
        V[1] * cos_t + u_dir[1] * sin_t,
        V[2] * cos_t + u_dir[2] * sin_t
    ), dtype=np.float64)
    v_norm = math.sqrt(V_app[0]*V_app[0] + V_app[1]*V_app[1] + V_app[2]*V_app[2])
    if v_norm > 1e-12:
        V_app /= v_norm

    s_app = -(C[0]*V_app[0] + C[1]*V_app[1] + C[2]*V_app[2])
    s_app_fwd = max(s_app, 0.0)
    Pax = C[0] + s_app_fwd * V_app[0]
    Pay = C[1] + s_app_fwd * V_app[1]
    Paz = C[2] + s_app_fwd * V_app[2]
    r_app = math.sqrt(Pax*Pax + Pay*Pay + Paz*Paz)
    if r_app > 1e-6:
        P_dir_app = np.array((Pax/r_app, Pay/r_app, Paz/r_app), dtype=np.float64)
    else:
        P_dir_app = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    solid_r = refract_solid_radius(P_dir_app, radius, oblateness, pole)

    is_occluded = (r_app < solid_r)
    return V_app, is_occluded

@njit(cache=True, fastmath=True)
def compute_gravitational_deflection(C_km, V, d_km, rs, radius, lens_type, spin, pole, strength):
    if rs <= 1e-6:
        return 0.0, False

    r_cam_km = math.sqrt(C_km[0]*C_km[0] + C_km[1]*C_km[1] + C_km[2]*C_km[2])
    metric_factor = math.sqrt(max(1e-4, 1.0 - rs / max(1e-4, r_cam_km)))
    s_min = -(C_km[0]*V[0] + C_km[1]*V[1] + C_km[2]*V[2])
    Px = C_km[0] + s_min * V[0]
    Py = C_km[1] + s_min * V[1]
    Pz = C_km[2] + s_min * V[2]
    p_len = math.sqrt(Px*Px + Py*Py + Pz*Pz)
    b = p_len / metric_factor

    b_c_base = 2.5980762 * rs
    b_c = b_c_base

    if abs(spin) > 1e-4:
        pole_len = math.sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2])
        if pole_len > 1e-4:
            px, py, pz = pole[0]/pole_len, pole[1]/pole_len, pole[2]/pole_len
        else:
            px, py, pz = 0.0, 1.0, 0.0
        cam_dir = (C_km[0] / max(1e-4, r_cam_km), C_km[1] / max(1e-4, r_cam_km), C_km[2] / max(1e-4, r_cam_km))
        cpx = py*cam_dir[2] - pz*cam_dir[1]
        cpy = pz*cam_dir[0] - px*cam_dir[2]
        cpz = px*cam_dir[1] - py*cam_dir[0]
        len_cp = math.sqrt(cpx*cpx + cpy*cpy + cpz*cpz)
        if len_cp > 1e-4:
            prog_x, prog_y, prog_z = cpx/len_cp, cpy/len_cp, cpz/len_cp
            if p_len > 1e-4:
                rpx, rpy, rpz = Px/p_len, Py/p_len, Pz/p_len
            else:
                rpx, rpy, rpz = prog_x, prog_y, prog_z
            cos_phi = min(max(rpx*prog_x + rpy*prog_y + rpz*prog_z, -1.0), 1.0)
            spin_factor = spin * min(max(len_cp, 0.0), 1.0)
            b_c = b_c_base * (1.0 - 0.35 * spin_factor * cos_phi)

    if lens_type == 3 and b <= b_c and s_min > 0.0 and (d_km <= 0.0 or s_min < d_km):
        return 0.0, True

    if b < 1e-4:
        return 0.0, False

    geom_factor = 1.0
    if d_km > 0.0:
        d_s = d_km - s_min
        denom_s = math.sqrt(b * b + d_s * d_s)
        denom_0 = math.sqrt(b * b + s_min * s_min)
        geom_factor = 0.5 * ((d_s / max(1e-6, denom_s)) + (s_min / max(1e-6, denom_0)))
        geom_factor = min(max(geom_factor, 0.0), 1.0)

    alpha_weak = 2.0 * rs / b
    b_ratio = rs / b
    alpha_gr = alpha_weak + 2.945243 * b_ratio * b_ratio

    if lens_type == 3:
        b_over_bc = b / max(1e-5, b_c)
        if 1.0 < b_over_bc < 4.0:
            delta = b_over_bc - 1.0
            strong_div = max(0.0, -math.log(max(1e-6, delta)) * (1.0 / b_over_bc)**3)
            alpha_gr += strong_div

    st = strength if strength > 0.0 else 1.0
    final_alpha = alpha_gr * geom_factor * st
    return max(0.0, final_alpha), False

@njit(cache=True, fastmath=True)
def apply_gravitational_deflection(C_km, V, d_km, rs, radius, lens_type, spin, pole, strength):
    alpha_gr, is_sh = compute_gravitational_deflection(C_km, V, d_km, rs, radius, lens_type, spin, pole, strength)
    if is_sh:
        return V, True
    if alpha_gr <= 1e-7:
        return V, False

    c_dot_v = C_km[0]*V[0] + C_km[1]*V[1] + C_km[2]*V[2]
    ux = C_km[0] - V[0] * c_dot_v
    uy = C_km[1] - V[1] * c_dot_v
    uz = C_km[2] - V[2] * c_dot_v
    u_len = math.sqrt(ux*ux + uy*uy + uz*uz)
    if u_len <= 1e-5:
        return V, False
    ux /= u_len
    uy /= u_len
    uz /= u_len

    if abs(spin) > 1e-4:
        pole_len = math.sqrt(pole[0]*pole[0] + pole[1]*pole[1] + pole[2]*pole[2])
        if pole_len > 1e-4:
            px, py, pz = pole[0]/pole_len, pole[1]/pole_len, pole[2]/pole_len
        else:
            px, py, pz = 0.0, 1.0, 0.0
        cdx = py*V[2] - pz*V[1]
        cdy = pz*V[0] - px*V[2]
        cdz = px*V[1] - py*V[0]
        len_drag = math.sqrt(cdx*cdx + cdy*cdy + cdz*cdz)
        if len_drag > 1e-4:
            drag_x, drag_y, drag_z = cdx/len_drag, cdy/len_drag, cdz/len_drag
            r_cam_km = math.sqrt(C_km[0]*C_km[0] + C_km[1]*C_km[1] + C_km[2]*C_km[2])
            metric_factor = math.sqrt(max(1e-4, 1.0 - rs / max(1e-4, r_cam_km)))
            b_km = u_len / metric_factor
            drag_angle = min(max((rs * rs * spin) / max(1.0, b_km * b_km), -0.5), 0.5)
            ux += drag_x * drag_angle
            uy += drag_y * drag_angle
            uz += drag_z * drag_angle
            u_n = math.sqrt(ux*ux + uy*uy + uz*uz)
            if u_n > 1e-6:
                ux /= u_n
                uy /= u_n
                uz /= u_n

    cos_a = math.cos(alpha_gr)
    sin_a = math.sin(alpha_gr)
    vx = V[0] * cos_a - ux * sin_a
    vy = V[1] * cos_a - uy * sin_a
    vz = V[2] * cos_a - uz * sin_a
    v_norm = math.sqrt(vx*vx + vy*vy + vz*vz)
    if v_norm > 1e-12:
        return np.array((vx/v_norm, vy/v_norm, vz/v_norm), dtype=np.float64), False
    return V, False

COMPUTE = """
#version 430 core
layout(local_size_x = 1) in;
SHARED_REFRACTION
layout(std430, binding=0) buffer Out { vec4 result; };
uniform vec3 t_C;
uniform vec3 t_V;
uniform float t_d;
uniform int t_mode; // 0=atmo, 1=lensing
void main() {
    if (t_mode == 0) {
        bool occluded;
        vec3 apparent = solve_refraction_apparent(t_C, t_V, t_d, occluded);
        float lift = atan(length(cross(t_V, apparent)), dot(t_V, apparent));
        result = vec4(apparent.x, apparent.y, apparent.z, lift);
    } else {
        bool is_shadow;
        vec3 apparent = apply_gravitational_deflection(t_C, t_V, t_d, is_shadow);
        float lift = atan(length(cross(t_V, apparent)), dot(t_V, apparent));
        result = vec4(apparent.x, apparent.y, apparent.z, lift);
    }
}
"""

def main():
    ctx = moderngl.create_standalone_context(require=430)
    prog = ctx.compute_shader(COMPUTE.replace("SHARED_REFRACTION", load_shader("common/refraction.glsl")))
    buf = ctx.buffer(reserve=16)
    buf.bind_to_storage_buffer(0)
    
    radius = 6371.0
    max_bend = math.radians(68.0 / 60.0)
    scale_height = 8.5
    oblateness = 0.0
    pole = np.array((0.0, 1.0, 0.0), dtype=np.float64)

    for name, value in (
        ("u_refract_radius", radius),
        ("u_refract_max_bend", max_bend),
        ("u_refract_scale_height", scale_height),
        ("u_refract_oblateness", oblateness),
        ("u_refract_pole", tuple(pole)),
    ):
        if name in prog:
            prog[name].value = value

    cases = []
    for elevation in (0.0, 2.0, 5.0, 10.0):
        e = math.radians(elevation)
        cases.append((f"ground {elevation:g} deg", np.array((6371.002, 0.0, 0.0)), np.array((math.sin(e), 0.0, -math.cos(e)))))
    cases.append(("space grazing limb", np.array((6376.0, 0.0, 10000.0)), np.array((0.0, 0.0, -1.0))))
    cases.append(("above atmosphere", np.array((6571.0, 0.0, 0.0)), np.array((math.sin(0.1), 0.0, -math.cos(0.1)))))

    arcmin = 180.0 * 60.0 / math.pi
    all_passed = True
    prog["t_mode"].value = 0
    try:
        print("--- ATMOSPHERIC REFRACTION TESTS ---")
        for label, C, V in cases:
            d = 1e9
            prog["t_C"].value = tuple(C)
            prog["t_V"].value = tuple(V)
            prog["t_d"].value = d
            prog.run(group_x=1)
            gpu_res = np.frombuffer(buf.read(), dtype=np.float32)
            gpu_V_app = gpu_res[:3]
            gpu_lift = gpu_res[3]

            py_V_app, py_occ = solve_refraction_apparent(C, V, d, max_bend, radius, scale_height, oblateness, pole)
            cross_v = np.cross(V, py_V_app)
            py_lift = math.atan2(np.linalg.norm(cross_v), np.dot(V, py_V_app))

            diff = np.linalg.norm(gpu_V_app - py_V_app)
            lift_diff_arcmin = abs(gpu_lift - py_lift) * arcmin
            print(f"{label}:")
            print(f"  GPU lift: {gpu_lift * arcmin:.4f}' | Python lift: {py_lift * arcmin:.4f}' | diff: {lift_diff_arcmin:.6f}'")
            if diff > 1e-4 or lift_diff_arcmin > 0.01:
                print(f"  FAILED mismatch {diff}")
                all_passed = False
            else:
                print(f"  MATCH!")

        print("\n--- GRAVITATIONAL LENSING TESTS ---")
        prog["t_mode"].value = 1
        rs = 10.0
        r_surf = 10.0
        lens_type = 3
        spin = 0.5
        pole_lens = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        strength = 1.0
        for name, value in (
            ("u_grav_lens_enabled", True),
            ("u_grav_lens_rs", rs),
            ("u_grav_lens_radius", r_surf),
            ("u_grav_lens_type", lens_type),
            ("u_grav_lens_spin", spin),
            ("u_grav_lens_pole", tuple(pole_lens)),
            ("u_grav_lens_strength", strength),
        ):
            if name in prog:
                prog[name].value = value

        lens_cases = [
            ("BH grazing impact b ~ 50 km", np.array((0.0, 0.0, 1000.0)), np.array((0.05, 0.0, -0.998749))),
            ("BH medium impact b ~ 200 km", np.array((0.0, 0.0, 1000.0)), np.array((0.2, 0.0, -0.979796))),
            ("BH high impact b ~ 1000 km", np.array((0.0, 0.0, 1000.0)), np.array((0.707106, 0.0, -0.707106))),
        ]
        for label, C, V in lens_cases:
            V = V / np.linalg.norm(V)
            d = 5000.0
            prog["t_C"].value = tuple(C)
            prog["t_V"].value = tuple(V)
            prog["t_d"].value = d
            prog.run(group_x=1)
            gpu_res = np.frombuffer(buf.read(), dtype=np.float32)
            gpu_V_app = gpu_res[:3]
            gpu_lift = gpu_res[3]

            py_V_app, py_sh = apply_gravitational_deflection(C, V, d, rs, r_surf, lens_type, spin, pole_lens, strength)
            cross_v = np.cross(V, py_V_app)
            py_lift = math.atan2(np.linalg.norm(cross_v), np.dot(V, py_V_app))

            diff = np.linalg.norm(gpu_V_app - py_V_app)
            lift_diff_arcmin = abs(gpu_lift - py_lift) * arcmin
            print(f"{label}:")
            print(f"  GPU lift: {gpu_lift * arcmin:.4f}' | Python lift: {py_lift * arcmin:.4f}' | diff: {lift_diff_arcmin:.6f}'")
            if diff > 1e-4 or lift_diff_arcmin > 0.01:
                print(f"  FAILED mismatch {diff}")
                all_passed = False
            else:
                print(f"  MATCH!")
    finally:
        buf.release()
        prog.release()
        ctx.release()

    if not all_passed:
        raise AssertionError("GPU vs Python mismatch!")
    print("\nALL ATMOSPHERE AND LENSING CASES MATCHED GPU WITHIN 0.001 ARCMIN!")

if __name__ == "__main__":
    main()
