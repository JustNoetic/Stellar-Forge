"""Numeric port of engine/glsl/common/refraction.glsl (current state, incl.
the s_min gate removal + forward-periapsis occlusion fixes) for validating
the observer-inside closed-form refraction branch."""
import numpy as np

AU_KM = 149597870.7

u_refract_radius = 6371.0
u_refract_max_bend = 0.0
u_refract_scale_height = 8.5
u_refract_oblateness = 0.0
u_refract_pole = np.array([0.0, 1.0, 0.0])


def refract_solid_radius(P_dir):
    return u_refract_radius


def Efun(x):
    return np.sign(x) * np.sqrt(max(0.0, 1.0 - np.exp(-1.239 * x * x)))


def _refraction_setup(C, V, d):
    s_min = -np.dot(C, V)
    P_min = C + s_min * V
    r_min = np.linalg.norm(P_min)
    local_r = refract_solid_radius(P_min / r_min if r_min > 1e-6 else np.array([0.0, 1.0, 0.0]))
    if r_min > local_r + u_refract_scale_height * 15.0:
        return -1.0, s_min, 0.0, 0.0
    r_min_clamped = max(r_min, local_r)
    delta_rmin = min(0.15, u_refract_max_bend * np.exp(-(r_min_clamped - local_r) / max(1e-4, u_refract_scale_height)))
    sigma = np.sqrt(max(1e-4, r_min_clamped * u_refract_scale_height))
    return r_min, s_min, delta_rmin, sigma


def compute_refraction_angle(C, V, d):
    r_min, s_min, delta_rmin, sigma = _refraction_setup(C, V, d)
    if r_min < 0.0:
        return 0.0
    if d < 0.1 * sigma:
        k0 = (delta_rmin / (sigma * 2.506628)) * np.exp(-(s_min * s_min) / (2.0 * sigma * sigma))
        return 0.5 * k0 * d
    x_d = (d - s_min) / (1.4142135 * sigma)
    x_0 = s_min / (1.4142135 * sigma)
    E_d, E_0 = Efun(x_d), Efun(x_0)
    G_d = np.exp(-np.clip(x_d * x_d, 0, 50))
    G_0 = np.exp(-np.clip(x_0 * x_0, 0, 50))
    a = delta_rmin * (0.5 * (E_d + E_0) * (1.0 - s_min / d) + (sigma / (d * 2.506628)) * (G_d - G_0))
    return max(0.0, a)


def compute_refraction_angle_v2(C, V, d):
    """PROPOSED: adds the observer-inside closed form for s_min < 0."""
    s_min = -np.dot(C, V)
    if s_min < 0.0:
        Rc = np.linalg.norm(C)
        if Rc < 1e-6:
            return 0.0
        up = C / Rc
        sin_e = np.dot(V, up)
        h = Rc - refract_solid_radius(up)
        if h > u_refract_scale_height * 15.0:
            return 0.0
        delta_cam = u_refract_max_bend * np.exp(-max(h, 0.0) / max(1e-4, u_refract_scale_height))
        if delta_cam <= 1e-9:
            return 0.0
        k = Rc / (2.0 * max(1e-4, u_refract_scale_height))
        x = np.sqrt(k) * min(max(sin_e, 0.0), 1.0)
        R = 0.5 * delta_cam * _erfcx(x)
        sigma_cam = np.sqrt(max(1e-4, Rc * u_refract_scale_height))
        x_d = d / (1.4142135 * sigma_cam)
        return max(0.0, R * Efun(x_d))
    return compute_refraction_angle(C, V, d)


def _erfcx(x):
    """exp(x^2)*erfc(x), x>=0 — A&S 7.1.26 poly equals exp(x^2)*(1-erf(x))."""
    t = 1.0 / (1.0 + 0.3275911 * x)
    return max(0.0, t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
               + t * (-1.453152027 + t * 1.061405429)))))


def solve(C, V, d, alpha_fn):
    s_min = -np.dot(C, V)
    if s_min >= d:
        return V, False, 0.0
    u = C - V * np.dot(C, V)
    ul = np.linalg.norm(u)
    if ul <= 1e-5:
        return V, True, 0.0
    u /= ul
    P_min = C + s_min * V
    r_min = np.linalg.norm(P_min)
    if r_min > u_refract_radius + u_refract_scale_height * 15.0:
        return V, False, 0.0
    a0 = alpha_fn(C, V, d)
    if a0 <= 1e-7:
        return V, False, 0.0
    th = a0 / (1.0 + (max(s_min, 0.0) / u_refract_scale_height) * a0)
    for _ in range(3):
        Vt = V * np.cos(th) + u * np.sin(th)
        a = alpha_fn(C, Vt, d)
        st = -np.dot(C, Vt)
        th = max(0.0, th - (th - a) / (1.0 + (max(st, 0.0) / u_refract_scale_height) * a))
    Va = V * np.cos(th) + u * np.sin(th)
    sa = -np.dot(C, Va)
    r_app = np.linalg.norm(C + max(sa, 0.0) * Va)
    return Va, r_app < u_refract_radius, th


def apply(star_pos, cam_km, d_km, alpha_fn):
    tv = star_pos - cam_km
    d_km = np.linalg.norm(tv)
    V = tv / d_km
    Va, occ, th = solve(cam_km, V, d_km, alpha_fn)
    if occ:
        return star_pos, True, th
    return cam_km + Va * d_km, False, th
