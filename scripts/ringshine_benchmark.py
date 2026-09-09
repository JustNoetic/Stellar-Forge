"""
Ringshine Radiative Transfer Benchmark:
Compares the Current Shader Implementation and the Proposed Quadrature Implementation
against a Brute-Force Monte Carlo / High-Density Grid Ground Truth (over 2 million points per surface evaluation).
"""
import os
import math
import numpy as np
from PIL import Image
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1. Physics & Phase Functions (Identical to ring.frag)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def henyey_greenstein(g, mu):
    denom = max(1e-6, 1.0 + g * g - 2.0 * g * mu)
    return (1.0 - g * g) / (4.0 * math.pi * denom * math.sqrt(denom))

@njit(fastmath=True)
def get_ring_phase_functions(mu, alpha, asym, back_asym):
    dust_to_chunks = min(max((alpha - 0.1) / 0.5, 0.0), 1.0)
    w_fwd = 0.95 * (1.0 - dust_to_chunks) + 0.5 * dust_to_chunks
    w_back = 0.05 * (1.0 - dust_to_chunks) + 0.5 * dust_to_chunks
    pf_fwd = henyey_greenstein(asym, mu)
    pf_back = henyey_greenstein(back_asym, mu)
    return w_fwd * pf_fwd + w_back * pf_back

@njit(fastmath=True)
def opposition_surge(cos_phase, column_density):
    phase_angle = math.acos(min(max(cos_phase, -1.0), 1.0))
    density_scale = min(max(column_density / 1.5, 0.0), 1.0)
    shoe = 1.0 + (0.8 * density_scale) / (1.0 + phase_angle / 0.07)
    cboe = 1.0 + (0.3 * density_scale) * math.exp(-phase_angle / 0.006)
    return shoe * cboe

@njit(fastmath=True)
def analytic_multiple_scattering(mu_v, mu_0, tau, on_lit_side):
    w0 = 0.92
    gamma = math.sqrt(max(1e-4, 1.0 - w0))
    Hv = (1.0 + 2.0 * mu_v) / (1.0 + 2.0 * mu_v * gamma)
    H0 = (1.0 + 2.0 * mu_0) / (1.0 + 2.0 * mu_0 * gamma)
    path_term = 1.0 - math.exp(-tau * (1.0 / max(1e-4, mu_v) + 1.0 / max(1e-4, mu_0)))
    mu_ratio = mu_0 / max(1e-4, mu_v + mu_0)
    inv_4pi = 1.0 / (4.0 * math.pi)
    if on_lit_side:
        return max(0.0, w0 * mu_ratio * (Hv * H0 - 1.0) * path_term * inv_4pi)
    else:
        return max(0.0, w0 * mu_ratio * (Hv * H0) * math.exp(-gamma * tau) * path_term * inv_4pi)


# -----------------------------------------------------------------------------
# 2. Brute-Force Radiative Transfer Ground Truth (2+ Million 2D Disk Elements)
# -----------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def compute_ground_truth_irradiance(
    phi_lat, phi_center, sun_elev,
    inner_r, outer_r, opacity,
    asym, back_asym, unlit_factor,
    tex_radii, tex_rgb, tex_alpha,
    n_r_steps=1024, n_alpha_steps=2048
):
    sin_lat = math.sin(phi_lat)
    cos_lat = math.cos(phi_lat)
    P_x = cos_lat * math.cos(phi_center)
    P_y = cos_lat * math.sin(phi_center)
    P_z = sin_lat
    
    sin_sun = math.sin(sun_elev)
    cos_sun = math.cos(sun_elev)
    L_x = -cos_sun
    L_y = 0.0
    L_z = sin_sun
    mu_0 = max(abs(sin_sun), 1e-4)

    total_irrad_r = 0.0
    total_irrad_g = 0.0
    total_irrad_b = 0.0

    dr = (outer_r - inner_r) / n_r_steps
    d_alpha = (2.0 * math.pi) / n_alpha_steps

    # Accumulate over 2D annular disk
    for i in prange(n_r_steps):
        r = inner_r + (i + 0.5) * dr
        frac = min(max((r - inner_r) / (outer_r - inner_r), 0.0), 1.0)
        
        # Sample 1D radial texture profile
        t_idx = frac * (len(tex_radii) - 1)
        i0 = int(math.floor(t_idx))
        i1 = min(i0 + 1, len(tex_radii) - 1)
        w1 = t_idx - i0
        w0 = 1.0 - w1
        
        raw_a = w0 * tex_alpha[i0] + w1 * tex_alpha[i1]
        a_phys = min(max(raw_a * opacity, 0.0), 0.999)
        if a_phys < 1e-5:
            continue
        tau_phys = -math.log(max(1e-4, 1.0 - a_phys))
        
        tex_r = w0 * tex_rgb[i0, 0] + w1 * tex_rgb[i1, 0]
        tex_g = w0 * tex_rgb[i0, 1] + w1 * tex_rgb[i1, 1]
        tex_b = w0 * tex_rgb[i0, 2] + w1 * tex_rgb[i1, 2]

        local_r = 0.0
        local_g = 0.0
        local_b = 0.0

        for j in range(n_alpha_steps):
            alpha = (j + 0.5) * d_alpha
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            
            # Position of ring element
            R_x = r * cos_a
            R_y = r * sin_a
            
            # 1. Shadow test: does the host planet sphere (radius 1.0) block Sun -> Ring Element?
            t_sun = -(R_x * L_x + R_y * L_y)
            if t_sun > 0.0:
                # Perpendicular distance squared from planet center to ray
                perp_sq = (r * r) - (t_sun * t_sun)
                if perp_sq <= 1.0:
                    continue # In umbral shadow of the planet

            # 2. Ray from surface point to ring element
            vx = R_x - P_x
            vy = R_y - P_y
            vz = 0.0 - P_z
            d2 = vx * vx + vy * vy + vz * vz
            d = math.sqrt(d2)
            if d < 1e-6:
                continue

            # Direction from surface point towards ring
            lx = vx / d
            ly = vy / d
            lz = vz / d

            # 3. Horizon occlusion: is the ring element above the local planetary surface horizon?
            cos_theta_surf = P_x * lx + P_y * ly + P_z * lz
            if cos_theta_surf <= 0.0:
                continue # Below local planetary horizon

            # 4. Vertical slant angles
            mu_v = max(abs(P_z) / d, 1e-4) # |cos(slant)| relative to ring normal (0, 0, 1)
            tau_v = tau_phys / mu_v
            tau_0 = tau_phys / mu_0

            # 5. 3D Scattering phase angle: cos(theta) between Sun L and ray pointing from ring to surface
            # Ray from ring to surface is -l
            cos_theta_phase = -(L_x * lx + L_y * ly + L_z * lz)

            # 6. Radiative transfer (lit side vs unlit side)
            on_lit_side = (sun_elev * P_z) >= 0.0
            pf = get_ring_phase_functions(cos_theta_phase, a_phys, asym, back_asym)
            dust_to_chunks = min(max((a_phys - 0.1) / 0.5, 0.0), 1.0)

            if on_lit_side:
                scat_single = (tau_v / (tau_v + tau_0)) * (1.0 - math.exp(-tau_v - tau_0))
                ms = analytic_multiple_scattering(mu_v, mu_0, tau_phys, True) * dust_to_chunks
                surge = opposition_surge(-cos_theta_phase, tau_phys)
                radiance_factor = scat_single * pf * surge + ms
            else:
                denom = tau_0 - tau_v
                if abs(denom) > 1e-6:
                    scat_unlit = (math.exp(-tau_v) - math.exp(-tau_0)) * tau_v / denom
                else:
                    scat_unlit = tau_v * math.exp(-tau_v)
                ms = analytic_multiple_scattering(mu_v, mu_0, tau_phys, False) * dust_to_chunks
                radiance_factor = scat_unlit * pf * unlit_factor + ms

            # Solid angle & differential irradiance: (cos_surf * mu_v / d^2) * r * dr * dalpha
            weight = (cos_theta_surf * mu_v / d2) * (r * dr * d_alpha)
            local_r += tex_r * radiance_factor * weight
            local_g += tex_g * radiance_factor * weight
            local_b += tex_b * radiance_factor * weight

        total_irrad_r += local_r
        total_irrad_g += local_g
        total_irrad_b += local_b

    # Convert radiance to Lambertian irradiance equivalent (matches engine convention)
    factor = math.pi * mu_0 * 0.318309886
    return np.array([total_irrad_r * factor, total_irrad_g * factor, total_irrad_b * factor], dtype=np.float64)


# -----------------------------------------------------------------------------
# 3. Precomputed LUT & CDF Builders (Exact matches to app.py)
# -----------------------------------------------------------------------------

def build_luts_numpy():
    res_x, res_y = 256, 256
    sin_lats = np.linspace(0.001, 0.999, res_x, dtype=np.float32)
    radii = np.linspace(1.001, 5.0, res_y, dtype=np.float32)

    sin_lat_grid = sin_lats[None, :]
    cos_lat_grid = np.sqrt(np.maximum(0.0, 1.0 - sin_lat_grid**2))
    r_grid = radii[:, None]

    num_alpha = 360
    alpha = np.linspace(0.0, 2.0 * np.pi, num_alpha, endpoint=False, dtype=np.float32)[:, None, None]

    cos_alpha = np.cos(alpha)
    d2 = r_grid**2 + 1.0 - 2.0 * r_grid * cos_lat_grid * cos_alpha
    d = np.sqrt(np.maximum(d2, 1e-6))

    ndotl = np.maximum(0.0, (r_grid * cos_lat_grid * cos_alpha - 1.0) / d)
    ring_mu = sin_lat_grid / d

    d_alpha = (2.0 * np.pi) / num_alpha
    diff_irradiance = (ndotl * ring_mu / np.maximum(d2, 1e-6)) * r_grid * d_alpha
    lut_data = np.sum(diff_irradiance, axis=0, dtype=np.float32)

    # CDF
    res_cdf_lat, res_cdf_r, res_cdf_theta = 64, 128, 128
    cdf_sin_lats = np.linspace(0.001, 0.999, res_cdf_lat, dtype=np.float32)[:, None, None]
    cdf_cos_lats = np.sqrt(np.maximum(0.0, 1.0 - cdf_sin_lats**2))
    cdf_radii = np.linspace(1.001, 5.0, res_cdf_r, dtype=np.float32)[None, :, None]
    cdf_thetas = np.linspace(0.0, np.pi, res_cdf_theta, dtype=np.float32)[None, None, :]

    cdf_cos_alpha = np.cos(cdf_thetas)
    cdf_d2 = cdf_radii**2 + 1.0 - 2.0 * cdf_radii * cdf_cos_lats * cdf_cos_alpha
    cdf_d = np.sqrt(np.maximum(cdf_d2, 1e-6))

    cdf_ndotl = np.maximum(0.0, (cdf_radii * cdf_cos_lats * cdf_cos_alpha - 1.0) / cdf_d)
    cdf_ring_mu = cdf_sin_lats / cdf_d
    cdf_d_alpha = np.pi / max(1, res_cdf_theta - 1)
    cdf_diff_irrad = (cdf_ndotl * cdf_ring_mu / np.maximum(cdf_d2, 1e-6)) * cdf_radii * cdf_d_alpha

    trapz_step = 0.5 * (cdf_diff_irrad[:, :, :-1] + cdf_diff_irrad[:, :, 1:])
    cdf_cum_irrad = np.zeros_like(cdf_diff_irrad)
    cdf_cum_irrad[:, :, 1:] = np.cumsum(trapz_step, axis=2)

    cdf_totals = cdf_cum_irrad[:, :, -1:]
    cdf_normalized = np.where(cdf_totals > 1e-12, cdf_cum_irrad / cdf_totals, 1.0).astype(np.float32)

    return lut_data, cdf_normalized, sin_lats, radii


def eval_cdf_lookup(cdf_3d, angle, v_tex, sin_lat):
    TWO_PI = 2.0 * math.pi
    a_mod = angle - TWO_PI * math.floor((angle + math.pi) / TWO_PI)
    k = math.floor((angle + math.pi) / TWO_PI)
    u = min(max(abs(a_mod) / math.pi, 0.0), 1.0)

    # 3D lookup dimensions: (lat:64, r:128, theta:128)
    i_lat = min(max(int(sin_lat * 63.0), 0), 63)
    i_r = min(max(int(v_tex * 127.0), 0), 127)
    i_th = min(max(int(u * 127.0), 0), 127)
    base_cdf = cdf_3d[i_lat, i_r, i_th]
    signed_cdf = -base_cdf if a_mod < 0.0 else base_cdf
    return 2.0 * k + signed_cdf


def sample_lut(lut_2d, sin_lat, v_tex):
    i_lat = min(max(int(sin_lat * 255.0), 0), 255)
    i_r = min(max(int(v_tex * 255.0), 0), 255)
    return lut_2d[i_r, i_lat]


# -----------------------------------------------------------------------------
# 4. Current Shader Model (reproduces ringshine_map.frag faithfully)
# -----------------------------------------------------------------------------

def evaluate_current_shader(
    phi_lat, phi_center, sun_elev,
    inner_r, outer_r, opacity,
    unlit_factor, tex_radii, tex_rgb, tex_alpha,
    lut_2d, cdf_3d, band_count=10
):
    sin_lat = min(max(abs(math.sin(phi_lat)), 0.001), 0.999)
    frag_elevation = math.sin(phi_lat)
    sin_sun_elev = min(max(abs(math.sin(sun_elev)), 1e-4), 1.0)
    cos_sun_elev = math.sqrt(max(0.0, 1.0 - sin_sun_elev * sin_sun_elev))
    sun_elevation = math.sin(sun_elev)

    same_hemisphere = sun_elevation * frag_elevation
    same_hemi_t = min(max((same_hemisphere - (-0.02)) / 0.04, 0.0), 1.0)

    total_irradiance = np.zeros(3, dtype=np.float64)
    log_r_ratio = math.log(max(1.001, outer_r / max(1e-5, inner_r)))

    for m in range(band_count):
        u0 = m / band_count
        u1 = (m + 1) / band_count
        u_mid = 0.5 * (u0 + u1)

        r_m0 = inner_r * math.exp(u0 * log_r_ratio)
        r_m1 = inner_r * math.exp(u1 * log_r_ratio)
        r_mid = inner_r * math.exp(u_mid * log_r_ratio)
        dr = (r_m1 - r_m0) / 1.0
        frac_mid = min(max((r_mid - inner_r) / max(1e-5, outer_r - inner_r), 0.0), 1.0)

        # Single point sample
        idx = int(frac_mid * (len(tex_radii) - 1))
        alpha = tex_alpha[idx]
        if alpha < 1e-4:
            continue
        rgb = tex_rgb[idx]

        alpha_phys = min(max(alpha * opacity, 0.0), 0.999)
        tau_phys = -math.log(max(1e-4, 1.0 - alpha_phys))

        # Current shader hardcoded parameters
        cosViewRayVertical = max(sin_lat, 1e-4)
        cosLightRayVertical = max(sin_sun_elev, 1e-4)
        viewDensity = tau_phys / cosViewRayVertical
        lightDensity = tau_phys / cosLightRayVertical

        # Asymmetry and backscatter default to 0.0 because they were never uploaded!
        layer_asym = 0.0
        layer_backasym = 0.0

        scatteredLight_sunlit = viewDensity / (viewDensity + lightDensity) * (1.0 - math.exp(-viewDensity - lightDensity))
        pf_sunlit = get_ring_phase_functions(0.70, alpha_phys, layer_asym, layer_backasym)

        # Multiple scattering
        w0 = 0.92
        gamma = math.sqrt(max(1e-4, 1.0 - w0))
        Hv = (1.0 + 2.0 * cosViewRayVertical) / (1.0 + 2.0 * cosViewRayVertical * gamma)
        H0 = (1.0 + 2.0 * cosLightRayVertical) / (1.0 + 2.0 * cosLightRayVertical * gamma)
        path_term = 1.0 - math.exp(-tau_phys * (1.0 / cosViewRayVertical + 1.0 / cosLightRayVertical))
        mu_ratio = cosLightRayVertical / max(1e-4, cosViewRayVertical + cosLightRayVertical)
        dust_to_chunks = min(max((alpha_phys - 0.1) / 0.5, 0.0), 1.0)
        ms_sunlit = max(0.0, w0 * mu_ratio * (Hv * H0 - 1.0) * path_term * dust_to_chunks) / (4.0 * math.pi)

        band_color_sunlit = rgb * (scatteredLight_sunlit * pf_sunlit + ms_sunlit)

        denom = lightDensity - viewDensity
        if abs(denom) > 1e-6:
            scatteredLight_unlit = (math.exp(-viewDensity) - math.exp(-lightDensity)) * viewDensity / denom
        else:
            scatteredLight_unlit = viewDensity * math.exp(-viewDensity)
        pf_unlit = get_ring_phase_functions(-0.70, alpha_phys, layer_asym, layer_backasym)
        band_color_unlit = rgb * (scatteredLight_unlit * pf_unlit * unlit_factor)

        band_color = (1.0 - same_hemi_t) * band_color_unlit + same_hemi_t * band_color_sunlit

        norm_r = r_mid / 1.0
        delta_alpha_shadow = 0.0
        if norm_r <= 1.0 / sin_sun_elev:
            arg = math.sqrt(max(0.0, 1.0 - 1.0 / (norm_r * norm_r))) / max(1e-4, cos_sun_elev)
            delta_alpha_shadow = math.acos(min(max(arg, 0.0), 1.0))

        psi1 = -delta_alpha_shadow - phi_center
        psi2 = +delta_alpha_shadow - phi_center
        v_tex = min(max((norm_r - 1.0) / 4.0, 0.0), 1.0)

        cdf1 = eval_cdf_lookup(cdf_3d, psi1, v_tex, sin_lat)
        cdf2 = eval_cdf_lookup(cdf_3d, psi2, v_tex, sin_lat)
        shadow_fraction = min(max(0.5 * (cdf2 - cdf1), 0.0), 1.0)
        band_illum = max(0.0, 1.0 - shadow_fraction)

        kernel_val = sample_lut(lut_2d, sin_lat, v_tex)
        total_irradiance += band_color * kernel_val * dr * band_illum

    # Multiply by PI and 1/PI scaling as in sphere.frag
    factor = math.pi * sin_sun_elev * 0.318309886
    return total_irradiance * factor


# -----------------------------------------------------------------------------
# 5. Proposed Quadrature Shader Model (Area-Weighted Quadrature + Dynamic Scattering)
# -----------------------------------------------------------------------------

def evaluate_proposed_shader(
    phi_lat, phi_center, sun_elev,
    inner_r, outer_r, opacity,
    asym, back_asym, unlit_factor,
    tex_radii, tex_rgb, tex_alpha,
    lut_2d, cdf_3d, band_count=10
):
    sin_lat = min(max(abs(math.sin(phi_lat)), 0.001), 0.999)
    cos_lat = math.sqrt(max(0.0, 1.0 - sin_lat * sin_lat))
    frag_elevation = math.sin(phi_lat)
    sin_sun_elev = min(max(abs(math.sin(sun_elev)), 1e-4), 1.0)
    cos_sun_elev = math.sqrt(max(0.0, 1.0 - sin_sun_elev * sin_sun_elev))
    sun_elevation = math.sin(sun_elev)

    same_hemisphere = sun_elevation * frag_elevation
    same_hemi_t = min(max((same_hemisphere - (-0.02)) / 0.04, 0.0), 1.0)

    total_irradiance = np.zeros(3, dtype=np.float64)
    log_r_ratio = math.log(max(1.001, outer_r / max(1e-5, inner_r)))

    # 4-point Gauss-Legendre quadrature offsets & weights
    quad_t = np.array([0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558])
    quad_w = np.array([0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226])

    for m in range(band_count):
        u0 = m / band_count
        u1 = (m + 1) / band_count
        u_mid = 0.5 * (u0 + u1)

        r_m0 = inner_r * math.exp(u0 * log_r_ratio)
        r_m1 = inner_r * math.exp(u1 * log_r_ratio)
        r_mid = inner_r * math.exp(u_mid * log_r_ratio)
        dr = (r_m1 - r_m0) / 1.0

        # --- AREA-WEIGHTED BAND QUADRATURE ---
        sum_area = 0.0
        sum_alpha = 0.0
        sum_color = np.zeros(3)

        for s in range(4):
            t_sub = u0 + quad_t[s] * (u1 - u0)
            r_s = inner_r * math.exp(t_sub * log_r_ratio)
            frac_s = min(max((r_s - inner_r) / max(1e-5, outer_r - inner_r), 0.0), 1.0)
            
            idx_s = int(frac_s * (len(tex_radii) - 1))
            a_s = tex_alpha[idx_s]
            c_s = tex_rgb[idx_s]

            # Annular area weight proportional to r
            w_area = r_s * quad_w[s]
            sum_area += w_area
            sum_alpha += a_s * w_area
            sum_color += c_s * a_s * w_area

        if sum_area < 1e-9 or sum_alpha < 1e-9:
            continue

        band_alpha = sum_alpha / sum_area
        band_rgb = sum_color / sum_alpha

        alpha_phys = min(max(band_alpha * opacity, 0.0), 0.999)
        tau_phys = -math.log(max(1e-4, 1.0 - alpha_phys))

        # --- EXACT GEOMETRIC SCATTERING ANGLE ---
        # Distance d from surface point P(phi, phi_center) to dominant ring element at r_mid
        d2 = r_mid * r_mid + 1.0 - 2.0 * r_mid * cos_lat * math.cos(0.0) # dominant meridian alpha=0
        d = math.sqrt(max(1e-6, d2))

        # True slant angle for slab vertical emission
        cosViewRayVertical = max(sin_lat / d, 0.001)
        cosLightRayVertical = max(sin_sun_elev, 0.001)

        viewDensity = tau_phys / cosViewRayVertical
        lightDensity = tau_phys / cosLightRayVertical

        # Dominant phase angle cos(theta) between Sun L and ray pointing from ring to surface
        # L = (-cos_sun, 0, sin_sun)
        # Vector from ring element (r_mid * cos(phi_center), r_mid * sin(phi_center), 0) to P:
        cos_theta_phase = ((r_mid - cos_lat) * cos_sun_elev * math.cos(phi_center) + sin_sun_elev * sin_lat) / d
        cos_theta_phase = min(max(cos_theta_phase, -1.0), 1.0)

        # Dynamic phase function with true asymmetry & backscatter!
        pf = get_ring_phase_functions(cos_theta_phase, alpha_phys, asym, back_asym)
        dust_to_chunks = min(max((alpha_phys - 0.1) / 0.5, 0.0), 1.0)

        # Sunlit side
        scat_sunlit = viewDensity / (viewDensity + lightDensity) * (1.0 - math.exp(-viewDensity - lightDensity))
        ms_sunlit = analytic_multiple_scattering(cosViewRayVertical, cosLightRayVertical, tau_phys, True) * dust_to_chunks
        surge = opposition_surge(-cos_theta_phase, tau_phys)
        band_color_sunlit = band_rgb * (scat_sunlit * pf * surge + ms_sunlit)

        # Unlit side
        denom = lightDensity - viewDensity
        if abs(denom) > 1e-6:
            scat_unlit = (math.exp(-viewDensity) - math.exp(-lightDensity)) * viewDensity / denom
        else:
            scat_unlit = viewDensity * math.exp(-viewDensity)
        ms_unlit = analytic_multiple_scattering(cosViewRayVertical, cosLightRayVertical, tau_phys, False) * dust_to_chunks
        band_color_unlit = band_rgb * (scat_unlit * pf * unlit_factor + ms_unlit)

        band_color = (1.0 - same_hemi_t) * band_color_unlit + same_hemi_t * band_color_sunlit

        norm_r = r_mid / 1.0
        delta_alpha_shadow = 0.0
        if norm_r <= 1.0 / sin_sun_elev:
            arg = math.sqrt(max(0.0, 1.0 - 1.0 / (norm_r * norm_r))) / max(1e-4, cos_sun_elev)
            delta_alpha_shadow = math.acos(min(max(arg, 0.0), 1.0))

        psi1 = -delta_alpha_shadow - phi_center
        psi2 = +delta_alpha_shadow - phi_center
        v_tex = min(max((norm_r - 1.0) / 4.0, 0.0), 1.0)

        cdf1 = eval_cdf_lookup(cdf_3d, psi1, v_tex, sin_lat)
        cdf2 = eval_cdf_lookup(cdf_3d, psi2, v_tex, sin_lat)
        shadow_fraction = min(max(0.5 * (cdf2 - cdf1), 0.0), 1.0)
        band_illum = max(0.0, 1.0 - shadow_fraction)

        kernel_val = sample_lut(lut_2d, sin_lat, v_tex)
        total_irradiance += band_color * kernel_val * dr * band_illum

    factor = math.pi * sin_sun_elev * 0.318309886
    return total_irradiance * factor


# -----------------------------------------------------------------------------
# Main Benchmark Execution
# -----------------------------------------------------------------------------

def main():
    print("================================================================================")
    print("  RINGSHINE ACCURACY BENCHMARK: BRUTE FORCE RADIATIVE TRANSFER vs SHADERS")
    print("================================================================================")

    # 1. Load texture
    tex_path = "textures/Solar System/Saturn/Rings.png"
    img = Image.open(tex_path).convert('RGBA')
    img_ds = img.resize((4096, 1), Image.Resampling.LANCZOS)
    arr = np.array(img_ds, dtype=np.float64) / 255.0
    arr = arr.reshape(4096, 4)

    tex_rgb = np.power(arr[:, 0:3], 2.2) # decode to linear space
    tex_alpha = arr[:, 3]
    tex_radii = np.linspace(0.0, 1.0, 4096)

    # Saturn parameters
    inner_r = 1.144886
    outer_r = 2.266241
    opacity = 1.0
    asym = 0.436
    back_asym = -0.65711
    unlit_factor = 1.0

    print("Building analytical LUTs (256x256 view factor + 64x128x128 CDF)...")
    lut_2d, cdf_3d, _, _ = build_luts_numpy()
    print("LUTs ready.")

    # Warmup Numba ground truth JIT
    print("Warming up Numba parallel ground truth kernel (JIT compilation)...")
    _ = compute_ground_truth_irradiance(
        0.5, 0.0, 0.2, inner_r, outer_r, opacity, asym, back_asym, unlit_factor,
        tex_radii, tex_rgb, tex_alpha, n_r_steps=32, n_alpha_steps=64
    )
    print("JIT compilation done.\n")

    test_scenarios = [
        # (Scenario Name, sun_elev_deg, list of (lat_deg, phi_center_deg))
        ("Scenario A: Sunlit Side, High Illumination (Sun Elev = +20°)", 20.0, [
            (10.0, 0.0,   "Equatorial Midnight Meridian (phi=0°)"),
            (10.0, 60.0,  "Equatorial Twilight Meridian (phi=60°)"),
            (10.0, 120.0, "Equatorial Daytime Meridian (phi=120°)"),
            (35.0, 0.0,   "Mid-Latitude Midnight (phi=0°)"),
            (35.0, 60.0,  "Mid-Latitude Twilight (phi=60°)"),
            (60.0, 0.0,   "Sub-Polar Midnight (phi=0°)"),
        ]),
        ("Scenario B: Grazing Sunlit Side (Sun Elev = +5°)", 5.0, [
            (10.0, 0.0,   "Equatorial Midnight (phi=0°)"),
            (25.0, 45.0,  "Low-Latitude Twilight (phi=45°)"),
            (45.0, 0.0,   "Mid-Latitude Midnight (phi=0°)"),
        ]),
        ("Scenario C: Unlit Transmission Side (Sun Elev = -15°, Observer Lat = +25°)", -15.0, [
            (25.0, 0.0,   "Unlit Northern Hemisphere Midnight (phi=0°)"),
            (25.0, 60.0,  "Unlit Northern Hemisphere Twilight (phi=60°)"),
            (45.0, 0.0,   "Unlit Northern Hemisphere Mid-Lat (phi=0°)"),
        ]),
    ]

    all_gt = []
    all_curr = []
    all_prop_10 = []
    all_prop_64 = []

    for scen_name, sun_deg, test_points in test_scenarios:
        sun_rad = math.radians(sun_deg)
        print("--------------------------------------------------------------------------------")
        print(f"  {scen_name}")
        print("--------------------------------------------------------------------------------")
        print(f"{'Surface Point':<42} | {'Ground Truth (L=RGB)':<22} | {'Current (10 bands)':<20} | {'Proposed (10 bands)':<20} | {'Proposed (64 bands)':<20}")
        print("-" * 135)

        for lat_deg, phi_deg, label in test_points:
            lat_rad = math.radians(lat_deg)
            phi_rad = math.radians(phi_deg)

            # Ground Truth: 1024 radial x 2048 azimuthal = 2,097,152 exact rays!
            gt = compute_ground_truth_irradiance(
                lat_rad, phi_rad, sun_rad,
                inner_r, outer_r, opacity,
                asym, back_asym, unlit_factor,
                tex_radii, tex_rgb, tex_alpha,
                n_r_steps=1024, n_alpha_steps=2048
            )

            curr = evaluate_current_shader(
                lat_rad, phi_rad, sun_rad,
                inner_r, outer_r, opacity,
                unlit_factor, tex_radii, tex_rgb, tex_alpha,
                lut_2d, cdf_3d, band_count=10
            )

            prop_10 = evaluate_proposed_shader(
                lat_rad, phi_rad, sun_rad,
                inner_r, outer_r, opacity,
                asym, back_asym, unlit_factor,
                tex_radii, tex_rgb, tex_alpha,
                lut_2d, cdf_3d, band_count=10
            )

            prop_64 = evaluate_proposed_shader(
                lat_rad, phi_rad, sun_rad,
                inner_r, outer_r, opacity,
                asym, back_asym, unlit_factor,
                tex_radii, tex_rgb, tex_alpha,
                lut_2d, cdf_3d, band_count=64
            )

            all_gt.append(gt)
            all_curr.append(curr)
            all_prop_10.append(prop_10)
            all_prop_64.append(prop_64)

            gt_str = f"({gt[0]:.4f}, {gt[1]:.4f}, {gt[2]:.4f})"
            curr_str = f"({curr[0]:.4f}, {curr[1]:.4f}, {curr[2]:.4f})"
            prop10_str = f"({prop_10[0]:.4f}, {prop_10[1]:.4f}, {prop_10[2]:.4f})"
            prop64_str = f"({prop_64[0]:.4f}, {prop_64[1]:.4f}, {prop_64[2]:.4f})"

            print(f"{label:<42} | {gt_str:<22} | {curr_str:<20} | {prop10_str:<20} | {prop64_str:<20}")
        print()

    # Calculate overall error statistics
    gt_arr = np.array(all_gt)
    curr_arr = np.array(all_curr)
    prop10_arr = np.array(all_prop_10)
    prop64_arr = np.array(all_prop_64)

    # Photopic luminance error L = 0.2126 R + 0.7152 G + 0.0722 B
    weights = np.array([0.2126, 0.7152, 0.0722])
    gt_lum = np.dot(gt_arr, weights)
    curr_lum = np.dot(curr_arr, weights)
    p10_lum = np.dot(prop10_arr, weights)
    p64_lum = np.dot(prop64_arr, weights)

    curr_rmse = np.sqrt(np.mean((curr_lum - gt_lum)**2))
    p10_rmse = np.sqrt(np.mean((p10_lum - gt_lum)**2))
    p64_rmse = np.sqrt(np.mean((p64_lum - gt_lum)**2))

    curr_mae = np.mean(np.abs(curr_lum - gt_lum))
    p10_mae = np.mean(np.abs(p10_lum - gt_lum))
    p64_mae = np.mean(np.abs(p64_lum - gt_lum))

    rel_mask = gt_lum > 1e-4
    curr_rel = np.mean(np.abs(curr_lum[rel_mask] - gt_lum[rel_mask]) / gt_lum[rel_mask]) * 100.0
    p10_rel = np.mean(np.abs(p10_lum[rel_mask] - gt_lum[rel_mask]) / gt_lum[rel_mask]) * 100.0
    p64_rel = np.mean(np.abs(p64_lum[rel_mask] - gt_lum[rel_mask]) / gt_lum[rel_mask]) * 100.0

    print("================================================================================")
    print("  ACCURACY SUMMARY vs GROUND TRUTH MONTE CARLO (2,097,152 rays/pt)")
    print("================================================================================")
    print(f"  Current Shader (10 bands, 1-pt midpoint, hardcoded phase):")
    print(f"    - Mean Absolute Error (MAE):    {curr_mae:.6f}")
    print(f"    - Root Mean Square Error (RMSE): {curr_rmse:.6f}")
    print(f"    - Mean Relative Error:          {curr_rel:.2f}%\n")

    print(f"  Proposed Shader (10 bands, 4-pt area-weighted quadrature, dynamic phase):")
    print(f"    - Mean Absolute Error (MAE):    {p10_mae:.6f}  ({(1.0 - p10_mae/curr_mae)*100:.1f}% error reduction)")
    print(f"    - Root Mean Square Error (RMSE): {p10_rmse:.6f}  ({(1.0 - p10_rmse/curr_rmse)*100:.1f}% error reduction)")
    print(f"    - Mean Relative Error:          {p10_rel:.2f}%\n")

    print(f"  Proposed Shader (64 bands, 4-pt area-weighted quadrature, dynamic phase):")
    print(f"    - Mean Absolute Error (MAE):    {p64_mae:.6f}  ({(1.0 - p64_mae/curr_mae)*100:.1f}% error reduction)")
    print(f"    - Root Mean Square Error (RMSE): {p64_rmse:.6f}  ({(1.0 - p64_rmse/curr_rmse)*100:.1f}% error reduction)")
    print(f"    - Mean Relative Error:          {p64_rel:.2f}%\n")
    print("================================================================================")

if __name__ == '__main__':
    main()
