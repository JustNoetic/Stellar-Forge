"""Shared Hapke photometry and cached disk integrals for surface materials.

Textures retain their existing linear reflectance at i=30, e=0, phase=30 deg.
The Hapke response is normalized there; w describes grains, not texture albedo.
Presets are illustrative starting points, not fitted planetary measurements.
"""
import math
from functools import lru_cache

import numpy as np
from numba import njit

PHASE_SAMPLES = 721
MATERIAL_PRESETS = {
    'lambert': {'model': 'lambert', 'brightness': 1.0},
    'lunar': {'model': 'hapke', 'single_scattering_albedo': 0.23,
              'phase_width': 0.35, 'backscatter_fraction': 0.9,
              'opposition_strength': 1.0, 'opposition_width': 0.06,
              'roughness_deg': 20.0, 'brightness': 1.0,
              'coherent_strength': 0.15, 'coherent_width': 0.006},
    'icy': {'model': 'hapke', 'single_scattering_albedo': 0.85,
            'phase_width': 0.3, 'backscatter_fraction': 0.75,
            'opposition_strength': 0.5, 'opposition_width': 0.035,
            'roughness_deg': 12.0, 'brightness': 1.0,
            'coherent_strength': 0.4, 'coherent_width': 0.008},
}
PARAMETERS = (
    ('single_scattering_albedo', 0.001, 0.999),
    ('phase_width', 0.0, 0.85),
    ('backscatter_fraction', 0.0, 1.0),
    ('opposition_strength', 0.0, 2.0),
    ('opposition_width', 0.003, 0.5),
    ('roughness_deg', 0.0, 45.0),
    ('brightness', 0.0, 2.0),
    ('coherent_strength', 0.0, 1.0),
    ('coherent_width', 0.003, 0.1),
)


def resolve_material(body):
    """Resolve legacy data without mutating it; unknown bodies retain Lambert."""
    raw = body.get('material')
    if isinstance(raw, str):
        raw = {'preset': raw}
    if not isinstance(raw, dict):
        raw = {}
    automatic = {'moon': 'lunar', 'luna': 'lunar', 'europa': 'icy'}
    preset = raw.get('preset', 'auto')
    if preset == 'auto':
        preset = automatic.get(str(body.get('name', '')).lower(), 'lambert')
    defaults = MATERIAL_PRESETS['lunar'] if preset == 'custom' else MATERIAL_PRESETS.get(preset, MATERIAL_PRESETS['lambert'])
    model = raw.get('model', defaults['model'])
    if body.get('type') in ('Star', 'Barycenter'):
        model = 'lambert'
    result = {'preset': preset, 'model': 'hapke' if model == 'hapke' else 'lambert'}
    for name, low, high in PARAMETERS:
        default = defaults.get(name, MATERIAL_PRESETS['lunar'].get(name, 1.0))
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError, OverflowError):
            value = default
        if not math.isfinite(value):
            value = default
        result[name] = min(high, max(low, value))
    return result


def material_key(body):
    m = resolve_material(body)
    return (float(m['model'] == 'hapke'),) + tuple(m[p[0]] for p in PARAMETERS)


@njit(cache=True)
def hapke_raw(mu0, mu, cos_phase, p):
    """Radiance factor I/F; Hapke 1984 roughness, HG2, SHOE and CBOE.

    Roughness equations follow the public-domain USGS ISIS Hapke algorithm:
    https://github.com/DOI-USGS/ISIS3/blob/dev/isis/src/base/objs/Hapke/Hapke.cpp
    H uses the usual isotropic multiple-scattering approximation.
    """
    if mu0 <= 0.0 or mu <= 0.0:
        return 0.0
    mu0, mu = min(1.0, mu0), min(1.0, mu)
    c = min(1.0, max(-1.0, cos_phase))
    b, back = p[2], p[3]
    pf = (1.0 - b*b) * ((1.0-back) / max(1e-9, 1.0+b*b+2.0*b*c)**1.5
                        + back / max(1e-9, 1.0+b*b-2.0*b*c)**1.5)
    tan_half = math.sqrt(max(0.0, (1.0-c)/max(1e-12, 1.0+c)))
    shadow_opposition = p[4] / (1.0 + tan_half/p[5])
    x = tan_half / p[9]
    coherent = p[8] * (1.0 + (-math.expm1(-x)/x if x > 1e-5 else 1.0)) / (2.0*(1.0+x)**2)
    u0, u, shadow = mu0, mu, 1.0
    if p[6] > 0.001:
        t = math.tan(p[6]*math.pi/180.0)
        chi = 1.0 / math.sqrt(1.0 + math.pi*t*t)
        si, se = math.sqrt(max(0.0, 1.0-mu0*mu0)), math.sqrt(max(0.0, 1.0-mu*mu))
        ci, ce = mu0/max(1e-10, si)/t, mu/max(1e-10, se)/t
        e1i, e1e = math.exp(-2.0*ci/math.pi), math.exp(-2.0*ce/math.pi)
        e2i, e2e = math.exp(-ci*ci/math.pi), math.exp(-ce*ce/math.pi)
        u00 = chi*(mu0 + si*t*e2i/(2.0-e1i))
        u10 = chi*(mu + se*t*e2e/(2.0-e1e))
        cp = min(1.0, max(-1.0, (c-mu0*mu)/max(1e-10, si*se))) if si*se > 1e-10 else 1.0
        psi = math.acos(cp)
        half_sin_sq = 0.5*(1.0-cp)
        f = math.exp(-2.0*math.sqrt(max(0.0, (1.0-cp)/max(1e-12, 1.0+cp))))
        if mu0 >= mu:
            den = max(1e-10, 2.0-e1e-psi/math.pi*e1i)
            u0 = chi*(mu0+si*t*(cp*e2e+half_sin_sq*e2i)/den)
            u = chi*(mu+se*t*(e2e-half_sin_sq*e2i)/den)
            q = chi*mu0/max(1e-10, u00)
        else:
            den = max(1e-10, 2.0-e1i-psi/math.pi*e1e)
            u0 = chi*(mu0+si*t*(e2i-half_sin_sq*e2e)/den)
            u = chi*(mu+se*t*(cp*e2i+half_sin_sq*e2e)/den)
            q = chi*mu/max(1e-10, u10)
        shadow = u*mu0*chi/max(1e-10, u10*u00*(1.0-f+f*q))
    gamma = math.sqrt(1.0-p[1])
    h0 = (1.0+2.0*u0)/(1.0+2.0*u0*gamma)
    h = (1.0+2.0*u)/(1.0+2.0*u*gamma)
    return max(0.0, 0.25*p[1]*u0/max(1e-10, u0+u) * ((1.0+shadow_opposition)*pf+h0*h-1.0) * shadow * (1.0+coherent))


@lru_cache(maxsize=128)
def packed_material(key):
    row = np.zeros(12, dtype=np.float32)
    row[:10] = key
    row[10] = 1.0
    if row[0] > 0.5:
        row[10] = math.cos(math.pi/6) / max(1e-8, hapke_raw(math.cos(math.pi/6), 1.0, math.cos(math.pi/6), row))
    row.setflags(write=False)
    return row


@njit(cache=True)
def surface_response(mu0, mu, cos_phase, p):
    if p[0] < 0.5:
        return max(0.0, mu0) * p[7]
    return hapke_raw(mu0, mu, cos_phase, p) * p[10] * p[7]


@njit(cache=True)
def integrate_disk(row, phases, z_nodes, z_weights, azimuth_count):
    """Integrate only the visible illuminated lune, resolving thin crescents."""
    result = np.zeros(len(phases), dtype=np.float32)
    for k in range(len(phases)):
        c, s = math.cos(phases[k]), math.sin(phases[k])
        total = 0.0
        z_max = max(0.0,s) if c < 0.0 else 1.0
        for j in range(len(z_nodes)):
            z = z_nodes[j]*z_max
            radial = math.sqrt(1.0-z*z)
            half_arc = math.acos(min(1.0,max(-1.0,-z*c/max(1e-12,radial*s)))) if s > 1e-12 else math.pi
            for a in range(azimuth_count):
                mu0 = z*c + radial*s*math.cos(half_arc*(a+0.5)/azimuth_count)
                total += surface_response(mu0, z, c, row)*z*z_weights[j]*z_max*half_arc
        result[k] = total * 2.0 / (math.pi*azimuth_count)
    return result


@lru_cache(maxsize=128)
def phase_table(key):
    row = packed_material(key)
    phases = np.linspace(0.0, math.pi, PHASE_SAMPLES)
    if row[0] < 0.5:
        values = (2.0/3.0 * row[7] * (np.sin(phases)+(math.pi-phases)*np.cos(phases))/math.pi).astype('f4')
    else:
        nodes, weights = np.polynomial.legendre.leggauss(24)
        values = integrate_disk(row, phases, (nodes+1.0)*0.5, weights*0.5, 48)
    values[-1] = 0.0
    values.setflags(write=False)
    return values


@njit(cache=True)
def lookup_phase(table, angle):
    pos = min(len(table)-1.0, max(0.0, angle/math.pi*(len(table)-1)))
    lo = min(len(table)-2, int(pos))
    return table[lo]*(1.0-(pos-lo)) + table[lo+1]*(pos-lo)


@njit(cache=True)
def disk_response(table, cos_phase, angular_radius=0.0):
    """Five-point stellar disk average, consistent with the surface shader."""
    c = min(1.0, max(-1.0, cos_phase))
    alpha = math.acos(c)
    if angular_radius < 1e-5:
        return lookup_phase(table, alpha)
    beta = min(1.0, angular_radius)*math.sqrt(0.625)
    s = math.sqrt(max(0.0, 1.0-c*c))
    cb, sb = math.cos(beta), math.sin(beta)
    result = lookup_phase(table, alpha)
    result += lookup_phase(table, math.acos(min(1.0, max(-1.0, c*cb+s*sb))))
    result += lookup_phase(table, math.acos(min(1.0, max(-1.0, c*cb-s*sb))))
    result += 2.0*lookup_phase(table, math.acos(min(1.0, max(-1.0, c*cb))))
    return result*0.2


def material_albedos(body, reflectance):
    key = material_key(body)
    values = phase_table(key)
    rgb = np.asarray(reflectance)*float(values[0])
    geometric = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
    phases = np.linspace(0.0, math.pi, PHASE_SAMPLES)
    integrand = values*np.sin(phases)
    integral = float(np.sum((integrand[1:]+integrand[:-1])*np.diff(phases)))
    bond = float(np.dot(reflectance, [0.2126, 0.7152, 0.0722]))*integral
    return geometric, min(1.0, max(0.0, bond)), integral/max(1e-10, float(values[0])), rgb


class SurfaceMaterialCache:
    """Body-indexed SSBO plus shared phase tables; update only on edits/reorder."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.key = None
        self.material_buffer = ctx.buffer(reserve=48)
        self.phase_buffer = ctx.buffer(reserve=PHASE_SAMPLES*4)

    def update(self, bodies):
        keys = tuple(material_key(body) for body in bodies)
        if keys != self.key:
            unique = list(dict.fromkeys(keys))
            offsets = {key: i*PHASE_SAMPLES for i, key in enumerate(unique)}
            self.rows = np.zeros((max(1, len(keys)), 12), dtype='f4')
            self.tables = np.zeros((max(1, len(keys)), PHASE_SAMPLES), dtype='f4')
            atlas = np.concatenate([phase_table(key) for key in unique]) if unique else np.zeros(PHASE_SAMPLES, dtype='f4')
            for i, key in enumerate(keys):
                self.rows[i] = packed_material(key)
                self.rows[i, 11] = offsets[key]
                self.tables[i] = phase_table(key)
            for buf, data in ((self.material_buffer, self.rows), (self.phase_buffer, atlas)):
                buf.orphan(data.nbytes)
                buf.write(data.tobytes())
            self.key = keys
        self.material_buffer.bind_to_storage_buffer(11)
        self.phase_buffer.bind_to_storage_buffer(12)
        return self.rows, self.tables

    def release(self):
        self.material_buffer.release()
        self.phase_buffer.release()
