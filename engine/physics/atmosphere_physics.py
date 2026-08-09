import math
import numpy as np

# Physical Constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
N_A = 6.02214076e23 # Avogadro constant (1/mol)
R = 8.314462618     # Universal gas constant (J/(mol*K))

# Standard conditions
T_STD = 288.15      # K
P_STD = 101325.0    # Pa (1 atm)
N_S = P_STD / (K_B * T_STD) # Number density at standard conditions (~ 2.547e25 m^-3)

# Default rendering wavelengths for RGB (meters)
WAVELENGTHS = np.array([680e-9, 550e-9, 440e-9])

# Gas properties dictionary
# format: "GasName": (RefractiveIndex, KingFactor, MolarMass(kg/mol), AbsorptionCrossSection_RGB(m^2))
# Absorption cross sections are at the rendering wavelengths (680, 550, 440 nm).
# CH4: strong visible/NIR bands (the red/NIR absorption makes Titan orange/dark);
# values approximate visible-band methane absorption (strongest toward red/NIR).
# SO2: strong UV absorber (Venus dark UV markings); cross sections increased to realistic values.
GAS_PROPERTIES = {
    "N2":  (1.000298, 1.034, 0.02801, np.array([0.0, 0.0, 0.0])),
    "O2":  (1.000271, 1.096, 0.03199, np.array([0.0, 0.0, 0.0])),
    "Ar":  (1.000281, 1.000, 0.03995, np.array([0.0, 0.0, 0.0])),
    "CO2": (1.000450, 1.150, 0.04401, np.array([0.0, 0.0, 0.0])),
    "CH4": (1.000444, 1.000, 0.01604, np.array([2.0e-30, 1.2e-30, 6.0e-31])),
    "H2":  (1.000132, 1.020, 0.002016, np.array([0.0, 0.0, 0.0])),
    "He":  (1.000036, 1.000, 0.004002, np.array([0.0, 0.0, 0.0])),
    "O3":  (1.000520, 1.030, 0.04800, np.array([1.036e-25, 3.0e-25, 0.1356e-25])),
    "SO2": (1.000686, 1.075, 0.06406, np.array([0.1e-28, 0.5e-28, 3.0e-28])),
    "Tholin": (1.000600, 1.000, 0.08000, np.array([0.2e-29, 1.2e-29, 3.0e-29])),
}

_atmo_cache = {}

# O2 UV photolysis cross-section (Schumann-Runge / Herzberg continuum,
# ~200-240 nm). Order-of-magnitude value used to locate the Chapman ozone
# peak altitude; the exact band-averaged value varies but ~1e-23 m^2 is the
# commonly adopted figure for the Hartley/Hertzberg region.
_SIGMA_O2_UV = 1.0e-23  # m^2


def _chapman_ozone_profile(x_o2, pressure_pa, temperature_k, scale_height_m):
    """Return (z_peak_m, width_m) of the Chapman ozone layer.

    Ozone is produced by photolysis of O2 in the upper atmosphere and destroyed
    by catalytic cycles, producing a Gaussian-like layer whose peak sits where
    the overhead O2 column has optical depth ~1 to UV-C. For a well-mixed O2
    fraction x_o2 and scale height H, the slant optical depth at altitude z is
    approximately

        tau(z) = x_o2 * sigma_O2 * n0 * H * exp(-z / H)

    where n0 = P/(k_B T) is the surface number density. Setting tau=1 gives

        z_peak = H * ln(x_o2 * sigma_O2 * n0 * H).

    The layer width is of order H (the hydrostatic scale height): a Chapman
    production profile is exp(-z/H) * exp(-tau(z)) whose maximum has a 1/e
    half-width of ~H/sqrt(2) in altitude, so we use sigma_z = H/sqrt(2).

    For bodies with no O2 (x_o2 -> 0) z_peak collapses to a large negative
    number and the caller should treat the ozone contribution as negligible
    (the Gaussian exp(-(z-z_peak)^2 / 2 sigma^2) -> 0 everywhere relevant).
    """
    if x_o2 <= 0.0 or scale_height_m <= 0.0 or temperature_k <= 0.0 or pressure_pa <= 0.0:
        # No O2 or unphysical parameters: return a profile that evaluates to ~0 in the atmosphere.
        # Place the peak far below the surface so rho_O -> 0 for z >= 0.
        return (-1.0e6, max(scale_height_m, 1.0))
    n0 = pressure_pa / (K_B * max(1.0, temperature_k))
    arg = x_o2 * _SIGMA_O2_UV * n0 * scale_height_m
    if arg > 0.0:
        z_peak = scale_height_m * math.log(arg)
    else:
        z_peak = -1.0e6
    width = scale_height_m / math.sqrt(2.0)
    return (z_peak, width)


def _chapman_ozone_enhancement(x_o2, pressure_pa, temperature_k, scale_height_m):
    """Peak-to-surface number-density ratio of the Chapman ozone layer.

    The classic '246.2x' Earth enhancement factor is just the ratio of the
    peak density of a Chapman profile to the well-mixed surface value. For a
    production rate P(z) = J * x_o2 * n(z) with destruction balanced by the
    overhead column, the peak density is

        n_peak = (x_o2 * n0) * (z_peak / H) * exp(1 - z_peak / H)

    so the enhancement over the well-mixed surface value (x_o2 * n0) is
        E = (z_peak / H) * exp(1 - z_peak / H).
    For Earth this reproduces the canonical ~246x when z_peak/H ~ 4.5.
    """
    if x_o2 <= 0.0 or scale_height_m <= 0.0 or temperature_k <= 0.0 or pressure_pa <= 0.0:
        return 0.0
    z_peak, _ = _chapman_ozone_profile(x_o2, pressure_pa, temperature_k, scale_height_m)
    u = z_peak / scale_height_m
    if u <= 0.0:
        return 0.0
    return u * math.exp(1.0 - u)


def compute_atmosphere_properties(pressure_atm, temperature_k, composition, gravity_m_s2):
    """
    Computes scattering and absorption coefficients based on physical properties.
    
    Args:
        pressure_atm: Surface pressure in atmospheres.
        temperature_k: Surface temperature in Kelvin.
        composition: Dictionary of gas fractions, e.g., {"N2": 0.78, "O2": 0.21, "O3": 0.0000003}
        gravity_m_s2: Surface gravity in m/s^2.
        
    Returns:
        dict containing 'beta_rayleigh', 'beta_absorption', 'scale_height_km', 'molar_mass'
    """
    if not composition:
        composition = {"N2": 1.0}
        
    try:
        pressure_atm = max(0.0, float(pressure_atm))
        if math.isnan(pressure_atm) or math.isinf(pressure_atm):
            pressure_atm = 1.0
    except (ValueError, TypeError):
        pressure_atm = 1.0

    try:
        temperature_k = max(1.0, float(temperature_k))
        if math.isnan(temperature_k) or math.isinf(temperature_k):
            temperature_k = 288.15
    except (ValueError, TypeError):
        temperature_k = 288.15

    try:
        gravity_m_s2 = max(1e-4, float(gravity_m_s2))
        if math.isnan(gravity_m_s2) or math.isinf(gravity_m_s2):
            gravity_m_s2 = 9.81
    except (ValueError, TypeError):
        gravity_m_s2 = 9.81

    comp_tuple = tuple(sorted(composition.items()))
    cache_key = (pressure_atm, temperature_k, comp_tuple, gravity_m_s2)
    
    if cache_key in _atmo_cache:
        return _atmo_cache[cache_key]

        
    total_fraction = sum(composition.values())
    if total_fraction <= 0:
        total_fraction = 1.0
        
    normalized_composition = {k: v / total_fraction for k, v in composition.items()}
    
    # Calculate aggregate physical properties
    avg_molar_mass = 0.0
    beta_abs_mixed = np.zeros(3)
    beta_abs_layered = np.zeros(3)
    sigma_rayleigh_mix = np.zeros(3)
    refractivity_mix = 0.0  # mixture (n-1) at STP, weighted by fraction
    x_o2 = 0.0              # O2 mole fraction (drives Chapman ozone layer)
    
    pressure_pa = pressure_atm * P_STD
    number_density = pressure_pa / (K_B * temperature_k)
    
    term1 = 24.0 * (math.pi ** 3) / (WAVELENGTHS ** 4 * N_S ** 2)
    
    # First pass: molar mass and O2 fraction, so we can compute the scale
    # height and the Chapman ozone profile before the absorption loop.
    for gas, fraction in normalized_composition.items():
        if gas not in GAS_PROPERTIES:
            continue
        n, king, m, abs_cross = GAS_PROPERTIES[gas]
        avg_molar_mass += m * fraction
        if gas == "O2":
            x_o2 = fraction
    
    # Scale height H = R * T / (M * g)
    if gravity_m_s2 > 0 and avg_molar_mass > 0:
        scale_height_m = (R * temperature_k) / (avg_molar_mass * gravity_m_s2)
    else:
        scale_height_m = 8000.0 # fallback
    scale_height_km = scale_height_m / 1000.0
    
    # Chapman ozone layer (peak altitude + Gaussian width) from O2, P, T, H.
    ozone_peak_m, ozone_width_m = _chapman_ozone_profile(
        x_o2, pressure_pa, temperature_k, scale_height_m)
    
    for gas, fraction in normalized_composition.items():
        if gas not in GAS_PROPERTIES:
            continue
            
        n, king, m, abs_cross = GAS_PROPERTIES[gas]
        
        # Rayleigh scattering cross section for this specific gas
        term2_i = ((n**2 - 1.0) / (n**2 + 2.0))**2
        sigma_i = term1 * term2_i * king
        
        # Add to mixture cross section (Bodhaine et al. 1999)
        sigma_rayleigh_mix += fraction * sigma_i
        
        # Mixture refractivity (n-1) at STP, weighted by fraction. Linear
        # additivity of (n-1) holds for ideal-gas mixtures (Gladstone-Dale).
        refractivity_mix += fraction * (n - 1.0)
        
        # Absorption coefficient: cross section * number density of this specific gas
        gas_number_density = number_density * fraction
        if gas == "O3":
            # Ozone is concentrated in a stratospheric Chapman layer rather
            # than being well-mixed. The peak density enhancement over the
            # well-mixed surface value is derived from the O2 fraction, P, T
            # and scale height via the Chapman production function (Earth's
            # canonical value is ~246x; this generalises it to any body).
            o3_enhancement = _chapman_ozone_enhancement(
                x_o2, pressure_pa, temperature_k, scale_height_m)
            beta_abs_layered += abs_cross * (gas_number_density * o3_enhancement)
        else:
            beta_abs_mixed += abs_cross * gas_number_density
            
    # Beta Rayleigh = mixture cross section * actual number density
    beta_rayleigh = sigma_rayleigh_mix * number_density
    
    # Ensure no NaN or Inf in output arrays
    beta_rayleigh = np.nan_to_num(beta_rayleigh, nan=0.0, posinf=0.0, neginf=0.0)
    beta_abs_mixed = np.nan_to_num(beta_abs_mixed, nan=0.0, posinf=0.0, neginf=0.0)
    beta_abs_layered = np.nan_to_num(beta_abs_layered, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Top-of-atmosphere altitude derived from the vertical optical depth of
    # the most scattering-dominated rendering channel (shortest wavelength,
    # 440 nm). We solve beta_R(440) * H * exp(-z_top/H) = epsilon for z_top:
    # the altitude at which the vertical optical depth drops below epsilon.
    # This scales correctly for both thin (Mars) and thick (Venus) atmospheres
    # without the previous log(pressure*1e6) heuristic.
    _EPSILON_TAU = 1e-6
    h_m = max(1.0, scale_height_m)
    # beta_rayleigh is the surface (1/m) coefficient per channel; column OD
    # at the surface is beta*H. Use the largest channel for a conservative cap.
    governing_tau = float(np.max(beta_rayleigh) * h_m)
    if governing_tau > _EPSILON_TAU:
        atmo_top_m = max(h_m, h_m * math.log(governing_tau / _EPSILON_TAU))
    else:
        atmo_top_m = h_m * 8.0  # very thin: still keep a few scale heights
    
    res = {
        "beta_rayleigh": beta_rayleigh.astype(np.float32),
        "beta_abs_mixed": beta_abs_mixed.astype(np.float32),
        "beta_abs_layered": beta_abs_layered.astype(np.float32),
        "scale_height_km": float(scale_height_km),
        "molar_mass": float(avg_molar_mass),
        # Surface refractivity (n-1) of the mixture at the actual surface P,T.
        # The per-gas (n-1) values in GAS_PROPERTIES are at STP (number density
        # N_S); scale to the actual surface number density so that refraction
        # (Snell bending) is correct for any surface pressure/temperature.
        "refractivity": float(refractivity_mix * (number_density / N_S)),
        # Chapman-layer ozone profile (peak altitude and Gaussian width, km).
        "ozone_peak_km": float(ozone_peak_m / 1000.0),
        "ozone_width_km": float(ozone_width_m / 1000.0),
        "atmo_height_km": float(atmo_top_m / 1000.0)
    }
    
    _atmo_cache[cache_key] = res
    return res


def compute_mie_coefficients(base_beta=2.0e-6, angstrom_exponent=None):
    """
    Computes Mie scattering coefficients using the Angstrom exponent approximation.
    base_beta is the scattering coefficient at 550nm.
    If angstrom_exponent is None, it is calculated automatically from the aerosol concentration:
    thick aerosol decks are assumed to have large particles (angstrom -> 0),
    while thin background hazes are assumed to have small particles (angstrom -> 1.2).
    """
    if angstrom_exponent is None:
        # Clear skies (base_beta -> 0) yields 1.2, thick aerosol decks (base_beta >= 2.0e-5) yields ~ 0.0
        angstrom_exponent = 1.2 * math.exp(-base_beta / 5.0e-6)
        
    lambda_0 = 550e-9
    # beta_mie = base_beta * (lambda / lambda_0)^(-alpha)
    beta_mie = base_beta * (WAVELENGTHS / lambda_0) ** (-angstrom_exponent)
    return beta_mie.astype(np.float32)


def compute_mie_absorption(beta_mie, single_scattering_albedo):
    """
    Computes the Mie absorption coefficient beta_abs = beta_ext * (1 - w0)
    from the Mie extinction (here `beta_mie` is treated as the extinction at
    each wavelength) and a per-channel single-scattering albedo w0 in [0,1].

    For physically absorbing aerosols (e.g. Titan tholins) w0 should be < 1
    and smaller in the blue/UV than in the red, producing dark, orange/brown
    haze instead of bright conservatively-scattering haze.

    Args:
        beta_mie: (3,) Mie extinction coefficients (1/m) at RGB wavelengths.
        single_scattering_albedo: scalar in [0,1] or (3,) array. Per-channel
            fraction of extinction that is scattering. Default 1.0 (no
            absorption, legacy behaviour).

    Returns:
        (3,) float32 Mie absorption coefficients (1/m).
    """
    beta_mie = np.asarray(beta_mie, dtype=np.float64)
    w0 = np.asarray(single_scattering_albedo, dtype=np.float64)
    if w0.ndim == 0:
        w0 = np.broadcast_to(w0, beta_mie.shape)
    w0 = np.clip(w0, 0.0, 1.0)
    beta_abs = beta_mie * (1.0 - w0)
    return beta_abs.astype(np.float32)


def compute_dynamic_mie_properties(pressure_atm, temperature_k, composition, gravity_m_s2=9.81):
    """
    Dynamically derives aerosol Mie scattering extinction (beta_mie), scale height (h_mie),
    asymmetry parameter (mie_g), single-scattering albedo (mie_albedo), and Angstrom exponent (mie_angstrom)
    based on atmospheric gas composition, surface pressure, and temperature.
    
    This physical heuristic models condensation, photolysis haze, and dust lifting regimes:
      - Titan-like (high Tholin or CH4 at T < 140K): Organic tholin haze (thick, amber/orange).
      - Venus-like (SO2 present): Sulfuric acid haze deck (thick, pale yellow).
      - Mars-like (thin CO2 atmosphere): Airborne mineral dust (thin/medium, ferric oxide blue-absorption).
      - Gas Giant-like (H2/He dominate): Ammonia/methane haze decks (moderate, white/cream).
      - Earth-like (temperate, N2/O2 dominate): Clean-sky background aerosol haze (light, white).
    """
    if not composition:
        composition = {"N2": 1.0}
    
    total_frac = sum(composition.values())
    comp = {k: v / (total_frac if total_frac > 0 else 1.0) for k, v in composition.items()}
    
    p0 = max(0.0001, float(pressure_atm))
    T0 = max(1.0, float(temperature_k))
    
    x_tholin = comp.get("Tholin", 0.0)
    x_ch4 = comp.get("CH4", 0.0)
    x_so2 = comp.get("SO2", 0.0)
    x_co2 = comp.get("CO2", 0.0)
    x_h2 = comp.get("H2", 0.0)
    x_he = comp.get("He", 0.0)
    
    props = compute_atmosphere_properties(p0, T0, comp, gravity_m_s2)
    H_km = props['scale_height_km']
    
    # 1. Titan-like photochemical tholin haze regime
    if x_tholin > 0.005 or (x_ch4 > 0.02 and T0 < 140.0 and p0 > 0.5):
        base_beta = 1.2e-4
        h_mie = max(15.0, min(2.5 * H_km, 60.0))
        mie_albedo = np.array([0.98, 0.75, 0.35], dtype=np.float32)
        mie_g = 0.88
        angstrom = 0.5
        
    # 2. Venus-like sulfuric acid haze regime (SO2 present)
    elif x_so2 > 0.00005:
        base_beta = 2.0e-4
        h_mie = max(20.0, min(2.0 * H_km, 50.0))
        mie_albedo = np.array([0.99, 0.98, 0.80], dtype=np.float32)
        mie_g = 0.85
        angstrom = 0.2
        
    # 3. Mars-like thin CO2 dust atmosphere regime (p0 < 0.05 atm, CO2 > 0.80)
    elif x_co2 > 0.80 and p0 < 0.05:
        base_beta = 4.0e-5
        h_mie = max(5.0, min(1.0 * H_km, 20.0))
        mie_albedo = np.array([0.95, 0.85, 0.65], dtype=np.float32)
        mie_g = 0.76
        angstrom = 0.0
        
    # 4. Gas Giant ammonia/methane upper haze regime (H2/He dominated)
    elif (x_h2 + x_he) > 0.70:
        base_beta = 1.5e-5
        h_mie = max(10.0, min(0.6 * H_km, 40.0))
        mie_albedo = np.array([0.99, 0.97, 0.92], dtype=np.float32)
        mie_g = 0.80
        angstrom = 0.4
        
    # 5. Earth-like temperate terrestrial background haze regime
    else:
        base_beta = 2.0e-6 * math.sqrt(max(0.1, p0))
        h_mie = max(0.8, min(0.2 * H_km, 3.0))
        mie_albedo = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        mie_g = 0.76
        angstrom = None  # auto-computed from base_beta
        
    return {
        "beta_mie": float(base_beta),
        "h_mie": float(h_mie),
        "mie_albedo": mie_albedo,
        "mie_g": float(mie_g),
        "mie_angstrom": angstrom
    }

