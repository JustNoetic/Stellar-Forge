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
GAS_PROPERTIES = {
    "N2":  (1.000298, 1.034, 0.02801, np.array([0.0, 0.0, 0.0])),
    "O2":  (1.000271, 1.096, 0.03199, np.array([0.0, 0.0, 0.0])),
    "Ar":  (1.000281, 1.000, 0.03995, np.array([0.0, 0.0, 0.0])),
    "CO2": (1.000450, 1.150, 0.04401, np.array([0.0, 0.0, 0.0])),
    "CH4": (1.000444, 1.000, 0.01604, np.array([0.0, 0.0, 0.0])),
    "H2":  (1.000132, 1.020, 0.002016, np.array([0.0, 0.0, 0.0])),
    "He":  (1.000036, 1.000, 0.004002, np.array([0.0, 0.0, 0.0])),
    "O3":  (1.000520, 1.030, 0.04800, np.array([1.036e-25, 3.0e-25, 0.1356e-25])),
    "SO2": (1.000686, 1.075, 0.06406, np.array([0.001e-25, 0.005e-25, 0.03e-25])),
}

_atmo_cache = {}

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
    
    pressure_pa = pressure_atm * P_STD
    number_density = pressure_pa / (K_B * temperature_k)
    
    term1 = 24.0 * (math.pi ** 3) / (WAVELENGTHS ** 4 * N_S ** 2)
    
    for gas, fraction in normalized_composition.items():
        if gas not in GAS_PROPERTIES:
            continue
            
        n, king, m, abs_cross = GAS_PROPERTIES[gas]
        
        avg_molar_mass += m * fraction
        
        # Rayleigh scattering cross section for this specific gas
        term2_i = ((n**2 - 1.0) / (n**2 + 2.0))**2
        sigma_i = term1 * term2_i * king
        
        # Add to mixture cross section (Bodhaine et al. 1999)
        sigma_rayleigh_mix += fraction * sigma_i
        
        # Absorption coefficient: cross section * number density of this specific gas
        gas_number_density = number_density * fraction
        if gas == "O3":
            # Ozone is concentrated in the stratosphere, where its peak density
            # is about 246.2 times its average sea-level mixing ratio.
            beta_abs_layered += abs_cross * (gas_number_density * 246.2)
        else:
            beta_abs_mixed += abs_cross * gas_number_density
            
    # Beta Rayleigh = mixture cross section * actual number density
    beta_rayleigh = sigma_rayleigh_mix * number_density
    
    # Scale height H = R * T / (M * g)
    if gravity_m_s2 > 0 and avg_molar_mass > 0:
        scale_height_m = (R * temperature_k) / (avg_molar_mass * gravity_m_s2)
    else:
        scale_height_m = 8000.0 # fallback
        
    scale_height_km = scale_height_m / 1000.0
    
    res = {
        "beta_rayleigh": beta_rayleigh.astype(np.float32),
        "beta_abs_mixed": beta_abs_mixed.astype(np.float32),
        "beta_abs_layered": beta_abs_layered.astype(np.float32),
        "scale_height_km": float(scale_height_km),
        "molar_mass": float(avg_molar_mass),
        "atmo_height_km": float(scale_height_km * max(1.0, math.log(max(1.0, pressure_atm * 1e6))))
    }
    
    _atmo_cache[cache_key] = res
    return res

def compute_mie_coefficients(base_beta=2.0e-6, angstrom_exponent=None):
    """
    Computes Mie scattering coefficients using the Angstrom exponent approximation.
    base_beta is the scattering coefficient at 550nm.
    If angstrom_exponent is None, it is calculated automatically from the aerosol concentration:
    thick aerosol decks (clouds) are assumed to have large particles (angstrom -> 0),
    while thin background hazes are assumed to have small particles (angstrom -> 1.2).
    """
    if angstrom_exponent is None:
        # Clear skies (base_beta -> 0) yields 1.2, thick cloud decks (base_beta >= 2.0e-5) yields ~ 0.0
        angstrom_exponent = 1.2 * math.exp(-base_beta / 5.0e-6)
        
    lambda_0 = 550e-9
    # beta_mie = base_beta * (lambda / lambda_0)^(-alpha)
    beta_mie = base_beta * (WAVELENGTHS / lambda_0) ** (-angstrom_exponent)
    return beta_mie.astype(np.float32)
