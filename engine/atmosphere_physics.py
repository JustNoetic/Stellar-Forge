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
    "H2":  (1.000132, 1.000, 0.002016, np.array([0.0, 0.0, 0.0])),
    "He":  (1.000036, 1.000, 0.004002, np.array([0.0, 0.0, 0.0])),
    "O3":  (1.000271, 1.096, 0.04800, np.array([0.5e-25, 3.0e-25, 0.2e-25])), # Ozone strongly absorbs green/red (Chappuis band)
}

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
        
    total_fraction = sum(composition.values())
    if total_fraction <= 0:
        total_fraction = 1.0
        
    normalized_composition = {k: v / total_fraction for k, v in composition.items()}
    
    # Calculate aggregate physical properties
    avg_molar_mass = 0.0
    avg_n_minus_1 = 0.0
    avg_king_factor = 0.0
    beta_absorption = np.zeros(3)
    
    pressure_pa = pressure_atm * P_STD
    number_density = pressure_pa / (K_B * temperature_k)
    
    for gas, fraction in normalized_composition.items():
        if gas not in GAS_PROPERTIES:
            continue
            
        n, king, m, abs_cross = GAS_PROPERTIES[gas]
        
        avg_molar_mass += m * fraction
        avg_n_minus_1 += (n - 1.0) * fraction
        avg_king_factor += king * fraction
        
        # Absorption coefficient: cross section * number density of this specific gas
        gas_number_density = number_density * fraction
        beta_absorption += abs_cross * gas_number_density
        
    avg_n = 1.0 + avg_n_minus_1
    
    # Rayleigh scattering cross section for the mixture
    # sigma = (24 * pi^3 / (lambda^4 * N_s^2)) * ((n^2 - 1)/(n^2 + 2))^2 * KingFactor
    
    term1 = 24.0 * (math.pi ** 3) / (WAVELENGTHS ** 4 * N_S ** 2)
    term2 = ((avg_n ** 2 - 1.0) / (avg_n ** 2 + 2.0)) ** 2
    
    sigma_rayleigh = term1 * term2 * avg_king_factor
    
    # Beta Rayleigh = cross section * actual number density
    beta_rayleigh = sigma_rayleigh * number_density
    
    # Scale height H = R * T / (M * g)
    if gravity_m_s2 > 0 and avg_molar_mass > 0:
        scale_height_m = (R * temperature_k) / (avg_molar_mass * gravity_m_s2)
    else:
        scale_height_m = 8000.0 # fallback
        
    scale_height_km = scale_height_m / 1000.0
    
    return {
        "beta_rayleigh": beta_rayleigh.astype(np.float32),
        "beta_absorption": beta_absorption.astype(np.float32),
        "scale_height_km": float(scale_height_km),
        "molar_mass": float(avg_molar_mass)
    }
