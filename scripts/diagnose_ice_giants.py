"""Deep diagnostic: Trace exactly how the accuracy test handles Pluto+Charon.

The key bug found: For START vectors, ALL bodies use center=500@10 (Sun body).
For END vectors, moons use their parent body center. 

For Charon: START fetches Sun-centered, END fetches Pluto-centered.
But more critically: ALL start vectors are Sun-centered and added directly to 
REBOUND at those Sun-centered coordinates.

The comparison then does: sim_pos[charon] - sim_pos[pluto] vs horizons_pluto_centered.

This should actually be fine IF the relative dynamics are correct.
But what about the effect on Pluto itself?
"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from fetch_horizons import query_horizons, parse_state_vector, HORIZONS_IDS

AU_TO_KM = 149597870.7

print("=" * 90)
print("DEEP DIVE: Pluto system accuracy")
print("=" * 90)

# The test fetches ALL start vectors from 500@10 (Sun center)
# This means Charon's start position is relative to the Sun, not Pluto
# When added to REBOUND, Charon is at the correct Sun-centered position
# The sim then integrates everything in heliocentric coordinates (post move_to_com)
# At the end, sim[charon] - sim[pluto] gives Pluto-relative position
# The END Charon vector is fetched relative to Pluto (500@999)
# So the comparison IS consistent for Charon.
# For Pluto itself, both start and end are relative to Sun (500@10)
# So the comparison IS consistent for Pluto too.

# Let's check: is the Pluto mass correct for the SYSTEM (Pluto+Charon)?
# JPL Horizons treats Pluto body (ID=999) as just Pluto, not the system
# But when fetching relative to Sun, do we get the system barycenter or Pluto body?

print("\n--- Checking: Pluto body vs Pluto system barycenter ---")
print("Horizons ID 999 = Pluto body center")
print("Horizons ID 9 = Pluto SYSTEM barycenter")

# Fetch Pluto body (999) relative to Sun
resp_body = query_horizons("999", "500@10", start_time="2025-01-01 12:00", stop_time="2025-01-02")
sv_body = parse_state_vector(resp_body)

# Fetch Pluto system bary (9) relative to Sun  
resp_bary = query_horizons("9", "500@10", start_time="2025-01-01 12:00", stop_time="2025-01-02")
sv_bary = parse_state_vector(resp_bary)

if sv_body and sv_bary:
    dx = (sv_body['x'] - sv_bary['x']) * AU_TO_KM
    dy = (sv_body['y'] - sv_bary['y']) * AU_TO_KM
    dz = (sv_body['z'] - sv_bary['z']) * AU_TO_KM
    dr = math.sqrt(dx**2 + dy**2 + dz**2)
    dvx = (sv_body['vx'] - sv_bary['vx']) * AU_TO_KM / 365.25 / 86400
    dvy = (sv_body['vy'] - sv_bary['vy']) * AU_TO_KM / 365.25 / 86400
    dvz = (sv_body['vz'] - sv_bary['vz']) * AU_TO_KM / 365.25 / 86400
    dv = math.sqrt(dvx**2 + dvy**2 + dvz**2)
    print(f"\n  Pluto body (999) vs system bary (9) @ 2025-01-01:")
    print(f"    Position offset: Dx={dx:.1f}, Dy={dy:.1f}, Dz={dz:.1f} km  |Dr|={dr:.1f} km")
    print(f"    Velocity offset: |Dv|={dv:.6f} km/s")

print("\n  The accuracy test uses HORIZONS_IDS['Pluto'] = '999' (Pluto body center)")
print("  Charon's start is fetched from Sun center, so Charon is at Sun-relative coords")
print("  This is actually consistent for the integration.")

# But let's check: what's the Charon orbit error?
# Charon: 1047.53 km total drift, 1047.14 km along-track
# This is reasonable for a 6.387-day orbit over 1 year (~57 orbits)
# 1047 km / (2*pi*19571 km) * 360 = ~0.97 degrees of phase error

print("\n\n--- Checking: Does Pluto's error come from missing Pluto-specific physics? ---")
print("Pluto error: 6622 km total, 2696 km along, -3221 km radial, 5120 km cross-track")
print("This is NOT typical phase drift (would be mostly along-track)")
print("The large cross-track component suggests an ORBITAL PLANE error")

# Let's check if Pluto's J2 is in the system
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/system.json')
with open(data_path, "r") as f:
    bodies = json.load(f)

pluto = next(b for b in bodies if b["name"] == "Pluto")
charon = next(b for b in bodies if b["name"] == "Charon")

print(f"\n  Pluto mass: {pluto['m']:.6e} Msun")
print(f"  Charon mass: {charon['m']:.6e} Msun")
print(f"  Charon/Pluto mass ratio: {charon['m']/pluto['m']:.4f}")
print(f"  (JPL value: ~0.1218)")

# Check if the test correctly accounts for Pluto's orbit
# Pluto's orbital period is ~248 years
# In 1 year it moves ~1.45 degrees
# At 39.5 AU, 1 degree = 39.5 * 2*pi/360 AU = 0.689 AU = 1.03e8 km
# So 6622 km = 0.006 degrees, which is very small

print(f"\n  Pluto orbital period: ~248 years")
print(f"  Angular motion in 1 year: ~1.45 degrees")  
print(f"  6622 km at 39.5 AU = {6622 / (39.5 * AU_TO_KM) * 180 / math.pi * 3600:.2f} arcseconds")

# Now let's check Neptune mass more carefully
# Neptune mass error was -0.0209%
# Over 1 year at 30 AU, Neptune moves ~2.2 degrees
# An error in Neptune's mass affects its gravitational pull on Pluto

neptune = next(b for b in bodies if b["name"] == "Neptune")
print(f"\n\n--- Neptune mass analysis ---")
print(f"  Neptune mass in system.json: {neptune['m']:.12e} Msun")
print(f"  JPL GM_Neptune = 6836529.9 km^3/s^2")
print(f"  GM_Sun = 1.32712440018e11 km^3/s^2")
print(f"  Neptune/Sun mass ratio = {6836529.9 / 1.32712440018e11:.12e}")
print(f"  Difference from system.json: {(neptune['m'] - 6836529.9/1.32712440018e11) / (6836529.9/1.32712440018e11) * 100:+.4f}%")

# Key insight: Neptune's mass error of -0.021% means the sim
# pulls Pluto slightly wrong. Over 248 years this matters;
# over 1 year it's small but not negligible for Pluto given
# the 3:2 resonance sensitivity.

print(f"\n\n--- Key question: Is the Pluto mass just Pluto, or Pluto+Charon system? ---")
# JPL GM of Pluto system = 977 km^3/s^2
# JPL GM of Pluto body = 869.6 km^3/s^2
# JPL GM of Charon = 105.88 km^3/s^2
pluto_sys_gm = 977.0
pluto_body_gm = 869.6138
charon_gm = 105.8799
print(f"  JPL GM Pluto body: {pluto_body_gm} km^3/s^2 -> {pluto_body_gm/1.32712440018e11:.6e} Msun")
print(f"  JPL GM Charon: {charon_gm} km^3/s^2 -> {charon_gm/1.32712440018e11:.6e} Msun")
print(f"  JPL GM system: {pluto_sys_gm} km^3/s^2 -> {pluto_sys_gm/1.32712440018e11:.6e} Msun")
print(f"  system.json Pluto: {pluto['m']:.6e} Msun")
print(f"  system.json Charon: {charon['m']:.6e} Msun")
print(f"  system.json total: {pluto['m'] + charon['m']:.6e} Msun")

# Now the big question: do we have a J2 for Uranus in the sim?
uranus = next(b for b in bodies if b["name"] == "Uranus")
print(f"\n\n--- Uranus configuration check ---")
print(f"  J2: {uranus.get('J2', 'NOT SET')}")
print(f"  J4: {uranus.get('j4', 'NOT SET')}")
print(f"  oblateness: {uranus.get('oblateness', 'NOT SET')}")
print(f"  pole_ra: {uranus.get('pole_ra', 'NOT SET')}")
print(f"  pole_dec: {uranus.get('pole_dec', 'NOT SET')}")
print(f"  rotation_period: {uranus.get('rotation_period', 'NOT SET')}")

# Check: Uranus pole is at RA=77.31, Dec=15.17
# This is very close to the ecliptic plane (Uranus is tilted ~98 degrees!)
# The pole dec of 15.17 in J2000 equatorial -> in ecliptic this is:
# Let's compute
from main import pole_to_ecliptic
pole_ecl = pole_to_ecliptic(uranus['pole_ra'], uranus['pole_dec'])
print(f"  Pole in ecliptic: [{pole_ecl[0]:.4f}, {pole_ecl[1]:.4f}, {pole_ecl[2]:.4f}]")
print(f"  (Uranus' pole is nearly in the orbital plane - 98-degree axial tilt)")

# Neptune check
print(f"\n--- Neptune configuration check ---")
print(f"  J2: {neptune.get('J2', 'NOT SET')}")
print(f"  J4: {neptune.get('j4', 'NOT SET')}")
print(f"  oblateness: {neptune.get('oblateness', 'NOT SET')}")
print(f"  pole_ra: {neptune.get('pole_ra', 'NOT SET')}")
print(f"  pole_dec: {neptune.get('pole_dec', 'NOT SET')}")
print(f"  rotation_period: {neptune.get('rotation_period', 'NOT SET')}")

pole_ecl_n = pole_to_ecliptic(neptune['pole_ra'], neptune['pole_dec'])
print(f"  Pole in ecliptic: [{pole_ecl_n[0]:.4f}, {pole_ecl_n[1]:.4f}, {pole_ecl_n[2]:.4f}]")

# Final: Check if Pluto is even in the J2 list or gets any special treatment
print(f"\n--- Pluto physics check ---")
print(f"  J2: {pluto.get('J2', 'NOT SET')}")
print(f"  pole_ra: {pluto.get('pole_ra', 'NOT SET')}")
print(f"  pole_dec: {pluto.get('pole_dec', 'NOT SET')}")
print(f"  type: {pluto.get('type', 'NOT SET')}")
print()
print("  NOTE: Pluto has no J2, which is fine (its J2 effect is negligible).")
print("  The error is likely from the 3:2 Neptune resonance sensitivity +")
print("  slight mass imprecisions propagating through the resonance dynamics.")

# Let's compute: how much does Neptune's mass error affect the resonance?
# The 3:2 MMR means Pluto orbits 2x for every 3 Neptune orbits
# Neptune's period ~ 164.8 yr, Pluto's ~ 248 yr
# If Neptune's mass is off by -0.021%, its period changes by:
# T = 2*pi*sqrt(a^3 / G*M_total) ~ 1/sqrt(M)
# dT/T ~ -0.5 * dM/M = -0.5 * (-0.021%) = +0.0105%
# Over 1 year: dT = 164.8 * 365.25 * 0.000105 * 86400 = ...
# Actually let's just compute the libration sensitivity

# The resonance argument phi = 3*lambda_Pluto - 2*lambda_Neptune - omega_Pluto
# librates around 180 degrees with a period of ~20,000 years
# A mass error shifts the libration center/frequency
# Over 1 year, the maximum displacement from a resonance shift is:
# dphi/dt * dt ~ (3*n_P - 2*n_N) error contribution
# This is the fundamental source of Pluto's anomalous error pattern

print("\n--- Summary of findings ---")
print("1. The accuracy_test.py centers ARE consistent for planets (all Sun-centered)")
print("2. Charon has a center mismatch (start=Sun, end=Pluto) but sim comparison")
print("   correctly uses sim[charon]-sim[pluto], so this is NOT causing Charon's error")
print("3. Masses are close to JPL but not exact:")
print("   - Neptune: -0.021% (affects Pluto via 3:2 resonance)")
print("   - Pluto: +0.079%")  
print("   - Charon: -0.642% <-- largest mass discrepancy!")
print("4. Pluto's unique error signature (large radial+cross-track) is likely from:")
print("   a) Sensitivity of the 3:2 Neptune resonance to mass precision")
print("   b) Charon's mass error affecting the Pluto-Charon barycenter dynamics")
print("5. Uranus 329 km drift: likely from unmodeled perturbations or subtle")
print("   interaction with its J2 harmonics at the planetary orbit level")
print("6. Neptune 22.8 km drift: cross-track component from unmodeled")
print("   Trans-Neptunian Object perturbations")
