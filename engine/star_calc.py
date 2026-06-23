import math

class StarCalculator:
    G = 6.67430e-11
    MSUN = 1.98847e30
    RSUN = 6.957e8
    LSUN = 3.828e26
    WIEN = 2.8977719e-3

    @staticmethod
    def calc_lum(r, t): 
        if r <= 0 or t <= 0: return 0
        return (r ** 2) * ((t / 5778) ** 4)

    @staticmethod
    def calc_rad(l, t): 
        if l <= 0 or t <= 0: return 0
        return math.sqrt(l / ((t / 5778) ** 4))

    @staticmethod
    def calc_temp(l, r): 
        if l <= 0 or r <= 0: return 0
        return 5778 * ((l / (r ** 2)) ** 0.25)

    @staticmethod
    def ms_mass_from_lum(l):
        if l <= 0: return 0.1
        if l < 0.03: return (l / 0.23) ** (1 / 2.3)
        if l < 16: return l ** 0.25
        if l < 1000000: return (l / 1.4) ** (1 / 3.5)
        return l / 32000

    @staticmethod
    def ms_lum_from_mass(m):
        if m < 0.43: return 0.23 * (m ** 2.3)
        if m < 2.0: return m ** 4
        if m < 55: return 2.0 * (m ** 3.5)
        return 32000 * m

    @classmethod
    def evolve_star(cls, initial_mass, age_val, evo_path="standard", metallicity=0.0):
        rad_mult = max(0.6, min(1.2, 1.0 + 0.15 * metallicity))
        lum_mult = max(0.5, min(2.0, 1.0 - 0.2 * metallicity))

        base_r = (initial_mass ** 0.8) * rad_mult
        
        if initial_mass < 1.5:
            r_tams_mult, l_tams_mult = 1.50, 2.00
            curve_r_ms, curve_l_ms = 2.0, 1.9
        else:
            r_tams_mult, l_tams_mult = 2.00, 2.50
            curve_r_ms, curve_l_ms = 1.5, 1.6

        r_zams, r_tams = base_r * 0.87, base_r * r_tams_mult
        
        base_l = cls.ms_lum_from_mass(initial_mass) * lum_mult
        l_zams, l_tams = base_l * 0.70, base_l * l_tams_mult

        # Phase 0: Pre-Main Sequence (Protostar)
        if age_val < 0.0:
            p = max(0.0, 1.0 + (age_val / 0.05))
            r = r_zams * (1.0 + 9.0 * ((1.0 - p) ** 2.0))
            l = l_zams * (1.0 + 4.0 * ((1.0 - p) ** 2.0))
            phase = "T-Tauri Star" if initial_mass < 2.0 else "Herbig Ae/Be Star"
            return initial_mass, r, l, phase

        # Phase 1: Main Sequence (0.0 to 1.0)
        elif age_val <= 1.0:
            r = r_zams + (r_tams - r_zams) * (age_val ** curve_r_ms)
            l = l_zams + (l_tams - l_zams) * (age_val ** curve_l_ms)
            return initial_mass, r, l, None

        # Phase 2: Giant / Post-MS Phase (1.0 to 1.1)
        elif age_val <= 1.1:
            progress = (age_val - 1.0) * 10.0 # 0.0 to 1.0
            
            # --- MASSIVE STAR STRIPPING PATHWAY (LBV -> WR) ---
            if (initial_mass >= 25.0 and evo_path == "stripping") or initial_mass >= 45.0:
                if progress < 0.5:
                    # 1.0 to 1.05: Luminous Blue Variable (LBV)
                    sub_prog = progress * 2.0
                    r = r_tams + (60.0 * math.sqrt(initial_mass) - r_tams) * sub_prog
                    l = l_tams * (1.5 + sub_prog * 0.5)
                    current_mass = initial_mass * (1.0 - 0.3 * sub_prog) # Huge mass loss
                    return current_mass, r, l, "Luminous Blue Variable (LBV)"
                else:
                    # 1.05 to 1.1: Wolf-Rayet Star (WR)
                    sub_prog = (progress - 0.5) * 2.0
                    r_lbv = 60.0 * math.sqrt(initial_mass)
                    # Envelope stripped! Shrinks down to a bare core (few solar radii)
                    r = r_lbv - (r_lbv - max(1.5, 0.15 * initial_mass)) * (sub_prog ** 0.3)
                    l = l_tams * 2.0 # Maintains huge luminosity
                    current_mass = (initial_mass * 0.7) * (1.0 - 0.4 * sub_prog) # Keeps shedding
                    return current_mass, r, l, "Wolf-Rayet (WR)"

            # --- STANDARD PATHWAY ---
            r_max = 200.0 * (initial_mass ** 0.6) * rad_mult
            l_max = 2500.0 * (initial_mass ** 1.5) * lum_mult
            
            if initial_mass >= 8.0:
                # Ensure extremely massive stars expand correctly
                r_max = max(r_max, r_tams * (40.0 * math.sqrt(initial_mass)))
                l_max = max(l_max, l_tams * (1.2 + (initial_mass * 0.015)))

            if initial_mass < 2.0:
                mass_lost = 0.45 * progress
                curve_r, curve_l = progress ** 3.0, progress ** 4.0
            elif initial_mass < 8.0:
                mass_lost = 0.6 * progress 
                curve_r, curve_l = progress ** 2.5, progress ** 3.0
            else:
                mass_lost = 0.4 * progress
                curve_r, curve_l = progress ** 2.0, progress ** 1.5

            r = r_tams + (r_max - r_tams) * curve_r
            l = l_tams + (l_max - l_tams) * curve_l
            current_mass = max(0.1, initial_mass * (1.0 - mass_lost))
            return current_mass, r, l, None

        # Phase 3: Remnant (1.1 to 1.2)
        else:
            progress = (age_val - 1.1) * 10.0
            if initial_mass < 8.0:
                rem_mass = min(1.4, initial_mass * 0.5)
                return rem_mass, 0.01, 1.0 * (0.0001 ** progress), None
            elif initial_mass < 25.0:
                return 1.4, 1.4e-5, 1e-5, None
            else:
                rem_mass = initial_mass * 0.3
                return rem_mass, (3.0 * rem_mass) / 695700.0, 0.0, None

    @classmethod
    def forge(cls, mode="evolution", evo_path="standard", mass=1.0, metallicity=0.0, rot_frac=0.0, inclination=0.0, age_pct=0.46, radius=None, temp=None, lum=None):
        if mass is None: mass = 1.0
        if age_pct is None: age_pct = 0.46
        if metallicity is None: metallicity = 0.0
        if rot_frac is None: rot_frac = 0.0
        if inclination is None: inclination = 0.0
        phase_override = None

        rad_mult = max(0.6, min(1.2, 1.0 + 0.15 * metallicity))
        lum_mult = max(0.5, min(2.0, 1.0 - 0.2 * metallicity))

        if mode == "evolution":
            current_mass, radius, lum, phase_override = cls.evolve_star(mass, age_pct, evo_path, metallicity)
            temp = cls.calc_temp(lum, radius)
            
        elif mode == "surface":
            if radius and temp and not lum: lum = cls.calc_lum(radius, temp)
            elif lum and temp and not radius: radius = cls.calc_rad(lum, temp)
            elif lum and radius and not temp: temp = cls.calc_temp(lum, radius)
            current_mass = cls.ms_mass_from_lum(lum / lum_mult)
            
            base_r = (current_mass ** 0.8) * rad_mult
            r_tams_mult = 1.50 if current_mass < 1.5 else 2.00
            r_zams, r_tams = base_r * 0.87, base_r * r_tams_mult
            
            if radius <= r_zams:
                age_pct = 0.0
            elif radius <= r_tams:
                age_pct = math.sqrt((radius - r_zams) / (r_tams - r_zams))
            else:
                r_max = 200.0 * (current_mass ** 0.6) * rad_mult
                if current_mass >= 8.0:
                    r_max = max(r_max, r_tams * (40.0 * math.sqrt(current_mass)))

                if current_mass < 2.0: exp_val = 3.0
                elif current_mass < 8.0: exp_val = 2.5
                else: exp_val = 2.0
                
                if r_max > r_tams:
                    progress = max(0, (radius - r_tams) / (r_max - r_tams)) ** (1.0 / exp_val)
                    age_pct = 1.0 + min(0.1, progress * 0.1)
                else:
                    age_pct = 1.1

        # Rotation Physics (Roche Model Approximation & Von Zeipel)
        # Conserve stellar volume: R_pole * R_eq^2 = R_base^3
        f_oblateness = 1.0 + 0.22 * (rot_frac ** 2)
        r_pole = radius / (f_oblateness ** (2/3))
        r_eq = radius * (f_oblateness ** (1/3))
        
        # Modern empirical beta for gravity darkening
        beta = 0.08
        g_ratio_eq_pole = ((r_pole / r_eq) ** 2) * max(0.01, (1.0 - rot_frac ** 2))
        t_ratio_eq_pole = g_ratio_eq_pole ** beta
        
        t_pole = temp * (1.0 + 0.08 * rot_frac)
        t_eq = t_pole * t_ratio_eq_pole
        
        inc_rad = math.radians(inclination)
        cos_i = math.cos(inc_rad)
        sin_i = math.sin(inc_rad)
        
        r_proj_y = math.sqrt((r_pole * sin_i)**2 + (r_eq * cos_i)**2)
        apparent_area_ratio = (r_eq * r_proj_y) / (radius ** 2)
        
        # Proper normalized interpolation for apparent temperature
        apparent_temp = t_pole * (1.0 - sin_i) + t_eq * sin_i
        apparent_lum = lum * apparent_area_ratio * ((apparent_temp / temp) ** 4) if temp > 0 else 0
        apparent_radius = math.sqrt(r_eq * r_proj_y)

        if apparent_radius <= 0 or current_mass <= 0:
            logg, density, escape_vel = 0, 0, 0
        else:
            logg = math.log10(max(1, ((cls.G * (current_mass * cls.MSUN)) / ((apparent_radius * cls.RSUN) ** 2)) * 100))
            density = ((current_mass * cls.MSUN) * 1000) / (((4/3) * math.pi * ((apparent_radius * cls.RSUN) ** 3)) * 1000000)
            escape_vel = math.sqrt((2 * cls.G * (current_mass * cls.MSUN)) / (apparent_radius * cls.RSUN)) / 1000

        is_bh = apparent_lum == 0 and apparent_radius > 0 and mode == "evolution" and mass >= 25.0
        is_ns = apparent_radius < 1e-4 and apparent_temp > 0
        is_wd = apparent_radius < 0.05 and current_mass <= 1.4 and not is_ns and not is_bh

        # Classification Routing
        if phase_override:
            full_designation = phase_override
            phase = "Envelope Stripping Phase"
        elif is_bh:
            full_designation = "Black Hole"
            phase = "Singularity"
        elif is_ns:
            full_designation = "Neutron Star"
            phase = "Degenerate Remnant"
        elif is_wd:
            full_designation = f"White Dwarf (D)"
            phase = "Degenerate Remnant"
        else:
            classes = [
                {"min": 30000, "c": 'O'}, {"min": 10000, "c": 'B'}, {"min": 7500, "c": 'A'},
                {"min": 6000, "c": 'F'}, {"min": 5200, "c": 'G'}, {"min": 3700, "c": 'K'},
                {"min": 2400, "c": 'M'}, {"min": 1300, "c": 'L'}, {"min": 550, "c": 'T'}
            ]
            spectral_type, subclass = 'Y', 0
            for i, c in enumerate(classes):
                if apparent_temp >= c["min"]:
                    spectral_type = c["c"]
                    max_t = 100000 if i == 0 else classes[i-1]["min"]
                    subclass = min(9, max(0, math.floor(((max_t - apparent_temp) / max(1, max_t - c["min"])) * 10)))
                    break

            if logg < 0.0: lum_class = 'Ia-0'
            elif logg < 1.5: lum_class = 'Ia'
            elif logg < 2.2: lum_class = 'II'
            elif logg < 3.2: lum_class = 'III'
            elif logg < 3.8: lum_class = 'IV'
            elif logg < 5.5: lum_class = 'V'
            else: lum_class = 'VI'

            full_designation = f"{spectral_type}{subclass} {lum_class}"
            
            if spectral_type == 'O': color_name = "Blue"
            elif spectral_type == 'B': color_name = "Blue-White"
            elif spectral_type == 'A': color_name = "White"
            elif spectral_type == 'F': color_name = "Yellow-White"
            elif spectral_type == 'G': color_name = "Yellow"
            elif spectral_type == 'K': color_name = "Orange"
            elif spectral_type == 'M': color_name = "Red"
            else: color_name = "Brown"

            if lum_class == 'Ia-0': size_name = "Hypergiant"
            elif lum_class == 'Ia': size_name = "Supergiant"
            elif lum_class == 'II': size_name = "Bright Giant"
            elif lum_class == 'III': size_name = "Giant"
            elif lum_class == 'IV': size_name = "Subgiant"
            elif lum_class == 'V': size_name = "Dwarf"
            else: size_name = "Subdwarf"

            phase = f"{color_name} {size_name}"

        # Color Math
        if is_bh:
            color_hex = "#000000"
        else:
            t = apparent_temp / 100
            r_c = 255 if t <= 66 else max(0, min(255, 329.69 * ((t - 60) ** -0.133)))
            g_c = max(0, min(255, 99.47 * math.log(max(1, t)) - 161.1)) if t <= 66 else max(0, min(255, 288.1 * ((t - 60) ** -0.075)))
            b_c = 255 if t >= 66 else (0 if t <= 19 else max(0, min(255, 138.5 * math.log(max(1, t - 10)) - 305)))
            color_hex = f"#{int(r_c):02x}{int(g_c):02x}{int(b_c):02x}"

        base_mass_for_life = mass if mode == "evolution" else current_mass
        base_l_for_life = cls.ms_lum_from_mass(base_mass_for_life) * lum_mult
        lifespan_gyr = 10 * (base_mass_for_life / max(1e-10, base_l_for_life))

        v_eq_km_s = 0.0
        if r_eq > 0 and current_mass > 0:
            v_crit_km_s = math.sqrt((cls.G * (current_mass * cls.MSUN)) / (r_eq * cls.RSUN)) / 1000
            v_eq_km_s = v_crit_km_s * rot_frac

        # Habitable Zone
        hz_inner = round(math.sqrt(apparent_lum / 1.1), 3) if apparent_lum > 0 else 0
        hz_outer = round(math.sqrt(apparent_lum / 0.53), 3) if apparent_lum > 0 else 0

        # Pulsation (Instability Strip)
        is_pulsating = False
        pulsation_type = ""
        pulsation_period_days = 0.0
        pulsation_period_str = ""

        if apparent_lum > 0 and mode == "evolution" and age_pct > 0.0 and apparent_radius > 0:
            log_l = math.log10(apparent_lum)
            t_blue = 8500 - 500 * log_l
            t_red = 7000 - 500 * log_l
            
            if t_red <= apparent_temp <= t_blue and not is_wd and not is_ns and not is_bh:
                is_pulsating = True
                if apparent_lum < 100:
                    pulsation_type = "Delta Scuti"
                elif apparent_lum < 1000:
                    pulsation_type = "RR Lyrae"
                else:
                    pulsation_type = "Classical Cepheid"
                
                pulsation_period_days = 0.04 * math.sqrt((apparent_radius ** 3) / current_mass)
                
                if pulsation_period_days < 1.0:
                    pulsation_period_str = f"{round(pulsation_period_days * 24, 1)} hours"
                else:
                    pulsation_period_str = f"{round(pulsation_period_days, 1)} days"

        return {
            "physical": {
                "mass_msun": round(current_mass, 4), "radius_rsun": round(apparent_radius, 6),
                "temp_k": round(apparent_temp), "lum_lsun": round(apparent_lum, 6), "logg": round(logg, 3)
            },
            "habitable_zone": {
                "inner": hz_inner,
                "outer": hz_outer
            },
            "pulsation": {
                "is_pulsating": is_pulsating,
                "type": pulsation_type,
                "period_days": pulsation_period_days,
                "period_str": pulsation_period_str
            },
            "classification": { "fullDesignation": full_designation },
            "evolution": {
                "lifespan_gyr": round(lifespan_gyr, 3), 
                "age_gyr": round(lifespan_gyr * min(1.2, age_pct), 3),
                "age_pct": round(age_pct * 100, 1),
                "phase": phase
            },
            "rotation": {
                "v_eq": round(v_eq_km_s, 1),
                "r_eq": round(r_eq, 6),
                "r_pole": round(r_pole, 6),
                "t_eq": round(t_eq),
                "t_pole": round(t_pole),
                "oblateness": round(1.0 - (r_pole / r_eq), 3) if r_eq > 0 else 0,
                "is_rotating": rot_frac > 0.05
            },
            "visual": { "colorHex": color_hex, "is_bh": is_bh, "is_wr": phase_override == "Wolf-Rayet (WR)", "visual_oblateness": round(1.0 - (r_proj_y / r_eq), 3) if r_eq > 0 else 0 }
        }