"""Canonicalize atmosphere records without rewriting unrelated system data."""

import argparse
import json
import re
from pathlib import Path


def migrate(record, body_name="", solar_presets=False):
    if record.get("schema_version", 1) > 2:
        raise ValueError("Unsupported atmosphere schema version")
    complete = ("pressure_pa" in record or "surface_pressure" in record or "reference_pressure_pa" in record) and (
        "temperature_k" in record or "temperature" in record) and bool(record.get("composition"))
    physical = record.get("model", "physical" if complete else "optical") == "physical"
    result = {"schema_version": 2, "enabled": record.get("enabled", complete),
              "model": "physical" if physical else "optical"}
    result["pressure_pa"] = record.get("pressure_pa", record.get("surface_pressure", 1) * 101325)
    if "reference_pressure_pa" in record:
        result.pop("pressure_pa")
        result["reference_pressure_pa"] = record["reference_pressure_pa"]
        result["bottom_altitude_km"] = record.get("bottom_altitude_km", 0)
    result["temperature_k"] = record.get("temperature_k", record.get("temperature", 288.15))
    result["composition"] = record.get("composition", {"N2": .78, "O2": .21, "Ar": .01})
    for key in ("ground_albedo", "ozone", "cloud_deck_pressure_pa"):
        if key in record:
            result[key] = record[key]
    layers = record.get("temperature_layers", record.get("temperature_profile", []))
    if layers:
        result["temperature_layers"] = layers
    automatic_height = record.get("automatic_height", record.get("height_mode", "auto") == "auto")
    if not automatic_height:
        result["automatic_height"] = False
    if not automatic_height or not physical:
        result["height_km"] = record.get("height_km", 100)
    if not physical:
        for key in ("rayleigh_scattering", "rayleigh_scale_height_km"):
            if key in record:
                result[key] = record[key]
    manual = any(key in record for key in ("beta_mie", "mie_extinction", "mie_scattering"))
    automatic = record.get("automatic_aerosols", record.get("aerosol_mode", "manual" if manual else "auto") == "auto")
    if physical and solar_presets and body_name in ("Mars", "Titan") and record.get("schema_version", 1) == 1:
        automatic = True
        if body_name == "Mars":
            result["aerosols"] = [dict(material="mineral_dust", optical_depth_550=.48,
                effective_radius_um=1.5, effective_variance=.3, scale_height_km=12)]
        else:
            result["aerosols"] = [dict(material="organic_haze", optical_depth_550=6,
                effective_radius_um=.2, effective_variance=.2, scale_height_km=50),
                dict(material="organic_haze", optical_depth_550=.02,
                effective_radius_um=.04, effective_variance=.1, peak_km=500, width_km=20)]
    if physical and automatic:
        if "aerosols" not in result:
            if "aerosols" in record or record.get("schema_version") == 2:
                result["aerosols"] = record.get("aerosols", [])
            else:
                gases = record.get("composition", {})
                total = sum(gases.values()) or 1
                fraction = lambda gas: gases.get(gas, 0) / total
                pressure = result["pressure_pa"] / 101325
                if fraction("CH4") > .02 and result["temperature_k"] < 140 and pressure > .5:
                    material, tau, radius, variance, height = "organic_haze", 6*pressure/1.45, .2, .2, 50
                elif fraction("SO2") > .00005:
                    material, tau, radius, variance, height = "sulfate", 10*pressure/92, 1, .2, 50
                elif fraction("CO2") > .8 and pressure < .05:
                    material, tau, radius, variance, height = "mineral_dust", .48*pressure/.006, 1.5, .3, 12
                else:
                    material, tau, radius, variance, height = "water", .0024*pressure, .1, .2, 1.2
                result["aerosols"] = [dict(material=material, optical_depth_550=tau,
                    effective_radius_um=radius, effective_variance=variance, scale_height_km=height)] if pressure > 0 else []
    else:
        result["automatic_aerosols"] = automatic
        for key in ("mie_scattering", "mie_extinction", "mie_scale_height_km", "mie_asymmetry",
                    "beta_mie", "h_mie", "mie_g", "mie_angstrom", "mie_albedo"):
            if key in record:
                result[key] = record[key]
    return result


def migrate_file(path, solar_presets=False):
    source = path.read_text(encoding="utf-8-sig")
    if json.loads(source) == {"source_preset": "Solar System"}:
        return 0
    decoder = json.JSONDecoder()
    changes = []
    position = source.index("[") + 1
    while True:
        while source[position].isspace() or source[position] == ",":
            position += 1
        if source[position] == "]":
            break
        body, length = decoder.raw_decode(source[position:])
        end = position + length
        matches = list(re.finditer(r'"atmosphere"\s*:\s*', source[position:end]))
        if matches:
            original = {}
            spans = []
            for match in matches:
                start = position + match.end()
                record, record_length = decoder.raw_decode(source[start:])
                original.update(record)
                spans.append((position + match.start(), start, start + record_length))
            result = migrate(original, body.get("name", ""), solar_presets)
            key_start, start, record_end = spans[0]
            line = source.rfind("\n", 0, key_start) + 1
            indent = source[line:key_start]
            encoded = json.dumps(result, indent=2, ensure_ascii=False).replace("\n", "\n" + indent)
            changes.append((start, record_end, encoded))
            for key_start, start, record_end in spans[1:]:
                line = source.rfind("\n", 0, key_start) + 1
                if source[record_end:record_end+1] != ",":
                    raise ValueError("Duplicate atmosphere at end of object needs manual repair")
                changes.append((line, record_end + 1, ""))
        position = end
    for start, end, encoded in reversed(changes):
        source = source[:start] + encoded + source[end:]
    path.write_text(source, encoding="utf-8")
    return len(changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--solar-presets", action="store_true", help="Repair legacy Mars and Titan presets with explicit particle columns")
    args = parser.parse_args()
    for path in args.paths:
        print(f"{path}: {migrate_file(path, args.solar_presets)} atmosphere records")
