"""Apply documented static Venus/giant profiles to the bundled Solar System definitions."""

import json
import math
import re
from pathlib import Path

MOLAR = {"H2": .00201588, "He": .004002602, "CH4": .0160425, "NH3": .01703052}
THERMAL = {
    "Jupiter": [(10, 330), (3, 220), (1, 165), (.5, 135), (.1, 110), (.01, 150), (.001, 165), (.00001, 170)],
    "Saturn": [(10, 270), (3, 180), (1, 134), (.5, 110), (.1, 82), (.01, 120), (.001, 140), (.00001, 150)],
    "Uranus": [(10, 150), (3, 105), (1, 76), (.5, 64), (.1, 53), (.01, 70), (.001, 95), (.00001, 110)],
    "Neptune": [(10, 150), (3, 100), (1, 72), (.5, 62), (.1, 55), (.01, 90), (.001, 130), (.00001, 150)],
}
CLOUD_DECK_BAR = {"Jupiter": .25, "Saturn": .12, "Uranus": .5, "Neptune": .5}


def pressure_geometry(body, thermal, pressure):
    mixture = body["atmosphere"]["composition"]
    molar = sum(MOLAR[gas]*weight for gas, weight in mixture.items()) / sum(mixture.values())
    radius = body["r"] * 696340
    gravity = 6.67430e-11 * 1.98847e30 * body["m"] / (radius*1000)**2
    points = sorted((math.log(p), t) for p, t in thermal)

    def temperature(x):
        for (x0, t0), (x1, t1) in zip(points, points[1:]):
            if x <= x1:
                return t0+(t1-t0)*(x-x0)/(x1-x0)
        return points[-1][1]

    target = math.log(pressure)
    knots = sorted({0., target, *[x for x, _ in points if min(0., target) < x < max(0., target)]})
    integral = sum((b-a)*(temperature(a)+temperature(b))/2 for a, b in zip(knots, knots[1:]))
    if target < 0:
        integral = -integral
    coefficient = 8.31446261815324/(molar*gravity*1000)
    height = radius*(1/(1+coefficient*integral/radius)-1)
    return height, temperature(target), coefficient*temperature(target)


def atmosphere_for(body):
    old = body["atmosphere"]
    result = dict(schema_version=2, enabled=True, model="physical", temperature_k=old["temperature_k"], composition=old["composition"])
    if body["name"] == "Venus":
        result["pressure_pa"] = old["pressure_pa"]
        result["temperature_layers"] = [[0, 737], [10, 658], [20, 580], [30, 495], [40, 416], [50, 340], [60, 260], [70, 230], [80, 200], [100, 180]]
        result["aerosols"] = [dict(material="sulfate", optical_depth_550=tau, effective_radius_um=size,
            effective_variance=.15, peak_km=peak, width_km=width) for tau, size, peak, width in [(1, .2, 65, 5), (15, 1, 60, 5), (10, 3, 53, 3)]]
        result["aerosols"][0]["material"] = "sulfate_with_uv_absorber"
        return result
    thermal = THERMAL[body["name"]]
    result["reference_pressure_pa"] = 100000
    result["cloud_deck_pressure_pa"] = CLOUD_DECK_BAR[body["name"]]*100000
    result["bottom_altitude_km"] = round(pressure_geometry(body, thermal, 10)[0], 2)
    pressures = []
    for (p0, _), (p1, _) in zip(thermal, thermal[1:]):
        pressures.extend(math.exp(math.log(p0)+(math.log(p1)-math.log(p0))*i/4) for i in range(4))
    pressures.append(thermal[-1][0])
    result["temperature_layers"] = [[round(h, 2), round(t, 2)] for h, t, _ in (pressure_geometry(body, thermal, p) for p in pressures)]
    result["aerosols"] = []
    return result


def update(path):
    source = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = source.index("[")+1
    changes = []
    while True:
        while source[position].isspace() or source[position] == ",":
            position += 1
        if source[position] == "]":
            break
        body, length = decoder.raw_decode(source[position:])
        if body.get("name") in {*THERMAL, "Venus"}:
            match = re.search(r'"atmosphere"\s*:\s*', source[position:position+length])
            start = position+match.end()
            _, old_length = decoder.raw_decode(source[start:])
            key = position+match.start()
            indent = source[source.rfind("\n", 0, key)+1:key]
            replacement = json.dumps(atmosphere_for(body), indent=2).replace("\n", "\n"+indent)
            changes.append((start, start+old_length, replacement))
        position += length
    for start, end, replacement in reversed(changes):
        source = source[:start]+replacement+source[end:]
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for path in [root/"data/system.json", root/"data/systems/Solar System/system.json"]:
        update(path)
