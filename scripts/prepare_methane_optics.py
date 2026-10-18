"""Reduce the archived Karkoschka (1994) cold-methane spectrum to RGB-band means."""

import argparse
from pathlib import Path

BANDS = ((600, 700), (500, 600), (406, 500))


def band_means(path):
    samples = [tuple(map(float, line.split())) for line in path.read_text().splitlines() if len(line.split()) == 2]
    result = []
    for lower, upper in BANDS:
        selected = [(wavelength, cross_section) for wavelength, cross_section in samples if lower <= wavelength < upper]
        if [w for w, _ in selected] != list(range(lower, upper)):
            raise ValueError("Expected complete, ordered 1 nm cross-section samples")
        result.append(sum(value for _, value in selected) / len(selected) * 1e-4)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path)
    args = parser.parse_args()
    print(", ".join(f"{value:.15g}" for value in band_means(args.spectrum)))
