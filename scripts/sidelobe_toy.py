#!/usr/bin/env python3
"""
sidelobe_toy.py - how many baselines a sparse tomographic stack needs before
its sidelobes stop looking like deep structure.

Uses no SAR data and makes no claim about any published image. It replaces
aliasing_simulation.py (now in deprecated/), whose printed "ambiguity height"
came from the smallest gap between two random baselines (399 km for its
6-baseline case) and whose text about periodic copies at 600 m and 1200 m did
not follow from that number.

Model: five point scatterers at 0, 21, 43, 60 and 140 m, random baselines,
5% noise, matched beamformer over -100 to +800 m. Nothing is planted deeper
than 140 m. The statistic is the largest reconstructed amplitude beyond
300 m, as a fraction of the main peak, over 2000 random baseline draws.

Usage:
    python3 scripts/sidelobe_toy.py
"""
import numpy as np

SCATTERERS = [(0, 1.0), (21, 0.3), (43, 0.5), (60, 0.2), (140, 0.8)]   # (height m, amplitude)
SLANT_RANGE = 650e3
INCIDENCE = np.radians(35)
DEPTHS = np.arange(-100, 800, 0.5)
FAR = DEPTHS > 300

CASES = [
    ("6 baselines, X-band, within +/-300 m", 6, 300, 0.031),
    ("15 baselines, C-band, within +/-170 m", 15, 170, 0.055),
    ("200 baselines, X-band, within +/-500 m", 200, 500, 0.031),
]


def run(n_images, b_max, wavelength, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    spurious, resolution = [], []
    for _ in range(draws):
        b = rng.uniform(-b_max, b_max, n_images)
        b[n_images // 2] = 0.0
        kz = 4 * np.pi * b / (wavelength * SLANT_RANGE * np.sin(INCIDENCE))
        y = sum(a * np.exp(1j * kz * h) for h, a in SCATTERERS)
        y = y + 0.05 * np.abs(y).max() * (rng.standard_normal(n_images) + 1j * rng.standard_normal(n_images))
        t = np.abs(np.exp(1j * np.outer(kz, DEPTHS)).conj().T @ y)
        spurious.append(t[FAR].max() / t.max())
        resolution.append(2 * np.pi / (kz.max() - kz.min()))
    return np.array(spurious), float(np.median(resolution))


if __name__ == "__main__":
    print("Largest reconstructed amplitude beyond 300 m, as a fraction of the main peak")
    print("(nothing is planted beyond 140 m; 2000 random baseline draws per case)\n")
    for label, n, b_max, lam in CASES:
        s, res = run(n, b_max, lam)
        print(f"  {label:40s} median {np.median(s):.2f}   5th-95th pct {np.percentile(s, 5):.2f}-"
              f"{np.percentile(s, 95):.2f}   median resolution {res:.0f} m")
    print("\nToy only. It shows that a handful of baselines cannot tell a deep feature")
    print("from a sidelobe of shallow ones. It does not diagnose any published figure.")
