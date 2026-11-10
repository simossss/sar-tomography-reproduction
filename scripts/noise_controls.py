#!/usr/bin/env python3
"""
noise_controls.py - negative and positive controls for the note's empirical tests.

Needs no SAR data. It feeds random numbers, and synthetic targets planted in
random numbers, through the same code that produced the v1 tables:

  Section 5.1  fix_coherence.local_coherence, with the band layout of
               fix_coherence.measure_coherence
  Section 5.2  sweep_v2.run_experiment over the full 240-point grid
  Section 5.3  the inversion lines of multi_site_tomo.extract_and_invert,
               with the baselines of Section 6.1

How to read the output: if random input reproduces a published table, the
table describes the processing and not the scene. The injected targets show
what each test could and could not have detected.

Usage:
    python3 scripts/noise_controls.py

Runtime: about four minutes. Seeds are fixed, so the printed numbers repeat.
"""
import itertools
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_coherence as fc  # noqa: E402
import sweep_v2 as sw  # noqa: E402

fc._uniform_filter = uniform_filter
rng = np.random.default_rng(0)


def noise(n, m, oversample_rows=1.0):
    """Unit-power complex Gaussian noise. Optionally band-limited along rows,
    which mimics an SLC that is oversampled in that direction."""
    x = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) / np.sqrt(2)
    if oversample_rows > 1:
        keep = np.abs(np.fft.fftfreq(n)) <= 0.5 / oversample_rows
        x = np.fft.ifft(np.fft.fft(x, axis=0) * keep[:, None], axis=0)
    return x


# ----------------------------------------------------------------------------
# Section 5.1  Sub-aperture coherence
# ----------------------------------------------------------------------------
DISTANCES = (1, 2, 5, 10, 14)


def coherence_table(data, n_sub=30, window=5):
    """Same band layout as fix_coherence.measure_coherence (tests 2 and 3)."""
    spectrum = np.fft.fftshift(np.fft.fft2(data), axes=1)
    total = spectrum.shape[1]
    usable, start = total // 2, total // 4
    bins_per = usable // n_sub
    ref = n_sub // 2
    sas = []
    for i in range(n_sub):
        sp = np.zeros_like(spectrum)
        s = start + i * bins_per
        sp[:, s:s + bins_per] = spectrum[:, s:s + bins_per]
        sas.append(np.fft.ifft2(np.fft.ifftshift(sp, axes=1)))
    table = [float(np.mean(fc.local_coherence(sas[ref + d], sas[ref], window))) for d in DISTANCES]
    return bins_per, total, table


def section_coherence():
    print("=" * 72)
    print("SECTION 5.1  sub-aperture coherence at distance " + ", ".join(map(str, DISTANCES)))
    print("=" * 72)
    print("  v1, Giza:                   0.443  0.419  0.335  0.109  0.050")
    print("  v1, Vesuvius:               0.446  0.422  0.331  0.108  0.051")
    for label, osf in (("white noise:                ", 1.0), ("noise, 1.35x oversampled:   ", 1.35)):
        bins, total, t = coherence_table(noise(1000, 1000, osf))
        print(f"  {label}" + "  ".join(f"{v:.3f}" for v in t))
    # Zero true coherence, L effective looks, and the phase ramp left between
    # two disjoint bands that are `d * bins` apart, summed over a 5-pixel window.
    looks = 3.9
    floor = np.sqrt(np.pi / (4 * looks))
    model = []
    for d in DISTANCES:
        df = d * bins / total
        model.append(floor * abs(np.sin(5 * np.pi * df) / (5 * np.sin(np.pi * df))))
    print("  model, zero true coherence: " + "  ".join(f"{v:.3f}" for v in model))
    print(f"  (the script cuts disjoint {bins}-bin slices of a {total}-bin spectrum; the model is the")
    print(f"   estimator's bias floor for about {looks} effective looks times a 5-point Dirichlet kernel)")


# ----------------------------------------------------------------------------
# Section 5.2  240-run sweep
# ----------------------------------------------------------------------------
# Geometry from the Section 3.2 table (v1 extraction). The sweep's depth axis
# is built from the withdrawn B_perp scalar; it is used here only because the
# point is to run the sweep exactly as it was run.
SWEEP_PARAMS = dict(velocity=7675.6, wavelength=0.03123, doppler_bw=29102.0,
                    slant_range=634971.0, inc_angle=np.radians(32.3), ka=5941.0)


def run_sweep(data):
    keys = list(sw.PARAM_GRID)
    return np.array([sw.run_experiment(data, SWEEP_PARAMS, **dict(zip(keys, c)))
                     for c in itertools.product(*sw.PARAM_GRID.values())])


def section_sweep():
    print()
    print("=" * 72)
    print("SECTION 5.2  240-run sweep, SNR_peak_mean")
    print("=" * 72)
    a, b = noise(800, 800), noise(800, 800)
    ma, mb = run_sweep(a), run_sweep(b)
    print("  v1:     Giza 1.118 +/- 0.077 | control 1.139 +/- 0.082 | diff -0.021 +/- 0.043 | "
          "corr 0.855 | 0 of 240 above 1.5x")
    print(f"  noise:  A    {ma.mean():.3f} +/- {ma.std():.3f} | B       {mb.mean():.3f} +/- {mb.std():.3f} | "
          f"diff {np.mean(ma - mb):+.3f} +/- {np.std(ma - mb):.3f} | corr {np.corrcoef(ma, mb)[0, 1]:.3f} | "
          f"{int(np.sum(ma > 1.5 * mb))} of {len(ma)} above 1.5x")
    print()
    print("  Positive control: one bright scatterer on the sampled line of patch A,")
    print("  scored against noise patch B with the sweep's own 1.5x criterion.")
    print("  scatterer over clutter | position on the line | configurations flagged")
    centre = a.shape[0] // 2
    for snr_db in (10, 20, 30):
        for offset in (-3, 0, 2):
            d = a.copy()
            d[centre + offset, :] += 10 ** (snr_db / 20)
            m = run_sweep(d)
            print(f"        {snr_db:2d} dB           |      {offset:+d} px          |   "
                  f"{int(np.sum(m > 1.5 * mb)):3d} of {len(m)}")
    print("  The response depends on where the scatterer sits along the line, by a few")
    print("  pixels. The sweep's 'depth' axis is that position, relabelled in metres.")


# ----------------------------------------------------------------------------
# Section 5.3  15-image inversion
# ----------------------------------------------------------------------------
B_PERP = np.array([167.3, 81.3, 146.4, 140.6, -24.3, 31.8, 37.3, 0.0, 89.6,
                   170.3, -16.5, 112.4, -41.5, 65.0, 93.1])       # Section 6.1
KZ = B_PERP * 0.0978 / (B_PERP.max() - B_PERP.min())              # kz span of Section 6.1
DEPTHS = np.arange(-300, 300, 0.5)
STEER = np.exp(1j * np.outer(KZ, DEPTHS))
PINV = np.linalg.pinv(STEER)                                      # as in multi_site_tomo.py
V1_HEIGHTS = {140: (0.106, 0.118), 43: (0.121, 0.130), 21: (0.049, 0.043), 0: (0.145, 0.145),
              -50: (0.144, 0.142), -100: (0.150, 0.146), -200: (0.203, 0.192)}


def cnoise(n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def section_multipass(trials=2000):
    print()
    print("=" * 72)
    print("SECTION 5.3  15-image inversion (pseudo-inverse, -300 to +300 m)")
    print("=" * 72)
    profiles, contrast, edge = [], [], 0
    for _ in range(trials):
        t = np.abs(PINV @ cnoise(15))
        profiles.append(t / t.max())
        contrast.append((t.max() - t.mean()) / t.std())
        edge += abs(DEPTHS[np.argmax(t)]) >= 290
    profiles = np.array(profiles)
    corr = [np.corrcoef(profiles[0], q)[0, 1] for q in profiles[1:]]
    mad = [np.mean(np.abs(profiles[0] - q)) for q in profiles[1:]]
    print("  v1:     peak contrast 8.2 to 8.6 sigma | pyramid-control correlation 0.9913 | mean abs diff 0.0106")
    print(f"  random: peak contrast {np.mean(contrast):.1f} +/- {np.std(contrast):.1f} sigma | correlation of two random "
          f"inputs {np.median(corr):.4f} (5th pct {np.percentile(corr, 5):.4f}) | mean abs diff {np.median(mad):.4f}")
    print(f"  random: peak within 10 m of the window edge in {100 * edge / trials:.0f}% of trials")
    print("  height     v1 pyramid / control     random input")
    for d, (py, ct) in V1_HEIGHTS.items():
        v = profiles[:, np.argmin(np.abs(DEPTHS - d))]
        print(f"  {d:+5d} m       {py:.3f} / {ct:.3f}           {v.mean():.3f} +/- {v.std():.3f}")
    sv = np.linalg.svd(STEER, compute_uv=False)
    print(f"  singular values of the steering matrix: largest {sv[0]:.1f}, smallest {sv[-1]:.3f}")
    print()
    print("  Positive control: one scatterer planted at +43 m with a common phase,")
    print("  plus unit noise per image. Share of trials with the peak within 32 m of +43 m.")
    print("  per-image SNR | pseudo-inverse (as in v1) | matched beamformer A^H Y")
    a = np.exp(1j * KZ * 43)
    for snr_db in (0, 10, 20, 40, 60):
        hits = {"pinv": 0, "ah": 0}
        for _ in range(trials):
            y = 10 ** (snr_db / 20) * a * np.exp(1j * rng.uniform(0, 2 * np.pi)) + cnoise(15)
            hits["pinv"] += abs(DEPTHS[np.argmax(np.abs(PINV @ y))] - 43) <= 32
            hits["ah"] += abs(DEPTHS[np.argmax(np.abs(STEER.conj().T @ y))] - 43) <= 32
        print(f"     {snr_db:3d} dB    |          {100 * hits['pinv'] / trials:5.1f}%           |        "
              f"{100 * hits['ah'] / trials:5.1f}%")
    hits = 0
    for _ in range(trials):
        y = 100 * a * np.exp(1j * rng.uniform(0, 2 * np.pi, 15)) + cnoise(15)
        hits += abs(DEPTHS[np.argmax(np.abs(STEER.conj().T @ y))] - 43) <= 32
    print(f"  40 dB scatterer with an unknown phase per image (no coregistration or")
    print(f"  flattening), matched beamformer: {100 * hits / trials:.1f}% (chance is about "
          f"{100 * 64 / 600:.0f}%)")


if __name__ == "__main__":
    section_coherence()
    section_sweep()
    section_multipass()
