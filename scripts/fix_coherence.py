#!/usr/bin/env python3
"""
fix_coherence.py — Measure interferometric coherence between Doppler sub-apertures.

Uses the standard SAR local-window coherence estimator (not global averaging,
which produces meaningless zeros).

Known limits (v3, see scripts/noise_controls.py and Section 5.1 of the note):
- The N-way sub-apertures cut here are disjoint slices of the spectrum, not the
  overlapping sub-apertures (bandwidth ratio 0.5) of the method under review.
- The carrier offset between two slices is not removed before the 5x5 window.
- Random noise reproduces the v1 coherence table to about 0.01. The table is
  the estimator's bias floor and says nothing about the scene.
The script is kept so the v1 numbers can be traced. It was not re-run for v3.

Usage:
    python3 fix_coherence.py

Requirements:
    pip install sarpy numpy scipy
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_dir import data_dir

DATA_DIR = data_dir()
FILES = {
    "giza": os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf"),
    "vesuvius": os.path.join(DATA_DIR, "umbra/vesuvius/2023-11-15-19-47-28_UMBRA-05_SICD.nitf"),
}


def _imports():
    global _sicd_io, _uniform_filter
    try:
        import sarpy.io.complex as sicd_io
        from scipy.ndimage import uniform_filter
    except ImportError:
        print("ERROR: pip install sarpy numpy scipy")
        print("No coherence was measured.")
        sys.exit(1)
    _sicd_io = sicd_io
    _uniform_filter = uniform_filter
    return sicd_io, uniform_filter


def local_coherence(img1, img2, window=5):
    """
    Standard SAR interferometric coherence estimator.
    Computes coherence in local windows, returns the coherence map.
    This is the textbook formula used in all SAR interferometry.
    """
    uniform_filter = _uniform_filter
    cross = uniform_filter(np.real(img1 * np.conj(img2)), window) + \
            1j * uniform_filter(np.imag(img1 * np.conj(img2)), window)
    pow1 = uniform_filter(np.abs(img1)**2, window)
    pow2 = uniform_filter(np.abs(img2)**2, window)

    denom = np.sqrt(pow1 * pow2)
    mask = denom > 0

    coh_map = np.zeros_like(denom)
    coh_map[mask] = np.abs(cross[mask]) / denom[mask]

    trim = window
    coh_interior = coh_map[trim:-trim, trim:-trim]

    return coh_interior


def measure_coherence(filepath, n_sub=30, patch=1000, window=5):
    print(f"\n{'='*60}")
    print(f"FILE: {filepath.split('/')[-1]}")
    print(f"N_sub={n_sub}, patch={patch}x{patch}, window={window}x{window}")
    print(f"{'='*60}")

    sicd_io, _uniform = _imports()
    reader = sicd_io.open(filepath)
    meta = reader.sicd_meta
    nrows = meta.ImageData.NumRows
    ncols = meta.ImageData.NumCols

    r0 = max(0, nrows//2 - patch//2)
    c0 = max(0, ncols//2 - patch//2)
    r1 = min(nrows, r0 + patch)
    c1 = min(ncols, c0 + patch)

    print(f"Reading [{r0}:{r1}, {c0}:{c1}]...")
    data = reader[(r0, r1, 1), (c0, c1, 1)]
    print(f"Shape: {data.shape}, dtype: {data.dtype}")

    if not np.iscomplexobj(data):
        print("ERROR: not complex")
        return None

    # 2D FFT
    print("FFT...")
    spectrum = np.fft.fftshift(np.fft.fft2(data), axes=1)
    total_az = spectrum.shape[1]

    # =============================================
    # TEST 1: Simple 2-way split
    # =============================================
    half = total_az // 2
    spec_lo = np.zeros_like(spectrum)
    spec_hi = np.zeros_like(spectrum)
    spec_lo[:, :half] = spectrum[:, :half]
    spec_hi[:, half:] = spectrum[:, half:]
    sa_lo = np.fft.ifft2(np.fft.ifftshift(spec_lo, axes=1))
    sa_hi = np.fft.ifft2(np.fft.ifftshift(spec_hi, axes=1))

    coh_map = local_coherence(sa_lo, sa_hi, window)
    mean_coh = np.mean(coh_map)
    median_coh = np.median(coh_map)
    std_coh = np.std(coh_map)

    print(f"\n2-WAY SPLIT:")
    print(f"  Mean coherence:   {mean_coh:.4f}")
    print(f"  Median coherence: {median_coh:.4f}")
    print(f"  Std coherence:    {std_coh:.4f}")
    print(f"  Min coherence:    {np.min(coh_map):.4f}")
    print(f"  Max coherence:    {np.max(coh_map):.4f}")

    phase_diff = np.angle(sa_lo * np.conj(sa_hi))
    print(f"  Phase diff std:   {np.std(phase_diff):.4f} rad")

    # =============================================
    # TEST 2: N-way sub-apertures vs reference
    # =============================================
    usable = total_az // 2
    start = total_az // 4
    bins_per = usable // n_sub

    if bins_per < 2:
        n_sub = max(2, usable // 4)
        bins_per = usable // n_sub
        print(f"\n  Reduced to {n_sub} sub-apertures ({bins_per} bins each)")

    ref_idx = n_sub // 2
    sas = []
    for i in range(n_sub):
        sa_spec = np.zeros_like(spectrum)
        s = start + i * bins_per
        e = min(s + bins_per, total_az)
        sa_spec[:, s:e] = spectrum[:, s:e]
        sas.append(np.fft.ifft2(np.fft.ifftshift(sa_spec, axes=1)))

    ref = sas[ref_idx]
    mean_cohs = []

    for i in range(n_sub):
        if i == ref_idx:
            continue
        coh_map_i = local_coherence(sas[i], ref, window)
        mean_cohs.append(np.mean(coh_map_i))

    mean_cohs = np.array(mean_cohs)

    print(f"\n{n_sub}-WAY SUB-APERTURES (vs reference SA {ref_idx}):")
    print(f"  Mean coherence:     {np.mean(mean_cohs):.4f}")
    print(f"  Min (most distant): {np.min(mean_cohs):.4f}")
    print(f"  Max (nearest):      {np.max(mean_cohs):.4f}")
    print(f"  Std:                {np.std(mean_cohs):.4f}")

    # =============================================
    # TEST 3: Coherence vs sub-aperture distance
    # =============================================
    print(f"\n  Coherence vs distance from reference:")
    for i in range(n_sub):
        if i == ref_idx:
            continue
        dist = abs(i - ref_idx)
        if dist in [1, 2, 5, 10, 14, 15, 20, 25, 29]:
            coh_i = local_coherence(sas[i], ref, window)
            print(f"    SA {i:2d} (dist={dist:2d}): coherence={np.mean(coh_i):.4f}")

    # =============================================
    # INTERPRETATION
    # =============================================
    overall = np.mean(mean_cohs)
    print(f"\n*** HOW TO READ THIS ***")
    print(f"  Overall coherence: {overall:.4f}")
    print(f"  These sub-apertures are disjoint spectral slices. For speckle they are")
    print(f"  uncorrelated, so the values printed above are the bias floor of a 5x5")
    print(f"  estimator, shaped by the carrier offset between slices. Random noise")
    print(f"  gives the same table (scripts/noise_controls.py). Do not read the")
    print(f"  numbers as evidence about the scene or about depth.")

    return overall


# ==========================================
# RUN ALL TESTS
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("SAR DOPPLER SUB-APERTURE COHERENCE VERIFICATION")
    print("Using local-window coherence estimator (standard InSAR method)")
    print("=" * 60)
    print(f"DATA_DIR = {DATA_DIR}")

    missing = [path for path in FILES.values() if not os.path.isfile(path)]
    if missing:
        print("\nLocal SICD files are absent. No coherence was measured.")
        for path in missing:
            print(f"  missing: {path}")
        print("Set SAR_DATA_DIR (or DATA_DIR). v1 coherence numbers are not recomputed.")
        sys.exit(2)

    # Test 1: Giza and Vesuvius with 30 sub-apertures
    for name, path in FILES.items():
        measure_coherence(path, n_sub=30, patch=1000, window=5)

    # Test 2: Giza with varying N_sub
    print(f"\n{'='*60}")
    print("GIZA — VARYING N_SUB (does coherence depend on sub-aperture count?)")
    print(f"{'='*60}")
    results = []
    for ns in [2, 5, 10, 20, 30]:
        c = measure_coherence(FILES["giza"], n_sub=ns, patch=600, window=5)
        if c is not None:
            results.append((ns, c))

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'N_sub':>6} | {'Mean Coherence':>15}")
    print(f"{'-'*6}-+-{'-'*15}")
    for ns, c in results:
        print(f"{ns:6d} | {c:15.4f}")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("The v1 single-pass elevation figures (~618 mm, ~285 m, ~8559 m)")
    print("are withdrawn. Azimuth sub-apertures are not an elevation aperture.")
    print("King's Chamber is ~43 m above the pavement, not a depth below grade.")
    print("")
    print("A monotonic coherence drop that is similar across geologically")
    print("distinct sites is consistent with spectral decorrelation. Similarity")
    print("is not identity, and it does not by itself rule out a small")
    print("target-dependent contribution.")
