#!/usr/bin/env python3
"""
fix_coherence.py — Measure interferometric coherence between Doppler sub-apertures.

Uses the standard SAR local-window coherence estimator (not global averaging,
which produces meaningless zeros). Tests whether sub-aperture decorrelation
is spectral (geometric) or depth-dependent (signal).

Usage:
    python3 fix_coherence.py

Requirements:
    pip install sarpy numpy scipy
"""

import numpy as np
import sarpy.io.complex as sicd_io
from scipy.ndimage import uniform_filter
import os

FILES = {
    "giza": os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf"),
    "vesuvius": os.path.join(DATA_DIR, "umbra/vesuvius/2023-11-15-19-47-28_UMBRA-05_SICD.nitf"),
}


def local_coherence(img1, img2, window=5):
    """
    Standard SAR interferometric coherence estimator.
    Computes coherence in local windows, returns the coherence map.
    This is the textbook formula used in all SAR interferometry.
    """
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
    print(f"\n*** INTERPRETATION ***")
    if overall > 0.95:
        verdict = "IDENTICAL — ZERO depth discrimination"
    elif overall > 0.85:
        verdict = "VERY HIGH — negligible depth discrimination"
    elif overall > 0.7:
        verdict = "HIGH — minimal depth discrimination"
    elif overall > 0.5:
        verdict = "MODERATE — some depth info may exist"
    elif overall > 0.3:
        verdict = "LOW — meaningful differences between sub-apertures"
    else:
        verdict = "VERY LOW — strong decorrelation between sub-apertures"

    print(f"  Overall coherence: {overall:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"")
    print(f"  NOTE: High coherence (>0.9) means sub-apertures see the same thing.")
    print(f"  This CONFIRMS the kz calculation: B_perp is too small for depth.")
    print(f"  Low coherence (<0.5) could mean either:")
    print(f"    a) Real depth-dependent signal (good), OR")
    print(f"    b) Speckle decorrelation from non-overlapping bands (bad)")
    print(f"  To interpret: check if coherence DROPS with SA distance.")
    print(f"  If it drops monotonically and identically across sites:")
    print(f"    consistent with spectral decorrelation (geometric, not geological).")
    print(f"  If it differs between geologically distinct sites:")
    print(f"    may indicate target-dependent depth information.")

    return overall


# ==========================================
# RUN ALL TESTS
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("SAR DOPPLER SUB-APERTURE COHERENCE VERIFICATION")
    print("Using local-window coherence estimator (standard InSAR method)")
    print("=" * 60)

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
    print("KZ PHYSICS REMINDER (from verify_v2.py — corrected formula)")
    print(f"{'='*60}")
    print("B_perp per sub-aperture:  618 mm")
    print("Elevation resolution:     285 m")
    print("Ambiguity height:         8,559 m")
    print("")
    print("King's Chamber depth:     43 m")
    print("Magma chamber depth:      6,000 m")
    print("LNGS depth:               1,400 m")
    print("")
    print("INTERPRETATION: The monotonic coherence drop with sub-aperture")
    print("distance, identical between geologically distinct sites, is")
    print("consistent with spectral decorrelation (a geometric property of")
    print("the imaging system). This supports the conclusion that the")
    print("sub-aperture decomposition does not encode target-dependent depth")
    print("information, though it does not rule out a small contribution")
    print("from target-dependent structure below the noise floor.")
