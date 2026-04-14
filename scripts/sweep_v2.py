#!/usr/bin/env python3
"""
sweep_v2.py — Corrected parameter sweep for SAR Doppler sub-aperture tomography.

Fixes from v1:
1. Uses corrected Δx = Δf·λ·R/(2V) instead of wrong Δx = V/Δf
2. Removed seismic_vel from sweep parameters (was never used in computation)
3. Cleaner SNR metric definition matching the paper

Sweeps: n_sub × bandwidth_ratio × depth_range × filter_type
Tests both Giza pyramid and empty desert control with identical parameters.

Usage:
    python3 sweep_v2.py
"""
import numpy as np
import sarpy.io.complex as sicd_io
import itertools
import json
import sys
import os

GIZA_SICD = os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf")

# Giza pyramid center and desert control (same scene)
PATCH_SIZE = 800
GIZA_CENTER = (5251, 10136)     # (row, col) — pyramid
DESERT_CENTER = (7251, 6986)    # (row, col) — empty desert ~5km SE

# Target depth band for scoring (known pyramid chamber range)
TARGET_DEPTH_MIN = 20   # Queen's Chamber ~21m
TARGET_DEPTH_MAX = 80   # Grand Gallery upper ~60m, with margin

# Parameter grid
PARAM_GRID = {
    'n_sub':       [5, 10, 15, 20, 30, 50],
    'bdl_ratio':   [0.3, 0.4, 0.5, 0.6, 0.7],
    'depth_max':   [200, 500, 800, 1200],
    'filter_type': ['hanning', 'none'],
}


def load_data():
    """Load SICD and extract both patches."""
    print("Loading SICD...")
    reader = sicd_io.open(GIZA_SICD)
    meta = reader.sicd_meta

    vx, vy, vz = meta.SCPCOA.ARPVel.X, meta.SCPCOA.ARPVel.Y, meta.SCPCOA.ARPVel.Z
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    wavelength = 2.0 / meta.Grid.Row.KCtr
    col_bw = meta.Grid.Col.ImpRespBW
    doppler_bw = col_bw * velocity

    arp = np.array([meta.SCPCOA.ARPPos.X, meta.SCPCOA.ARPPos.Y, meta.SCPCOA.ARPPos.Z])
    scp = np.array([meta.GeoData.SCP.ECF.X, meta.GeoData.SCP.ECF.Y, meta.GeoData.SCP.ECF.Z])
    slant_range = np.linalg.norm(arp - scp)
    cos_inc = (np.linalg.norm(arp)**2 + slant_range**2 - np.linalg.norm(scp)**2) / \
              (2 * np.linalg.norm(arp) * slant_range)
    inc_angle = np.arccos(np.clip(cos_inc, -1, 1))

    # Doppler FM rate (corrected formula)
    ka = 2 * velocity**2 / (wavelength * slant_range)

    params = {
        'velocity': velocity,
        'wavelength': wavelength,
        'doppler_bw': doppler_bw,
        'slant_range': slant_range,
        'inc_angle': inc_angle,
        'ka': ka,
    }

    print(f"  Velocity: {velocity:.1f} m/s")
    print(f"  Wavelength: {wavelength*100:.3f} cm")
    print(f"  Doppler BW: {doppler_bw:.0f} Hz")
    print(f"  Doppler FM rate (ka): {ka:.1f} Hz/s")
    print(f"  Range: {slant_range/1000:.0f} km")
    print(f"  Inc angle: {np.degrees(inc_angle):.1f}°")

    # Read patches
    half = PATCH_SIZE // 2
    gr, gc = GIZA_CENTER
    dr, dc = DESERT_CENTER

    print(f"\n  Reading Giza patch ({PATCH_SIZE}x{PATCH_SIZE})...")
    giza_data = reader[(gr - half, gr + half, 1), (gc - half, gc + half, 1)]
    print(f"  Reading Desert patch ({PATCH_SIZE}x{PATCH_SIZE})...")
    desert_data = reader[(dr - half, dr + half, 1), (dc - half, dc + half, 1)]

    print(f"  Giza amp: {np.mean(np.abs(giza_data)):.6f}")
    print(f"  Desert amp: {np.mean(np.abs(desert_data)):.6f}")

    return giza_data, desert_data, params


def run_experiment(data, params, n_sub, bdl_ratio, depth_max, filter_type):
    """
    Run single-pass Doppler sub-aperture tomography on a patch.
    
    Returns the SNR metric: peak amplitude in the target depth band
    (20-80m) divided by mean amplitude across the full depth range,
    averaged over all columns in the patch.
    
    Uses CORRECTED along-track formula: Δx = Δf·λ·R/(2V)
    """
    velocity = params['velocity']
    wavelength = params['wavelength']
    doppler_bw = params['doppler_bw']
    slant_range = params['slant_range']
    inc_angle = params['inc_angle']

    # Sub-aperture parameters
    processing_bw = doppler_bw * (1 - bdl_ratio)
    doppler_step = processing_bw / n_sub

    if doppler_step < 10:
        return 0.0  # degenerate case

    # CORRECTED along-track shift
    dx = doppler_step * wavelength * slant_range / (2 * velocity)
    delta_theta = doppler_step * wavelength / (2 * velocity)
    bperp = dx * np.sin(delta_theta)

    # kz per sub-aperture step
    kz_step = 4 * np.pi * bperp / (wavelength * slant_range * np.sin(inc_angle))

    if kz_step < 1e-12:
        return 0.0

    # Build kz array
    ref_idx = n_sub // 2
    kz = np.array([(i - ref_idx) * kz_step for i in range(n_sub)])

    # Depth vector
    depth_step = max(1.0, depth_max / 500)  # at most 500 bins
    depths = np.arange(0, depth_max, depth_step)
    n_depths = len(depths)

    if n_depths < 10:
        return 0.0

    # Target depth band indices
    target_mask = (depths >= TARGET_DEPTH_MIN) & (depths <= TARGET_DEPTH_MAX)
    if not np.any(target_mask):
        return 0.0

    # Steering matrix
    A = np.exp(1j * np.outer(kz, depths))

    # Process a subset of columns for speed
    n_rows, n_cols = data.shape
    col_step = max(1, n_cols // 100)  # ~100 columns
    col_indices = range(0, n_cols, col_step)

    snr_values = []

    for col in col_indices:
        # Extract one column, FFT along azimuth
        column = data[:, col]
        if np.mean(np.abs(column)) < 1e-10:
            continue

        spectrum = np.fft.fftshift(np.fft.fft(column))
        total_bins = len(spectrum)
        usable_bins = int(total_bins * (1 - bdl_ratio))
        start_bin = int(total_bins * bdl_ratio / 2)
        bins_per_sa = usable_bins // n_sub

        if bins_per_sa < 2:
            continue

        # Extract sub-aperture complex values
        sa_values = np.zeros(n_sub, dtype=np.complex64)
        for si in range(n_sub):
            s = start_bin + si * bins_per_sa
            e = s + bins_per_sa

            # Apply window
            sa_spec = np.zeros(total_bins, dtype=np.complex64)
            if filter_type == 'hanning':
                win = np.hanning(e - s)
            else:
                win = np.ones(e - s)
            sa_spec[s:e] = spectrum[s:e] * win
            sa_signal = np.fft.ifft(np.fft.ifftshift(sa_spec))

            # Use central portion of the reconstructed signal
            center = len(sa_signal) // 2
            hw = min(5, center)
            sa_values[si] = np.mean(sa_signal[center - hw:center + hw])

        # Beamformer
        tomo = np.abs(A.conj().T @ sa_values)

        if np.max(tomo) < 1e-15:
            continue

        # SNR: peak in target band / mean of full profile
        target_peak = np.max(tomo[target_mask])
        full_mean = np.mean(tomo)

        if full_mean > 0:
            snr_values.append(target_peak / full_mean)

    if len(snr_values) == 0:
        return 0.0

    return float(np.mean(snr_values))


# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    giza_data, desert_data, params = load_data()

    # Generate parameter combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    n_total = len(combos)

    print(f"\n{'='*60}")
    print(f"PARAMETER SWEEP — {n_total} EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Parameters: {', '.join(keys)}")
    print(f"Grid sizes: {[len(v) for v in values]}")
    print(f"Target depth band: {TARGET_DEPTH_MIN}-{TARGET_DEPTH_MAX} m")
    print(f"Using corrected Δx = Δf·λ·R/(2V) formula")
    print()

    results = []
    giza_wins = 0
    n_valid = 0

    for i, combo in enumerate(combos):
        p = dict(zip(keys, combo))

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{n_total}] n_sub={p['n_sub']}, bdl={p['bdl_ratio']}, "
                  f"depth={p['depth_max']}, filt={p['filter_type']}")

        snr_giza = run_experiment(giza_data, params, **p)
        snr_desert = run_experiment(desert_data, params, **p)

        results.append({
            **p,
            'snr_giza': snr_giza,
            'snr_desert': snr_desert,
            'diff': snr_giza - snr_desert,
            'ratio': snr_giza / snr_desert if snr_desert > 0 else 0,
        })

        if snr_giza > 0 and snr_desert > 0:
            n_valid += 1
            if snr_giza > 1.5 * snr_desert:
                giza_wins += 1

    # =============================================
    # ANALYSIS
    # =============================================
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")

    valid = [r for r in results if r['snr_giza'] > 0 and r['snr_desert'] > 0]
    n_valid = len(valid)

    if n_valid == 0:
        print("ERROR: No valid experiments")
        sys.exit(1)

    giza_snrs = np.array([r['snr_giza'] for r in valid])
    desert_snrs = np.array([r['snr_desert'] for r in valid])
    diffs = giza_snrs - desert_snrs

    print(f"Valid experiments: {n_valid} of {n_total}")
    print(f"")
    print(f"Giza SNR:   mean={np.mean(giza_snrs):.4f}, std={np.std(giza_snrs):.4f}")
    print(f"Desert SNR: mean={np.mean(desert_snrs):.4f}, std={np.std(desert_snrs):.4f}")
    print(f"Difference: mean={np.mean(diffs):.4f}, std={np.std(diffs):.4f}")
    print(f"")
    print(f"Experiments where Giza > 1.5× Desert: {giza_wins} of {n_valid}")
    print(f"Max Giza SNR: {np.max(giza_snrs):.4f}")
    print(f"Max Desert SNR: {np.max(desert_snrs):.4f}")
    print(f"")

    # Correlation between Giza and Desert SNR across experiments
    corr = np.corrcoef(giza_snrs, desert_snrs)[0, 1]
    print(f"Giza-Desert SNR correlation across experiments: {corr:.4f}")

    # Top 5 experiments by Giza SNR
    print(f"\nTop 5 experiments by Giza SNR:")
    sorted_valid = sorted(valid, key=lambda r: r['snr_giza'], reverse=True)
    for r in sorted_valid[:5]:
        print(f"  Giza={r['snr_giza']:.3f} Desert={r['snr_desert']:.3f} "
              f"n_sub={r['n_sub']} bdl={r['bdl_ratio']} "
              f"depth={r['depth_max']} filt={r['filter_type']}")

    # Largest Giza-Desert difference
    print(f"\nTop 5 experiments by Giza-Desert difference:")
    sorted_diff = sorted(valid, key=lambda r: r['diff'], reverse=True)
    for r in sorted_diff[:5]:
        print(f"  Diff={r['diff']:+.4f} Giza={r['snr_giza']:.3f} Desert={r['snr_desert']:.3f} "
              f"n_sub={r['n_sub']} bdl={r['bdl_ratio']}")

    # Save
    with open('sweep_v2_results.json', 'w') as f:
        json.dump({
            'n_total': n_total,
            'n_valid': n_valid,
            'giza_mean': float(np.mean(giza_snrs)),
            'desert_mean': float(np.mean(desert_snrs)),
            'diff_mean': float(np.mean(diffs)),
            'diff_std': float(np.std(diffs)),
            'giza_wins': giza_wins,
            'correlation': float(corr),
            'formula': 'corrected: Δx = Δf·λ·R/(2V)',
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to sweep_v2_results.json")

    print(f"\n{'='*60}")
    print(f"CONCLUSION")
    print(f"{'='*60}")
    if giza_wins == 0:
        print(f"No parameter combination produced a tomographic response that")
        print(f"differentiated the pyramid from empty desert.")
    else:
        print(f"{giza_wins} of {n_valid} experiments showed Giza > 1.5× Desert.")
        print(f"Manual inspection recommended.")
