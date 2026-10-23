#!/usr/bin/env python3
"""
sweep_v2.py — Parameter sweep of this Doppler beamformer only.

Scope (do not over-read the output):
- Averages sub-aperture spectra and applies A^H Y on sampled SLC columns.
- Does not estimate sub-pixel displacements by cross-correlation.
- Does not implement the published pseudoinverse A† Y.
- A null result here is not a failure of the full Biondi processing chain
  and is not a faithful reproduction.

Geometry notes:
- Δx = Δf·λ·R/(2V) is an along-track shift, not an elevation baseline.
- The kz step still uses the withdrawn scalar B = Δx·sin(Δθ), because that
  is the beamformer this sweep implements. Do not cite that scalar as
  single-pass elevation resolution (~285 m remains withdrawn).
- Incidence is SICD SCPCOA.IncidenceAng when present, else the angle at
  the target. A future run is not a reissue of the Zenodo v1 SNR table.

The southeast patch offsets are not geolocated in this script. v1 called
that patch empty desert; that label is withdrawn (Shabramant / settled fringe).

Usage:
    SAR_DATA_DIR=/path/to/data python3 scripts/sweep_v2.py
"""
import numpy as np
import itertools
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_dir import data_dir
from incidence import target_incidence_rad

DATA_DIR = data_dir()
GIZA_SICD = os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf")

# Pixel offsets in the Giza SICD. Not geolocated here.
PATCH_SIZE = 800
GIZA_CENTER = (5251, 10136)       # (row, col) — Khufu patch
CONTROL_CENTER = (7251, 6986)     # (row, col) — southeast control; not empty Sahara

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
    try:
        import sarpy.io.complex as sicd_io
    except ImportError:
        print("ERROR: pip install sarpy numpy")
        print("No sweep was run. v1 SNR figures are not recomputed.")
        sys.exit(1)

    print("Loading SICD...")
    print("Beamformer only: A^H Y on columns. No displacement cross-correlation. No A†Y.")
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
    inc_angle, inc_source, look_angle = target_incidence_rad(meta, arp, scp, slant_range)

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
    print(f"  Incidence: {np.degrees(inc_angle):.2f}° ({inc_source})")
    print(f"  Look angle (not incidence): {np.degrees(look_angle):.2f}°")
    print("  Southeast patch is not geolocated here and is not an empty-desert control.")

    # Read patches
    half = PATCH_SIZE // 2
    gr, gc = GIZA_CENTER
    dr, dc = CONTROL_CENTER

    print(f"\n  Reading Giza patch ({PATCH_SIZE}x{PATCH_SIZE})...")
    giza_data = reader[(gr - half, gr + half, 1), (gc - half, gc + half, 1)]
    print(f"  Reading southeast control patch ({PATCH_SIZE}x{PATCH_SIZE})...")
    desert_data = reader[(dr - half, dr + half, 1), (dc - half, dc + half, 1)]

    print(f"  Giza amp: {np.mean(np.abs(giza_data)):.6f}")
    print(f"  Desert amp: {np.mean(np.abs(desert_data)):.6f}")

    return giza_data, desert_data, params


def run_experiment(data, params, n_sub, bdl_ratio, depth_max, filter_type):
    """
    Run single-pass Doppler sub-aperture tomography on a patch.
    
    Returns SNR_peak_mean: peak amplitude in the 20-80 m band divided by
    the mean amplitude across the profile, averaged over sampled columns.

    This is not §4.3 SNR_energy. Δx below is along-track only.
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
def main():
    print(f"DATA_DIR = {DATA_DIR}")
    print("Partial beamformer sweep (A^H Y). Not the published A†Y chain.")
    print("v1 ~285 m elevation claim is not computed and stays withdrawn.")
    if not os.path.isfile(GIZA_SICD):
        print("\nLocal Giza SICD is absent. No sweep was run.")
        print(f"  missing: {GIZA_SICD}")
        print("Set SAR_DATA_DIR (or DATA_DIR). This exit does not invent SNR numbers.")
        return 2

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
    print("Metric: SNR_peak_mean (column peak/mean). Not SNR_energy.")
    print("Along-track Δx = Δf·λ·R/(2V). Not an elevation baseline.")
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
    print(f"Giza SNR_peak_mean:    mean={np.mean(giza_snrs):.4f}, std={np.std(giza_snrs):.4f}")
    print(f"Control SNR_peak_mean: mean={np.mean(desert_snrs):.4f}, std={np.std(desert_snrs):.4f}")
    print(f"Difference: mean={np.mean(diffs):.4f}, std={np.std(diffs):.4f}")
    print(f"")
    print(f"Experiments where Giza > 1.5× control: {giza_wins} of {n_valid}")
    print(f"Max Giza SNR_peak_mean: {np.max(giza_snrs):.4f}")
    print(f"Max control SNR_peak_mean: {np.max(desert_snrs):.4f}")
    print("These figures are this run only. They do not replace the v1 table")
    print("and they do not show that the full published chain fails.")
    print(f"")

    # Correlation between Giza and Desert SNR across experiments
    corr = np.corrcoef(giza_snrs, desert_snrs)[0, 1]
    print(f"Giza-control SNR_peak_mean correlation across experiments: {corr:.4f}")

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
            'formula': 'partial beamformer A^H Y; along-track Δx = Δf·λ·R/(2V); incidence from SICD or target angle',
            'scope': 'not displacement cross-correlation; not A†Y; not a full-chain reproduction',
            'metric': 'SNR_peak_mean',
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to sweep_v2_results.json")

    print(f"\n{'='*60}")
    print(f"CONCLUSION")
    print(f"{'='*60}")
    if giza_wins == 0:
        print("No parameter combination in this partial beamformer crossed")
        print("the 1.5× threshold between the pyramid patch and the control patch.")
    else:
        print(f"{giza_wins} of {n_valid} experiments showed Giza > 1.5× control.")
        print("Manual inspection recommended.")
    print("Not a claim that the full published micro-motion chain fails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
