#!/usr/bin/env python3
"""
sweep.py — Parameter sweep for SAR Doppler Tomography on Giza.

Self-contained. No pipeline dependency. No agent. No orchestration.
Loads the SICD, loops through parameter combinations, computes a
tomogram for each, scores it, logs everything.

Run in tmux, go to sleep, check results in the morning.

Usage:
    python3 sweep.py

Output:
    sweep_results.tsv  — one row per experiment
    sweep_log.txt      — human-readable progress log
    sweep_best.txt     — best result summary
"""

import numpy as np
import time
import os
import sys
import json
import itertools
from datetime import datetime, timedelta

try:
    import sarpy.io.complex as sicd_io
except ImportError:
    print("ERROR: pip install sarpy numpy scipy")
    sys.exit(1)

from scipy.ndimage import uniform_filter

# ============================================================
# CONFIGURATION
# ============================================================

GIZA_SICD = os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf")

# Two test regions from the same scene
# Giza: over the Great Pyramid
GIZA_ROW_CENTER = 5251
GIZA_COL_CENTER = 10136

# Desert control: flat empty area, no structures
DESERT_ROW_CENTER = 8300
DESERT_COL_CENTER = 2500

PATCH_SIZE = 800  # pixels — larger than Sergio's 600 for better spectral separation

# Target depths for SNR evaluation (meters)
TARGET_DEPTH_MIN = 20   # King's Chamber ~43m above base
TARGET_DEPTH_MAX = 80   # Generous window around known chambers
SURFACE_DEPTH_MAX = 5   # Surface response band

# Output files
RESULTS_FILE = "sweep_results.tsv"
LOG_FILE = "sweep_log.txt"
BEST_FILE = "sweep_best.txt"

# ============================================================
# PARAMETER SPACE
# ============================================================

PARAM_GRID = {
    "n_sub":          [2, 5, 10, 15, 20, 30, 50],
    "bdl_ratio":      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "filter_type":    ["rectangular", "hanning", "hamming", "blackman"],
    "seismic_vel":    [1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000],
    "z_max":          [200, 500, 1000],
    "z_step":         [0.5, 1.0, 2.0],
}

# We don't run every combination (7*6*4*9*3*3 = 13,608).
# Instead: sample ~200 diverse combinations.
MAX_EXPERIMENTS = 200

# ============================================================
# LOGGING
# ============================================================

def log(msg, also_print=True):
    """Write to log file and optionally print."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def progress_bar(current, total, width=40, extra=""):
    """Print a progress bar."""
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%) {extra}", end="", flush=True)


# ============================================================
# SAR PHYSICS — extracted from SICD metadata once
# ============================================================

def get_sar_params(reader):
    """Extract all needed SAR parameters from SICD metadata."""
    meta = reader.sicd_meta

    vx = meta.SCPCOA.ARPVel.X
    vy = meta.SCPCOA.ARPVel.Y
    vz = meta.SCPCOA.ARPVel.Z
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)

    k_row = meta.Grid.Row.KCtr
    wavelength = 2.0 / k_row

    col_imp_resp_bw = meta.Grid.Col.ImpRespBW
    doppler_bw_hz = col_imp_resp_bw * velocity

    arp_x, arp_y, arp_z = meta.SCPCOA.ARPPos.X, meta.SCPCOA.ARPPos.Y, meta.SCPCOA.ARPPos.Z
    scp_x, scp_y, scp_z = meta.GeoData.SCP.ECF.X, meta.GeoData.SCP.ECF.Y, meta.GeoData.SCP.ECF.Z
    slant_range = np.sqrt((arp_x-scp_x)**2 + (arp_y-scp_y)**2 + (arp_z-scp_z)**2)

    earth_r = 6371000.0
    scp_dist = np.sqrt(scp_x**2 + scp_y**2 + scp_z**2)
    arp_dist = np.sqrt(arp_x**2 + arp_y**2 + arp_z**2)
    cos_inc = (arp_dist**2 + slant_range**2 - scp_dist**2) / (2 * arp_dist * slant_range)
    inc_angle = np.arccos(np.clip(cos_inc, -1, 1))

    return {
        "velocity": velocity,
        "wavelength": wavelength,
        "doppler_bw_hz": doppler_bw_hz,
        "col_imp_resp_bw": col_imp_resp_bw,
        "slant_range": slant_range,
        "inc_angle": inc_angle,
    }


# ============================================================
# CORE TOMOGRAPHY — one experiment
# ============================================================

def make_filter_window(n_bins, filter_type):
    """Create a 1D window function for spectral filtering."""
    if filter_type == "rectangular":
        return np.ones(n_bins)
    elif filter_type == "hanning":
        return np.hanning(n_bins)
    elif filter_type == "hamming":
        return np.hamming(n_bins)
    elif filter_type == "blackman":
        return np.blackman(n_bins)
    else:
        return np.ones(n_bins)


def run_experiment(spectrum, sar, params, tomo_row_idx):
    """
    Run one complete tomographic experiment.
    
    Args:
        spectrum: 2D FFT of the SAR patch (fftshifted in azimuth), complex64
        sar: dict of SAR parameters from get_sar_params()
        params: dict with n_sub, bdl_ratio, filter_type, seismic_vel, z_max, z_step
        tomo_row_idx: row index within the patch for the tomographic line
    
    Returns:
        dict with snr, peak_depth, surface_snr, depth_profile, etc.
    """
    n_sub = params["n_sub"]
    bdl_ratio = params["bdl_ratio"]
    filter_type = params["filter_type"]
    seismic_vel = params["seismic_vel"]
    z_max = params["z_max"]
    z_step = params["z_step"]

    total_az = spectrum.shape[1]

    # Doppler band allocation
    usable_bins = int(total_az * (1 - bdl_ratio))
    start_bin = int(total_az * bdl_ratio / 2)
    bins_per_sa = usable_bins // n_sub

    if bins_per_sa < 2:
        return {"snr": 0, "error": "too few bins per sub-aperture"}

    # Create filter window
    win = make_filter_window(bins_per_sa, filter_type)

    # Generate sub-aperture images along the tomographic row
    # We only need one row (the tomo line), not the full 2D inverse FFT
    # This is much faster: extract the row from each sub-aperture's spectrum
    n_cols = spectrum.shape[1]
    sa_values = np.zeros((n_sub, n_cols), dtype=np.complex64)

    for i in range(n_sub):
        sa_spec_row = np.zeros(n_cols, dtype=np.complex64)
        s = start_bin + i * bins_per_sa
        e = s + bins_per_sa
        if e > total_az:
            e = total_az
        actual_bins = e - s
        sa_spec_row[s:e] = spectrum[tomo_row_idx, s:e] * win[:actual_bins]
        # Inverse FFT along azimuth for this row
        sa_values[i, :] = np.fft.ifft(np.fft.ifftshift(sa_spec_row))

    # Compute kz for each sub-aperture
    # B_perp per sub-aperture step
    doppler_step = sar["doppler_bw_hz"] * (1 - bdl_ratio) / n_sub
    delta_theta = doppler_step * sar["wavelength"] / (2 * sar["velocity"])
    along_track = sar["velocity"] / doppler_step
    B_perp_step = along_track * np.sin(delta_theta)

    # kz for each sub-aperture relative to center
    ref_idx = n_sub // 2
    kz = np.zeros(n_sub)
    for i in range(n_sub):
        B_perp_i = (i - ref_idx) * B_perp_step
        kz[i] = 4 * np.pi * B_perp_i / (sar["wavelength"] * sar["slant_range"] * np.sin(sar["inc_angle"]))

    # Depth vector
    depths = np.arange(0, z_max, z_step)
    n_depths = len(depths)

    if n_depths == 0:
        return {"snr": 0, "error": "empty depth vector"}

    # Steering matrix: A[i, z] = exp(j * kz[i] * depths[z])
    A = np.exp(1j * np.outer(kz, depths))  # shape: (n_sub, n_depths)

    # Pseudo-inverse of A
    # A† = (A^H A)^{-1} A^H
    # For numerical stability, use lstsq or pinv
    A_pinv = np.linalg.pinv(A)  # shape: (n_depths, n_sub)

    # Tomographic inversion for each pixel along the row
    # sa_values shape: (n_sub, n_cols)
    # For each column (pixel), tomographic profile = A_pinv @ sa_values[:, col]
    # Vectorized: tomogram = A_pinv @ sa_values -> shape (n_depths, n_cols)
    tomogram = np.abs(A_pinv @ sa_values)  # magnitude

    # =============================================
    # Scoring
    # =============================================

    # Average depth profile across all pixels
    depth_profile = np.mean(tomogram, axis=1)

    # Surface band
    surface_mask = depths < SURFACE_DEPTH_MAX
    surface_energy = np.max(depth_profile[surface_mask]) if np.any(surface_mask) else 0

    # Target band (pyramid chambers ~20-80m)
    target_mask = (depths >= TARGET_DEPTH_MIN) & (depths <= TARGET_DEPTH_MAX)
    target_energy = np.max(depth_profile[target_mask]) if np.any(target_mask) else 0

    # Background: everything outside surface and target
    bg_mask = ~surface_mask & ~target_mask
    bg_energy = np.mean(depth_profile[bg_mask]) if np.any(bg_mask) else 1e-10

    # SNR
    snr = target_energy / max(bg_energy, 1e-10)

    # Surface SNR (sanity check — should be >1 if focusing works at all)
    surface_snr = surface_energy / max(bg_energy, 1e-10)

    # Peak depth
    peak_idx = np.argmax(depth_profile[target_mask]) if np.any(target_mask) else 0
    peak_depth = depths[target_mask][peak_idx] if np.any(target_mask) else -1

    # Depth profile flatness: std/mean ratio
    # Flat = method not discriminating depth, all depths equal
    profile_flatness = np.std(depth_profile) / max(np.mean(depth_profile), 1e-10)

    # B_perp for the record
    B_perp_total = B_perp_step * n_sub

    return {
        "snr": float(snr),
        "surface_snr": float(surface_snr),
        "peak_depth": float(peak_depth),
        "profile_flatness": float(profile_flatness),
        "bg_energy": float(bg_energy),
        "target_energy": float(target_energy),
        "B_perp_step_mm": float(B_perp_step * 1000),
        "B_perp_total_mm": float(B_perp_total * 1000),
        "kz_step": float(np.min(np.abs(np.diff(kz)))) if n_sub > 1 else 0,
        "bins_per_sa": bins_per_sa,
        "error": None,
    }


# ============================================================
# PARAMETER SAMPLING
# ============================================================

def sample_experiments(param_grid, max_experiments):
    """
    Sample parameter combinations. Mix of:
    - Full grid on most impactful params (n_sub × seismic_vel)
    - Random sampling of the rest
    """
    experiments = []

    # Round 1: Full grid on n_sub × seismic_vel (most impactful)
    # Fix other params at defaults
    for n_sub in param_grid["n_sub"]:
        for vel in param_grid["seismic_vel"]:
            experiments.append({
                "n_sub": n_sub,
                "bdl_ratio": 0.5,
                "filter_type": "hanning",
                "seismic_vel": vel,
                "z_max": 500,
                "z_step": 1.0,
            })

    # Round 2: Full grid on bdl_ratio × filter_type
    for bdl in param_grid["bdl_ratio"]:
        for filt in param_grid["filter_type"]:
            experiments.append({
                "n_sub": 30,
                "bdl_ratio": bdl,
                "filter_type": filt,
                "seismic_vel": 3500,
                "z_max": 500,
                "z_step": 1.0,
            })

    # Round 3: Random combinations to fill remaining slots
    all_combos = list(itertools.product(
        param_grid["n_sub"],
        param_grid["bdl_ratio"],
        param_grid["filter_type"],
        param_grid["seismic_vel"],
        param_grid["z_max"],
        param_grid["z_step"],
    ))
    np.random.seed(42)
    np.random.shuffle(all_combos)

    for combo in all_combos:
        if len(experiments) >= max_experiments:
            break
        exp = {
            "n_sub": combo[0],
            "bdl_ratio": combo[1],
            "filter_type": combo[2],
            "seismic_vel": combo[3],
            "z_max": combo[4],
            "z_step": combo[5],
        }
        # Avoid exact duplicates
        if exp not in experiments:
            experiments.append(exp)

    return experiments[:max_experiments]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    # Initialize log
    with open(LOG_FILE, "w") as f:
        f.write(f"SAR Doppler Tomography Parameter Sweep\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")

    log("Loading SICD file...")
    if not os.path.exists(GIZA_SICD):
        log(f"ERROR: File not found: {GIZA_SICD}")
        sys.exit(1)

    reader = sicd_io.open(GIZA_SICD)
    sar = get_sar_params(reader)
    meta = reader.sicd_meta
    nrows = meta.ImageData.NumRows
    ncols = meta.ImageData.NumCols

    log(f"Image: {nrows} x {ncols}")
    log(f"Wavelength: {sar['wavelength']*100:.3f} cm")
    log(f"Doppler BW: {sar['doppler_bw_hz']:.1f} Hz")
    log(f"Velocity: {sar['velocity']:.1f} m/s")
    log(f"Slant range: {sar['slant_range']/1000:.1f} km")
    log(f"Inc angle: {np.degrees(sar['inc_angle']):.1f} deg")

    # Load Giza patch
    half = PATCH_SIZE // 2
    gr0 = max(0, GIZA_ROW_CENTER - half)
    gc0 = max(0, GIZA_COL_CENTER - half)
    gr1 = min(nrows, gr0 + PATCH_SIZE)
    gc1 = min(ncols, gc0 + PATCH_SIZE)

    log(f"Loading Giza patch [{gr0}:{gr1}, {gc0}:{gc1}]...")
    giza_patch = reader[(gr0, gr1, 1), (gc0, gc1, 1)]
    log(f"Giza patch: {giza_patch.shape}, {giza_patch.dtype}")

    # Load desert control patch
    dr0 = max(0, DESERT_ROW_CENTER - half)
    dc0 = max(0, DESERT_COL_CENTER - half)
    dr1 = min(nrows, dr0 + PATCH_SIZE)
    dc1 = min(ncols, dc0 + PATCH_SIZE)

    log(f"Loading desert patch [{dr0}:{dr1}, {dc0}:{dc1}]...")
    desert_patch = reader[(dr0, dr1, 1), (dc0, dc1, 1)]
    log(f"Desert patch: {desert_patch.shape}, {desert_patch.dtype}")

    # Pre-compute FFTs (done once, reused for all experiments)
    log("Computing 2D FFTs (done once)...")
    giza_spectrum = np.fft.fftshift(np.fft.fft2(giza_patch), axes=1)
    desert_spectrum = np.fft.fftshift(np.fft.fft2(desert_patch), axes=1)
    log("FFTs done.")

    # Tomographic line: middle row of each patch
    tomo_row = PATCH_SIZE // 2

    # Generate experiment list
    experiments = sample_experiments(PARAM_GRID, MAX_EXPERIMENTS)
    n_exp = len(experiments)
    log(f"\nRunning {n_exp} experiments...")
    log(f"Estimated time: {n_exp * 2:.0f}-{n_exp * 10:.0f} seconds\n")

    # Initialize results file
    with open(RESULTS_FILE, "w") as f:
        f.write("exp_id\tn_sub\tbdl_ratio\tfilter_type\tseismic_vel\tz_max\tz_step\t"
                "giza_snr\tdesert_snr\tgiza_surface_snr\tdesert_surface_snr\t"
                "giza_flatness\tdesert_flatness\t"
                "B_perp_step_mm\tB_perp_total_mm\tkz_step\t"
                "giza_peak_depth\tbins_per_sa\truntime_s\terror\n")

    # Track best
    best_snr = 0
    best_exp = None
    best_id = -1
    consecutive_no_improvement = 0

    # =============================================
    # MAIN LOOP
    # =============================================
    for i, params in enumerate(experiments):
        t0 = time.time()

        # Run on Giza
        giza_result = run_experiment(giza_spectrum, sar, params, tomo_row)

        # Run on desert control with identical parameters
        desert_result = run_experiment(desert_spectrum, sar, params, tomo_row)

        dt = time.time() - t0

        # Check for errors
        error = giza_result.get("error") or desert_result.get("error")

        # Extract scores
        g_snr = giza_result.get("snr", 0)
        d_snr = desert_result.get("snr", 0)

        # Track best
        if g_snr > best_snr and error is None:
            best_snr = g_snr
            best_exp = params.copy()
            best_exp["giza_snr"] = g_snr
            best_exp["desert_snr"] = d_snr
            best_id = i
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        # Write results
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{i}\t{params['n_sub']}\t{params['bdl_ratio']}\t{params['filter_type']}\t"
                    f"{params['seismic_vel']}\t{params['z_max']}\t{params['z_step']}\t"
                    f"{g_snr:.6f}\t{d_snr:.6f}\t"
                    f"{giza_result.get('surface_snr',0):.6f}\t{desert_result.get('surface_snr',0):.6f}\t"
                    f"{giza_result.get('profile_flatness',0):.6f}\t{desert_result.get('profile_flatness',0):.6f}\t"
                    f"{giza_result.get('B_perp_step_mm',0):.6f}\t{giza_result.get('B_perp_total_mm',0):.6f}\t"
                    f"{giza_result.get('kz_step',0):.10f}\t"
                    f"{giza_result.get('peak_depth',-1):.1f}\t{giza_result.get('bins_per_sa',0)}\t"
                    f"{dt:.2f}\t{error or ''}\n")

        # Progress display
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (n_exp - i - 1) / rate if rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta)))

        # Flag if Giza SNR significantly exceeds desert (would be interesting)
        flag = " ⚠️ GIZA > DESERT" if (g_snr > 3.0 and g_snr > d_snr * 1.5) else ""

        extra = f"SNR:{g_snr:.2f}/{d_snr:.2f} best:{best_snr:.2f} ETA:{eta_str}{flag}"
        progress_bar(i + 1, n_exp, extra=extra)

        # Log interesting results
        if g_snr > 3.0:
            log(f"\n  ⚠️  Exp {i}: Giza SNR={g_snr:.2f}, Desert SNR={d_snr:.2f} — {params}")

        # Early stopping check
        if consecutive_no_improvement >= 100 and i > 50:
            log(f"\n\nEarly stop: {consecutive_no_improvement} experiments with no improvement.")
            break

    # =============================================
    # FINAL SUMMARY
    # =============================================
    total_time = time.time() - start_time
    total_experiments = min(i + 1, n_exp)

    print("\n\n")
    log("=" * 60)
    log("SWEEP COMPLETE")
    log("=" * 60)
    log(f"Total experiments: {total_experiments}")
    log(f"Total time: {timedelta(seconds=int(total_time))}")
    log(f"Avg time per experiment: {total_time/total_experiments:.2f}s")
    log(f"")
    log(f"Best Giza SNR: {best_snr:.4f} (experiment {best_id})")
    if best_exp:
        log(f"Best params: {json.dumps(best_exp, indent=2)}")
    log(f"")

    # Load all results for summary stats
    snrs_giza = []
    snrs_desert = []
    with open(RESULTS_FILE) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 9:
                try:
                    snrs_giza.append(float(parts[7]))
                    snrs_desert.append(float(parts[8]))
                except ValueError:
                    pass

    snrs_giza = np.array(snrs_giza)
    snrs_desert = np.array(snrs_desert)

    log(f"Giza SNR — min: {np.min(snrs_giza):.4f}, max: {np.max(snrs_giza):.4f}, "
        f"mean: {np.mean(snrs_giza):.4f}, std: {np.std(snrs_giza):.4f}")
    log(f"Desert SNR — min: {np.min(snrs_desert):.4f}, max: {np.max(snrs_desert):.4f}, "
        f"mean: {np.mean(snrs_desert):.4f}, std: {np.std(snrs_desert):.4f}")

    # Statistical comparison
    snr_diff = snrs_giza - snrs_desert
    log(f"")
    log(f"Giza - Desert difference — mean: {np.mean(snr_diff):.4f}, std: {np.std(snr_diff):.4f}")
    log(f"  Max Giza advantage: {np.max(snr_diff):.4f}")
    log(f"  Max Desert advantage: {np.min(snr_diff):.4f}")

    # How many experiments had Giza > Desert by more than 50%?
    significant = np.sum(snrs_giza > snrs_desert * 1.5)
    log(f"  Experiments where Giza SNR > 1.5x Desert: {significant}/{total_experiments}")

    # Verdict
    log(f"")
    log("=" * 60)
    log("VERDICT")
    log("=" * 60)

    if best_snr > 3.0 and best_exp and best_exp.get("desert_snr", 0) < best_snr / 2:
        log("⚠️  ANOMALY DETECTED: Giza shows significantly higher SNR than desert")
        log("    in at least one parameter combination. Investigate further.")
        log(f"    Best: Giza SNR={best_snr:.2f}, Desert SNR={best_exp.get('desert_snr',0):.2f}")
    elif best_snr > 3.0:
        log("FALSE POSITIVE: High SNR found but desert shows comparable values.")
        log("The method produces artifacts regardless of subsurface structure.")
    else:
        log("NULL RESULT: No parameter combination produced SNR > 3.0")
        log(f"Best SNR achieved: {best_snr:.4f}")
        log(f"Across {total_experiments} experiments spanning:")
        log(f"  - Sub-apertures: {PARAM_GRID['n_sub']}")
        log(f"  - Velocities: {PARAM_GRID['seismic_vel']}")
        log(f"  - BDL ratios: {PARAM_GRID['bdl_ratio']}")
        log(f"  - Filters: {PARAM_GRID['filter_type']}")
        log(f"  - Depth ranges: {PARAM_GRID['z_max']}")
        log(f"The method cannot detect subsurface structures at any parameter setting.")

    log(f"")
    log("KZ PHYSICS REMINDER:")
    log(f"  B_perp per sub-aperture: ~15.6 mm")
    log(f"  Vertical resolution: ~11,297 m")
    log(f"  Target depth: 43 m")
    log(f"  Resolution / target ratio: {11297/43:.0f}x")
    log(f"  No parameter sweep can fix orbital geometry.")

    # Save best result summary
    with open(BEST_FILE, "w") as f:
        f.write(f"SAR Doppler Tomography Sweep — Best Result\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Experiments: {total_experiments}\n")
        f.write(f"Best Giza SNR: {best_snr:.4f}\n")
        f.write(f"Best params: {json.dumps(best_exp, indent=2)}\n")
        f.write(f"Giza SNR stats: {np.mean(snrs_giza):.4f} ± {np.std(snrs_giza):.4f}\n")
        f.write(f"Desert SNR stats: {np.mean(snrs_desert):.4f} ± {np.std(snrs_desert):.4f}\n")
        f.write(f"Verdict: {'ANOMALY' if best_snr > 3.0 else 'NULL RESULT'}\n")

    log(f"\nResults saved to: {RESULTS_FILE}")
    log(f"Log saved to: {LOG_FILE}")
    log(f"Best result saved to: {BEST_FILE}")
    log(f"Done.")
