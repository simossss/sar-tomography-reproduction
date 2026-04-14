#!/usr/bin/env python3
"""
verify.py — Independent verification of SAR Doppler Tomography physics constraints.

This script checks whether single-pass Doppler sub-aperture SAR can discriminate
depth at geological scales. It does NOT depend on any pipeline code. It reads
SICD files directly, extracts metadata, computes the kz geometry, and measures
sub-aperture coherence from first principles.

Usage:
    python3 verify.py

Requirements:
    pip install sarpy numpy
"""

import numpy as np
import sys
import os

try:
    import sarpy.io.complex as sicd_io
except ImportError:
    print("ERROR: sarpy not installed. Run: pip install sarpy")
    sys.exit(1)

# ============================================================
# FILE PATHS — edit these if they move
# ============================================================
FILES = {
    "giza_large": os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf"),
    "giza_feb07": os.path.join(DATA_DIR, "umbra/giza/2023-02-07-07-58-27_UMBRA-05_SICD.nitf"),
    "giza_feb08": os.path.join(DATA_DIR, "umbra/giza/2023-02-08-07-54-55_UMBRA-04_SICD.nitf"),
    "vesuvius":   os.path.join(DATA_DIR, "umbra/vesuvius/2023-11-15-19-47-28_UMBRA-05_SICD.nitf"),
    "mosul":      os.path.join(DATA_DIR, "umbra/mosul/2023-08-08-18-30-59_UMBRA-05_SICD.nitf"),
}


def extract_metadata(filepath):
    """Extract all relevant SAR parameters from SICD metadata."""
    print(f"\n{'='*70}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*70}")

    reader = sicd_io.open(filepath)
    meta = reader.sicd_meta

    # Image dimensions
    nrows = meta.ImageData.NumRows
    ncols = meta.ImageData.NumCols
    print(f"\nImage size: {nrows} rows x {ncols} cols")

    # Pixel spacing
    row_spacing = meta.Grid.Row.SS  # range sample spacing (meters)
    col_spacing = meta.Grid.Col.SS  # azimuth sample spacing (meters)
    print(f"Pixel spacing: {row_spacing:.4f} m (range) x {col_spacing:.4f} m (azimuth)")

    # Wavelength and frequency
    # Grid.Row.KCtr is in cycles/meter, convert to wavelength
    k_row = meta.Grid.Row.KCtr  # cycles/m in range
    wavelength = 2.0 / k_row  # SAR convention: k = 2/lambda for two-way
    freq_hz = 3e8 / wavelength
    print(f"Center frequency: {freq_hz/1e9:.3f} GHz")
    print(f"Wavelength: {wavelength*100:.3f} cm")

    # Doppler bandwidth (azimuth processing bandwidth)
    col_imp_resp_bw = meta.Grid.Col.ImpRespBW  # cycles/m
    doppler_bw_hz = col_imp_resp_bw * meta.Grid.Col.SS * meta.Grid.Col.SS
    # Better method: compute from velocity and azimuth resolution
    # ImpRespBW in cycles/m, velocity needed to convert to Hz

    # Platform velocity
    vx = meta.SCPCOA.ARPVel.X
    vy = meta.SCPCOA.ARPVel.Y
    vz = meta.SCPCOA.ARPVel.Z
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    print(f"Platform velocity: {velocity:.1f} m/s")

    # Doppler bandwidth in Hz = ImpRespBW (cycles/m) * velocity (m/s)
    doppler_bw_hz = col_imp_resp_bw * velocity
    print(f"Azimuth ImpRespBW: {col_imp_resp_bw:.4f} cycles/m")
    print(f"Doppler bandwidth: {doppler_bw_hz:.1f} Hz")

    # Azimuth resolution
    az_resolution = 1.0 / col_imp_resp_bw  # meters
    print(f"Azimuth resolution: {az_resolution:.3f} m")

    # Slant range to scene center
    arp_x = meta.SCPCOA.ARPPos.X
    arp_y = meta.SCPCOA.ARPPos.Y
    arp_z = meta.SCPCOA.ARPPos.Z
    scp_x = meta.GeoData.SCP.ECF.X
    scp_y = meta.GeoData.SCP.ECF.Y
    scp_z = meta.GeoData.SCP.ECF.Z
    slant_range = np.sqrt((arp_x - scp_x)**2 + (arp_y - scp_y)**2 + (arp_z - scp_z)**2)
    print(f"Slant range to SCP: {slant_range:.1f} m ({slant_range/1000:.1f} km)")

    # Incidence angle
    if hasattr(meta.SCPCOA, 'SideOfTrack'):
        print(f"Side of track: {meta.SCPCOA.SideOfTrack}")
    # Compute incidence angle from geometry
    # Earth radius approximation
    earth_radius = 6371000.0
    scp_dist = np.sqrt(scp_x**2 + scp_y**2 + scp_z**2)
    arp_dist = np.sqrt(arp_x**2 + arp_y**2 + arp_z**2)
    altitude = arp_dist - earth_radius
    print(f"Approximate altitude: {altitude/1000:.1f} km")

    # Grazing angle / incidence angle
    cos_inc = (arp_dist**2 + slant_range**2 - scp_dist**2) / (2 * arp_dist * slant_range)
    inc_angle_rad = np.arccos(np.clip(cos_inc, -1, 1))
    inc_angle_deg = np.degrees(inc_angle_rad)
    print(f"Approximate incidence angle: {inc_angle_deg:.1f} deg")

    # Collect type
    if hasattr(meta, 'CollectionInfo') and hasattr(meta.CollectionInfo, 'RadarMode'):
        print(f"Radar mode: {meta.CollectionInfo.RadarMode.ModeType}")

    # Range bandwidth
    range_bw = meta.Grid.Row.ImpRespBW  # cycles/m
    range_bw_hz = range_bw * 3e8 / 2  # convert to Hz
    print(f"Range bandwidth: {range_bw_hz/1e6:.1f} MHz")

    return {
        "reader": reader,
        "nrows": nrows,
        "ncols": ncols,
        "wavelength": wavelength,
        "freq_hz": freq_hz,
        "velocity": velocity,
        "doppler_bw_hz": doppler_bw_hz,
        "col_imp_resp_bw": col_imp_resp_bw,
        "slant_range": slant_range,
        "inc_angle_rad": inc_angle_rad,
        "inc_angle_deg": inc_angle_deg,
        "row_spacing": row_spacing,
        "col_spacing": col_spacing,
    }


def compute_kz_physics(meta_dict, n_subapertures=30):
    """
    Compute the kz geometry for single-pass Doppler sub-aperture tomography.
    This is the core physics check.
    """
    print(f"\n--- kz PHYSICS (N_sub = {n_subapertures}) ---")

    wavelength = meta_dict["wavelength"]
    velocity = meta_dict["velocity"]
    doppler_bw = meta_dict["doppler_bw_hz"]
    slant_range = meta_dict["slant_range"]
    inc_angle = meta_dict["inc_angle_rad"]

    # Sub-aperture Doppler step
    # Using BDL_RATIO = 0.5 (Biondi convention: leave out half the bandwidth)
    bdl_ratio = 0.5
    processing_bw = doppler_bw * (1 - bdl_ratio)
    doppler_step = processing_bw / n_subapertures
    print(f"Processing bandwidth: {processing_bw:.1f} Hz")
    print(f"Doppler step per sub-aperture: {doppler_step:.2f} Hz")

    # Along-track shift per sub-aperture step
    # Each Doppler step corresponds to a different azimuth look angle
    # delta_theta_az = doppler_step * wavelength / (2 * velocity)
    delta_theta_az = doppler_step * wavelength / (2 * velocity)
    along_track_shift = velocity / doppler_step  # temporal separation * velocity
    print(f"Delta azimuth angle per SA: {np.degrees(delta_theta_az)*3600:.4f} arcsec")
    print(f"Along-track shift per SA: {along_track_shift:.3f} m")

    # Perpendicular baseline per sub-aperture
    # The perpendicular baseline is the along-track shift projected
    # perpendicular to the line of sight
    # For a spotlight geometry, this projection is tiny because the
    # sub-aperture angular change is in the azimuth plane
    # B_perp = along_track_shift * sin(delta_theta_az / 2) for small angles
    # More precisely: B_perp per step = wavelength * slant_range * doppler_step / (2 * velocity^2 * sin(inc_angle))
    # But the geometric derivation gives:
    B_perp_per_sa = along_track_shift * np.sin(delta_theta_az)
    print(f"B_perp per sub-aperture step: {B_perp_per_sa*1000:.4f} mm")

    # Total perpendicular baseline span across all sub-apertures
    B_perp_total = B_perp_per_sa * n_subapertures
    print(f"B_perp total span: {B_perp_total*1000:.4f} mm")

    # kz per sub-aperture step
    kz_step = 4 * np.pi * B_perp_per_sa / (wavelength * slant_range * np.sin(inc_angle))
    kz_span = kz_step * n_subapertures
    print(f"kz step: {kz_step:.10f} rad/m")
    print(f"kz span: {kz_span:.10f} rad/m")

    # Ambiguity and resolution
    if kz_step > 0:
        ambiguity = 2 * np.pi / kz_step
    else:
        ambiguity = float('inf')

    if kz_span > 0:
        resolution = 2 * np.pi / kz_span
    else:
        resolution = float('inf')

    print(f"\n*** RESULTS ***")
    print(f"Vertical resolution: {resolution:.1f} m")
    print(f"Ambiguity height: {ambiguity:.1f} m")
    print(f"  (= max unambiguous depth range)")

    # Context
    print(f"\n*** CONTEXT ***")
    print(f"Pyramid King's Chamber depth: ~43 m")
    print(f"  -> {'WITHIN' if ambiguity > 43 else 'OUTSIDE'} ambiguity range")
    print(f"  -> {'RESOLVABLE' if resolution < 20 else 'NOT resolvable'} (need <20m)")
    print(f"Vesuvius magma chamber: ~6000 m")
    print(f"  -> {'WITHIN' if ambiguity > 6000 else 'OUTSIDE'} ambiguity range")
    print(f"Gran Sasso LNGS: ~1400 m")
    print(f"  -> {'WITHIN' if ambiguity > 1400 else 'OUTSIDE'} ambiguity range")

    return {
        "B_perp_per_sa": B_perp_per_sa,
        "B_perp_total": B_perp_total,
        "kz_step": kz_step,
        "kz_span": kz_span,
        "ambiguity": ambiguity,
        "resolution": resolution,
        "doppler_step": doppler_step,
        "along_track_shift": along_track_shift,
    }


def measure_coherence(reader, meta_dict, n_subapertures=30, patch_size=1000):
    """
    Directly measure coherence between Doppler sub-apertures.
    High coherence (~1.0) = sub-apertures are identical = no depth info.
    Low coherence (<0.5) = sub-apertures differ = potential depth info.
    """
    print(f"\n--- COHERENCE MEASUREMENT (N_sub={n_subapertures}, patch={patch_size}x{patch_size}) ---")

    nrows = meta_dict["nrows"]
    ncols = meta_dict["ncols"]

    # Read a patch from the center of the image
    row_start = max(0, nrows // 2 - patch_size // 2)
    col_start = max(0, ncols // 2 - patch_size // 2)
    row_end = min(nrows, row_start + patch_size)
    col_end = min(ncols, col_start + patch_size)

    print(f"Reading patch [{row_start}:{row_end}, {col_start}:{col_end}]...")
    data = reader.read_chip((slice(row_start, row_end), slice(col_start, col_end)))
    print(f"Patch shape: {data.shape}, dtype: {data.dtype}")
    print(f"Complex: {np.iscomplexobj(data)}")

    if not np.iscomplexobj(data):
        print("ERROR: Data is not complex. Cannot measure phase coherence.")
        return None

    # 2D FFT
    print("Computing 2D FFT...")
    spectrum = np.fft.fft2(data)
    spec_shape = spectrum.shape
    print(f"Spectrum shape: {spec_shape}")

    # Split azimuth (columns) into sub-apertures
    # Apply BDL_RATIO = 0.5: use only the central 50% of azimuth bandwidth
    total_az_bins = spec_shape[1]
    usable_bins = total_az_bins // 2  # central 50%
    start_bin = total_az_bins // 4    # start of usable band

    bins_per_sa = usable_bins // n_subapertures
    if bins_per_sa < 1:
        print(f"ERROR: Not enough azimuth bins for {n_subapertures} sub-apertures.")
        print(f"  Total azimuth bins: {total_az_bins}")
        print(f"  Usable bins (50%): {usable_bins}")
        print(f"  Bins per sub-aperture: {bins_per_sa}")
        # Reduce sub-aperture count
        n_subapertures = max(2, usable_bins // 4)
        bins_per_sa = usable_bins // n_subapertures
        print(f"  Reduced to {n_subapertures} sub-apertures ({bins_per_sa} bins each)")

    print(f"Azimuth bins per sub-aperture: {bins_per_sa}")

    # Generate sub-aperture images
    ref_idx = n_subapertures // 2  # reference = center sub-aperture
    subapertures = []

    for i in range(n_subapertures):
        sa_spectrum = np.zeros_like(spectrum)
        sa_start = start_bin + i * bins_per_sa
        sa_end = sa_start + bins_per_sa
        if sa_end > total_az_bins:
            sa_end = total_az_bins
        sa_spectrum[:, sa_start:sa_end] = spectrum[:, sa_start:sa_end]
        sa_image = np.fft.ifft2(sa_spectrum)
        subapertures.append(sa_image)

    print(f"Generated {len(subapertures)} sub-aperture images")

    # Measure coherence between each sub-aperture and the reference
    ref = subapertures[ref_idx]
    coherences = []
    phase_stds = []

    for i in range(n_subapertures):
        if i == ref_idx:
            continue
        sa = subapertures[i]

        # Coherence magnitude
        numerator = np.abs(np.mean(sa * np.conj(ref)))
        denominator = np.sqrt(np.mean(np.abs(sa)**2) * np.mean(np.abs(ref)**2))
        if denominator > 0:
            coh = numerator / denominator
        else:
            coh = 0.0
        coherences.append(coh)

        # Phase difference statistics
        phase_diff = np.angle(sa * np.conj(ref))
        phase_stds.append(np.std(phase_diff))

    coherences = np.array(coherences)
    phase_stds = np.array(phase_stds)

    print(f"\n*** COHERENCE RESULTS ***")
    print(f"Mean coherence: {np.mean(coherences):.4f}")
    print(f"Min coherence:  {np.min(coherences):.4f}")
    print(f"Max coherence:  {np.max(coherences):.4f}")
    print(f"Std coherence:  {np.std(coherences):.4f}")

    print(f"\n*** PHASE DIFFERENCE RESULTS ***")
    print(f"Mean phase std: {np.mean(phase_stds):.4f} rad")
    print(f"Min phase std:  {np.min(phase_stds):.4f} rad")
    print(f"Max phase std:  {np.max(phase_stds):.4f} rad")

    print(f"\n*** INTERPRETATION ***")
    mean_coh = np.mean(coherences)
    if mean_coh > 0.95:
        print(f"Coherence {mean_coh:.3f} ≈ 1.0: Sub-apertures are IDENTICAL.")
        print(f"  -> ZERO depth discrimination capability.")
        print(f"  -> Method CANNOT work for subsurface imaging.")
    elif mean_coh > 0.7:
        print(f"Coherence {mean_coh:.3f}: Sub-apertures are VERY SIMILAR.")
        print(f"  -> Minimal depth discrimination.")
        print(f"  -> Method unlikely to work.")
    elif mean_coh > 0.3:
        print(f"Coherence {mean_coh:.3f}: Sub-apertures show MODERATE differences.")
        print(f"  -> Some depth discrimination may exist.")
        print(f"  -> Investigate further.")
    else:
        print(f"Coherence {mean_coh:.3f}: Sub-apertures are SIGNIFICANTLY different.")
        print(f"  -> Meaningful depth discrimination possible.")
        print(f"  -> Method may work. Run full tomography.")

    # Also check: simple 2-way split (most basic possible test)
    print(f"\n--- SIMPLEST POSSIBLE TEST: 2-WAY SPLIT ---")
    half = total_az_bins // 2
    spec_lo = np.zeros_like(spectrum)
    spec_hi = np.zeros_like(spectrum)
    spec_lo[:, :half] = spectrum[:, :half]
    spec_hi[:, half:] = spectrum[:, half:]
    sa_lo = np.fft.ifft2(spec_lo)
    sa_hi = np.fft.ifft2(spec_hi)

    num2 = np.abs(np.mean(sa_lo * np.conj(sa_hi)))
    den2 = np.sqrt(np.mean(np.abs(sa_lo)**2) * np.mean(np.abs(sa_hi)**2))
    coh2 = num2 / den2 if den2 > 0 else 0
    phase2 = np.angle(sa_lo * np.conj(sa_hi))

    print(f"2-way coherence: {coh2:.4f}")
    print(f"2-way phase std: {np.std(phase2):.4f} rad")
    if coh2 > 0.95:
        print(f"  -> Even splitting the ENTIRE bandwidth in half produces identical images.")
        print(f"  -> No depth info exists in this data at any sub-aperture count.")

    return {
        "coherences": coherences,
        "phase_stds": phase_stds,
        "mean_coherence": np.mean(coherences),
        "two_way_coherence": coh2,
    }


def run_verification(filepath, n_subapertures=30, patch_size=1000):
    """Run the complete verification on one SICD file."""
    if not os.path.exists(filepath):
        print(f"FILE NOT FOUND: {filepath}")
        return

    # Step 1: Extract metadata
    meta = extract_metadata(filepath)

    # Step 2: Compute kz physics
    kz = compute_kz_physics(meta, n_subapertures)

    # Step 3: Measure coherence
    coh = measure_coherence(meta["reader"], meta, n_subapertures, patch_size)

    # Step 4: Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    print(f"Doppler bandwidth:    {meta['doppler_bw_hz']:.1f} Hz")
    print(f"B_perp per SA:        {kz['B_perp_per_sa']*1000:.4f} mm")
    print(f"B_perp total:         {kz['B_perp_total']*1000:.4f} mm")
    print(f"kz step:              {kz['kz_step']:.10f} rad/m")
    print(f"Vertical resolution:  {kz['resolution']:.1f} m")
    print(f"Ambiguity height:     {kz['ambiguity']:.1f} m")
    if coh:
        print(f"Mean coherence:       {coh['mean_coherence']:.4f}")
        print(f"2-way coherence:      {coh['two_way_coherence']:.4f}")

    verdict = "CANNOT" if (coh and coh['mean_coherence'] > 0.9) else "MIGHT"
    print(f"\nVERDICT: Single-pass Doppler sub-aperture tomography {verdict}")
    print(f"         discriminate depth with this data.")

    return meta, kz, coh


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SAR DOPPLER TOMOGRAPHY — INDEPENDENT PHYSICS VERIFICATION")
    print("=" * 70)
    print("This script checks whether single-pass Doppler sub-aperture")
    print("SAR can discriminate depth. It reads SICD metadata and measures")
    print("sub-aperture coherence directly. No pipeline dependency.")
    print("=" * 70)

    # Run on Giza (primary target)
    print("\n\n" + "#" * 70)
    print("# TEST 1: GIZA (primary target)")
    print("#" * 70)
    run_verification(
        FILES["giza_large"],
        n_subapertures=30,
        patch_size=1000
    )

    # Run on Vesuvius (secondary target)
    print("\n\n" + "#" * 70)
    print("# TEST 2: VESUVIUS (secondary target)")
    print("#" * 70)
    run_verification(
        FILES["vesuvius"],
        n_subapertures=30,
        patch_size=1000
    )

    # Run on Giza with different sub-aperture counts
    # to show the result doesn't depend on N_sub
    print("\n\n" + "#" * 70)
    print("# TEST 3: GIZA — varying sub-aperture count")
    print("#" * 70)
    for n_sub in [2, 5, 10, 30, 50]:
        print(f"\n>>> N_sub = {n_sub}")
        meta = extract_metadata(FILES["giza_large"])
        kz = compute_kz_physics(meta, n_sub)
        coh = measure_coherence(meta["reader"], meta, n_sub, patch_size=600)
        print(f"    kz_step={kz['kz_step']:.10f}, ambiguity={kz['ambiguity']:.1f}m, coherence={coh['mean_coherence']:.4f}")

    print("\n\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("""
KEY QUESTIONS ANSWERED:

1. What is B_perp per sub-aperture?
   -> If ~1mm: method cannot work (confirmed by both kz math and coherence)
   -> If >1m: method might work (investigate further)

2. What is sub-aperture coherence?
   -> If ~1.0: sub-apertures are identical, no depth info exists
   -> If <0.5: sub-apertures differ, depth info may exist

3. Does the result change with N_sub?
   -> If coherence stays ~1.0 for all N_sub: fundamental limitation
   -> If coherence drops for some N_sub: parameter-dependent, worth tuning

If ALL tests show B_perp ~1mm and coherence ~1.0, the Biondi-Malanga
method as published cannot image subsurface structures from single-pass
SAR data. This is a geometry constraint, not an implementation issue.
""")
