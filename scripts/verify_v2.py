#!/usr/bin/env python3
"""
verify_v2.py — Metadata and along-track geometry for one spotlight SLC.

What this script does
- Reads SICD metadata when a local file is present.
- Reports along-track Doppler quantities (velocity, bandwidth, ka, Δx).
- Reads ground incidence from SICD SCPCOA.IncidenceAng when present.
  If that field is missing, it uses the geometric angle at the target.
  The angle at the spacecraft is printed only as a look-angle diagnostic.

What this script does not do
- It does not treat B_perp = Δx·sin(Δθ) as an elevation baseline.
  That v1 construction is printed, labeled withdrawn, and is not a result.
- It does not reinstate the ~285 m single-pass elevation-resolution claim.
  A single monostatic SAR image has no elevation aperture: azimuth
  sub-apertures are not B_perp.
- It does not reproduce the published micro-motion chain
  (no displacement cross-correlation, no A†Y).

Usage:
    SAR_DATA_DIR=/path/to/data python3 scripts/verify_v2.py

Exit code 2 means the local SICD files are absent. That is not a measurement.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_dir import data_dir
from incidence import target_incidence_rad

DATA_DIR = data_dir()

FILES = {
    "giza": os.path.join(DATA_DIR, "umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf"),
    "vesuvius": os.path.join(DATA_DIR, "umbra/vesuvius/2023-11-15-19-47-28_UMBRA-05_SICD.nitf"),
}


def _open_reader(filepath):
    try:
        import sarpy.io.complex as sicd_io
    except ImportError:
        print("ERROR: pip install sarpy numpy")
        print("No scene was read. No resolution figure was computed.")
        sys.exit(1)
    return sicd_io.open(filepath)


def analyze(filepath, n_sub=30, bdl_ratio=0.5):
    print(f"\n{'='*70}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*70}")

    reader = _open_reader(filepath)
    meta = reader.sicd_meta

    vx, vy, vz = meta.SCPCOA.ARPVel.X, meta.SCPCOA.ARPVel.Y, meta.SCPCOA.ARPVel.Z
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    k_row = meta.Grid.Row.KCtr
    wavelength = 2.0 / k_row
    col_bw = meta.Grid.Col.ImpRespBW
    doppler_bw = col_bw * velocity

    arp = np.array([meta.SCPCOA.ARPPos.X, meta.SCPCOA.ARPPos.Y, meta.SCPCOA.ARPPos.Z])
    scp = np.array([meta.GeoData.SCP.ECF.X, meta.GeoData.SCP.ECF.Y, meta.GeoData.SCP.ECF.Z])
    slant_range = np.linalg.norm(arp - scp)

    inc_angle, inc_source, look_angle = target_incidence_rad(meta, arp, scp, slant_range)
    azimuth_res = velocity / doppler_bw if doppler_bw else float("nan")

    print("\nSAR parameters (descriptive; not an elevation resolution):")
    print(f"  Velocity:       {velocity:.1f} m/s")
    print(f"  Wavelength:     {wavelength*100:.3f} cm")
    print(f"  Doppler BW:     {doppler_bw:.1f} Hz")
    print(f"  Slant range:    {slant_range/1000:.1f} km")
    print(f"  Incidence:      {np.degrees(inc_angle):.2f}°  ({inc_source})")
    print(f"  Look angle:     {np.degrees(look_angle):.2f}°  (spacecraft; not incidence)")
    print(f"  Azimuth res:    {azimuth_res:.3f} m  (V / Doppler BW)")

    processing_bw = doppler_bw * (1 - bdl_ratio)
    doppler_step = processing_bw / n_sub
    ka = 2 * velocity**2 / (wavelength * slant_range)

    print(f"\nSub-aperture configuration (azimuth):")
    print(f"  N_sub:          {n_sub}")
    print(f"  BDL ratio:      {bdl_ratio}")
    print(f"  Processing BW:  {processing_bw:.1f} Hz")
    print(f"  Doppler step:   {doppler_step:.2f} Hz")
    print(f"  ka = 2V²/(λR) = {ka:.2f} Hz/s")

    # Along-track shift from the Doppler FM rate. This is an azimuth
    # distance, not an elevation baseline.
    dx = doppler_step * wavelength * slant_range / (2 * velocity)
    dx_old = velocity / doppler_step
    delta_theta = doppler_step * wavelength / (2 * velocity)

    print(f"\nAlong-track shift per sub-aperture step:")
    print(f"  Δx = Δf·λ·R/(2V) = {dx:.1f} m")
    print(f"  v1 wrong Δx = V/Δf = {dx_old:.1f} m")
    print(f"\nAzimuth look-angle change per SA step:")
    print(f"  Δθ = {np.degrees(delta_theta):.4f}° ({np.degrees(delta_theta)*3600:.1f} arcsec)")

    # Withdrawn v1 scalar. Kept so the arithmetic can be inspected.
    # It is not B_perp of SAR tomography.
    b_withdrawn = dx * np.sin(delta_theta)
    kz_withdrawn = 4 * np.pi * b_withdrawn / (wavelength * slant_range * np.sin(inc_angle))
    kz_span = kz_withdrawn * n_sub
    res_withdrawn = 2 * np.pi / kz_span if kz_span > 0 else float("inf")
    amb_withdrawn = 2 * np.pi / kz_withdrawn if kz_withdrawn > 0 else float("inf")

    print(f"\n{'='*70}")
    print("WITHDRAWN v1 SCALAR (not an elevation aperture, not a result)")
    print(f"{'='*70}")
    print("B = Δx·sin(Δθ) projects along-track diversity into a scalar.")
    print("That scalar is not the tomographic baseline normal to the")
    print("flight track and the line of sight. Do not cite the meters")
    print("below as single-pass elevation resolution.")
    print(f"  withdrawn scalar per SA step: {b_withdrawn*1000:.1f} mm")
    print(f"  withdrawn scalar span:        {b_withdrawn*n_sub:.3f} m")
    print(f"  implied v1 'resolution':      {res_withdrawn:.1f} m  WITHDRAWN")
    print(f"  implied v1 'ambiguity':       {amb_withdrawn:.1f} m  WITHDRAWN")
    print("Operative statement: zero single-pass elevation aperture.")

    return {
        "dx": dx,
        "velocity": velocity,
        "wavelength": wavelength,
        "doppler_bw": doppler_bw,
        "slant_range": slant_range,
        "inc_angle": inc_angle,
        "inc_source": inc_source,
        "ka": ka,
        "azimuth_res": azimuth_res,
    }


def _varying_n(meta_r):
    print(f"\n\n{'='*70}")
    print("VARYING N_SUB — along-track Δx only (Giza metadata)")
    print("The withdrawn v1 scalar is shown so it is not mistaken for a result.")
    print(f"{'='*70}")
    print(f"{'N_sub':>6s}  {'Δx (m)':>8s}  {'withdrawn mm':>12s}  {'DO NOT CITE m':>14s}")
    print("-" * 50)
    for n in [2, 5, 10, 15, 20, 30, 50]:
        processing_bw = meta_r["doppler_bw"] * 0.5
        ds = processing_bw / n
        dx = ds * meta_r["wavelength"] * meta_r["slant_range"] / (2 * meta_r["velocity"])
        dtheta = ds * meta_r["wavelength"] / (2 * meta_r["velocity"])
        bp = dx * np.sin(dtheta)
        kzs = 4 * np.pi * bp / (
            meta_r["wavelength"] * meta_r["slant_range"] * np.sin(meta_r["inc_angle"])
        )
        res = 2 * np.pi / (kzs * n) if kzs > 0 else float("inf")
        print(f"{n:6d}  {dx:8.1f}  {bp*1000:12.1f}  {res:14.1f}")


def main():
    print("=" * 70)
    print("SAR DOPPLER TOMOGRAPHY — METADATA / ALONG-TRACK CHECK")
    print("=" * 70)
    print(f"DATA_DIR = {DATA_DIR}")
    print("Single-pass elevation aperture: none.")
    print("v1 ~285 m / ~618 mm elevation claim: withdrawn.")
    print("=" * 70)

    missing = [path for path in FILES.values() if not os.path.isfile(path)]
    if missing:
        print("\nLocal SICD files are not in DATA_DIR. Nothing was measured.")
        print("Set SAR_DATA_DIR (or DATA_DIR) to the directory that contains:")
        for path in missing:
            print(f"  missing: {path}")
        print("\nThis exit does not recompute or restore the withdrawn 285 m figure.")
        return 2

    results = {}
    for name, path in FILES.items():
        results[name] = analyze(path)

    _varying_n(results["giza"])

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(
        """
Azimuth sub-apertures inside one monostatic SLC are not an elevation
baseline. The v1 map B = Δx·sin(Δθ) → hundreds of meters of elevation
resolution is withdrawn and is not reinstated by the incidence-angle fix.

Incidence used above is SICD SCPCOA.IncidenceAng when the file provides
it, otherwise the geometric angle at the target. The spacecraft look
angle is not that incidence.

This script does not run displacement cross-correlation or A†Y and is
not a reproduction of the published micro-motion chain.
"""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
