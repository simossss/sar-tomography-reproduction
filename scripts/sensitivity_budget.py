#!/usr/bin/env python3
"""
sensitivity_budget.py - every number in Section 4 of the note.

Inputs are the acquisition parameters printed in Table 1 of Biondi & Malanga
(2022, retracted), the displacement scale of their Figure 6, the dominant
frequency measured inside the pyramid by ElGabry et al. (2026), and the
velocity error reported for the bridge validation of Lotti et al. (2026).
No SAR data is used.

Usage:
    python3 scripts/sensitivity_budget.py
"""
import numpy as np

C = 299_792_458.0

# Table 1 of the paper under review
F_RADAR = 9.6e9            # Hz
CHIRP_BW = 400e6           # Hz
DOPPLER_BW = 22e3          # Hz
PRF = 2.0e3                # Hz
PRT = 0.23e-3              # s (does not equal 1/PRF)
T_ACQ = 15.0               # s
V_PLATFORM = 7_000.0       # m/s
RANGE = 650_000.0          # m (used as R in the paper's resolution formula)

# The claim
D_CLAIM = 1e-7             # m, axis scale of the vibration series (Figure 6)
F_CHOSEN = 12_500.0        # Hz, "frequency of investigation set by us"
V_ROCK = 6_000.0           # m/s
APERTURE = 42_000.0        # m, "about 42,000 m"

# Independent references
F_MEASURED = 2.3           # Hz, ElGabry et al. (2026)
ACCEL_RES_G = 1e-6         # g, accelerometer resolution in the same survey
V_ERR_VALIDATED = 1e-3     # m/s, Lotti et al. (2026)

lam = C / F_RADAR
az_cell = V_PLATFORM / DOPPLER_BW
rg_cell = C / (2 * CHIRP_BW)


def phase_noise(snr_db):
    """Single-look phase standard deviation of a point target, high SNR."""
    return 1 / np.sqrt(2 * 10 ** (snr_db / 10))


def shift_noise(cell, snr_db):
    """Lower bound on the error of locating a flat-spectrum response
    (Bamler & Eineder 2005): sqrt(3)/pi * cell / sqrt(2 SNR)."""
    return np.sqrt(3) / np.pi * cell / np.sqrt(2 * 10 ** (snr_db / 10))


print("4.1  The claimed quantity")
print(f"  radar wavelength                      {lam * 1e3:.2f} mm")
print(f"  claimed displacement                  {D_CLAIM * 1e6:.1f} um")
print(f"  resolution cell (azimuth x range)     {az_cell:.3f} m x {rg_cell:.3f} m")

print("\n4.2  Phase budget")
dphi = 4 * np.pi * D_CLAIM / lam
print(f"  phase change for the claimed motion   {dphi:.2e} rad")
for snr_db in (40, 20):
    s1 = phase_noise(snr_db)
    s2 = np.sqrt(2) * s1
    print(f"  SNR {snr_db} dB: one look {s1:.4f} rad = {lam * s1 / (4 * np.pi) * 1e6:6.1f} um | "
          f"difference of two looks {s2:.3f} rad = {lam * s2 / (4 * np.pi) * 1e6:6.1f} um | "
          f"noise / claim = {lam * s2 / (4 * np.pi) / D_CLAIM:,.0f}")

print("\n4.3  Sampling and sub-aperture duration")
half_period = 1 / (2 * F_MEASURED)
frac = half_period / T_ACQ
print(f"  piece length to follow {F_MEASURED} Hz          under {half_period:.3f} s = {100 * frac:.1f}% of the aperture")
print(f"  azimuth cell of such a piece          {az_cell / frac:.0f} m")
print(f"  SNR of such a piece                   full-aperture SNR minus {-10 * np.log10(frac):.1f} dB")
print(f"  half-band piece                       {T_ACQ / 2:.1f} s = {F_MEASURED * T_ACQ / 2:.0f} cycles at {F_MEASURED} Hz, "
      f"{F_CHOSEN * T_ACQ / 2:,.0f} cycles at {F_CHOSEN:,.0f} Hz")
print(f"  pulse interval                        {1e3 / PRF:.2f} ms at the stated PRF; Table 1 PRT {PRT * 1e3:.2f} ms "
      f"= {1 / PRT:,.0f} Hz")
print(f"  period of a {F_CHOSEN:,.0f} Hz vibration       {1e3 / F_CHOSEN:.3f} ms")

print("\n4.4  Comparison with a validated measurement")
v_claim = 2 * np.pi * F_MEASURED * D_CLAIM
print(f"  velocity of the claim at {F_MEASURED} Hz        {v_claim * 1e3:.4f} mm/s")
print(f"  validated error / claimed velocity    {V_ERR_VALIDATED / v_claim:,.0f}")
print(f"  frequency at which {D_CLAIM * 1e6:.1f} um is 1 mm/s    {V_ERR_VALIDATED / (2 * np.pi * D_CLAIM):,.0f} Hz "
      f"(period {1e3 * 2 * np.pi * D_CLAIM / V_ERR_VALIDATED:.2f} ms)")
shift = RANGE * v_claim / V_PLATFORM
print(f"  azimuth shift of the image at {F_MEASURED} Hz   {shift * 1e3:.2f} mm ({RANGE * 1e-3 / V_PLATFORM * 1e3:.0f} mm per mm/s)")
print("  offset-tracking bound against that shift (difference of two pieces):")
for label, f in (("two half-band pieces (7.5 s)", 0.5), (f"two {half_period:.2f} s pieces        ", frac)):
    for snr_db in (40, 20):
        sig = np.sqrt(2) * shift_noise(az_cell / f, snr_db + 10 * np.log10(f))
        print(f"    {label}, full-aperture SNR {snr_db} dB: {sig * 1e3:9.1f} mm = {sig / shift:9,.0f} x the shift")
acc = ACCEL_RES_G * 9.80665
print(f"  1 ug at {F_MEASURED} Hz as a displacement      {acc / (2 * np.pi * F_MEASURED) ** 2 * 1e6:.3f} um")

print("\n4.5  The assumed frequency and the resolution formula")
w = 2 * np.pi * F_CHOSEN
print(f"  {D_CLAIM * 1e6:.1f} um at {F_CHOSEN:,.0f} Hz                 velocity {w * D_CLAIM * 1e3:.2f} mm/s, "
      f"acceleration {w ** 2 * D_CLAIM:.0f} m/s^2 = {w ** 2 * D_CLAIM / 9.80665:.0f} g")
lam_stated = V_ROCK / F_CHOSEN
print(f"  stated inputs                         lambda = {lam_stated:.2f} m, "
      f"lambda R / (2 A) = {lam_stated * RANGE / (2 * APERTURE):.2f} m")
print(f"  printed in the journal version        lambda = {V_ROCK / 25_000:.2f} m, "
      f"{V_ROCK / 25_000:.2f} x 650,000 / (2 x 84,000) = {V_ROCK / 25_000 * RANGE / (2 * 84_000):.2f} m")
lam_meas = V_ROCK / F_MEASURED
print(f"  at the measured {F_MEASURED} Hz                lambda = {lam_meas / 1e3:.2f} km, formula gives "
      f"{lam_meas * RANGE / (2 * APERTURE) / 1e3:.1f} km (stated factors) or "
      f"{lam_meas / 2 * RANGE / (2 * 2 * APERTURE) / 1e3:.1f} km (printed factors)")
q = 100
print(f"  amplitude left after 100 m, Q = {q}     {np.exp(-np.pi * F_CHOSEN * 100 / (q * V_ROCK)):.4f} at {F_CHOSEN:,.0f} Hz, "
      f"{np.exp(-np.pi * F_MEASURED * 100 / (q * V_ROCK)):.4f} at {F_MEASURED} Hz")

# What a surface moving 0.1 um at 12,500 Hz would do to the air and to the rock.
RHO_AIR, C_AIR = 1.2, 343.0            # kg/m^3, m/s
RHO_ROCK = 2400.0                      # kg/m^3, limestone (assumed)
VOLUME = 2.6e6                         # m^3, the pyramid
v_peak = w * D_CLAIM
p_rms = RHO_AIR * C_AIR * v_peak / np.sqrt(2)     # plane radiator, face much larger than the 27 mm wavelength
faces = 4 * 0.5 * 230.33 * np.hypot(138.5, 230.33 / 2)
print(f"  sound at the faces                    {20 * np.log10(p_rms / 20e-6):.0f} dB SPL, "
      f"{p_rms ** 2 / (RHO_AIR * C_AIR) * faces / 1e3:.1f} kW radiated from {faces:,.0f} m^2")
energy = 0.5 * RHO_ROCK * v_peak ** 2 * VOLUME    # whole volume moving at that amplitude
for qq in (200, 100, 50):
    print(f"  power to sustain the motion, Q = {qq:<3d}  {w * energy / qq / 1e6:.0f} MW")

print("\n4.6  Averaging")
cell_area = az_cell * rg_cell
for snr_db in (40, 20):
    sigma = lam * np.sqrt(2) * phase_noise(snr_db) / (4 * np.pi)
    n = (sigma / D_CLAIM) ** 2
    print(f"  SNR {snr_db} dB: {n:,.0f} independent cells per time sample = {n * cell_area:,.0f} m^2 "
          f"= a square {np.sqrt(n * cell_area):.0f} m on a side")
print(f"  footprint of the pyramid              {230.33 ** 2:,.0f} m^2")
