#!/usr/bin/env python3
"""
aliasing_simulation.py — Demonstrate how tomographic aliasing creates
the illusion of deep underground structures from real shallow detections.

This script uses NO real data. It creates a synthetic SAR tomography scenario
with known ground truth, then shows how the beamformer output produces
periodic ghost copies of shallow structures extending to arbitrary depth.

The simulation models what would happen if Biondi's multi-pass processing
genuinely detects the King's Chamber at 43m — and why that real detection
would create the appearance of structures at 600m and beyond.

Usage:
    python3 aliasing_simulation.py

Output:
    aliasing_demo.png — Visual comparison of ground truth vs tomographic output
    aliasing_data.json — Numerical results
"""
import numpy as np
import json

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — will produce ASCII output only")


def simulate_tomography(n_images, B_perp_max, wavelength, slant_range, inc_angle,
                        scatterers, z_range, z_step, label=""):
    """
    Simulate multi-pass SAR tomography with known ground truth.
    
    Args:
        n_images: number of SAR images in the stack
        B_perp_max: maximum perpendicular baseline (meters)
        wavelength: radar wavelength (meters)
        slant_range: distance to target (meters)
        inc_angle: incidence angle (radians)
        scatterers: list of (depth, amplitude) tuples — the ground truth
        z_range: (z_min, z_max) depth range to image
        z_step: depth step size
        label: description string
    
    Returns:
        dict with depths, tomogram, ground_truth, baselines, ambiguity, resolution
    """
    # Generate baselines — realistic distribution
    np.random.seed(42)
    B_perps = np.random.uniform(-B_perp_max, B_perp_max, n_images)
    B_perps[n_images // 2] = 0  # master image
    B_perps = np.sort(B_perps)
    
    # Compute kz for each image
    kz = 4 * np.pi * B_perps / (wavelength * slant_range * np.sin(inc_angle))
    
    # Resolution and ambiguity
    kz_span = np.max(kz) - np.min(kz)
    kz_sorted = np.sort(kz)
    kz_diffs = np.diff(kz_sorted)
    kz_step_min = np.min(kz_diffs[kz_diffs > 0]) if np.any(kz_diffs > 0) else 1e-10
    
    resolution = 2 * np.pi / kz_span if kz_span > 0 else float('inf')
    ambiguity = 2 * np.pi / kz_step_min
    
    # Simulate the received signal at each image
    # Y[i] = sum over scatterers of amplitude * exp(j * kz[i] * depth) + noise
    Y = np.zeros(n_images, dtype=np.complex128)
    for depth, amplitude in scatterers:
        Y += amplitude * np.exp(1j * kz * depth)
    
    # Add noise
    noise_level = 0.05 * np.max(np.abs(Y))
    Y += noise_level * (np.random.randn(n_images) + 1j * np.random.randn(n_images))
    
    # Tomographic inversion
    depths = np.arange(z_range[0], z_range[1], z_step)
    A = np.exp(1j * np.outer(kz, depths))
    
    # Beamformer output
    tomo = np.abs(A.conj().T @ Y) / n_images
    tomo_norm = tomo / np.max(tomo) if np.max(tomo) > 0 else tomo
    
    # Ground truth profile
    truth = np.zeros_like(depths)
    for depth, amplitude in scatterers:
        idx = np.argmin(np.abs(depths - depth))
        truth[idx] = amplitude
    truth_norm = truth / np.max(truth) if np.max(truth) > 0 else truth
    
    return {
        'label': label,
        'depths': depths,
        'tomogram': tomo_norm,
        'ground_truth': truth_norm,
        'B_perps': B_perps,
        'kz': kz,
        'resolution': resolution,
        'ambiguity': ambiguity,
        'n_images': n_images,
        'scatterers': scatterers,
    }


# =============================================
# SCENARIO PARAMETERS
# =============================================

# Common SAR parameters
WAVELENGTH_X = 0.031      # X-band (COSMO-SkyMed)
WAVELENGTH_C = 0.055      # C-band (Sentinel-1)
SLANT_RANGE = 650000      # ~650 km
INC_ANGLE = np.radians(35)

# Ground truth: known pyramid features
PYRAMID_SCATTERERS = [
    (0,   1.0),    # Surface / pyramid base (strongest return)
    (21,  0.3),    # Queen's Chamber
    (43,  0.5),    # King's Chamber
    (60,  0.2),    # Grand Gallery upper
    (140, 0.8),    # Pyramid apex (exterior surface)
]

# =============================================
# RUN SIMULATIONS
# =============================================
print("=" * 60)
print("TOMOGRAPHIC ALIASING SIMULATION")
print("=" * 60)
print("\nGround truth scatterers (the REAL pyramid features):")
for depth, amp in PYRAMID_SCATTERERS:
    name = {0: 'Base', 21: "Queen's Ch.", 43: "King's Ch.", 
            60: 'Grand Gallery', 140: 'Apex'}.get(depth, f'{depth}m')
    print(f"  {name:20s}  depth={depth:4d}m  amplitude={amp:.1f}")

scenarios = []

# Scenario 1: 6 CSK images (what the paper's Table 2 shows)
print(f"\n{'='*60}")
print("SCENARIO 1: 6 COSMO-SkyMed X-band images")
print("(matches Table 2 of the published paper)")
print(f"{'='*60}")
s1 = simulate_tomography(
    n_images=6, B_perp_max=300, wavelength=WAVELENGTH_X,
    slant_range=SLANT_RANGE, inc_angle=INC_ANGLE,
    scatterers=PYRAMID_SCATTERERS,
    z_range=(-100, 800), z_step=0.5,
    label="6 CSK X-band (paper Table 2)"
)
scenarios.append(s1)
print(f"Resolution: {s1['resolution']:.1f} m")
print(f"Ambiguity height: {s1['ambiguity']:.1f} m")
print(f"B_perp range: {min(s1['B_perps']):.0f} to {max(s1['B_perps']):.0f} m")

# Scenario 2: 200 CSK images (what Biondi claims on Rogan)
print(f"\n{'='*60}")
print("SCENARIO 2: 200 COSMO-SkyMed X-band images")
print("(matches '200 scans' claim from interview)")
print(f"{'='*60}")
s2 = simulate_tomography(
    n_images=200, B_perp_max=500, wavelength=WAVELENGTH_X,
    slant_range=SLANT_RANGE, inc_angle=INC_ANGLE,
    scatterers=PYRAMID_SCATTERERS,
    z_range=(-100, 800), z_step=0.5,
    label="200 CSK X-band (claimed)"
)
scenarios.append(s2)
print(f"Resolution: {s2['resolution']:.1f} m")
print(f"Ambiguity height: {s2['ambiguity']:.1f} m")
print(f"B_perp range: {min(s2['B_perps']):.0f} to {max(s2['B_perps']):.0f} m")

# Scenario 3: 15 Sentinel-1 (what we actually have)
print(f"\n{'='*60}")
print("SCENARIO 3: 15 Sentinel-1 C-band images")
print("(our actual experiment)")
print(f"{'='*60}")
s3 = simulate_tomography(
    n_images=15, B_perp_max=170, wavelength=WAVELENGTH_C,
    slant_range=SLANT_RANGE, inc_angle=INC_ANGLE,
    scatterers=PYRAMID_SCATTERERS,
    z_range=(-100, 800), z_step=0.5,
    label="15 Sentinel-1 C-band (ours)"
)
scenarios.append(s3)
print(f"Resolution: {s3['resolution']:.1f} m")
print(f"Ambiguity height: {s3['ambiguity']:.1f} m")
print(f"B_perp range: {min(s3['B_perps']):.0f} to {max(s3['B_perps']):.0f} m")

# Scenario 4: Extended depth — show aliasing to 1200m with 6 images
print(f"\n{'='*60}")
print("SCENARIO 4: 6 CSK images — extended to 1200m depth")
print("(shows aliasing creating 'underground city')")
print(f"{'='*60}")
s4 = simulate_tomography(
    n_images=6, B_perp_max=300, wavelength=WAVELENGTH_X,
    slant_range=SLANT_RANGE, inc_angle=INC_ANGLE,
    scatterers=PYRAMID_SCATTERERS,
    z_range=(-100, 1300), z_step=0.5,
    label="6 CSK X-band — 1200m depth"
)
scenarios.append(s4)
print(f"Resolution: {s4['resolution']:.1f} m")
print(f"Ambiguity height: {s4['ambiguity']:.1f} m")

# =============================================
# ANALYSIS: Find ghost peaks
# =============================================
print(f"\n{'='*60}")
print("ALIASING ANALYSIS — SCENARIO 4 (6 images, extended depth)")
print(f"{'='*60}")

tomo = s4['tomogram']
depths = s4['depths']
amb = s4['ambiguity']

# Find all peaks above 0.3 threshold
from scipy.signal import find_peaks
try:
    peaks, props = find_peaks(tomo, height=0.3, distance=10)
    
    print(f"\nAmbiguity height: {amb:.1f} m")
    print(f"\nPeaks found above 0.3 threshold:")
    print(f"{'Depth':>8s}  {'Amplitude':>10s}  {'Alias of':>20s}")
    print("-" * 45)
    
    for p in peaks:
        d = depths[p]
        a = tomo[p]
        # Which real scatterer is this an alias of?
        best_match = None
        for real_d, real_a in PYRAMID_SCATTERERS:
            remainder = (d - real_d) % amb
            if remainder < 5 or (amb - remainder) < 5:
                best_match = f"{real_d}m (n={(d-real_d)/amb:.0f})"
                break
        if best_match is None:
            best_match = "unknown"
        print(f"{d:+8.1f}m  {a:10.3f}  {best_match:>20s}")

except ImportError:
    print("scipy not available for peak finding — showing raw profile instead")
    for d_val in range(-100, 1300, 25):
        idx = np.argmin(np.abs(depths - d_val))
        amp = tomo[idx]
        bar = '#' * int(amp * 40)
        marker = ''
        for rd, ra in PYRAMID_SCATTERERS:
            if abs(d_val - rd) < 3:
                marker = f' <-- REAL ({rd}m)'
        print(f"{d_val:+6d}m  {amp:.3f}  |{bar}|{marker}")

# =============================================
# ASCII VISUALIZATION
# =============================================
print(f"\n{'='*60}")
print("DEPTH PROFILES — ALL SCENARIOS")
print(f"{'='*60}")

for scenario in scenarios:
    print(f"\n--- {scenario['label']} ---")
    print(f"    Resolution: {scenario['resolution']:.1f}m, Ambiguity: {scenario['ambiguity']:.1f}m")
    print(f"    {'Depth':>8s}  {'Tomo':>6s}  {'Truth':>6s}  Profile")
    
    depths = scenario['depths']
    tomo = scenario['tomogram']
    truth = scenario['ground_truth']
    
    step = max(1, len(depths) // 60)  # show ~60 lines
    for i in range(0, len(depths), step):
        d = depths[i]
        t = tomo[i]
        tr = truth[i]
        bar_t = '#' * int(t * 30)
        bar_tr = '*' if tr > 0.1 else ' '
        
        marker = ''
        for rd, ra in PYRAMID_SCATTERERS:
            if abs(d - rd) < 2:
                marker = f' <-- REAL'
        
        print(f"    {d:+8.1f}m  {t:.3f}  {tr:.3f}  |{bar_t:30s}| {bar_tr}{marker}")

# =============================================
# KEY COMPARISON: Scenario 4 peaks vs ground truth
# =============================================
print(f"\n{'='*60}")
print("THE ALIASING EXPLANATION")
print(f"{'='*60}")
print(f"""
With 6 COSMO-SkyMed images and baselines up to 300m:
  - Vertical resolution: {s1['resolution']:.0f}m (can resolve King's Chamber at 43m)
  - Ambiguity height: {s1['ambiguity']:.0f}m

The REAL features (0-140m) are genuinely detected.
But every real feature REPEATS at intervals of {s1['ambiguity']:.0f}m:

  King's Chamber (43m) appears also at:
    43 + {s1['ambiguity']:.0f} = {43 + s1['ambiguity']:.0f}m
    43 + 2x{s1['ambiguity']:.0f} = {43 + 2*s1['ambiguity']:.0f}m
    43 + 3x{s1['ambiguity']:.0f} = {43 + 3*s1['ambiguity']:.0f}m
    ... and so on to any depth

  Surface (0m) appears also at:
    0 + {s1['ambiguity']:.0f} = {s1['ambiguity']:.0f}m
    0 + 2x{s1['ambiguity']:.0f} = {2*s1['ambiguity']:.0f}m
    ... etc.

  Multiple scatterers at different depths, each aliasing with
  different phase, create a COMPLEX PATTERN that looks like
  real subsurface architecture — columns, chambers, corridors.

  Biondi's claimed depth of 600m = {600/s1['ambiguity']:.1f} ambiguity periods.
  Biondi's claimed depth of 1200m = {1200/s1['ambiguity']:.1f} ambiguity periods.

  The 'underground city' is the pyramid's own internal structure,
  reflected {int(1200/s1['ambiguity'])}+ times down the depth axis by aliasing.
""")

# =============================================
# MATPLOTLIB FIGURE (if available)
# =============================================
if HAS_MPL:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Tomographic Aliasing: How Shallow Detections Create\n'
                 'the Illusion of Deep Underground Structures', 
                 fontsize=14, fontweight='bold')
    
    for ax, scenario in zip(axes.flat, scenarios):
        depths = scenario['depths']
        tomo = scenario['tomogram']
        truth = scenario['ground_truth']
        
        ax.plot(tomo, depths, 'b-', linewidth=1, label='Tomogram output', alpha=0.8)
        
        # Mark real scatterers
        for d, a in PYRAMID_SCATTERERS:
            if depths[0] <= d <= depths[-1]:
                ax.axhline(y=d, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
                ax.annotate(f'REAL: {d}m', xy=(0.95, d), fontsize=7, color='red',
                           ha='right', va='bottom')
        
        # Mark ambiguity periods
        amb = scenario['ambiguity']
        if amb < (depths[-1] - depths[0]):
            for n in range(1, int((depths[-1] - depths[0]) / amb) + 1):
                ax.axhline(y=n*amb, color='gray', linestyle=':', alpha=0.2)
        
        ax.set_xlabel('Normalized Amplitude')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f"{scenario['label']}\n"
                     f"Res: {scenario['resolution']:.0f}m, "
                     f"Ambiguity: {scenario['ambiguity']:.0f}m",
                     fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('aliasing_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to aliasing_demo.png")

# =============================================
# Save data
# =============================================
save_data = {}
for s in scenarios:
    save_data[s['label']] = {
        'resolution': s['resolution'],
        'ambiguity': s['ambiguity'],
        'n_images': s['n_images'],
        'B_perp_range': [float(min(s['B_perps'])), float(max(s['B_perps']))],
        'scatterers': s['scatterers'],
        'peak_depths': [float(s['depths'][i]) for i in range(len(s['depths'])) 
                        if s['tomogram'][i] > 0.3],
    }

with open('aliasing_data.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"Data saved to aliasing_data.json")
print("\nDone.")
