"""
Multi-pass tomography on Giza — using IW2 (the correct subswath).
"""
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from glob import glob
import os
import json

SAFE_DIR = os.path.join(DATA_DIR, "sentinel1_giza_safe")

SITES = {
    'pyramid':  {'lat': 29.9792, 'lon': 31.1342, 'desc': 'Great Pyramid'},
    'desert':   {'lat': 29.9500, 'lon': 31.1800, 'desc': 'Empty desert'},
    'sphinx':   {'lat': 29.9753, 'lon': 31.1376, 'desc': 'Sphinx'},
    'cairo':    {'lat': 30.0444, 'lon': 31.2357, 'desc': 'Downtown Cairo'},
    'farmland': {'lat': 30.0800, 'lon': 31.1000, 'desc': 'Farmland'},
}

def find_pixel_iw2(safe_path, target_lat, target_lon):
    """Find pixel coords in IW2 VV for a given lat/lon."""
    anno_dir = os.path.join(safe_path, 'annotation')
    meas_dir = os.path.join(safe_path, 'measurement')
    
    xmls = sorted(glob(os.path.join(anno_dir, 's1a-iw2-slc-vv-*.xml')))
    if not xmls:
        return None
    
    tree = ET.parse(xmls[0])
    root = tree.getroot()
    
    geo_points = root.findall('.//geolocationGridPoint')
    lats = np.array([float(gp.find('latitude').text) for gp in geo_points])
    lons = np.array([float(gp.find('longitude').text) for gp in geo_points])
    lines = np.array([int(gp.find('line').text) for gp in geo_points])
    pixels = np.array([int(gp.find('pixel').text) for gp in geo_points])
    
    if not (min(lats) < target_lat < max(lats) and
            min(lons) < target_lon < max(lons)):
        return None
    
    dist = np.sqrt((lats - target_lat)**2 + (lons - target_lon)**2)
    nearest = np.argsort(dist)[:6]
    w = 1.0 / (dist[nearest] + 1e-12)
    w /= w.sum()
    
    tgt_line = int(round(np.sum(lines[nearest] * w)))
    tgt_pixel = int(round(np.sum(pixels[nearest] * w)))
    
    tiff = sorted(glob(os.path.join(meas_dir, 's1a-iw2-slc-vv-*.tiff')))
    return {'line': tgt_line, 'pixel': tgt_pixel, 'tiff': tiff[0] if tiff else None}

# Load baselines
with open('baselines.json') as f:
    bl = json.load(f)
B_perps = np.array(bl['B_perps'])
dates = bl['dates']
master_idx = bl['master_idx']
wavelength = bl['wavelength']
slant_range = bl['slant_range']
inc_angle = bl['inc_angle']

safes = sorted(glob(os.path.join(SAFE_DIR, '*.SAFE')))
print(f"Images: {len(safes)}, Master: {dates[master_idx]}")
print(f"B_perp range: {min(B_perps):.1f} to {max(B_perps):.1f} m\n")

# =============================================
# Process each site
# =============================================
all_results = {}

for site_name, site_info in SITES.items():
    print(f"{'='*60}")
    print(f"{site_info['desc']} ({site_info['lat']:.4f}N, {site_info['lon']:.4f}E)")
    print(f"{'='*60}")
    
    complex_vals = []
    valid_idx = []
    patch_half = 25
    
    for i, safe in enumerate(safes):
        r = find_pixel_iw2(safe, site_info['lat'], site_info['lon'])
        if r is None or r['tiff'] is None:
            print(f"  [{i+1:2d}] {dates[i]} — not in IW2")
            continue
        
        with tifffile.TiffFile(r['tiff']) as tif:
            h, w = tif.pages[0].shape
            line = min(max(r['line'], patch_half), h - patch_half)
            pixel = min(max(r['pixel'], patch_half), w - patch_half)
            patch = tif.pages[0].asarray()[line-patch_half:line+patch_half,
                                           pixel-patch_half:pixel+patch_half]
        
        amp = np.mean(np.abs(patch))
        if amp < 1:
            print(f"  [{i+1:2d}] {dates[i]} — zero data, skip")
            continue
        
        # Central 10x10 mean
        cr, cc = patch.shape[0]//2, patch.shape[1]//2
        val = np.mean(patch[cr-5:cr+5, cc-5:cc+5])
        
        complex_vals.append(val)
        valid_idx.append(i)
        print(f"  [{i+1:2d}] {dates[i]} — amp={np.abs(val):.1f} phase={np.angle(val):+.2f} B={B_perps[i]:+.0f}m")
    
    n_valid = len(complex_vals)
    print(f"\n  Valid: {n_valid}/{len(safes)}")
    
    if n_valid < 3:
        print(f"  SKIP — too few images")
        continue
    
    # Tomographic inversion
    Y = np.array(complex_vals)
    bp = B_perps[valid_idx]
    kz = 4 * np.pi * bp / (wavelength * slant_range * np.sin(inc_angle))
    
    depths = np.arange(-300, 300, 0.5)
    A = np.exp(1j * np.outer(kz, depths))
    tomo = np.abs(np.linalg.pinv(A) @ Y)
    tomo_norm = tomo / np.max(tomo)
    tomo_db = 20 * np.log10(tomo_norm + 1e-10)
    
    peak_idx = np.argmax(tomo)
    peak_depth = depths[peak_idx]
    bg = np.mean(tomo)
    bg_std = np.std(tomo)
    snr = (tomo[peak_idx] - bg) / bg_std if bg_std > 0 else 0
    
    print(f"\n  Peak: {peak_depth:.1f}m, SNR: {snr:.1f}σ")
    
    print(f"\n  {'Depth':>25s}  {'Norm':>6s}  {'dB':>6s}")
    print(f"  {'-'*45}")
    for name, d in [('Apex +140m', 140), ('King Ch +43m', 43),
                     ('Queen Ch +21m', 21), ('Corridor +7m', 7),
                     ('Base 0m', 0), ('-25m', -25), ('-50m', -50),
                     ('-100m', -100), ('-200m', -200)]:
        idx = np.argmin(np.abs(depths - d))
        print(f"  {name:>25s}  {tomo_norm[idx]:6.3f}  {tomo_db[idx]:6.1f}")
    
    all_results[site_name] = {
        'peak_depth': float(peak_depth),
        'snr': float(snr),
        'n_valid': n_valid,
        'depths': depths.tolist(),
        'amplitude': tomo_norm.tolist(),
    }

# =============================================
# Comparison
# =============================================
print(f"\n\n{'='*60}")
print("SITE COMPARISON")
print(f"{'='*60}")
print(f"{'Site':>20s}  {'Peak':>8s}  {'SNR':>6s}  {'N':>3s}")
print(f"{'-'*45}")
for name, r in all_results.items():
    print(f"{SITES[name]['desc']:>20s}  {r['peak_depth']:+7.1f}m  {r['snr']:5.1f}σ  {r['n_valid']:3d}")

# Key question: does pyramid differ from desert?
if 'pyramid' in all_results and 'desert' in all_results:
    p = np.array(all_results['pyramid']['amplitude'])
    d = np.array(all_results['desert']['amplitude'])
    corr = np.corrcoef(p, d)[0, 1]
    diff = np.mean(np.abs(p - d))
    print(f"\nPyramid vs Desert correlation: {corr:.4f}")
    print(f"Mean absolute difference: {diff:.4f}")
    if corr > 0.9:
        print("CONCLUSION: Profiles are nearly identical — no subsurface detection")
    elif corr < 0.5:
        print("CONCLUSION: Profiles differ significantly — possible subsurface signal")
    else:
        print("CONCLUSION: Moderate difference — inconclusive")

# Save
with open('tomo_results.json', 'w') as f:
    json.dump(all_results, f)
print(f"\nResults saved to tomo_results.json")
