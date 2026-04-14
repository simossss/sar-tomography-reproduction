#!/usr/bin/env python3
"""
multi_site_tomo.py — Run multi-pass tomography on 20 locations from the same
Sentinel-1 data. Produces a comparison grid showing that every location
generates similar "depth structure" regardless of what's actually underground.

Usage:
    python3 multi_site_tomo.py

Requires: step3_baselines.py to have been run first (produces baselines.json)
"""
import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from glob import glob
import os
import json

SAFE_DIR = os.path.join(DATA_DIR, "sentinel1_giza_safe")

# 20 sites across the scene — monuments, urban, desert, water, farmland
SITES = {
    # Ancient monuments
    'great_pyramid':    {'lat': 29.9792, 'lon': 31.1342, 'desc': 'Great Pyramid of Khufu'},
    'khafre_pyramid':   {'lat': 29.9761, 'lon': 31.1309, 'desc': 'Pyramid of Khafre'},
    'menkaure_pyramid': {'lat': 29.9725, 'lon': 31.1278, 'desc': 'Pyramid of Menkaure'},
    'sphinx':           {'lat': 29.9753, 'lon': 31.1376, 'desc': 'Great Sphinx'},
    'saqqara':          {'lat': 29.8712, 'lon': 31.2164, 'desc': 'Step Pyramid Saqqara'},
    
    # Urban
    'cairo_downtown':   {'lat': 30.0444, 'lon': 31.2357, 'desc': 'Downtown Cairo'},
    'cairo_mosque':     {'lat': 30.0325, 'lon': 31.2628, 'desc': 'Al-Azhar Mosque area'},
    'giza_city':        {'lat': 30.0131, 'lon': 31.2089, 'desc': 'Giza city center'},
    'cairo_airport':    {'lat': 30.1219, 'lon': 31.4056, 'desc': 'Cairo Airport area'},
    
    # Desert
    'desert_south':     {'lat': 29.9500, 'lon': 31.1800, 'desc': 'Empty desert S'},
    'desert_west':      {'lat': 29.9900, 'lon': 31.0500, 'desc': 'Empty desert W'},
    'desert_sw':        {'lat': 29.9200, 'lon': 31.1000, 'desc': 'Empty desert SW'},
    
    # Water
    'nile_giza':        {'lat': 30.0050, 'lon': 31.2200, 'desc': 'Nile at Giza'},
    'nile_downtown':    {'lat': 30.0500, 'lon': 31.2300, 'desc': 'Nile downtown'},
    
    # Agricultural
    'farm_north':       {'lat': 30.0800, 'lon': 31.1000, 'desc': 'Farmland N'},
    'farm_delta':       {'lat': 30.1200, 'lon': 31.1500, 'desc': 'Nile Delta farmland'},
    
    # Random featureless spots
    'random_1':         {'lat': 29.9300, 'lon': 31.2500, 'desc': 'Random spot 1'},
    'random_2':         {'lat': 30.0600, 'lon': 31.0800, 'desc': 'Random spot 2'},
    'random_3':         {'lat': 29.9600, 'lon': 31.3000, 'desc': 'Random spot 3'},
    'random_4':         {'lat': 30.0000, 'lon': 31.1600, 'desc': 'Random spot 4'},
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


def extract_and_invert(safes, dates, B_perps, site_lat, site_lon, bl):
    """Extract complex values and run tomographic inversion for one site."""
    complex_vals = []
    valid_idx = []
    patch_half = 25
    
    for i, safe in enumerate(safes):
        r = find_pixel_iw2(safe, site_lat, site_lon)
        if r is None or r['tiff'] is None:
            continue
        
        with tifffile.TiffFile(r['tiff']) as tif:
            h, w = tif.pages[0].shape
            line = min(max(r['line'], patch_half), h - patch_half)
            pixel = min(max(r['pixel'], patch_half), w - patch_half)
            patch = tif.pages[0].asarray()[line-patch_half:line+patch_half,
                                           pixel-patch_half:pixel+patch_half]
        
        amp = np.mean(np.abs(patch))
        if amp < 1:
            continue
        
        cr, cc = patch.shape[0]//2, patch.shape[1]//2
        val = np.mean(patch[cr-5:cr+5, cc-5:cc+5])
        complex_vals.append(val)
        valid_idx.append(i)
    
    if len(complex_vals) < 3:
        return None
    
    Y = np.array(complex_vals)
    bp = B_perps[valid_idx]
    kz = 4 * np.pi * bp / (bl['wavelength'] * bl['slant_range'] * np.sin(bl['inc_angle']))
    
    depths = np.arange(-300, 300, 0.5)
    A = np.exp(1j * np.outer(kz, depths))
    tomo = np.abs(np.linalg.pinv(A) @ Y)
    tomo_norm = tomo / np.max(tomo) if np.max(tomo) > 0 else tomo
    
    peak_idx = np.argmax(tomo)
    bg = np.mean(tomo)
    bg_std = np.std(tomo)
    snr = (tomo[peak_idx] - bg) / bg_std if bg_std > 0 else 0
    
    return {
        'depths': depths,
        'amplitude': tomo_norm,
        'peak_depth': float(depths[peak_idx]),
        'snr': float(snr),
        'n_valid': len(complex_vals),
    }


# =============================================
# MAIN
# =============================================
print("=" * 60)
print("MULTI-SITE TOMOGRAPHY — 20 LOCATIONS")
print("=" * 60)

with open('baselines.json') as f:
    bl = json.load(f)
B_perps = np.array(bl['B_perps'])
dates = bl['dates']

safes = sorted(glob(os.path.join(SAFE_DIR, '*.SAFE')))
print(f"Images: {len(safes)}\n")

results = {}
ref_profile = None

for site_name, site_info in SITES.items():
    print(f"  Processing {site_info['desc']:30s} ...", end='', flush=True)
    
    r = extract_and_invert(safes, dates, B_perps, 
                           site_info['lat'], site_info['lon'], bl)
    
    if r is None:
        print(f" SKIP (not in IW2 or no valid data)")
        continue
    
    results[site_name] = r
    
    # Compute correlation with Great Pyramid if available
    corr_str = ""
    if ref_profile is not None:
        corr = np.corrcoef(r['amplitude'], ref_profile)[0, 1]
        corr_str = f" corr={corr:.4f}"
    
    if site_name == 'great_pyramid':
        ref_profile = r['amplitude'].copy()
    
    print(f" peak={r['peak_depth']:+.0f}m SNR={r['snr']:.1f}σ n={r['n_valid']}{corr_str}")

# =============================================
# Correlation matrix
# =============================================
print(f"\n{'='*60}")
print("CORRELATION WITH GREAT PYRAMID")
print(f"{'='*60}")

if 'great_pyramid' not in results:
    print("ERROR: Great Pyramid not processed")
else:
    ref = results['great_pyramid']['amplitude']
    print(f"\n{'Site':>30s}  {'Correlation':>12s}  {'Peak':>8s}  {'SNR':>6s}")
    print("-" * 65)
    
    corrs = []
    for name, r in results.items():
        corr = np.corrcoef(r['amplitude'], ref)[0, 1]
        corrs.append(corr)
        desc = SITES[name]['desc']
        print(f"{desc:>30s}  {corr:12.4f}  {r['peak_depth']:+7.0f}m  {r['snr']:5.1f}σ")
    
    corrs = np.array(corrs)
    print(f"\n  Mean correlation: {np.mean(corrs):.4f}")
    print(f"  Min correlation:  {np.min(corrs):.4f}")
    print(f"  Max correlation:  {np.max(corrs):.4f}")
    
    if np.mean(corrs) > 0.9:
        print(f"\n  CONCLUSION: All {len(results)} sites produce nearly identical")
        print(f"  depth profiles (mean correlation {np.mean(corrs):.3f}).")
        print(f"  The tomographic output is dominated by the inversion geometry,")
        print(f"  not subsurface structure. The method generates 'underground")
        print(f"  structures' everywhere — pyramids, desert, farmland, river,")
        print(f"  city streets — all look the same.")

# =============================================
# Save amplitude at key depths for all sites
# =============================================
print(f"\n{'='*60}")
print("AMPLITUDE AT KEY DEPTHS")
print(f"{'='*60}")

key_depths = [140, 43, 21, 0, -50, -100, -200]
depths_arr = results['great_pyramid']['depths'] if 'great_pyramid' in results else None

if depths_arr is not None:
    header = f"{'Site':>20s}"
    for d in key_depths:
        header += f"  {d:+4d}m"
    print(header)
    print("-" * (20 + len(key_depths) * 7))
    
    for name, r in results.items():
        row = f"{SITES[name]['desc'][:20]:>20s}"
        for d in key_depths:
            idx = np.argmin(np.abs(r['depths'] - d))
            row += f"  {r['amplitude'][idx]:.3f}"
        print(row)

# Save
save_data = {}
for name, r in results.items():
    save_data[name] = {
        'desc': SITES[name]['desc'],
        'lat': SITES[name]['lat'],
        'lon': SITES[name]['lon'],
        'peak_depth': r['peak_depth'],
        'snr': r['snr'],
        'n_valid': r['n_valid'],
        'amplitude': r['amplitude'].tolist(),
    }

with open('multi_site_results.json', 'w') as f:
    json.dump(save_data, f)
print(f"\nResults saved to multi_site_results.json")
print("Done.")
