"""
Step 3: Unzip Sentinel-1 SAFE files and compute perpendicular baselines.
This tells us the actual kz geometry before we do any heavy processing.
"""
import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from glob import glob

DATA_DIR = os.path.join(DATA_DIR, "sentinel1_giza")
SAFE_DIR = os.path.join(DATA_DIR, "sentinel1_giza_safe")

# Giza coordinates
GIZA_LAT = 29.9792
GIZA_LON = 31.1342

# =============================================
# Step 3a: Unzip all files
# =============================================
print("=" * 60)
print("STEP 3a: Unzipping SAFE files")
print("=" * 60)

os.makedirs(SAFE_DIR, exist_ok=True)
zips = sorted(glob(os.path.join(DATA_DIR, '*.zip')))
print(f"Found {len(zips)} zip files\n")

for i, zf in enumerate(zips):
    name = os.path.basename(zf).replace('.zip', '.SAFE')
    safe_path = os.path.join(SAFE_DIR, name)
    if os.path.exists(safe_path):
        print(f"  [{i+1}/{len(zips)}] SKIP {name} — already extracted")
        continue
    print(f"  [{i+1}/{len(zips)}] Extracting {os.path.basename(zf)}...")
    with zipfile.ZipFile(zf, 'r') as z:
        z.extractall(SAFE_DIR)
    print(f"    DONE")

# =============================================
# Step 3b: Read orbit state vectors from annotation XML
# =============================================
print(f"\n{'=' * 60}")
print("STEP 3b: Reading orbit state vectors")
print("=" * 60)

def get_orbit_state_vectors(safe_path):
    """Extract orbit state vectors from Sentinel-1 annotation XML."""
    # Find annotation XML for IW2 (the subswath most likely covering Giza)
    anno_dir = os.path.join(safe_path, 'annotation')
    if not os.path.exists(anno_dir):
        return None
    
    # Pick the first VV polarization annotation file
    xmls = sorted(glob(os.path.join(anno_dir, 's1a-iw*-slc-vv-*.xml')))
    if not xmls:
        xmls = sorted(glob(os.path.join(anno_dir, '*.xml')))
    if not xmls:
        return None
    
    # Try each subswath to find one covering Giza
    best_xml = xmls[0]
    for xml_path in xmls:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Check if this subswath covers Giza latitude
        geo_grid = root.findall('.//geolocationGridPoint')
        lats = [float(g.find('latitude').text) for g in geo_grid if g.find('latitude') is not None]
        if lats and min(lats) < GIZA_LAT < max(lats):
            best_xml = xml_path
            break
    
    tree = ET.parse(best_xml)
    root = tree.getroot()
    
    # Get orbit state vectors
    orbits = root.findall('.//orbit')
    if not orbits:
        return None
    
    positions = []
    velocities = []
    times = []
    for orb in orbits:
        t = orb.find('time')
        px = orb.find('position/x')
        py = orb.find('position/y')
        pz = orb.find('position/z')
        vx = orb.find('velocity/x')
        vy = orb.find('velocity/y')
        vz = orb.find('velocity/z')
        if all(v is not None for v in [t, px, py, pz, vx, vy, vz]):
            times.append(t.text)
            positions.append([float(px.text), float(py.text), float(pz.text)])
            velocities.append([float(vx.text), float(vy.text), float(vz.text)])
    
    # Get acquisition date
    start_time = root.find('.//startTime')
    date = start_time.text[:10] if start_time is not None else 'unknown'
    
    # Get subswath info
    swath = os.path.basename(best_xml).split('-')[1] if '-' in os.path.basename(best_xml) else 'unknown'
    
    return {
        'date': date,
        'swath': swath,
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'times': times,
        'xml': os.path.basename(best_xml),
    }

# Read all SAFE directories
safes = sorted(glob(os.path.join(SAFE_DIR, '*.SAFE')))
print(f"Found {len(safes)} SAFE directories\n")

all_orbits = []
for i, safe in enumerate(safes):
    name = os.path.basename(safe)
    orb = get_orbit_state_vectors(safe)
    if orb is None:
        print(f"  [{i+1}] FAIL: {name}")
        continue
    orb['safe'] = name
    all_orbits.append(orb)
    pos_mid = orb['positions'][len(orb['positions'])//2]
    alt = np.linalg.norm(pos_mid) - 6371000
    print(f"  [{i+1}] {orb['date']}  swath={orb['swath']}  "
          f"alt={alt/1000:.1f}km  n_orbits={len(orb['positions'])}")

# =============================================
# Step 3c: Compute perpendicular baselines
# =============================================
print(f"\n{'=' * 60}")
print("STEP 3c: Computing perpendicular baselines")
print("=" * 60)

if len(all_orbits) < 2:
    print("ERROR: Need at least 2 images for baseline computation")
    exit(1)

# Use the middle image as master (best baseline distribution)
master_idx = len(all_orbits) // 2
master = all_orbits[master_idx]
print(f"Master image: {master['date']} (index {master_idx})")

# For each image, compute B_perp relative to master
# Use the middle orbit state vector as representative position
def compute_bperp(master_orb, slave_orb):
    """
    Compute perpendicular baseline between two SAR acquisitions.
    Uses the middle orbit state vector from each.
    """
    # Middle positions
    m_pos = master_orb['positions'][len(master_orb['positions'])//2]
    s_pos = slave_orb['positions'][len(slave_orb['positions'])//2]
    m_vel = master_orb['velocities'][len(master_orb['velocities'])//2]
    
    # Baseline vector
    baseline = s_pos - m_pos
    
    # Look direction: from satellite to ground (approximate as nadir - satellite)
    # More precisely: from satellite to Giza
    # Convert Giza lat/lon to ECEF
    lat_r = np.radians(GIZA_LAT)
    lon_r = np.radians(GIZA_LON)
    R_earth = 6371000.0
    giza_ecef = np.array([
        R_earth * np.cos(lat_r) * np.cos(lon_r),
        R_earth * np.cos(lat_r) * np.sin(lon_r),
        R_earth * np.sin(lat_r),
    ])
    
    look = giza_ecef - m_pos
    look_norm = look / np.linalg.norm(look)
    
    # Along-track direction (velocity)
    vel_norm = m_vel / np.linalg.norm(m_vel)
    
    # Cross-track direction (perpendicular to both look and velocity)
    cross = np.cross(look_norm, vel_norm)
    cross_norm = cross / np.linalg.norm(cross)
    
    # Perpendicular baseline = projection of baseline onto cross-track
    B_perp = np.dot(baseline, cross_norm)
    
    # Parallel baseline = projection onto look direction
    B_par = np.dot(baseline, look_norm)
    
    # Slant range
    slant_range = np.linalg.norm(look)
    
    return B_perp, B_par, slant_range

print(f"\n{'Date':>12} {'B_perp (m)':>12} {'B_par (m)':>12} {'Slant R (km)':>14}")
print("-" * 55)

B_perps = []
dates = []
for i, orb in enumerate(all_orbits):
    bp, bpar, sr = compute_bperp(master, orb)
    B_perps.append(bp)
    dates.append(orb['date'])
    marker = " <-- MASTER" if i == master_idx else ""
    print(f"{orb['date']:>12} {bp:12.1f} {bpar:12.1f} {sr/1000:14.1f}{marker}")

B_perps = np.array(B_perps)

# =============================================
# Step 3d: Compute kz and resolution
# =============================================
print(f"\n{'=' * 60}")
print("STEP 3d: Tomographic parameters")
print("=" * 60)

wavelength = 0.05546  # C-band Sentinel-1
slant_range = np.mean([compute_bperp(master, orb)[2] for orb in all_orbits])
inc_angle = np.radians(35)  # approximate for IW mode

kz = 4 * np.pi * B_perps / (wavelength * slant_range * np.sin(inc_angle))

# Remove master (kz=0)
kz_nonzero = kz[kz != 0]
kz_sorted = np.sort(kz)
kz_diffs = np.diff(kz_sorted)
kz_diffs_nonzero = kz_diffs[kz_diffs > 0]

kz_span = np.max(kz) - np.min(kz)
kz_step_min = np.min(kz_diffs_nonzero) if len(kz_diffs_nonzero) > 0 else 0

resolution = 2 * np.pi / kz_span if kz_span > 0 else float('inf')
ambiguity = 2 * np.pi / kz_step_min if kz_step_min > 0 else float('inf')

print(f"Number of images: {len(B_perps)}")
print(f"B_perp range: {np.min(B_perps):.1f} to {np.max(B_perps):.1f} m")
print(f"B_perp span: {np.max(B_perps) - np.min(B_perps):.1f} m")
print(f"")
print(f"kz span: {kz_span:.4f} rad/m")
print(f"kz min step: {kz_step_min:.6f} rad/m")
print(f"")
print(f"*** RESULTS ***")
print(f"Vertical resolution: {resolution:.1f} m")
print(f"Ambiguity height: {ambiguity:.1f} m")
print(f"")

# Context
targets = [
    ("King's Chamber", 43),
    ("Queen's Chamber", 21),
    ("Recently found corridor", 7),
    ("Below pyramid base", -50),
    ("Biondi deep claims", -600),
]

print(f"*** FEASIBILITY ***")
for name, depth in targets:
    within = abs(depth) < ambiguity
    resolvable = resolution < abs(depth) / 2 if depth != 0 else True
    status = "YES" if (within and resolvable) else "NO"
    reason = ""
    if not within:
        reason = f"outside ambiguity ({ambiguity:.0f}m)"
    elif not resolvable:
        reason = f"resolution too coarse ({resolution:.0f}m)"
    print(f"  {name:30s} {depth:6d}m  -> {status}  {reason}")

# Save for next step
import json
result = {
    'master_date': master['date'],
    'master_idx': master_idx,
    'dates': dates,
    'B_perps': B_perps.tolist(),
    'kz': kz.tolist(),
    'wavelength': wavelength,
    'slant_range': float(slant_range),
    'inc_angle': float(inc_angle),
    'resolution': float(resolution),
    'ambiguity': float(ambiguity),
}
with open('baselines.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nBaselines saved to baselines.json")
