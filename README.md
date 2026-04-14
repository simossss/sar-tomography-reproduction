# Independent Reproduction Attempt: SAR Doppler Tomography for Subsurface Imaging

An independent analysis of the SAR Doppler Tomography method published by Biondi & Malanga (2022) for imaging subsurface structures beneath the Great Pyramid of Giza using satellite radar.

**Paper:** [link to TechRxiv/arXiv when available]

## Summary

In March 2025, claims that satellite radar had discovered underground structures beneath the Giza pyramids went viral after a podcast appearance. The underlying method was published in *Remote Sensing* (2022, 14, 5231). No independent reproduction had been attempted.

We performed the first independent analysis. Our findings:

- **The published single-pass method has an elevation resolution of ~285 m.** The King's Chamber sits at 43 m depth — 6.6× smaller than one resolution cell. The method cannot distinguish it from the surface.
- **A 240-experiment parameter sweep finds no difference between the pyramid and empty desert.** Mean SNR difference: -0.021 ± 0.043. Zero of 240 parameter combinations favor the pyramid.
- **Sub-aperture coherence is identical between Giza (limestone) and Vesuvius (volcanic rock)** to three decimal places, consistent with spectral decorrelation rather than depth-dependent signal.
- **The deep claims (600 m to 6 km) are consistent with sidelobe artifacts** from sparse kz sampling. With 6 satellite images, the beamformer produces structured features at 70-90% of peak amplitude at all depths. With 200 images, these artifacts are suppressed to 10-15%.
- **The paper acknowledges SARPROZ**, a multi-pass interferometric tool, while describing the method as operating on a single image. These are fundamentally different techniques.

The shallow chamber detections (20-80 m) may be real — but from standard multi-pass SAR tomography, an established technique, not from the novel single-pass method described in the paper.

## Repository Structure

```
├── paper/
│   └── technical_note_SAR_tomography.md    # Full technical note
│
├── scripts/
│   ├── verify_v2.py              # Core geometric proof (corrected Doppler FM rate)
│   ├── sweep_v2.py               # 240-experiment parameter sweep (Giza vs desert)
│   ├── fix_coherence.py          # Sub-aperture coherence (Giza vs Vesuvius)
│   ├── step3_baselines.py        # Sentinel-1 multi-pass baseline computation
│   ├── run_tomo.py               # Multi-pass tomographic inversion (simplified)
│   ├── multi_site_tomo.py        # 20-site comparison (simplified)
│   └── aliasing_simulation.py    # Sidelobe contamination PSF illustration
│
├── deprecated/
│   ├── verify.py                 # v1 — contains Δx formula error (kept for transparency)
│   ├── sweep.py                  # v1 — unused seismic_vel parameter, old formula
│   └── tomogram_2d.py            # Produced unusable images
│
└── README.md
```

### Script Roles

| Script | Type | What it does |
|--------|------|-------------|
| `verify_v2.py` | **First-principles proof** | Extracts SAR metadata, computes Doppler FM rate, derives elevation resolution (~285 m) from kz geometry |
| `sweep_v2.py` | **Empirical null test** | Sweeps 240 parameter combinations, compares pyramid vs desert control. Same general processing family as published method, not exact replication |
| `fix_coherence.py` | **Supporting evidence** | Measures sub-aperture coherence decay. Consistent with spectral decorrelation, identical between geologically distinct sites |
| `aliasing_simulation.py` | **PSF illustration** | Synthetic toy model showing how 6 images produce sidelobes at all depths while 200 images suppress them |
| `step3_baselines.py` | **Approximate geometry** | Computes Sentinel-1 perpendicular baselines from orbit state vectors |
| `run_tomo.py` | **Preliminary** | Multi-pass tomographic inversion without full InSAR coregistration (see caveats in paper) |
| `multi_site_tomo.py` | **Preliminary** | 20-site comparison without full coregistration (see caveats in paper) |

## Data Sources (All Free)

### Umbra X-band Spotlight SICD
- **Source:** AWS Open Data Program, `s3://umbra-open-data-catalog/`
- **License:** CC-BY 4.0
- **Scenes used:**
  - Giza: `5aa49658-ecf9-4504-afee-281f43fb076e` (UMBRA-04, 2023-03-08)
  - Vesuvius: `a73184a7-2b25-4dfd-b750-1da35653be44` (UMBRA-05, 2023-11-15)

Download with AWS CLI (no account required):
```bash
aws s3 cp --no-sign-request \
  s3://umbra-open-data-catalog/sar-data/tasks/5aa49658-ecf9-4504-afee-281f43fb076e/ \
  ./data/umbra/giza/ --recursive

aws s3 cp --no-sign-request \
  s3://umbra-open-data-catalog/sar-data/tasks/a73184a7-2b25-4dfd-b750-1da35653be44/ \
  ./data/umbra/vesuvius/ --recursive
```

### Sentinel-1 C-band IW SLC
- **Source:** Alaska Satellite Facility (https://search.asf.alaska.edu/)
- **Account required:** Free Earthdata account (https://urs.earthdata.nasa.gov/)
- **Track:** 58 Ascending
- **Period:** January – December 2023
- **Images:** 15 acquisitions, ~82 GB total
- **Subswath:** IW2 covers Giza (not IW1)

Search parameters for ASF:
```
Platform: Sentinel-1A
Beam Mode: IW
Flight Direction: Ascending
Relative Orbit: 58
Start Date: 2023-01-01
End Date: 2023-12-31
Processing Level: SLC
```

## Reproducing the Results

### Requirements
```bash
pip install sarpy numpy scipy matplotlib tifffile
```

### Core analysis (single-pass geometry)
```bash
# Requires Umbra Giza + Vesuvius SICD files
python scripts/verify_v2.py
python scripts/fix_coherence.py
python scripts/sweep_v2.py
```

`verify_v2.py` runs in seconds. `fix_coherence.py` takes ~1 minute. `sweep_v2.py` takes ~20-30 minutes (240 experiments on 800×800 patches).

### Multi-pass analysis (Sentinel-1)
```bash
# Requires 15 Sentinel-1 SLC files downloaded and extracted
python scripts/step3_baselines.py
python scripts/run_tomo.py
python scripts/multi_site_tomo.py
```

### Sidelobe illustration (no data required)
```bash
python scripts/aliasing_simulation.py
```

## Key Numbers

| Quantity | Value |
|----------|-------|
| Single-pass B_⊥ per sub-aperture | 618 mm |
| Single-pass elevation resolution | 285 m |
| King's Chamber depth | 43 m |
| Resolution / target ratio | 6.6× (unresolvable) |
| Sweep: Giza mean SNR | 1.118 ± 0.077 |
| Sweep: Desert mean SNR | 1.139 ± 0.082 |
| Sweep: experiments favoring pyramid | 0 of 240 |
| Coherence (Giza, distance 1) | 0.443 |
| Coherence (Vesuvius, distance 1) | 0.446 |
| Multi-pass 15-image resolution | 64.2 m |
| Multi-pass: pyramid-desert correlation | 0.991 |

## Citation

```
Pomposi, S. "Independent Reproduction Attempt of SAR Doppler Tomography 
for Subsurface Imaging of the Great Pyramid of Giza." Zenodo, 2026.
```

## References

1. Biondi, F.; Malanga, C. "Synthetic Aperture Radar Doppler Tomography Reveals Details of Undiscovered High-Resolution Internal Structure of the Great Pyramid of Giza." *Remote Sensing* 2022, 14, 5231.
2. Reigber, A.; Moreira, A. "First Demonstration of Airborne SAR Tomography Using Multibaseline L-Band Data." *IEEE TGRS* 2000.
3. Zhu, X.X.; Bamler, R. "Tomographic SAR Inversion by L1-Norm Regularization." *IEEE TGRS* 2010.
4. Morishima, K. et al. "Discovery of a Big Void in Khufu's Pyramid by Observation of Cosmic-Ray Muons." *Nature* 2017.
5. Perissin, D. et al. "The SARPROZ InSAR tool." *ISRSE* 2011.

## License

Code: MIT  
Paper: CC-BY 4.0
