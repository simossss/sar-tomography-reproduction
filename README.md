# Independent assessment: single-pass SAR Doppler tomography

Manuscript draft for review of the Zenodo v1 note [10.5281/zenodo.19574701](https://zenodo.org/records/19574701). This repository revision does **not** upload a new Zenodo version and does not change the punkjazz.ai site.

**Manuscript:** `paper/technical_note_SAR_tomography.md`  
**Short changelog:** `CHANGELOG.md`

Biondi & Malanga, *Remote Sensing* 2022, 14, 5231, was retracted on 10 August 2026 (*Remote Sensing* 2026, 18, 2679; [doi:10.3390/rs18162679](https://doi.org/10.3390/rs18162679)). The retraction is a status update. It is not evidence for this note’s geometry.

## What v2 holds

- A single monostatic SAR image has **no elevation aperture**. Azimuth Doppler sub-apertures are not a perpendicular baseline \(B_\perp\). The v1 claim of ~618 mm \(B_\perp\) and ~285 m single-pass elevation resolution is **withdrawn**, including the 6.6× / 13.6× comparisons and the Vesuvius ~660 m bound.
- The 240-experiment sweep tests **this beamformer** (\(A^H Y\) on sampled columns). It does not estimate displacements by cross-correlation and does not apply \(A^\dagger Y\). It does not show that the full published chain fails. Reported v1 `SNR_peak_mean` difference: −0.021 ± 0.043 (0 of 240 above a 1.5× threshold). That figure was not recomputed here. It is not the §4.3 `SNR_energy` value (1.87 ± 0.54).
- Sub-aperture coherence is **similar** between the Giza and Vesuvius patches (differences 0.001–0.004), consistent with spectral decorrelation.
- The southeast control is not empty Sahara (Shabramant / settled fringe). Gran Sasso is a public claim only; this note has no Gran Sasso results.
- Multi-pass geometry can improve elevation resolution. **X- and C-band do not reach** the Khufu chambers through intact limestone. The 15-image Sentinel-1 result is a preliminary diagnostic (pyramid–control correlation 0.9913 under simplified processing), not a physical negative baseline.
- Section 6.3 is a toy sidelobe illustration. The retracted paper’s Table 2 lists 12 products, not 6.

## Titles (not yet aligned)

| Surface | Title |
|---------|--------|
| This manuscript | On the Reproducibility of Single-Pass SAR Doppler Tomography for Subsurface Imaging: An Independent Assessment |
| Zenodo v1 record | Independent Reproduction Attempt of SAR Doppler Tomography for Subsurface Imaging of the Great Pyramid of Giza |

Use one title on the PDF and the Zenodo record when a v2 deposit is approved. This branch does not retitle the record.

## Repository structure

```
├── paper/technical_note_SAR_tomography.md
├── CHANGELOG.md
├── scripts/
│   ├── data_dir.py               # SAR_DATA_DIR or DATA_DIR, else ./data
│   ├── incidence.py              # SICD incidence, else angle at the target
│   ├── verify_v2.py              # Metadata and along-track quantities
│   ├── sweep_v2.py               # Partial beamformer sweep
│   ├── fix_coherence.py          # Sub-aperture coherence
│   ├── step3_baselines.py        # Sentinel-1 baselines
│   ├── run_tomo.py               # Preliminary multi-pass inversion
│   ├── multi_site_tomo.py        # Preliminary multi-site comparison
│   └── aliasing_simulation.py    # Toy PSF; no SAR data required
└── deprecated/                   # v1 scripts kept for the old Δx formula
```

| Script | Role |
|--------|------|
| `verify_v2.py` | Reads SICD metadata. Prints the withdrawn \(\Delta x\sin\Delta\theta\) scalar only as a non-result. |
| `sweep_v2.py` | 240 combinations of this \(A^H Y\) beamformer. Not a full-chain reproduction. |
| `fix_coherence.py` | Local-window sub-aperture coherence. |
| `aliasing_simulation.py` | Toy PSF. A 6-baseline case is not “Table 2.” |
| `step3_baselines.py`, `run_tomo.py`, `multi_site_tomo.py` | Preliminary Sentinel-1 geometry and inversion. Missing coregistration. |

## Data

`SAR_DATA_DIR` (or `DATA_DIR`) points at the data root. The default is `<repo>/data`.

Umbra X-band Spotlight SICD, AWS Open Data, CC-BY 4.0:

- Giza: `5aa49658-ecf9-4504-afee-281f43fb076e` (UMBRA-04, 2023-03-08)
- Vesuvius: `a73184a7-2b25-4dfd-b750-1da35653be44` (UMBRA-05, 2023-11-15)

Expected paths:

```
$SAR_DATA_DIR/umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf
$SAR_DATA_DIR/umbra/vesuvius/2023-11-15-19-47-28_UMBRA-05_SICD.nitf
$SAR_DATA_DIR/sentinel1_giza/          # Sentinel-1A SLC zips
$SAR_DATA_DIR/sentinel1_giza_safe/     # extracted SAFE trees
```

Sentinel-1A IW SLC, Track 58 ascending, January–December 2023, from the Alaska Satellite Facility. The Copernicus Open Access Hub was retired in October 2023. Product IDs are not in this repo.

```bash
aws s3 cp --no-sign-request \
  s3://umbra-open-data-catalog/sar-data/tasks/5aa49658-ecf9-4504-afee-281f43fb076e/ \
  ./data/umbra/giza/ --recursive

aws s3 cp --no-sign-request \
  s3://umbra-open-data-catalog/sar-data/tasks/a73184a7-2b25-4dfd-b750-1da35653be44/ \
  ./data/umbra/vesuvius/ --recursive
```

## Running the scripts

```bash
pip install sarpy numpy scipy matplotlib tifffile
```

Without the local SLC files, the data scripts exit and do not invent measurements:

```bash
python3 scripts/incidence.py          # synthetic triangle self-check
python3 scripts/verify_v2.py          # exit 2 if the Umbra SICDs are absent
python3 scripts/sweep_v2.py           # exit 2 if the Giza SICD is absent
python3 scripts/fix_coherence.py      # exit 2 if the SICDs are absent
python3 scripts/aliasing_simulation.py  # toy only; no SAR scene
```

`sweep_v2.py` is the long run (240 experiments) once a SICD is present. A run with the incidence fix is a new execution, not a reissue of the v1 SNR table, and it still is not the published chain.

## Numbers kept from v1 (not re-measured)

| Quantity | Value | Status |
|----------|-------|--------|
| Single-pass elevation resolution | ~285 m | **Withdrawn** |
| Single-pass \(B_\perp\) per sub-aperture | ~618 mm | **Withdrawn** as an elevation baseline |
| §4.3 `SNR_energy` (Giza) | 1.87 ± 0.54 | v1 partial chain; not re-run |
| §5 mean `SNR_peak_mean` difference | −0.021 ± 0.043 | v1 beamformer; not re-run |
| Coherence, SA distance 1 | 0.443 (Giza), 0.446 (Vesuvius) | similar, not identical |
| 15-image elevation resolution | 64.2 m | multi-pass geometry only |
| Pyramid–control correlation | 0.9913 | preliminary; control is not empty desert |

## Citation

Manuscript title and Zenodo v1 title differ. Until they are aligned:

```
Pomposi, S. "On the Reproducibility of Single-Pass SAR Doppler Tomography
for Subsurface Imaging: An Independent Assessment." Manuscript draft, 2026.
Zenodo v1: 10.5281/zenodo.19574701
```

## References

1. Biondi, F.; Malanga, C. *Remote Sensing* 2022, 14, 5231. **Retracted** 2026-08-10. Notice: doi:10.3390/rs18162679.
2. Biondi, F. *Remote Sensing* 2022, 14, 3828. doi:10.3390/rs14153828. Published volcano imaging is on the order of ~3 km.
3. Reigber, A.; Moreira, A. *IEEE TGRS* 2000.
4. Zhu, X.X.; Bamler, R. *IEEE TGRS* 2010.
5. Morishima, K. et al. *Nature* 2017.
6. Perissin, D. et al. SARPROZ. ISRSE 2011.

## License

Code: MIT  
Paper: CC-BY 4.0
