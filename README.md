# Independent assessment: single-pass SAR Doppler tomography

Version 3 of the technical note, November 2026.

**Manuscript:** `paper/technical_note_SAR_tomography.md`
**PDF:** `paper/technical_note_SAR_tomography.pdf` (rebuild with `scripts/build_pdf.sh`; pandoc and XeLaTeX or LuaLaTeX)
**Changes:** `CHANGELOG.md`, and Appendix A of the note

Earlier versions on Zenodo: v1 [10.5281/zenodo.19574701](https://doi.org/10.5281/zenodo.19574701) (April 2026), v2 [10.5281/zenodo.23593771](https://doi.org/10.5281/zenodo.23593771) (October 2026).

The paper under review, Biondi & Malanga, *Remote Sensing* 2022, 14, 5231, was retracted on 10 August 2026 (*Remote Sensing* 2026, 18, 2679; [doi:10.3390/rs18162679](https://doi.org/10.3390/rs18162679)). The retraction is a change of status. It is not evidence for this note.

## What version 3 says

- **Geometry.** One monostatic SAR image has no elevation aperture. Azimuth Doppler sub-apertures are not a perpendicular baseline. The v1 figures built on such a baseline (618 mm, 285 m, 8,559 m, 660 m) stay withdrawn.
- **Sensitivity (new).** The vibration series behind the published interior image has a scale of 1e-7 m. At X-band that is 4e-5 rad of phase, 250 to 2,500 times below the phase noise of one measurement, and at the pyramid's measured 2.3 Hz it is 0.0014 mm/s, about 700 times below the error of the validated spaceborne measurement of this kind. The 0.92 m resolution comes from a chosen frequency of 12,500 Hz; the printed arithmetic uses 25,000 Hz and twice the stated aperture, and the arXiv preprint prints 3.71 m.
- **Our own tests, with controls (new).** Random input reproduces the coherence table, the 240-run sweep summary and the 15-image inversion table of v1. Planted targets show what each test could have detected: nothing about height or depth. All four empirical tests are withdrawn as evidence.
- **Multi-pass.** Real baselines give elevation resolution (64.2 m for the 15-image Sentinel-1 stack). X- and C-band do not reach the chambers through intact limestone. Sparse stacks produce deep sidelobes as strong as the scatterers that cause them.

This is not a reproduction of the published processing chain. For one with controls, see Foreman (2026), <https://github.com/Hassanforeman/subsurface-sar-tomo>. For an audit of v1 of this repository, see Pastore (2026), [doi:10.5281/zenodo.21965821](https://doi.org/10.5281/zenodo.21965821); the note answers it point by point in Section 7.3.

## Repository structure

```
├── paper/technical_note_SAR_tomography.md
├── paper/technical_note_SAR_tomography.pdf
├── CHANGELOG.md
├── scripts/
│   ├── sensitivity_budget.py     # Section 4; no data needed
│   ├── noise_controls.py         # Section 5; no data needed
│   ├── sidelobe_toy.py           # Section 6.3; no data needed
│   ├── build_pdf.sh              # pandoc + XeLaTeX/LuaLaTeX
│   ├── data_dir.py               # SAR_DATA_DIR or DATA_DIR, else ./data
│   ├── incidence.py              # SICD incidence, else angle at the target
│   ├── verify_v2.py              # SICD metadata (needs data)
│   ├── fix_coherence.py          # v1 coherence table (needs data)
│   ├── sweep_v2.py               # v1 sweep (needs data)
│   ├── step3_baselines.py        # Sentinel-1 baselines (needs data)
│   ├── run_tomo.py               # v1 multi-pass inversion (needs data)
│   └── multi_site_tomo.py        # v1 multi-site inversion (needs data)
└── deprecated/                   # v1 scripts, including aliasing_simulation.py
```

## Running the scripts that need no data

```bash
pip install numpy scipy
python3 scripts/sensitivity_budget.py   # seconds
python3 scripts/noise_controls.py       # about four minutes
python3 scripts/sidelobe_toy.py         # under a minute
```

Seeds are fixed, so the printed numbers match the note.

## The data scripts

They are kept so the v1 numbers can be traced. Each states its known limits in its header. They were not re-run for v2 or v3, because the scene files used for v1 were not retained. Without local data they exit with a message and print no measurements.

`SAR_DATA_DIR` (or `DATA_DIR`) points at the data root. The default is `<repo>/data`.

Umbra X-band spotlight SICD, AWS Open Data, CC BY 4.0:

- Giza: `5aa49658-ecf9-4504-afee-281f43fb076e` (UMBRA-04, 2023-03-08)
- Vesuvius: `a73184a7-2b25-4dfd-b750-1da35653be44` (UMBRA-05, 2023-11-15)

```
$SAR_DATA_DIR/umbra/giza/2023-03-08-07-57-53_UMBRA-04_SICD.nitf
$SAR_DATA_DIR/umbra/vesuvius/2023-11-15-19-47-28_UMBRA-05_SICD.nitf
$SAR_DATA_DIR/sentinel1_giza/          # Sentinel-1A SLC zips
$SAR_DATA_DIR/sentinel1_giza_safe/     # extracted SAFE trees
```

Sentinel-1A IW SLC, Track 58 ascending, January to December 2023, from the Alaska Satellite Facility. Product IDs were not recorded.

```bash
aws s3 cp --no-sign-request \
  s3://umbra-open-data-catalog/sar-data/tasks/5aa49658-ecf9-4504-afee-281f43fb076e/ \
  ./data/umbra/giza/ --recursive
pip install sarpy tifffile matplotlib
```

## Status of the v1 numbers

| Quantity | Value | Status in v3 |
|----------|-------|--------------|
| Single-pass elevation resolution | ~285 m | Withdrawn (v2) |
| Single-pass baseline per sub-aperture | ~618 mm | Withdrawn (v2) |
| Coherence at sub-aperture distance 1 | 0.443 Giza, 0.446 Vesuvius | Withdrawn. Noise gives 0.443. |
| Fixed-parameter `SNR_energy` | 1.87 ± 0.54 | Withdrawn. No script in the repository computes it as described. |
| Sweep, mean `SNR_peak_mean` difference | −0.021 ± 0.043, 0 of 240 above 1.5x | Withdrawn. Noise against noise gives +0.028 ± 0.048, 0 of 240. |
| Pyramid and control profile correlation | 0.9913 | Withdrawn. Random input gives a median of 0.9995. |
| 15-image elevation resolution | 64.2 m | Kept. Orbit geometry only. |

## Citation

```
Pomposi, S. "On the Reproducibility of Single-Pass SAR Doppler Tomography
for Subsurface Imaging: An Independent Assessment." Version 3. Zenodo, 2026.
```

## License

Code: MIT
Paper: CC BY 4.0
