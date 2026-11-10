# Changelog

## Version 3 (November 2026)

The scientific text is `paper/technical_note_SAR_tomography.md`.

### Added

- Section 4, a sensitivity budget for the displacement series of the paper under review, with `scripts/sensitivity_budget.py`. It follows analysis shared by Marcin Gerwin in correspondence and uses Lotti et al. (2026) and ElGabry et al. (2026).
- `scripts/noise_controls.py`: random input and planted targets through the code behind the v1 coherence table, the 240-run sweep and the 15-image inversion.
- `scripts/sidelobe_toy.py`, replacing `aliasing_simulation.py` (moved to `deprecated/`).
- References to Foreman (2026) and Pastore (2026), with a point-by-point response to the Pastore audit in Section 7.3.

### Withdrawn as evidence

- Sub-aperture coherence comparison. Random noise reproduces the table to about 0.01. The script cuts disjoint spectral slices and the values are the estimator's bias floor.
- 240-run sweep. Noise against noise gives the same summary. The script transforms lines along the SICD row dimension, so its sub-bands are range sub-bands. Its depth axis uses the withdrawn baseline scalar.
- 15-image inversion. Random complex numbers reproduce every entry. The untruncated pseudo-inverse misses a planted scatterer at 40 dB per-image SNR.
- Fixed-parameter `SNR_energy` test. No script in the repository computes the statistic as described.

### Corrected

- The 0.92 m resolution: the journal text states 12,500 Hz and 42,000 m, the printed equations use 25,000 and 84,000, and the arXiv preprint prints 3.71 m. v2 called this a factor-of-two ambiguity.
- Grand Gallery height (was about 60 m in two places; it runs between roughly 21 and 43 m above the pavement).
- The 32.3 degrees in the scene table is the look angle at the spacecraft, not the incidence angle. The listed centre frequency (9.605 GHz) and wavelength (3.123 cm) disagreed.
- The "ambiguity height" of about 4,560 m is the smallest-gap figure and is no longer presented as a clean window.
- Multi-pass X-band resolution for a 500 to 1,000 m baseline spread: roughly 6 to 12 m (was 5 to 10 m).

### Housekeeping

- The manuscript is a standalone paper again. Draft status lines, the title-alignment note and the front-matter changelog are gone; changes are listed in Appendix A.
- The v2 PDF on Zenodo carried a draft header and unrendered mathematics. The v3 PDF is built from this repository with `scripts/build_pdf.sh`.
- Data scripts carry a "known limits" block in their headers and no longer print conclusions. Their logic is unchanged and they were not re-run: the v1 scene files were not retained.

## Version 2 (October 2026)

Intended Zenodo v2 changes for 10.5281/zenodo.19574701. This repository revision is a manuscript and script draft for review. It does not upload a new Zenodo version and does not edit the punkjazz.ai site.

The scientific text is `paper/technical_note_SAR_tomography.md`. The v1 ~285 m single-pass elevation-resolution claim stays withdrawn.

### Critical

- Retraction banner for Biondi & Malanga (2022), in the abstract, introduction, and reference [1]. Notice: *Remote Sensing* 2026, 18, 2679, doi:10.3390/rs18162679 (retracted 2026-08-10). The retraction is a publication-status update. It does not validate this note’s geometry or null tests.
- Withdraw ~618 mm \(B_\perp\), ~285 m elevation resolution, the 6.6× / 13.6× chamber comparisons, and the Vesuvius ~660 m bound. Azimuth Doppler sub-apertures are not an elevation baseline. A single monostatic SAR image has no single-pass elevation aperture.
- Relabel the 240-run sweep as this beamformer only (\(A^H Y\)). It does not run displacement cross-correlation and does not run \(A^\dagger Y\). It does not show that the full published chain fails.

### High

- Separate `SNR_energy` (§4.3) from `SNR_peak_mean` (§5). Coherence is similar within 0.001–0.004, not identical.
- Chambers and the Grand Gallery are heights above the pavement/base. “Depth” is reserved for the tomographic coordinate.
- Table 2 of the retracted paper lists 12 products, not 6. Section 6.3 is a speculative toy illustration and does not explain the published structures.
- Ambiguity uses minimum baseline spacing. The universal 100–200 m plateau is withdrawn. Deep claims are limited by electromagnetic penetration and the forward model.
- Multi-pass stacks can improve resolution geometry. X- and C-band do not reach the chambers through intact limestone.
- The southeast “desert” control (29.950°N, 31.180°E, and the ~3 km SE patch) is Shabramant / settled fringe, not empty Sahara. Gran Sasso is absent from the abstract because this note has no Gran Sasso results.

### Provenance and wording

- Joe Rogan Experience #2443: 23 January 2026.
- Volcano paper: about 3 km, full DOI 10.3390/rs14153828.
- Surface vibration is discussed as phonons, not as a validated “phononic” mechanism.
- ALOS-2/4 L-band spotlight exists. That does not demonstrate chamber detection.
- Sentinel-1 stack: Sentinel-1A, Track 58 ascending, Alaska Satellite Facility. The Copernicus Open Access Hub was retired in October 2023. Product IDs are not in this repo and are not invented here.
- Acoustic \(\lambda R/(2A)\): 0.92 m is the authors’ stated figure; the stated inputs do not uniquely recover it.
- Zenodo v1 title and this manuscript title differ. Align them at upload. This draft does not retitle the record.
- AI disclosure is in the manuscript. No model is a co-author.

### Scripts

- `verify_v2.py` and `sweep_v2.py` parse. `DATA_DIR` comes from `SAR_DATA_DIR` or `DATA_DIR`, otherwise `<repo>/data`.
- Incidence is SICD `SCPCOA.IncidenceAng` when present, otherwise the geometric angle at the target. The spacecraft look angle is not incidence.
- Without local SICD or SAFE inputs the data scripts exit with a message. They were not re-run on the Umbra or Sentinel-1 scenes. No new experimental numbers are issued. This is not a full-chain reproduction.
