# Changelog

Intended Zenodo v2 changes for 10.5281/zenodo.19574701. This repository revision is a manuscript and script draft for review. It does not upload a new Zenodo version and does not edit the punkjazz.ai site.

The scientific text is `paper/technical_note_SAR_tomography.md`. The v1 ~285 m single-pass elevation-resolution claim stays withdrawn.

## Critical

- Retraction banner for Biondi & Malanga (2022), in the abstract, introduction, and reference [1]. Notice: *Remote Sensing* 2026, 18, 2679, doi:10.3390/rs18162679 (retracted 2026-08-10). The retraction is a publication-status update. It does not validate this note’s geometry or null tests.
- Withdraw ~618 mm \(B_\perp\), ~285 m elevation resolution, the 6.6× / 13.6× chamber comparisons, and the Vesuvius ~660 m bound. Azimuth Doppler sub-apertures are not an elevation baseline. A single monostatic SAR image has no single-pass elevation aperture.
- Relabel the 240-run sweep as this beamformer only (\(A^H Y\)). It does not run displacement cross-correlation and does not run \(A^\dagger Y\). It does not show that the full published chain fails.

## High

- Separate `SNR_energy` (§4.3) from `SNR_peak_mean` (§5). Coherence is similar within 0.001–0.004, not identical.
- Chambers and the Grand Gallery are heights above the pavement/base. “Depth” is reserved for the tomographic coordinate.
- Table 2 of the retracted paper lists 12 products, not 6. Section 6.3 is a speculative toy illustration and does not explain the published structures.
- Ambiguity uses minimum baseline spacing. The universal 100–200 m plateau is withdrawn. Deep claims are limited by electromagnetic penetration and the forward model.
- Multi-pass stacks can improve resolution geometry. X- and C-band do not reach the chambers through intact limestone.
- The southeast “desert” control (29.950°N, 31.180°E, and the ~3 km SE patch) is Shabramant / settled fringe, not empty Sahara. Gran Sasso is absent from the abstract because this note has no Gran Sasso results.

## Provenance and wording

- Joe Rogan Experience #2443: 23 January 2026.
- Volcano paper: about 3 km, full DOI 10.3390/rs14153828.
- Surface vibration is discussed as phonons, not as a validated “phononic” mechanism.
- ALOS-2/4 L-band spotlight exists. That does not demonstrate chamber detection.
- Sentinel-1 stack: Sentinel-1A, Track 58 ascending, Alaska Satellite Facility. The Copernicus Open Access Hub was retired in October 2023. Product IDs are not in this repo and are not invented here.
- Acoustic \(\lambda R/(2A)\): 0.92 m is the authors’ stated figure; the stated inputs do not uniquely recover it.
- Zenodo v1 title and this manuscript title differ. Align them at upload. This draft does not retitle the record.
- AI disclosure is in the manuscript. No model is a co-author.

## Scripts

- `verify_v2.py` and `sweep_v2.py` parse. `DATA_DIR` comes from `SAR_DATA_DIR` or `DATA_DIR`, otherwise `<repo>/data`.
- Incidence is SICD `SCPCOA.IncidenceAng` when present, otherwise the geometric angle at the target. The spacecraft look angle is not incidence.
- Without local SICD or SAFE inputs the data scripts exit with a message. They were not re-run on the Umbra or Sentinel-1 scenes. No new experimental numbers are issued. This is not a full-chain reproduction.
