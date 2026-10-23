# On the Reproducibility of Single-Pass SAR Doppler Tomography for Subsurface Imaging: An Independent Assessment

**Authors:** Simone Pomposi, Independent Researcher

**Date:** April 2026 (v1 Zenodo deposit)  
**Version:** 2 manuscript draft (2026-10-23, Europe/Paris UTC+2)  
**Zenodo v1:** 10.5281/zenodo.19574701  
**Status:** Manuscript draft only — no Zenodo upload; no site edits.

**Title alignment (note only; record not edited).** The Zenodo v1 record title is “Independent Reproduction Attempt of SAR Doppler Tomography for Subsurface Imaging of the Great Pyramid of Giza” (10.5281/zenodo.19574701). This manuscript title is “On the Reproducibility of Single-Pass SAR Doppler Tomography for Subsurface Imaging: An Independent Assessment.” Intended v2 action, after review and not performed here: use one title on the PDF and on the Zenodo record. This draft keeps the manuscript title and does not retitle the record.

---

## Changelog vs Zenodo v1

| # | Severity | Change |
|---|----------|--------|
| 1 | **Critical** | Front-matter **retraction banner** for Biondi & Malanga (2022); Ref [1] marked RETRACTED; cite notice DOI 10.3390/rs18162679. Independent critique is consistent with, **not validated by**, the retraction. |
| 2 | **Critical** | **Withdraw** numerical single-pass elevation resolution **~285 m**, **618 mm \(B_\perp\)** elevation claim, **6.6× / 13.6×**, and Vesuvius **~660 m** bound pending a valid elevation-aperture derivation from state vectors. Argue **zero single-pass elevation aperture** (azimuth sub-apertures ≠ elevation baseline) + vibration-model dependence. Conclusion **strengthened**. |
| 3 | **Critical** | Relabel §5 sweep as a test of **this Doppler beamformer** (averaging sub-apertures / \(A^H Y\)); does **not** implement displacement cross-correlation or stated \(A^\dagger Y\); does **not** assess the full published micro-motion chain. |
| 4 | **High** | Unify SNR labels: §4.3 = `SNR_energy` (per-pixel peak/mean then average); §5 = `SNR_peak_mean` (column peak/mean over ~100 columns). Same word “SNR” no longer conflated. Soften coherence “identical” → “similar within 0.001–0.004.” |
| 5 | **High** | Chambers / Gallery: **height above pavement/base** throughout; reserve “depth” for tomographic coordinate with explicit zero. |
| 6 | **High** | Sidelobe §6.3: original Table 2 lists **12** products (6 external pairs + 6 internal single-image), not 6; downgrade to **speculative illustration**; drop “explains published structures.” |
| 7 | **High** | Ambiguity: fix mean- vs min-spacing; withdraw universal **100–200 m plateau**; lead with **EM penetration**, not ambiguity cap. |
| 8 | **High** | Abstract / §6.2: multi-pass can improve **resolution geometry**; **X/C-band cannot reach chambers** through solid limestone. Drop Gran Sasso from abstract (no results in paper). |
| 9 | **High/Med** | “Desert” controls: 29.950°N 31.180°E ≈ Shabramant (irrigated/settled); ~3 km SE of Khufu = urban fringe—not empty Sahara. Relabel; recommend true desert W/SW. Sweep patch geolocation caveat. |
| 10 | **Med+** | Rogan date **2026-01-23** (Episode #2443); volcano imaging ~**3 km** (not 6 km as headline claim) + full DOI 10.3390/rs14153828; “phonons”/surface vibration wording; **ALOS-2/4 L-band spotlight exists**; ASF provenance (Open Access Hub retired Oct 2023); Sentinel-1A mission letter for the Track 58 stack (the v1 Sentinel-1C Gran Sasso row had no results); AI disclosure; acoustic \(\lambda R/(2A)\) arithmetic note; Zenodo title ≠ manuscript title (align at upload, not here). `verify_v2.py` / `sweep_v2.py` parse, define `DATA_DIR`, and take incidence from SICD metadata. No new scene numbers; full-chain reproduction is not claimed. Sentinel-1 product IDs are not in the repo and are not invented. |

---

> ### Publication status notice (added in v2)
>
> Biondi & Malanga (2022), *Remote Sensing* 14, 5231 ([doi:10.3390/rs14205231](https://doi.org/10.3390/rs14205231)), was **retracted on 10 August 2026** by the journal (*Remote Sensing* 2026, 18, 2679; [doi:10.3390/rs18162679](https://doi.org/10.3390/rs18162679)). The editors cited serious methodological and statistical flaws; the authors disagreed. This technical note was deposited on Zenodo in **April 2026**, before the retraction. We retain the independent geometric and null-test assessment. Consistency between our critique and the journal’s later action does **not** mean the retraction validates our specific formulas, Umbra measurements, or null experiments. Readers should treat [1] as retracted literature.

---

**Abstract.** We present, to our knowledge, an independent assessment (first deposited April 2026) of the SAR Doppler Tomography method described by Biondi & Malanga (2022; **retracted 2026-08-10**) for imaging subsurface structures using satellite radar. Through first-principles analysis of single-pass Doppler sub-aperture geometry and empirical testing on Umbra X-band scenes over Giza (Egypt) and Vesuvius (Italy), plus exploratory Sentinel-1 multi-pass work over Giza, we argue that **azimuth Doppler sub-apertures within a single monostatic SAR image do not furnish an elevation aperture** in the sense required by conventional SAR tomography. Prior numerical claims in v1 of this note (~618 mm \(B_\perp\) → ~285 m elevation resolution) are **withdrawn** pending a derivation from actual state vectors and height-dependent phase; the qualitative conclusion—that the published single-pass chain does not justify chamber- or kilometer-scale detection—is **strengthened**. A 240-experiment parameter sweep over the Giza plateau and an adjacent control patch tests a **partial Doppler beamformer** (not the full published micro-motion + \(A^\dagger Y\) chain) and finds no evidence of target-specific subsurface discrimination under that implementation (mean `SNR_peak_mean` difference: −0.021 ± 0.043). Multi-pass interferometric SAR tomography using real orbital baselines can improve **elevation resolution geometry**, but under standard electromagnetic theory **X- and C-band radar do not penetrate intact limestone** to the heights of known internal chambers (~21–43 m above the pavement). This established multi-pass technique differs fundamentally from the single-pass method described in the retracted literature.

---

## 1. Introduction

Biondi & Malanga (2022) published a method for imaging the internal structure of the Great Pyramid of Giza using Synthetic Aperture Radar (SAR) Doppler Tomography [1]. That article was **retracted on 10 August 2026** [1a]; we cite it as the object of assessment, not as standing peer-reviewed method guidance. The method processes a single SAR image in the Single-Look Complex (SLC) configuration, decomposing it into Doppler sub-apertures to estimate surface micro-movements generated by seismic background activity. Through tomographic focusing of these micro-movements, the authors claim to achieve sub-meter elevation resolution at depths of several kilometers, enabling reconstruction of internal pyramid chambers and subsurface features.

Subsequent claims by the same research group extended the technique to imaging volcanic interiors beneath Vesuvius to roughly **~3 km** scale in the published volcano paper [2] (public commentary has also cited deeper figures) and subsurface structures at depths exceeding 600 m beneath the Khafre Pyramid [3]. These claims, particularly the 2025 Giza announcements, received extensive media coverage but have not been subjected to a complete independent reproduction of the proprietary processing chain.

We present an independent attempt to assess this method. Our analysis proceeds on three levels: (i) first-principles evaluation of single-pass Doppler geometry and why it does not supply a conventional elevation aperture; (ii) empirical testing on real X-band SAR data over two primary target sites (Giza, Vesuvius); and (iii) an exhaustive parameter sweep comparing a **partial** beamformer’s output over known structures versus a control patch. Gran Sasso / LNGS appears in public claims [7] but is **not** analyzed with results in this note; it is therefore omitted from the abstract.

## 2. Method Summary

The published method [1] involves the following processing chain, as described in Sections 3–4 of the original paper:

1. Acquire a single SAR SLC image (COSMO-SkyMed Second Generation, X-band, Spotlight mode)
2. Compute the 2D Discrete Fourier Transform of the SLC data
3. Decompose the azimuth (Doppler) spectrum into \(N_D\) overlapping sub-apertures via bandpass filtering
4. For each pixel along a tomographic line, estimate sub-pixel displacements between sub-aperture pairs through cross-correlation (the micro-movement estimation)
5. Construct a steering matrix \(A(K_z, z)\) encoding the phase relationship between Doppler frequency and depth (Equation 22 of [1])
6. Invert the steering matrix to obtain a tomographic depth profile \(h(z) = A^\dagger Y\) (Equation 24 of [1])

The paper reports elevation resolution of approximately 0.92 m, computed from \(\delta_z = \lambda R/(2A)\) where \(\lambda\) is described as “the sound wavelength over the Earth” (approximately 0.24 m), \(R\) is the slant range (~650 km), and \(A\) is the orbital aperture. **Arithmetic note (v2):** substituting the paper’s stated inputs into \(\lambda R/(2A)\) does not uniquely recover 0.92 m without additional choices for \(A\) (a factor-of-two scale ambiguity appears in naive recomputation, e.g. ~0.92 vs ~1.8 m); we treat 0.92 m as the authors’ stated figure, not as an independently verified evaluation of their formula.

## 3. Theoretical Analysis: Baseline Geometry

### 3.1 Why single-pass Doppler sub-apertures are not an elevation aperture

The elevation resolution and ambiguity of **conventional** SAR tomography depend on the perpendicular baseline \(B_\perp\) between observations that differ in **elevation look angle** (the angle in the plane normal to the flight track). In multi-pass tomography, different orbital passes provide such baselines of tens to hundreds of meters (Reigber & Moreira, 2000).

In single-pass Doppler sub-aperture decomposition, the diversity is **along-track (azimuth)**. Sub-apertures sample different Doppler frequencies corresponding to different azimuth squint within the synthetic aperture. A single monostatic SAR image does not contain independent elevation-angle measurements of the same scatterer: the radar line of sight for a given range-azimuth cell is fixed by the acquisition geometry. Converting along-track angular diversity into an effective **elevation** \(B_\perp\) via

\[
B_\perp = \Delta x \cdot \sin(\Delta\theta)
\]

(with \(\Delta x\) and \(\Delta\theta\) derived from Doppler steps as in v1) lacks a valid elevation-aperture derivation. That construction projects azimuth diversity into a scalar that is not the tomographic baseline normal to both flight direction and line of sight. DLR / textbook SAR tomography treats elevation aperture as absent from a single image; our own Umbra metadata imply an **azimuth** resolution of 0.264 m, not an elevation resolution of hundreds of meters from the same sub-aperture split.

**v2 withdrawal.** Numerical v1 claims of \(B_\perp \approx 618\) mm per sub-aperture step, total span 18.6 m, \(\Delta k_z \approx 7.34\times10^{-4}\) rad/m, elevation resolution ≈ **285 m**, ambiguity height ≈ 8,559 m, King’s Chamber “6.6× smaller,” Queen’s Chamber “13.6× smaller,” and Vesuvius single-pass bounds ~**660 m** are **withdrawn** pending a derivation from actual state vectors and a height-dependent phase model. Along-track quantities that remain well-defined from metadata (Doppler bandwidth, platform velocity, slant range, azimuth resolution) may still be reported as descriptive SAR parameters; they are not elevation-tomographic resolution.

### 3.2 Application to real data (descriptive parameters only)

We extracted orbital parameters directly from SICD metadata of an Umbra X-band Spotlight acquisition over Giza (2023-03-08, UMBRA-04). The 32.3° incidence in the v1 table is the value reported with that extraction. `verify_v2.py` previously used the triangle angle at the spacecraft (off-nadir). It now prefers SICD `SCPCOA.IncidenceAng` and, if that field is absent, the geometric angle at the target, and it prints the spacecraft angle only as a look-angle diagnostic. No new incidence or resolution table is issued here: the SICD files are not reprocessed in this revision. The ~285 m elevation figure stays withdrawn either way. Along-track diversity is still not an elevation aperture.

| Parameter | Value (as extracted in v1) |
|-----------|----------------------------|
| Center frequency | 9.605 GHz |
| Radar wavelength | 3.123 cm |
| Platform velocity | 7,675.6 m/s |
| Doppler bandwidth | 29,102 Hz |
| Doppler FM rate (\(k_a\)) | 5,941 Hz/s |
| Slant range | 634,971 m |
| Incidence angle (reported) | 32.3° |
| Azimuth resolution | 0.264 m |

For \(N_D = 30\) sub-apertures with \(r = 0.5\) (the paper’s convention), v1 derived Doppler step 485 Hz and along-track shift per SA ≈ 627 m. These along-track figures may be retained as descriptive; **elevation \(B_\perp\) and the 285 m resolution are not**.

### 3.3 Implications without a numerical elevation resolution

Even without a 285 m number, the qualitative geometric case against the published single-pass claims is strong:

1. **Zero single-pass elevation aperture** under conventional SAR tomography: azimuth Doppler splits do not create multi-baseline elevation sampling.
2. Known Khufu chambers sit at approximately **43 m** (King’s Chamber) and **21 m** (Queen’s Chamber) **above the pyramid pavement/base** (Petrie-scale references), with the Grand Gallery spanning intermediate heights—not “depths” below ground. Distinguishing those heights from the exterior masonry requires an elevation (or other depth-encoding) observable that a single SLC does not provide under standard models.
3. Deep claims (hundreds of meters to kilometers) additionally require either electromagnetic penetration incompatible with X-band skin depths in limestone, or an unvalidated surface-vibration (“phonon”) forward model mapping subsurface structure into the Doppler stack—with positive controls that this note does not claim to have fully implemented.

The empirical tests in Sections 4 and 5 address a **partial** processing implementation and serve as supporting null diagnostics; they are not a substitute for a full-chain reproduction.

### 3.4 Note on the paper’s resolution formula

The paper computes \(\delta_z = \lambda R/(2A) \approx 0.92\) m using \(\lambda = v_{\mathrm{seismic}}/f \approx 0.24\) m (acoustic wavelength) rather than the radar wavelength. This formula is presented as the potential resolution of the beamformer, but it conflates several notions of resolution and range. Because the acoustic-\(\lambda\) construction is internal to the retracted paper’s model and is not independently calibrated here, we do not use it to claim a validated sub-meter capability. Conventional multi-pass SAR tomography resolution remains governed by the elevation \(k_z\) span from real orbital baselines [4, 5].

## 4. Empirical Testing

### 4.1 Data Sources

We tested processing on the following datasets (Gran Sasso listed only as a public-claim target, not as a result site in this note):

| Target | Sensor | Band | Mode | Known features / notes |
|--------|--------|------|------|------------------------|
| Giza, Egypt | Umbra-04 | X (9.6 GHz) | Spotlight SICD | Chambers ~21–43 m **above base** |
| Vesuvius, Italy | Umbra-05 | X (9.6 GHz) | Spotlight SICD | Magma-chamber claims; published paper discusses ~3 km scale [2] |
| (Public claim only) Gran Sasso, Italy | — | — | — | LNGS ~1,400 m rock overburden; **no results reported here** |

All Umbra data were obtained free of charge from the AWS Open Data Program (Creative Commons CC-BY 4.0). Sentinel-1 stacks used later (Section 6) were obtained from the **Alaska Satellite Facility (ASF)**; the Copernicus Open Access Hub referenced in v1 was **retired in October 2023**—provenance text is corrected accordingly. While Umbra sensors are not identical to COSMO-SkyMed Second Generation data used in [1], orbital and radar parameters are comparable for descriptive purposes within roughly 5% for LEO X-band spotlight systems.

### 4.2 Sub-Aperture Coherence Analysis

We measured interferometric coherence between Doppler sub-apertures using the standard local-window estimator (5×5 pixel windows) on 1000×1000 pixel patches from both the Giza and Vesuvius datasets.

Coherence as a function of sub-aperture distance from the reference (SA 15 of 30):

| SA distance | Giza coherence | Vesuvius coherence |
|-------------|----------------|-------------------|
| 1 | 0.443 | 0.446 |
| 2 | 0.419 | 0.422 |
| 5 | 0.335 | 0.331 |
| 10 | 0.109 | 0.108 |
| 14 | 0.050 | 0.051 |

Coherence drops monotonically with sub-aperture distance and is **similar** between Giza and Vesuvius (pairwise differences 0.001–0.004), not literally identical. This pattern is consistent with spectral decorrelation—a geometric property of the imaging system—rather than subsurface structure. If the decorrelation encoded target-dependent depth information, two geologically distinct sites (limestone plateau vs. volcanic edifice) would be expected to produce more divergent coherence profiles. The similarity supports geometry-dominated coherence, though it does not by itself rule out a small target-dependent contribution.

### 4.3 Giza vs. Control Patch (Fixed-Parameter Null Test)

As an illustrative fixed-parameter test, we applied a **partial** tomographic processing chain (Section 5 caveats) to two regions within the same Giza SICD scene using the paper’s default parameters (\(N_D = 30\), \(r = 0.5\)):

- **Giza site:** 800×800 pixel patch centered on the Great Pyramid of Khufu (29.979°N, 31.134°E), containing known internal chambers at ~21–43 m **height above base**.
- **Control patch (v1 label: “desert”):** 800×800 pixel patch approximately 3 km southeast. **Geolocation caveat (v2):** this location is **not** empty Sahara. Coordinates near 29.95°N, 31.18°E fall in the Shabramant / settled–irrigated fringe southeast of Giza; urban and agricultural clutter are present. The v1 “featureless Sahara” description is withdrawn. Results remain a same-scene control comparison under identical processing, but they are **not** a clean desert-null. Recommended follow-up: relocate controls to true desert west/southwest of the plateau and geolocate the sweep patch from SICD annotation grids.

Identical processing parameters were applied to both regions. The tomographic output was scored using **`SNR_energy`**: for each pixel, the beamformer produces a profile along the reconstruction coordinate; `SNR_energy` is the ratio of the peak amplitude within the band corresponding to ~20–80 m **height above base** (King’s Chamber ~43 m, Queen’s Chamber ~21 m, Grand Gallery ~60 m above pavement) to the mean amplitude across the full reconstruction range, computed per-pixel in linear amplitude, then averaged across the 800×800 patch.

Result: Giza `SNR_energy` = 1.87 ± 0.54. Control `SNR_energy` = 1.87 ± 0.55. Mean difference smaller than \(10^{-4}\).

Under this partial implementation and metric, the method produces numerically indistinguishable output for both locations. This does **not** by itself prove absence of signal under the full published chain.

## 5. Parameter Sweep (Partial Beamformer Null Test)

To ensure the null result of Section 4.3 is not an artifact of a specific parameter choice, we conducted a systematic sweep of 240 parameter combinations, evaluating both the Giza and control sites under identical conditions for each combination.

**Scope limitation (Critical, v2).** The sweep implements a Doppler sub-aperture beamformer that averages or combines sub-aperture spectra / applies forms equivalent to \(A^H Y\) on single columns of the SLC spectrum. It **omits** the published displacement cross-correlation micro-movement step and does **not** implement the stated pseudoinverse \(A^\dagger Y\) chain of Equations 22–24 of [1]. The sweep is therefore a null test of **this beamformer family** across a broad parameter space—not a claim that the full Biondi & Malanga processing chain fails, and not a line-for-line reproduction of proprietary code.

Parameters varied:

| Parameter | Range tested |
|-----------|-------------|
| Sub-apertures (\(N_D\)) | 5, 10, 15, 20, 30, 50 |
| Bandwidth ratio (\(B_{DL}/B_{CD}\)) | 0.3, 0.4, 0.5, 0.6, 0.7 |
| Filter type | Rectangular, Hanning |
| Maximum reconstruction range | 200, 500, 800, 1200 m |

The sweep metric is **`SNR_peak_mean`**: ratio of peak beamformer amplitude in the 20–80 m height-above-base band to the mean amplitude across the full profile, averaged over ~100 sampled columns per patch. This is **not** the same estimator as §4.3 `SNR_energy` (full-patch per-pixel averaging); the two must not be compared as if one were a subset of the other. The along-track distance mapping uses \(\Delta x = \Delta f\cdot\lambda\cdot R/(2V)\) based on the Doppler FM rate (azimuth geometry only).

Results across all 240 experiments:

| Metric | Value |
|--------|-------|
| Best Giza `SNR_peak_mean` | 1.39 |
| Best control `SNR_peak_mean` (same parameters) | 1.32 |
| Mean Giza `SNR_peak_mean` | 1.118 ± 0.077 |
| Mean control `SNR_peak_mean` | 1.139 ± 0.082 |
| Mean Giza − control | −0.021 ± 0.043 |
| Experiments where Giza > 1.5× control | 0 of 240 |
| Giza–control correlation across experiments | 0.855 |

No parameter combination produced a response that differentiated the pyramid site from the control under the chosen 1.5× threshold. The mean difference is slightly negative (control scores marginally higher), and the 0.855 correlation indicates that both sites respond similarly to parameter changes—consistent with processing-dominated behavior in this partial implementation. We do **not** claim statistical equivalence of the sites; we report the threshold result and correlation as descriptive.

## 6. Multi-Pass Baseline Analysis

To evaluate what standard multi-pass SAR tomography could achieve over Giza **in elevation resolution geometry**, we downloaded 15 Sentinel-1A IW SLC images from a single ascending track (Track 58) spanning January to December 2023, obtained from the Alaska Satellite Facility. We computed perpendicular baselines from orbital state vectors relative to a master image (2023-06-24).

### 6.1 Measured Baselines

| Date | \(B_\perp\) (m) |
|------|---------|
| 2023-01-07 | +167.3 |
| 2023-01-31 | +81.3 |
| 2023-02-24 | +146.4 |
| 2023-03-20 | +140.6 |
| 2023-04-13 | −24.3 |
| 2023-05-07 | +31.8 |
| 2023-05-31 | +37.3 |
| 2023-06-24 | 0.0 (master) |
| 2023-07-18 | +89.6 |
| 2023-08-11 | +170.3 |
| 2023-09-04 | −16.5 |
| 2023-09-28 | +112.4 |
| 2023-10-22 | −41.5 |
| 2023-11-15 | +65.0 |
| 2023-12-09 | +93.1 |

Total \(B_\perp\) span: 211.8 m (−41.5 to +170.3 m). This yields:

\[
k_z\ \mathrm{span} = 0.0978\ \mathrm{rad/m},\quad
\text{Elevation resolution} = 2\pi / k_{z,\mathrm{span}} = 64.2\ \mathrm{m}.
\]

Ambiguity height depends on the **minimum spacing** of baselines in \(k_z\) (not the mean spacing alone). Using the minimum step implied by this stack, v1 reported ambiguity ≈ 4,559 m; that figure is retained as stack-specific, **not** as a universal satellite SAR cap.

### 6.2 Implications: resolution geometry ≠ penetration

The 64.2 m elevation resolution from 15 C-band images is insufficient to resolve the King’s Chamber (~43 m above the pyramid base) as a distinct feature within the pyramid body. The chamber falls within a single resolution cell. Multi-pass geometry can avoid the single-pass elevation-aperture problem, but **resolution is not penetration**.

For comparison, a hypothetical stack of ~200 COSMO-SkyMed X-band images (\(\lambda \approx 3.1\) cm), assuming a \(B_\perp\) spread of 500–1000 m achievable over many repeat cycles, would yield elevation resolution on the order of **5–10 m**—fine enough, geometrically, to place the King’s and Queen’s Chambers within a resolvable height scale. Under standard electromagnetic theory, however, X-band radar (\(\lambda \approx 3\) cm) has limited penetration into intact dry limestone, typically on the order of **centimeters to decimeters** depending on moisture and heterogeneity. Reaching internal chambers through solid limestone would therefore require either a non-electromagnetic surface-vibration hypothesis as proposed in [1] or propagation through air-filled passages and structural discontinuities. **v2 abstract stance:** multi-pass may supply resolution geometry; **X/C-band cannot be assumed to reach the chambers.**

Structures beyond the stack’s ambiguity height can appear as sidelobe or alias artifacts from shallower features; the limiting factor for deep claims is primarily **physics of penetration / forward model**, not a universal 100–200 m ambiguity plateau (withdrawn).

### 6.3 The Sidelobe Contamination Hypothesis (speculative illustration)

We propose—as a **hypothesis illustrated by a toy model**, not as a diagnosis of the published images—that deep subsurface claims *can* be produced by tomographic sidelobe contamination from sparse \(k_z\) sampling.

We constructed a synthetic point spread function (PSF) illustration using known ground truth: five scatterers placed at real pyramid feature heights (base at 0 m, Queen’s Chamber at 21 m, King’s Chamber at 43 m, Grand Gallery at 60 m, apex at 140 m **above base**). This is a toy forward model, not a faithful electromagnetic model of the pyramid.

**Image count correction (v2).** Table 2 of [1] lists **twelve** SAR products associated with the study (six external-pair / multi-date entries plus six internal single-image entries in the published tabulation), not six alone. v1’s “With 6 images (matching Table 2)” wording is corrected: the simulation uses a **sparse** \(k_z\) sampling scenario comparable in *order of magnitude* to a handful of baselines, labeled here as a **6-baseline toy**, and separately a 200-baseline toy matching public “200 scans” commentary [7]. These runs do **not** claim to reproduce the exact published baseline set, and they do **not** “explain” the published structures.

**With a 6-baseline toy:** The tomographic PSF has high sidelobes (v1: 70–90% of main peak) throughout a deep reconstruction range, with spurious peaks comparable in magnitude to the planted shallow scatterers.

**With a 200-baseline toy:** Sidelobe amplitudes drop (v1: 10–15%); planted shallow scatterers stand out; deep artifacts are suppressed.

**With our 15 C-band images:** Intermediate behavior (v1: sidelobes 30–50%).

If deep features were genuine and SNR-limited, adding baselines should strengthen them; in the toy, adding baselines **suppresses** deep artifacts. That pattern is diagnostic of sidelobe contamination **in the toy**. Applying the same story to [1]’s figures remains **speculative**.

### 6.4 Multi-Pass Tomographic Inversion (preliminary)

We performed standard multi-pass tomographic inversion using the 15 Sentinel-1 images over five sites within the same radar scene:

| Site | Coordinates | Geology (corrected labels) | Peak contrast (σ) |
|------|------------|----------------------------|-------------------|
| Great Pyramid of Khufu | 29.979°N, 31.134°E | Limestone, known internal chambers | 8.5 |
| Control (v1: “empty desert”) | 29.950°N, 31.180°E | **Shabramant / irrigated–settled fringe**, not empty desert | 8.6 |
| Great Sphinx | 29.975°N, 31.138°E | Limestone, known subsurface enclosure | 8.2 |
| Downtown Cairo | 30.044°N, 31.236°E | Urban, building foundations | 8.4 |
| Agricultural land | 30.080°N, 31.100°E | Irrigated soil | 8.3 |

For each site, complex pixel values were extracted from all 15 images at nominally corresponding locations determined by annotation geolocation grids, **without** full interferometric coregistration. Tomographic inversion used a standard pseudo-inverse beamformer. The reconstruction axis spanned −300 m to +300 m in 0.5 m steps (zero ≈ local reference height; positive ≈ height above reference).

All five sites produced highly similar profiles. The Pearson correlation between the Great Pyramid and the southeast control was 0.9913, with a mean absolute difference of 0.0106 (normalized amplitude). Profiles are dominated by the steering-matrix PSF.

Normalized amplitude at key heights (Great Pyramid / Control):

| Height above reference | Pyramid | Control | Difference |
|-------|---------|--------|------------|
| +140 m (apex) | 0.106 | 0.118 | −0.012 |
| +43 m (King’s Chamber) | 0.121 | 0.130 | −0.009 |
| +21 m (Queen’s Chamber) | 0.049 | 0.043 | +0.006 |
| 0 m (base) | 0.145 | 0.145 | 0.000 |
| −50 m | 0.144 | 0.142 | +0.002 |
| −100 m | 0.150 | 0.146 | +0.004 |
| −200 m | 0.203 | 0.192 | +0.011 |

No height shows a measurable pyramid-specific excess. At +43 m, the control is slightly higher. The 64.2 m elevation resolution is too coarse to resolve the chambers, and C-band does not penetrate limestone blocks regardless of baseline configuration.

**Processing limitations.** Missing coregistration, flat-earth/topographic phase removal, and coherent pixel selection mean this null is a **preliminary diagnostic**, not a physical negative baseline proving absence of signal under full InSAR tomography (softened vs v1 §7.4). Given 64.2 m resolution alone, chambers remain unresolved. A definitive multi-pass test would require X-band data with proper coregistration and phase calibration.

**L-band availability (v2).** Civilian L-band spotlight exists: **ALOS-2 / PALSAR-2** (and ALOS-4) provide L-band spotlight on the order of ~1×3 m. v1’s claim that “no current high-resolution civilian SAR satellite operates at L/P-band in spotlight mode” is **incorrect**. L-band improves penetration relative to X/C-band but still does not by itself demonstrate chamber detection at Giza.

## 7. Discussion

### 7.1 What we conclude

The single-pass SAR Doppler sub-aperture tomography method, as described in Sections 3–4 of Biondi & Malanga (2022; retracted), does not supply a conventional elevation aperture from azimuth Doppler splits. Numerical v1 elevation-resolution bounds (~285 m, 6.6×, etc.) are withdrawn; the qualitative conclusion stands and is strengthened: the published single-pass description does not justify resolving chambers at ~21–43 m above the pavement, nor kilometer-scale subsurface architecture, under standard SAR tomography. Deep claims remain unsupported by electromagnetic penetration physics and by our partial-implementation null tests.

Empirical testing supports this reading within stated limits. We find no evidence that the **tested partial beamformer** produces target-specific discrimination between the pyramid and a same-scene control under the 240-run sweep (Section 5). Sub-aperture coherence is similar between Giza and Vesuvius (Section 4.2). These results are supporting diagnostics; they are **not** a full-chain disproof of every possible proprietary variant.

Multi-pass tomography with real orbital baselines achieves meaningful elevation resolution (tens of meters with 15 C-band scenes; potentially ~5–10 m with large X-band stacks), but sparse stacks produce high sidelobes in toy models. Public deep features remain more plausibly discussed as artifacts or unverified claims than as demonstrated detections—especially given X/C-band penetration limits.

### 7.2 What remains open

**Multi-pass processing and the “200 scans” claim.** Table 2 of [1] tabulates multiple SAR products (twelve entries in the published table). Acknowledgments thank Daniele Perissin for making SARPROZ available, “through which many calculations were carried out.” SARPROZ is a professional multi-temporal InSAR tool [8] for PSI/QPS and related multi-pass techniques; it is not a single-pass Doppler sub-aperture engine. If SARPROZ performed material steps on multi-date stacks, the implemented workflow may have differed from the single-SLC narrative in Sections 3–4. The paper does not clarify the split of labor.

In a podcast interview released **23 January 2026** (The Joe Rogan Experience, Episode #2443) [7]—not “March 2025” as v1 body text once stated—the lead author stated that “more than 200” scans have been performed on the Giza area using COSMO-SkyMed, Capella Space, and other systems. If those are 200 independent single passes, each inherits the single-pass elevation-aperture problem. If they form a multi-pass stack with hundreds of meters of \(B_\perp\) spread, elevation resolution could reach ~5 m scale—still subject to **penetration** limits for X-band through limestone.

We withdraw the claim that ambiguity height “plateaus at 100–200 m regardless of the number of passes.” Ambiguity depends on baseline sampling; deep claims fail first on penetration / forward-model grounds.

**Validation claims.** The same interview claims imaging of LNGS at ~1.4 km overburden and states that the method grabs “entropy” on the surface rather than penetrating. Surface-vibration interpretations still require a validated mapping from subsurface structure to the measured stack; RF geometry arguments alone do not fully refute an elastic inverse problem, nor does a successful reconstruction prove the mechanism (artifacts remain possible). Forward modeling and positive controls are required.

**Proprietary processing.** Patent filings (Malta application 4451, July 2022; later US filing discussed in [7]) mean published equations may not fully represent implemented code.

**Different physical mechanism.** The paper describes the signal in terms of **phonons** / surface vibration rather than electromagnetic penetration. Geometry constraints on elevation sampling still apply to any inversion that relies on Doppler sub-aperture diversity alone; an entirely different undocumented approach cannot be ruled out.

### 7.3 Summary of feasibility

**Standard SAR tomography (EM signal must reach target):**

| Claimed target | Height / depth scale | Single-pass Doppler | 15-image C-band | ~200-image X-band |
|---|---|---|---|---|
| Khufu chambers | ~21–43 m above base | No elevation aperture from azimuth SAs | No (res. ~64 m; no penetration) | Geometry maybe; **X-band skin depth ~cm–dm in limestone** |
| Gran Sasso LNGS | ~1,400 m overburden | Not demonstrated here | No | No |
| Khafre deep claims | ~600 m+ | Not demonstrated | No | No |
| Vesuvius magma | km-scale (~3 km in [2]; deeper in commentary) | Not demonstrated | No | No |

Under standard SAR tomography, X- and C-band have limited penetration into intact limestone. Even with ~5 m geometric resolution, the electromagnetic signal is unlikely to reach internal chambers through solid rock. L-band (ALOS-2/4 spotlight available) improves penetration but does not automatically yield chamber maps.

**Surface-vibration / phonon mechanism as claimed in [1]:**

| Claimed target | Single-pass Doppler | Multi-pass |
|---|---|---|
| Khufu chambers | No conventional elevation aperture from azimuth SAs; mechanism unvalidated | Resolution geometry plausible (~5 m) if mechanism valid—**unvalidated** |
| Deep claims (600 m–km) | Not supported by our tests | Sparse stacks → sidelobe risk; mechanism unvalidated |

### 7.4 Recommendations

- Authors of [1] (and related claims) should clarify how many SAR images enter each inversion and which steps SARPROZ performed.
- Independent reproduction should use COSMO-SkyMed Second Generation Spotlight SLC via ASI/ESA programs, with documented displacement estimation, geometry, and positive controls—before claiming faithful full-chain reproduction.
- Our 15-image Sentinel-1 experiment is a **preliminary** negative diagnostic under simplified processing, not a definitive physical null.
- Relocate “desert” controls to true desert W/SW; publish geolocation of every patch.
- `verify_v2.py` and `sweep_v2.py` now define `DATA_DIR` (`SAR_DATA_DIR` or `DATA_DIR`, otherwise `<repo>/data`) and read incidence from SICD `SCPCOA.IncidenceAng` when present. They still omit displacement estimation and \(A^\dagger Y\). Archive code and data with a Zenodo version only after review; this draft is not that archive, and it does not replace the v1 numerical tables with a re-run.
- Do not treat the 2026 retraction as proof of this note’s geometry.

## 8. Reproducibility

All code used in this analysis is available at https://github.com/simossss/sar-tomography-reproduction. The analysis requires freely available data and open-source software:

- Umbra SICD data: AWS Open Data Program, `s3://umbra-open-data-catalog/`
- Sentinel-1 SLC data: **Alaska Satellite Facility** (https://search.asf.alaska.edu/), Sentinel-1A, Track 58 Ascending, dates in §6.1 (2023). Product IDs are not stored in this repository and are not invented here. The Copernicus Open Access Hub was retired in October 2023 and is not the source of this stack.
- Analysis code: Python 3.12, sarpy, numpy, scipy
- Data root: `SAR_DATA_DIR` or `DATA_DIR`, otherwise `<repo>/data`
- Scripts:
  - `verify_v2.py` — SICD metadata and along-track quantities. Incidence from `SCPCOA.IncidenceAng` when present. Prints the withdrawn v1 \(\Delta x\sin\Delta\theta\) scalar only as a non-result. Not trusted as elevation \(B_\perp\), and not a reinstatement of ~285 m.
  - `fix_coherence.py` — sub-aperture coherence
  - `sweep_v2.py` — 240-experiment **partial beamformer** sweep (\(A^H Y\); no displacement cross-correlation; no \(A^\dagger Y\))
  - `step3_baselines.py` — multi-pass baselines from Sentinel-1 state vectors
  - `run_tomo.py` / `multi_site_tomo.py` — preliminary multi-pass inversion (Section 6.4 caveats)
  - `aliasing_simulation.py` — toy PSF sidelobe illustration (sparse 6-baseline toy vs 200-baseline toy; not a match to the 12-product Table 2)

Running the data scripts without local SICD or SAFE inputs exits with a message. That exit does not regenerate the v1 tables. This revision does not claim a faithful full-chain reproduction. Compute time for a full local run remains on the order of the v1 workstation figure once the inputs are present; it was not re-timed here.

---

## AI Disclosure

Manuscript drafting, code assistance, and revision editing for this technical note used large language model tools (including Claude / Anthropic and ChatGPT / OpenAI in the original workflow). Accuracy reviews informing v2 were performed with Grok Bot, Codex/Astra, and Claude Opus. All scientific claims, code execution, and final wording are the author’s responsibility. No AI system is a co-author.

---

## References

[1] Biondi, F.; Malanga, C. “Synthetic Aperture Radar Doppler Tomography Reveals Details of Undiscovered High-Resolution Internal Structure of the Great Pyramid of Giza.” *Remote Sensing* 2022, 14, 5231. https://doi.org/10.3390/rs14205231 — **RETRACTED 2026-08-10**.

[1a] Retraction notice: *Remote Sensing* 2026, 18, 2679. https://doi.org/10.3390/rs18162679

[2] Biondi, F. “Scanning Inside Volcanoes with Synthetic Aperture Radar Echography Tomographic Doppler Imaging.” *Remote Sensing* 2022, 14, 3828. https://doi.org/10.3390/rs14153828 (full DOI; volcano paper discusses imaging on the order of ~**3 km**, not a validated 6 km detection).

[3] Biondi, F.; Malanga, C.; Mei, A. Press conference, Castel San Pietro Terme, Bologna, Italy, March 15–16, 2025. Claims of subsurface structures beneath the Khafre Pyramid extending to approximately 2 km depth. (Public / press claims, not peer-reviewed.)

[4] Reigber, A.; Moreira, A. “First Demonstration of Airborne SAR Tomography Using Multibaseline L-Band Data.” *IEEE Trans. Geosci. Remote Sens.* 2000, 38, 2142–2152.

[5] Zhu, X.X.; Bamler, R. “Tomographic SAR Inversion by L1-Norm Regularization — The Compressive Sensing Approach.” *IEEE Trans. Geosci. Remote Sens.* 2010, 48, 3839–3846.

[6] Morishima, K. et al. “Discovery of a Big Void in Khufu’s Pyramid by Observation of Cosmic-Ray Muons.” *Nature* 2017, 552, 386–390.

[7] Biondi, F. Interview on *The Joe Rogan Experience*, Episode #2443, released **23 January 2026**. Public claims cited include: Gran Sasso validation at 1.4 km depth, “more than 200” scans on Giza using multiple satellite platforms, and structures extending to ~1.2 km below the pyramids. Cited as documentation of public statements, not as peer-reviewed evidence.

[8] Perissin, D.; Wang, Z.; Wang, T. “The SARPROZ InSAR tool for urban subsidence/manmade structure stability monitoring in China.” Proceedings of 34th International Symposium on Remote Sensing of Environment, Sydney, Australia, 2011. Software: https://www.sarproz.com/

---

**Data availability:** Umbra SICD products via AWS Open Data Program (CC-BY 4.0). Sentinel-1 SLC products via the Alaska Satellite Facility. Analysis code and reproduction instructions: companion GitHub repository; archive with Zenodo v2 when Simone approves upload.

**Competing interests:** The authors declare no competing interests.

**Funding:** This research received no external funding.

---

*End of SAR technical note v2 manuscript draft. No Zenodo upload.*
