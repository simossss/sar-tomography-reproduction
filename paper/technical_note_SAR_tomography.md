# On the Reproducibility of Single-Pass SAR Doppler Tomography for Subsurface Imaging: An Independent Assessment

**Author:** Simone Pomposi, Independent Researcher  
**Version:** 3, November 2026  
**Earlier versions:** v1, April 2026, [10.5281/zenodo.19574701](https://doi.org/10.5281/zenodo.19574701); v2, October 2026, [10.5281/zenodo.23593771](https://doi.org/10.5281/zenodo.23593771)  
**Code:** <https://github.com/simossss/sar-tomography-reproduction>

> **Status of the paper under review.** Biondi & Malanga (2022), *Remote Sensing* 14, 5231 [1], was retracted by the journal on 10 August 2026 (*Remote Sensing* 2026, 18, 2679 [1a]). The editors cited serious methodological and statistical flaws; the authors disagreed. Version 1 of this note was deposited before the retraction. The retraction is a change of publication status. It is not evidence for anything argued here, and readers should treat [1] as retracted literature.

**Abstract.** This note assesses the single-image SAR Doppler tomography method that Biondi & Malanga (2022; retracted) used to report the internal structure of the Great Pyramid of Giza. Version 3 rests on two arguments that need no reproduction of the processing chain. The first is geometric: Doppler sub-apertures of one monostatic image differ in azimuth, not in elevation, so a single image has no elevation aperture in the sense of conventional SAR tomography. The second is a sensitivity budget: the vibration series from which the interior image is computed has a scale of \(10^{-7}\) m. At X-band that is a phase change of \(4\times10^{-5}\) rad, 250 to 2,500 times below the phase noise of a single measurement, and at the pyramid's measured dominant frequency of about 2.3 Hz it is a velocity about 700 times below the error of the one spaceborne measurement of this kind that has been validated against ground sensors. The quoted resolution of 0.92 m follows from a vibration frequency of 12,500 Hz chosen by the authors; at the measured frequency the same formula returns kilometres. This version also reports that the empirical null tests of versions 1 and 2 are reproduced by random input, and withdraws them as evidence about the scene. Multi-pass tomography remains a valid route to elevation resolution, but X- and C-band radar do not penetrate intact limestone to the known chambers.

## 1. Introduction

Biondi & Malanga (2022) described a method for imaging the interior of the Great Pyramid of Giza from Synthetic Aperture Radar (SAR) data [1]. The method takes one Single-Look Complex (SLC) image, splits its Doppler band into sub-apertures, estimates surface micro-movements attributed to background seismic activity, and focuses those movements into a tomogram. The paper reports a tomographic resolution of 0.92 m and a three-dimensional reconstruction of known and unknown chambers. It was retracted on 10 August 2026 [1a]; we cite it as the object of assessment.

The same group applied the technique to Vesuvius, where the published paper discusses imaging on the order of 3 km [2], and announced structures hundreds of metres to kilometres beneath the Khafre pyramid at a press conference in March 2025 [3] and in a widely heard interview in January 2026 [7]. Established non-invasive work on the Great Pyramid, such as the muon radiography that found a large void above the Grand Gallery [6], has not confirmed any of these structures.

Versions 1 and 2 of this note combined a geometric argument with empirical null tests on open SAR data. Version 3 changes the balance of the note in three ways:

1. It adds a sensitivity budget (Section 4), which asks whether the displacement the method starts from can be recorded by the radar at all. This is the step the earlier versions did not examine.
2. It runs negative and positive controls on our own empirical tests (Section 5). Random input reproduces the tables we published, so those tests are withdrawn as evidence about the pyramid or about the method.
3. It corrects the record against two assessments that appeared after version 1: a controlled reproduction by Foreman [11] and an audit of our version 1 by Pastore [12].

The geometric argument (Section 3) and the multi-pass analysis (Section 6) are kept, with corrections.

## 2. The method under review

As described in Sections 3 and 4 of [1], the processing chain is:

1. Acquire one SAR SLC image (COSMO-SkyMed Second Generation, X-band, spotlight).
2. Compute the two-dimensional Fourier transform of the SLC.
3. Split the azimuth (Doppler) spectrum into sub-apertures by band-pass filtering.
4. For each pixel along a tomographic line, estimate sub-pixel displacements between sub-aperture pairs by cross-correlation. These displacements are read as vibrations.
5. Build a steering matrix \(A(K_z, z)\) relating Doppler frequency to depth (Equation 22 of [1]).
6. Invert it to obtain a depth profile, \(h(z) = A^\dagger Y\) (Equation 24 of [1]).

Table 1 of [1] gives the acquisition parameters used below: centre frequency 9.6 GHz, chirp bandwidth 400 MHz, Doppler bandwidth 22 kHz, PRF 2.0 kHz, PRT 0.23 ms, spotlight mode, acquisition duration 15 s, platform velocity 7 km/s, observation height 650,000 m. Table 2 of [1] lists twelve SAR products: six used as master and slave pairs for external interferometry and six single images used for the interior.

The tomographic resolution is given as \(\delta_z = \lambda R/(2A)\), where \(\lambda\) is "the sound wavelength over the Earth", \(R\) the slant range and \(A\) the orbital aperture. The text states a seismic wave speed of about 6,000 m/s, "a frequency of investigation set by us equal to 12,500 Hz", an aperture of "about 42,000 m" and \(R = 650{,}000\) m. The equations printed beside that text use different numbers:

\[
\lambda = \frac{v}{f} \approx \frac{6000}{25{,}000} \approx 0.24\ \mathrm{m},
\qquad
\delta_z = \frac{0.24\cdot 650{,}000}{2\cdot 84{,}000} \approx 0.92\ \mathrm{m}.
\]

With the stated 12,500 Hz and 42,000 m the same formula gives \(0.48\cdot650{,}000/(2\cdot42{,}000)\approx 3.71\) m, which is the value printed in the arXiv preprint of the paper [1b]. The journal figure of 0.92 m therefore requires doubling both the frequency and the aperture relative to the accompanying text. Version 2 of this note described this only as a factor-of-two ambiguity.

## 3. Geometry: a single image has no elevation aperture

### 3.1 Azimuth diversity is not elevation diversity

In conventional SAR tomography, resolution along elevation comes from observations that differ in elevation look angle, that is, from a spread of perpendicular baselines \(B_\perp\) between passes. Stacks of tens of passes with baselines of tens to hundreds of metres are the norm [4, 5].

Doppler sub-apertures of a single image differ along the flight track. They sample different squint angles within one synthetic aperture. For a given range and azimuth cell the elevation look angle is fixed by the acquisition, so a single monostatic image holds no independent elevation samples of the same scatterer. Version 1 of this note converted the along-track diversity into an "effective" baseline,

\[
B_\perp = \Delta x\cdot\sin(\Delta\theta),
\]

and from it derived an elevation resolution of about 285 m. That scalar is not a baseline normal to the flight direction and the line of sight, and no derivation supports it. Version 2 withdrew it together with every number built on it (618 mm per sub-aperture step, 285 m resolution, 8,559 m ambiguity, the comparisons with chamber sizes, and a 660 m bound for Vesuvius). The withdrawal stands. The statement that survives is the qualitative one: by the standard model, one image has zero elevation aperture.

### 3.2 Descriptive parameters of the test scene

For reference, these are the along-track parameters extracted in version 1 from the SICD metadata of the Umbra X-band spotlight acquisition over Giza used in Section 5 (UMBRA-04, 8 March 2023). They describe the scene. None of them is an elevation-tomographic quantity.

| Parameter | Value (v1 extraction) |
|-----------|-----------------------|
| Radar wavelength | 3.123 cm |
| Platform velocity | 7,675.6 m/s |
| Doppler bandwidth | 29,102 Hz |
| Doppler FM rate | 5,941 Hz/s |
| Slant range | 634,971 m |
| Azimuth resolution | 0.264 m |
| Look angle at the spacecraft | 32.3° |

Two corrections to earlier versions. The 32.3° was labelled incidence angle; it was computed as the angle at the spacecraft, so the ground incidence is larger. The centre frequency was listed as 9.605 GHz, but the wavelength above corresponds to 9.600 GHz. The scene files were not retained after version 1, so neither value was re-extracted for this version (Section 8).

### 3.3 What the geometric argument does and does not cover

The argument closes the electromagnetic route. Distinguishing a chamber 21 m or 43 m above the pavement from the masonry around it needs an observable that encodes elevation, and a single SLC does not supply one under conventional SAR tomography.

It does not by itself close the route the paper actually claims. In [1] the depth axis is not attached to the radar wave. It is attached to a vibration: the steering matrix is called "sonic", and the resolution formula of Section 2 uses an acoustic wavelength. A critique that inserts the radar wavelength into a conventional \(k_z\) relation, as our version 1 did, tests a different model, a point made by Pastore [12] and accepted here. The vibration route has to be judged on its own inputs: whether the displacement series exists in the data, and whether the frequency in the resolution formula is physical. Section 4 takes up both.

## 4. Sensitivity: the claimed displacement against what the radar can record

The outline of this section follows an analysis that Marcin Gerwin shared with the author in correspondence [13]. All numbers were recomputed for this note and are printed by `scripts/sensitivity_budget.py`. The inputs are Table 1 of [1], the figures of [1], and two independent studies published in 2026 [9, 10].

### 4.1 The claimed quantity

Every tomogram in [1] is computed from one kind of input: the displacement time series of each pixel on the tomographic line. Figures 4 to 6 of [1] show such series. Figure 6 spans 12 seconds, and its axis reads "Vibrational displacement (m)" with a scale factor of \(10^{-7}\). The claimed motion is therefore about 0.1 µm, measured from a distance of 650 km with a radar wavelength of 31.2 mm.

We found no signal-to-noise ratio or error estimate for this series in [1], no series over a stable control, and no recovery of a known motion. For calibration and validation the paper refers to the authors' earlier work on other targets. The coherence map it shows (Figure 11) is the repeat-pass coherence of the external image pairs, and its lidar comparison (Figures 20 and 21) concerns the trend of an interferometric fringe on the west face. Neither bears on the vibration series.

### 4.2 Phase budget

The most sensitive way a radar senses motion is the phase of the echo. A displacement \(d\) along the line of sight changes the phase by \(4\pi d/\lambda\). For \(d = 0.1\) µm this is \(4.0\times10^{-5}\) rad.

Each phase reading carries noise. For a point-like target with signal-to-noise ratio \(S\) in the focused image, the standard deviation is about \(1/\sqrt{2S}\) radians [15]. A displacement is the difference of two readings, which multiplies the noise by \(\sqrt{2}\).

| Target | One reading | Difference of two readings | Noise relative to the claim |
|--------|-------------|----------------------------|-----------------------------|
| Excellent, 40 dB | 0.007 rad, 18 µm | 0.010 rad, 25 µm | 250 times larger |
| Typical, 20 dB | 0.071 rad, 176 µm | 0.100 rad, 248 µm | 2,500 times larger |

The 40 dB case is generous. It is the signal-to-noise ratio of a strong reflector in a pixel focused from the whole 15-second aperture, and it corresponds to a coherence of 0.9999, which covers the "coherence levels very close to 1" that [1] reports for its repeat-pass pairs. The faces of the pyramid are rough limestone, and a time series has to be built from pieces of the aperture, each with less signal than the whole. If the motion is hundreds of times smaller than the noise of the measurement, the series records the noise.

### 4.3 Sampling and sub-aperture duration

A time series follows a motion only if its samples are shorter than half the period of that motion. In a sub-aperture series each sample is built from a slice of the Doppler band, and for one target a slice that covers a fraction \(f\) of the band is built from about a fraction \(f\) of the acquisition.

Versions 1 and 2 of this note took the bandwidth ratio of the sub-apertures in [1] to be 0.5. A half-band slice lasts about 7.5 s. Over 7.5 s a 2.3 Hz oscillation completes 17 cycles and a 12,500 Hz oscillation completes about 94,000, so the displacement recorded by such a slice is an average over many cycles, close to zero.

To follow the pyramid's measured 2.3 Hz (Section 4.4) a slice has to be shorter than 0.22 s, which is 1.4% of the aperture. Its azimuth resolution cell grows from 0.32 m to about 22 m, and its signal-to-noise ratio falls by 18 dB. Both make the noise of Section 4.2 worse.

A vibration at 12,500 Hz has a period of 0.08 ms. The radar transmits a pulse every 0.5 ms at the stated PRF of 2.0 kHz. Table 1 of [1] also lists a PRT of 0.23 ms, which corresponds to 4.3 kHz and not to 2.0 kHz. On either reading the vibration completes several cycles between two consecutive pulses, so no series built from this acquisition can follow it. The text of [1] describes the SLC as "synthesized at 24 kHz", and the chosen 12,500 Hz is about half of that figure. The Doppler bandwidth of a spotlight SLC (22 kHz in Table 1) is a spatial bandwidth of the image. It is not the rate at which a point on the ground is sampled in time.

### 4.4 Comparison with a validated measurement

Two studies published in 2026 fix the scales.

ElGabry et al. [10] recorded ambient vibration at 37 points inside and around the Great Pyramid with a three-axis servo accelerometer (0.1 to 200 Hz, resolution 1 µg, 100 samples per second, 15 minutes per point). Of these points, 76% show a fundamental frequency between 2.0 and 2.6 Hz, with an average of about 2.3 Hz; the surrounding soil is at about 0.6 Hz. The survey gives frequencies and relative amplification, not absolute amplitudes, and its sampling rate limits it to 50 Hz.

Lotti et al. [9] measured the line-of-sight vibration velocity of a bridge deck from high-resolution spaceborne SAR and validated it against synchronous ground truth. They report an error on the order of 1 mm/s and a frequency resolution of 0.06 Hz for a 16-second acquisition. One of the authors, C. Clemente, co-authored with Biondi the ship micro-motion work by pixel tracking that [1] cites [14]. The validation therefore comes from the line of work the method builds on.

Against these:

- At 2.3 Hz, an amplitude of 0.1 µm is a velocity of 0.0014 mm/s, about 700 times below the validated error.
- A radial velocity \(v\) displaces a target in azimuth by \(R v/V\), which is 93 mm per mm/s for the parameters of [1]. The claimed motion shifts the image by 0.13 mm. The lower bound on the error of locating a target [16], applied to the difference of two half-band pieces, is 5 mm at 40 dB and 50 mm at 20 dB. For pieces short enough to follow 2.3 Hz it is about 1 m at 40 dB.
- An amplitude of 0.1 µm reaches 1 mm/s only above about 1.6 kHz, where the period (0.63 ms) is comparable to the interval between two pulses. There is no frequency at which the claimed amplitude is both above the validated error and followed by a series built from sub-apertures.
- The 1 µg resolution of the accelerometer in [10] corresponds to a displacement of about 0.05 µm at 2.3 Hz. The claim is that a radar at 650 km matched, in 15 seconds, the resolution of an instrument placed on the stone.

### 4.5 The assumed frequency and the resolution formula

The resolution formula of Section 2 contains the wavelength of the vibration, and that wavelength is the wave speed divided by a frequency that was "set by us". The formula returns the resolution of whatever frequency is entered.

- At the measured 2.3 Hz and 6,000 m/s the wavelength is 2.6 km, more than ten times the width of the pyramid. The formula then gives 20 km with the factors stated in the text of [1], or 5 km with the factors printed in its equations. A wave of that length moves the monument as one body.
- If the plotted series were at 12,500 Hz, an amplitude of 0.1 µm would mean a surface velocity of 7.9 mm/s and an acceleration of about 617 m/s², or 63 g. Either the motion is at 12,500 Hz and physically implausible, or it is at lower frequencies and says nothing about 12,500 Hz.
- The paper names its sources: "The vibration energy is generated from many sources such as wind", with a "great contribution" from the city of Cairo and the Nile. Ambient ground motion of that kind carries very little energy at kilohertz frequencies, and rock attenuates what there is. For a quality factor of 100, which is an assumption, the amplitude remaining after 100 m of rock at 12,500 Hz is about 0.1% of the original; at 2.3 Hz it is 99.9%. The survey in [10] does not test this, because it stops at 50 Hz.
- A stone face moving 0.1 µm at 12,500 Hz is a loudspeaker. It would radiate a tone of about 100 dB at the faces, within the range of human hearing and comparable to a chainsaw at close range. No such tone is heard at the site.
- Keeping the whole body of the pyramid in that motion would take a continuous supply of power. For limestone at 2,400 kg/m³, a volume of 2.6 million m³ and the same quality factor of 100, the motion dissipates about 150 MW; for quality factors between 200 and 50 the figure is 75 to 300 MW. Wind and city traffic do not deliver that at 12,500 Hz.

### 4.6 Averaging

Averaging \(N\) independent measurements divides the noise by \(\sqrt{N}\), which invites the objection that the noise of Section 4.2 can be averaged down. It can, at a price that removes the image.

Averaging in time does not apply: a vibration changes from one sample to the next, and the 40 dB of Section 4.2 already uses every pulse of the acquisition. Averaging in space needs \(6\times10^{4}\) independent resolution cells per time sample at 40 dB, and \(6\times10^{6}\) at 20 dB, to bring 25 µm or 248 µm down to 0.1 µm. With the cell of [1], 0.32 m by 0.375 m, that is a square 86 m on a side in the first case and 860 m in the second. The footprint of the pyramid is about 230 m on a side. Spatial averaging at this level yields at most one time series for the whole monument, with nothing left to form an image from, and it cannot remove errors common to all pixels, such as the atmosphere and the platform.

### 4.7 Summary

The interior image depends on a displacement that is hundreds to thousands of times below the phase noise of the radar, on a time resolution the sub-apertures do not have, and on a vibration frequency that was chosen and that the radar could not sample. None of this requires access to the processing code, and none of it depends on the tests of Section 5.

## 5. The empirical tests of versions 1 and 2, with controls

Versions 1 and 2 reported three tests on open data: a coherence comparison between Doppler sub-apertures, a 240-run parameter sweep over the pyramid and a control patch, and a 15-image multi-pass inversion at five sites. In each, the pyramid and the comparison site gave nearly the same numbers, and the note read that as a sign of processing-dominated output.

A null result is informative only if the test could have returned something else. For this version we ran two kinds of control through the same code: random complex numbers in place of SAR data, and a synthetic target planted in random numbers. The script is `scripts/noise_controls.py`. It needs no SAR data, and its seeds are fixed.

### 5.1 Sub-aperture coherence

Version 1 measured coherence between Doppler sub-apertures with a 5 by 5 window on 1000 by 1000 pixel patches of the Giza and Vesuvius scenes, and found similar values at both sites.

| Sub-aperture distance | Giza (v1) | Vesuvius (v1) | White noise | Noise, oversampled 1.35 times | Model with zero coherence |
|---|---|---|---|---|---|
| 1 | 0.443 | 0.446 | 0.401 | 0.443 | 0.444 |
| 2 | 0.419 | 0.422 | 0.387 | 0.429 | 0.431 |
| 5 | 0.335 | 0.331 | 0.310 | 0.338 | 0.343 |
| 10 | 0.109 | 0.108 | 0.102 | 0.111 | 0.110 |
| 14 | 0.050 | 0.051 | 0.047 | 0.052 | 0.051 |

Random noise reproduces the table to about 0.01. The reason is in the script. It cuts the spectrum into disjoint 16-bin slices, not the overlapping sub-apertures of the method, and disjoint slices of speckle are uncorrelated. A 5 by 5 window on such narrow-band images holds about four independent samples, and the expected coherence magnitude of an estimator with four looks and zero true coherence is about 0.45. The fall with distance is the phase ramp left between two slices with different centre frequencies, summed over five pixels. The last column is that model, with no free parameter except the number of looks. White noise sits slightly lower because it has five independent samples per window; band-limiting it along range by a factor of 1.35, as in an oversampled SLC, brings it onto the published values. We did not check that factor against the scene.

The table therefore measures the estimator. The agreement between Giza and Vesuvius carries no information about either site, and the inference drawn from it in versions 1 and 2 is withdrawn.

### 5.2 Parameter sweep

The sweep ran a Doppler beamformer over 240 parameter combinations (6 sub-aperture counts, 5 bandwidth ratios, 2 windows, 4 reconstruction ranges) on a patch centred on the pyramid and a control patch in the same scene. It scored each run by the peak of the profile in a 20 to 80 m band divided by the profile mean (`SNR_peak_mean`) and counted runs where the pyramid exceeded 1.5 times the control.

| Statistic | Giza vs control (v1) | Noise vs noise |
|---|---|---|
| Mean `SNR_peak_mean`, first patch | 1.118 ± 0.077 | 1.130 ± 0.075 |
| Mean `SNR_peak_mean`, second patch | 1.139 ± 0.082 | 1.102 ± 0.073 |
| Mean difference | −0.021 ± 0.043 | +0.028 ± 0.048 |
| Correlation across the 240 runs | 0.855 | 0.793 |
| Runs above the 1.5 times criterion | 0 of 240 | 0 of 240 |

Two patches of random noise give the same summary as the pyramid and its control. Four further problems limit what the sweep can mean:

- **Not the published chain.** The sweep has no displacement estimation, no master and slave pairing, and uses \(A^H Y\) in place of \(A^\dagger Y\). Version 2 said so; Pastore [12] documents it in detail.
- **Wrong axis.** `sweep_v2.py` transforms a line along the row dimension of the SICD. Its own metadata handling, and the coherence script, treat the column dimension as azimuth. By the repository's own convention the sweep therefore split the range spectrum, and its "sub-apertures" were range sub-bands. We could not re-check this against the scene for this version.
- **Withdrawn depth axis.** The steering matrix of the sweep is built from the scalar \(B_\perp = \Delta x\sin\Delta\theta\) that Section 3.1 withdraws. The metres on its depth axis have no physical meaning.
- **Criterion.** "0 of 240" counts runs above a ratio of 1.5. It is not a count of runs in which the pyramid scored higher than the control [12].

The positive control shows what the sweep does respond to. We added one bright scatterer to a noise patch, on the line the script samples, and scored it against a second noise patch:

| Scatterer above clutter | 3 px before centre | At centre | 2 px after centre |
|---|---|---|---|
| 10 dB | 0 of 240 | 0 of 240 | 9 of 240 |
| 20 dB | 15 of 240 | 49 of 240 | 107 of 240 |
| 30 dB | 34 of 240 | 68 of 240 | 128 of 240 |

The sweep can flag an isolated scatterer 20 dB or more above the clutter, and how often it does so depends on the position of that scatterer along the line, by a few pixels. Its depth axis is that position, relabelled in metres. The result of version 1 therefore shows that neither patch had such a scatterer on the sampled line. It says nothing about height, depth or subsurface structure, and it is withdrawn as evidence.

### 5.3 Fifteen-image inversion

Version 1 took complex pixel values from 15 Sentinel-1 images at five sites, without interferometric coregistration or removal of the flat-earth and topographic phase, and inverted them with a pseudo-inverse over −300 to +300 m. All five sites gave similar profiles, with a correlation of 0.9913 between the pyramid and the control.

| Quantity | Pyramid / control (v1) | Random complex input |
|---|---|---|
| Peak contrast | 8.2 to 8.6 σ over five sites | 8.4 ± 0.3 σ |
| Correlation between two profiles | 0.9913 | median 0.9995, 5th percentile 0.989 |
| Amplitude at +140 m | 0.106 / 0.118 | 0.113 ± 0.008 |
| Amplitude at +43 m | 0.121 / 0.130 | 0.127 ± 0.007 |
| Amplitude at +21 m | 0.049 / 0.043 | 0.046 ± 0.007 |
| Amplitude at 0 m | 0.145 / 0.145 | 0.147 ± 0.006 |
| Amplitude at −100 m | 0.150 / 0.146 | 0.152 ± 0.007 |
| Amplitude at −200 m | 0.203 / 0.192 | 0.202 ± 0.009 |

Fifteen random complex numbers reproduce every entry. The steering matrix for these baselines has singular values from 58 down to 0.013, and the untruncated pseudo-inverse amplifies the weakest one. The output is that singular vector, with its peak at the edge of the reconstruction window in every trial, whatever goes in.

The positive control confirms that the inversion could not have detected a real scatterer. With one scatterer planted at +43 m and a common phase across the 15 images, the share of trials in which the peak falls within 32 m of +43 m is:

| Per-image SNR | Pseudo-inverse, as in v1 | Matched beamformer \(A^H Y\) |
|---|---|---|
| 0 dB | 0.0% | 99.3% |
| 20 dB | 0.0% | 100% |
| 40 dB | 0.7% | 100% |
| 60 dB | 42.0% | 100% |

A matched beamformer finds the scatterer at 0 dB. The pseudo-inverse used in version 1 misses it at 40 dB. When each image is given an unknown phase, which is the situation without coregistration and phase flattening, the matched beamformer also falls to chance (11.9% against 11% for a 64 m window in a 600 m range). The test of version 1 had both defects. Its table, and the reading that five sites "look the same", are withdrawn.

### 5.4 Status of the earlier tests

| Test | Earlier reading | Status in version 3 |
|---|---|---|
| Sub-aperture coherence, Giza and Vesuvius | Geometry-dominated decorrelation | Withdrawn. Estimator bias; noise gives the same table. |
| Fixed-parameter test, `SNR_energy` 1.87 ± 0.54 against 1.87 ± 0.55 | Pyramid and control indistinguishable | Withdrawn. The repository holds no script that computes this statistic as described, so it can be neither regenerated nor controlled. |
| 240-run sweep | No target-specific discrimination | Withdrawn. Noise gives the same summary; range axis; withdrawn depth scale. |
| 15-image inversion at five sites | Preliminary negative diagnostic | Withdrawn. Random input gives the same table; a planted scatterer is not recovered. |

Other points from earlier versions no longer bear on any conclusion but are recorded for completeness. The control patch about 3 km southeast of the pyramid is in the Shabramant settled and irrigated fringe, not in empty desert. Its pixel offsets were never geolocated. None of the scene numbers was re-run after version 1.

These tests do not show that the published method fails, and they do not show that it works. We agree with Pastore [12] that they are not a computational falsification of the published chain. The argument of Section 4 does not rest on them.

## 6. Multi-pass tomography: resolution geometry and penetration

### 6.1 Measured baselines

To see what standard multi-pass tomography offers over Giza, version 1 took 15 Sentinel-1A IW SLC images from one ascending track (Track 58), January to December 2023, obtained from the Alaska Satellite Facility, and computed perpendicular baselines from the orbit state vectors relative to the image of 24 June 2023.

| Date | \(B_\perp\) (m) | Date | \(B_\perp\) (m) | Date | \(B_\perp\) (m) |
|------|------|------|------|------|------|
| 2023-01-07 | +167.3 | 2023-05-31 | +37.3 | 2023-10-22 | −41.5 |
| 2023-01-31 | +81.3 | 2023-06-24 | 0.0 | 2023-11-15 | +65.0 |
| 2023-02-24 | +146.4 | 2023-07-18 | +89.6 | 2023-12-09 | +93.1 |
| 2023-03-20 | +140.6 | 2023-08-11 | +170.3 | | |
| 2023-04-13 | −24.3 | 2023-09-04 | −16.5 | | |
| 2023-05-07 | +31.8 | 2023-09-28 | +112.4 | | |

The baselines span 211.8 m, which gives a \(k_z\) span of 0.0978 rad/m and an elevation resolution of \(2\pi/0.0978 = 64.2\) m. These numbers come from orbit geometry and do not depend on the inversion of Section 5.3.

Irregular baselines have no single ambiguity height. Earlier versions quoted about 4,560 m, which comes from the smallest gap in the stack (3.0 m). That is not a clean window: the mean spacing is 15 m, and Section 6.3 shows that a stack of this size has high sidelobes at every depth.

### 6.2 Resolution is not penetration

A resolution of 64 m cannot separate the King's Chamber (about 43 m above the pavement) or the Queen's Chamber (about 21 m) from the body of the pyramid. A larger stack does better geometrically. For an X-band system with \(\lambda = 3.1\) cm, a slant range of 650 km and 35° incidence, a baseline spread of 500 to 1,000 m gives a resolution of roughly 6 to 12 m.

Geometry is half of the requirement. The wave also has to reach the target and return. In intact dry limestone, X-band and C-band penetration is on the order of centimetres to decimetres, depending on moisture and heterogeneity. No baseline configuration changes that. L-band penetrates further, and civilian L-band spotlight data exist (ALOS-2 and ALOS-4), which still does not amount to detecting a chamber behind tens of metres of masonry.

Multi-pass tomography is therefore a different technique from the one in [1], with real elevation resolution and a real penetration limit. If any of the "more than 200" scans mentioned in [7] were combined as a multi-pass stack, the resolution could reach a few metres and the penetration limit would still apply.

### 6.3 Sparse stacks and sidelobes: a toy

Table 2 of [1] lists twelve products and the acknowledgments thank the author of SARPROZ, a multi-temporal InSAR package [8], "through which many calculations were carried out". This raises the question of what a multi-pass inversion with a handful of baselines looks like. The following toy answers that question only. It makes no claim about any published figure.

`scripts/sidelobe_toy.py` plants five point scatterers at 0, 21, 43, 60 and 140 m, draws random baselines, adds 5% noise, and applies a matched beamformer over −100 to +800 m. Nothing is planted deeper than 140 m. The statistic is the largest reconstructed amplitude beyond 300 m as a fraction of the main peak, over 2,000 baseline draws.

| Stack | Median | 5th to 95th percentile |
|---|---|---|
| 6 baselines, X-band, within ±300 m | 0.95 | 0.78 to 1.00 |
| 15 baselines, C-band, within ±170 m | 0.63 | 0.46 to 0.85 |
| 200 baselines, X-band, within ±500 m | 0.24 | 0.20 to 0.30 |

With six baselines, a spurious feature as strong as the real ones appears at depth in almost every draw. With 200 it falls to about a quarter of the main peak. A sparse stack cannot tell a deep feature from a sidelobe of shallow ones.

This replaces the simulation of earlier versions, which reported an "ambiguity height" taken from the smallest gap between random baselines (399 km in its six-baseline case) alongside text about periodic copies at 600 m and 1,200 m that did not follow from it, as Pastore [12] pointed out. The earlier sidelobe percentages are superseded by the table above.

## 7. Discussion

### 7.1 What this note supports

- One SAR image has no elevation aperture. Conventional tomography cannot place a scatterer in height from azimuth sub-apertures (Section 3).
- The method in [1] instead reads height from a vibration. The vibration it plots is 250 to 2,500 times below the phase noise of the radar and about 700 times below the error of the validated spaceborne measurement of the same kind. Its sub-apertures are too long to follow the pyramid's measured motion, and the frequency behind its resolution could not be sampled by the radar (Section 4).
- The resolution of 0.92 m is the output of a formula for a chosen frequency, and the printed arithmetic uses twice the stated frequency and twice the stated aperture (Sections 2 and 4.5).
- With real baselines, elevation resolution is available, and X- and C-band still do not reach the chambers through intact limestone. Sparse stacks produce deep sidelobes as strong as the scatterers that cause them (Section 6).

### 7.2 What this note does not show

- It is not a reproduction of the published chain. We did not implement the displacement estimation of [1] or its pseudo-inverse, and the processing code is not public.
- Its own empirical tests are withdrawn (Section 5). They neither refute nor support the method.
- The sensitivity budget uses nominal signal-to-noise ratios, because [1] reports none. A measured noise floor for the displacement series on a real spotlight scene would be stronger than the bound given here.
- The retraction of [1] is not evidence for this note.

### 7.3 Other independent assessments

Foreman [11] rebuilt the single-image chain from the papers and the patent, including the displacement-tracking front end, and ran it with controls on five sites from two sensors. That work reports that the outputs are statistically indistinguishable from their nulls while in-data positive controls pass, that a confident "detection" over a mapped mining district matches none of the documented workings, and that the reported depths follow from an investigation frequency that is not physical. It is the closest thing to a full-chain reproduction that we know of, and it does what Section 5 of this note failed to do: run the controls first.

Pastore [12] audited version 1 of this note and its repository. The parent report of that audit concludes in favour of the method, a conclusion we do not share and do not address here. Its specific findings about our work are largely correct, and this version accepts them:

| Finding in [12] | Response |
|---|---|
| The 240-run sweep is not equivalent to the published chain | Accepted in version 2; extended here with the noise control and the axis problem (Section 5.2). |
| "0 of 240" comes from a 1.5 times ratio criterion | Accepted (Section 5.2). |
| Coherence values described as identical are not | Corrected in version 2; the comparison itself is withdrawn here (Section 5.1). |
| The 285 m bound applies a conventional \(k_z\) model to a different observable | Withdrawn in version 2; scope stated in Section 3.3. |
| The aliasing script reports a 399 km ambiguity next to a 600 m and 1,200 m narrative | Accepted; script replaced (Section 6.3). |
| Baseline conventions differ between scripts | Moot after the withdrawal of the single-pass baseline. |

### 7.4 Open questions and recommendations

- **Which data entered which inversion.** Table 2 of [1] lists six pairs and six single images, and the acknowledgments credit a multi-temporal InSAR package. The paper does not say which steps used it.
- **The 200 scans.** In [7] the lead author says more than 200 scans of Giza have been made with several systems. Two hundred single images each inherit the limits of Sections 3 and 4. A multi-pass stack would have real resolution and the penetration limit of Section 6.2.
- **Proprietary processing.** Patent filings mean that the published equations may not describe the code that was run. That is a reason to ask for controls, not a reason to accept the output.

For the authors of [1], three results would settle most of Section 4: the displacement series over a stable control next to the series over the pyramid, the same series with a known synthetic vibration added to the raw data, and the signal-to-noise ratio or coherence of the pixels on the tomographic line.

For independent work, controls come before results. Run random input and a planted target through the pipeline before any real scene, and report both.

## 8. Reproducibility

Code: <https://github.com/simossss/sar-tomography-reproduction>.

Three scripts need no SAR data and reproduce every new number in this version:

- `scripts/sensitivity_budget.py`: Section 4.
- `scripts/noise_controls.py`: Section 5 (about four minutes).
- `scripts/sidelobe_toy.py`: Section 6.3.

The remaining scripts read SAR data and are kept so that the version 1 numbers can be traced: `verify_v2.py` (SICD metadata), `fix_coherence.py`, `sweep_v2.py`, `step3_baselines.py`, `run_tomo.py` and `multi_site_tomo.py`. Each now states its known limits in its header. They were not re-run for version 2 or version 3, because the scene files used for version 1 were not retained. The products remain public:

- Umbra X-band spotlight SICD, AWS Open Data (CC BY 4.0): Giza task `5aa49658-ecf9-4504-afee-281f43fb076e` (UMBRA-04, 8 March 2023) and Vesuvius task `a73184a7-2b25-4dfd-b750-1da35653be44` (UMBRA-05, 15 November 2023).
- Sentinel-1A IW SLC, Track 58 ascending, dates of Section 6.1, from the Alaska Satellite Facility. Product identifiers were not recorded.

Requirements: Python 3 with numpy and scipy; sarpy and tifffile for the data scripts. `scripts/build_pdf.sh` rebuilds this PDF with pandoc and XeLaTeX.

## Appendix A. Changes between versions

**Version 3 (November 2026), relative to version 2**

- New Section 4, sensitivity budget.
- New controls for the empirical tests; all four tests withdrawn as evidence (Section 5).
- The arithmetic behind the 0.92 m resolution is traced to the journal text and the arXiv preprint (Section 2).
- The sidelobe simulation is replaced by a toy with a defined statistic (Section 6.3).
- The Grand Gallery was placed at about 60 m in two places; it runs between roughly 21 and 43 m above the pavement. A height of 60 m is near the top of the relieving chambers above the King's Chamber.
- The 32.3° in the scene table is the look angle at the spacecraft, not the incidence angle; the listed centre frequency and wavelength disagreed (Section 3.2).
- References added for Foreman [11], Pastore [12], Lotti et al. [9] and ElGabry et al. [10].
- The PDF deposited as version 2 carried a draft header and unrendered mathematics. This version is typeset from the repository source.

**Version 2 (October 2026), relative to version 1**

- Retraction of [1] recorded.
- Withdrawn: the single-pass baseline of 618 mm per step, the 285 m elevation resolution, the 8,559 m ambiguity, the 6.6 and 13.6 times chamber comparisons, and the 660 m bound for Vesuvius.
- The sweep relabelled as a test of a partial beamformer.
- Chamber positions given as heights above the pavement.
- Table 2 of [1] has twelve products, not six.
- The universal 100 to 200 m ambiguity plateau withdrawn.
- The control patch is not empty desert.
- Corrections of dates and provenance: the interview [7] was released on 23 January 2026; the volcano paper [2] discusses about 3 km; L-band spotlight data exist; the Sentinel-1 stack came from the Alaska Satellite Facility.

## AI disclosure

Large language models were used throughout this work for drafting, code and review: Claude (Anthropic) and ChatGPT (OpenAI) for versions 1 and 2, with accuracy reviews by Grok, Codex and Claude. For version 3, Claude (Anthropic) wrote and ran the control and sensitivity scripts and drafted the revision under the author's direction. The author is responsible for all claims and for the final text. No AI system is an author.

## Acknowledgments

Thanks to Marcin Gerwin for sharing the analysis that Section 4 follows and for pointing to the two 2026 studies it rests on, and to Emmanuel Pio Pastore for a detailed audit of version 1.

## References

[1] Biondi, F.; Malanga, C. "Synthetic Aperture Radar Doppler Tomography Reveals Details of Undiscovered High-Resolution Internal Structure of the Great Pyramid of Giza." *Remote Sensing* 2022, 14, 5231. https://doi.org/10.3390/rs14205231. **Retracted 10 August 2026.**

[1a] Retraction notice. *Remote Sensing* 2026, 18, 2679. https://doi.org/10.3390/rs18162679

[1b] Biondi, F.; Malanga, C. Preprint of [1]. arXiv:2208.00811. https://arxiv.org/abs/2208.00811

[2] Biondi, F. "Scanning Inside Volcanoes with Synthetic Aperture Radar Echography Tomographic Doppler Imaging." *Remote Sensing* 2022, 14, 3828. https://doi.org/10.3390/rs14153828

[3] Biondi, F.; Malanga, C.; Mei, A. Press conference, Castel San Pietro Terme, Bologna, Italy, 15 to 16 March 2025. Public claims, not peer reviewed.

[4] Reigber, A.; Moreira, A. "First Demonstration of Airborne SAR Tomography Using Multibaseline L-Band Data." *IEEE Trans. Geosci. Remote Sens.* 2000, 38, 2142–2152.

[5] Zhu, X.X.; Bamler, R. "Tomographic SAR Inversion by L1-Norm Regularization: The Compressive Sensing Approach." *IEEE Trans. Geosci. Remote Sens.* 2010, 48, 3839–3846.

[6] Morishima, K. et al. "Discovery of a Big Void in Khufu's Pyramid by Observation of Cosmic-Ray Muons." *Nature* 2017, 552, 386–390.

[7] Biondi, F. Interview, *The Joe Rogan Experience*, Episode 2443, released 23 January 2026. Cited as a record of public statements.

[8] Perissin, D.; Wang, Z.; Wang, T. "The SARPROZ InSAR Tool for Urban Subsidence/Manmade Structure Stability Monitoring in China." *Proc. 34th International Symposium on Remote Sensing of Environment*, Sydney, 2011. https://www.sarproz.com/

[9] Lotti, A.; Vattulainen, A.B.; Diaz Riofrio, S.; Suppi, C.; Tubaldi, E.; Zonta, D.; Milillo, P.; Clemente, C. "Monitoring Bridge Vibrations via Spaceborne SAR Micro-Doppler." *Structural Control and Health Monitoring* 2026, 3858095. https://doi.org/10.1155/stc/3858095

[10] ElGabry, M.; Hamed, A.; Yoshimura, S.; Hussein, H.M.; Maklad, M.; Salama, A. "Architectural and Geotechnical Aspects Affecting Earthquake Resilience for the Antique Egyptian Khufu Pyramid." *Scientific Reports* 2026, 16, 14032. https://doi.org/10.1038/s41598-026-49962-6

[11] Foreman, H. "No Reproducible Evidence for Deep Subsurface Structures Beneath the Giza Plateau: A Controlled Reproduction of Single-Source SAR Doppler Micro-Motion Tomography." Preprint and code, 2026. https://doi.org/10.5281/zenodo.21065675; https://github.com/Hassanforeman/subsurface-sar-tomo

[12] Pastore, E.P. "Addendum I to the Technical Audit and Reproducibility of SAR Doppler Echography: Source-Fidelity and Report-Code Consistency Audit of Pomposi (2026)." Technical report, version 2.6, 16 August 2026. https://doi.org/10.5281/zenodo.21965821

[13] Gerwin, M. Personal communication, 9 and 10 November 2026.

[14] Biondi, F.; Addabbo, P.; Orlando, D.; Clemente, C. "Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in COSMO-SkyMed Synthetic Aperture Radar Data: An Operative Assessment." *Remote Sensing* 2019, 11, 1637.

[15] Hanssen, R.F. *Radar Interferometry: Data Interpretation and Error Analysis.* Kluwer Academic, Dordrecht, 2001.

[16] Bamler, R.; Eineder, M. "Accuracy of Differential Shift Estimation by Correlation and Split-Bandwidth Interferometry for Wideband and Delta-k SAR Systems." *IEEE Geosci. Remote Sens. Lett.* 2005, 2, 151–155.

**Data availability.** Umbra SICD products, AWS Open Data Program (CC BY 4.0). Sentinel-1 SLC products, Alaska Satellite Facility. Code under the MIT licence at the repository above; this note under CC BY 4.0.

**Competing interests.** None declared.

**Funding.** No external funding.
