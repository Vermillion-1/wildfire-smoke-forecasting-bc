# Known Limitations & Roadmap

What this project does not establish, where the measurements are weaker than the headline numbers
suggest, and what would be built next. Kept alongside the code because the boundary of a result is
part of the result.

Every limitation below was verified against the source, the notebooks, or a recomputation from
`data/processed/merged/dataset.csv` — not assumed.

---

## 1. Measurement and evaluation

### 1.1 The classifier's headline AUPRC depends on how folds are pooled

`train_smoke_detector.py:300` reports **0.331**, the mean of five per-fold AUPRC values. Each fold's
test window is built as `date >= cutoff` (`train_smoke_detector.py:227-232`), so the windows run from
each cutoff to the end of the data and are *nested* rather than disjoint. The severe post-2017
episodes therefore fall inside several test windows and are scored repeatedly, which pulls the mean
above the pooled value.

Pooling every out-of-fold prediction into a single curve — which is what
`figures/smoke_detector_pr_curves.png` actually plots — gives **AP = 0.239** against a random
baseline of 0.006. Both numbers are now reported; 0.239 is the one to quote as a single figure.

**Fix:** make test windows disjoint (`cutoff <= date < next_cutoff`) so per-fold means and pooled
values agree, and report the pooled AP as primary.

### 1.2 The classifier's F2 threshold is tuned on the fold it is scored on

`train_smoke_detector.py:270-272` picks the decision threshold by maximising F2 on `y_te`, then
reports that same maximised F2. The published 0.455 is therefore optimistically biased. AUPRC is
threshold-free and unaffected, which is why it is the better headline metric here.

**Fix:** select the threshold on a validation slice carved from the training window, then apply it
unchanged to the test fold.

### 1.3 `scale_pos_weight` is computed once over the whole dataset

`train_smoke_detector.py:217-219` derives the class-imbalance weight from every row, including future
test folds, and reuses it in all five folds. The positive rate is stable at ~0.5% across the record,
so the practical impact is small — but it is future information entering a training hyperparameter.

**Fix:** recompute from `y_tr` inside each fold.

### 1.4 The regression evaluation is sound, and for a non-obvious reason

The same nested folds appear in the regression scripts, but there the predictions are written as
`oof_preds[test_idx] = pred`, so later folds overwrite earlier ones and each day ends up predicted by
the model trained up to the most recent cutoff before it. That is a correct expanding-window
evaluation. This is worth writing down because the fold construction *looks* wrong and should not be
"fixed" without understanding why the overwrite saves it.

It also explains the two persistence baselines that appear across the docs: **1.694** is measured over
all 9,125 modeling rows, **1.668** over the post-2004 evaluation window that the folds actually cover.

### 1.5 The LSTM result is not reproducible to the quoted precision

`train_lstm.py` sets no random seed. Three runs gave MAE 2.342, 2.292 and 2.305 (−27%, −24.4%,
−25.1% against persistence). The conclusion — a sequence model does not beat persistence here — is
stable; the third decimal is not.

**Fix:** seed `torch`, `numpy` and `random`, or report a median over several runs.

### 1.6 The RQ3 trend statistics are not computed by any committed code

The four Spearman results (frequency ρ = 0.480 p = 0.015; peak severity ρ = 0.648 p = 0.043;
duration ρ = 0.483 p = 0.015; fire-season mean ρ = 0.202 p = 0.330) all reproduce exactly from
`dataset.csv`, but `grep -rni spearman` across `scripts/` and `notebooks/` returns nothing. The
analysis was run outside the committed code.

Two details that matter and were not previously written down: "episode duration" is the **longest
episode in each year**, not the mean; and the four tests do not share a sample definition — three run
over all 25 years with non-event years zero-filled, while peak severity runs over only the 10 years
that contain an episode.

**Fix:** commit the trend analysis as a notebook cell or script so the numbers are re-derivable.

### 1.7 No multiple-comparison correction

Four trend tests are reported at α = 0.05 with no correction. Under Holm, the peak-severity result
(p = 0.043) would not survive; the frequency result (p = 0.015) would.

---

## 2. Data limitations

### 2.1 The MODIS → VIIRS transition is not adjusted for

Fire detections come from MODIS for 2000–2011 and VIIRS for 2012 onward
(`download_all_25years.py:49-64`). VIIRS resolves at 375 m against MODIS's 1 km and detects
substantially more fires. Mean detections per day are 20.98 pre-2012 and 246.85 from 2012 on.

That ratio is **not** a clean estimate of the sensor effect — 2017, 2018, 2021 and 2023 were
genuinely severe BC fire seasons, so real fire activity rose over the same period. The honest
statement is that a sensor discontinuity and a real trend are confounded in every fire feature, and
nothing in the pipeline separates them. No sensor or era flag is carried into the merged dataset.

**Fix:** carry a `sensor` column, and either fit era-specific normalisation against the overlap
period or restrict fire-trend claims to within-era comparisons.

### 2.2 The monitoring network grows sevenfold across the record

`n_stations` rises from a mean of 2.96 in 2000 to 19.64 in 2024. The `pm25` column is a
city-wide average, so early years average 2–3 stations and recent years average ~20, with no
normalisation. More stations smooth extremes, so the measured upward trend in extremes is if anything
conservative — but the comparison is not like-for-like and this is not disclosed in the README.

**Fix:** re-run the trend tests on a fixed subset of stations reporting across the whole period, as a
robustness check.

### 2.3 The fire bounding box cannot see the events that matter most

`download_historical_fires.py:33` uses `lat_min: 48.0`, which clips a narrow strip of northern
Washington but stops far north of the Oregon and northern California fires that caused the September
2020 record. On 2020-09-14 the data shows `fire_count_total = 0` and `fire_frp_sum_total = 0.0`
against `pm25 = 163.5` — verified directly.

This is the project's central data ceiling and it is correctly identified in the analysis. It is
listed here because it bounds every fire-based result, not because it is unrecognised.

**Fix:** extend the FIRMS query to the US Pacific Northwest, or add HYSPLIT back-trajectories so
transport direction is represented independently of detection location.

### 2.4 Scope is Metro Vancouver, not BC

`download_bc_air_quality.py:53-79` filters to a hardcoded 25-station Metro Vancouver allowlist. This
is the correct scope for the research questions but the README's "BC Ministry of Environment FTP"
phrasing does not make the narrowing explicit.

### 2.5 Raw and intermediate data are not committed

`.gitignore` excludes everything under `data/` except the final merged CSV. A fresh clone cannot
re-derive `dataset.csv` without re-running all three downloads, and the upstream sources may serve
revised values. No credentials are needed — all three endpoints are public.

---

## 3. Scope and modelling gaps

| Gap | Detail |
|---|---|
| **Daily resolution** | Cross-correlation peaks at lag 0, which at daily resolution means only "within the same calendar day". Sub-day transport timing is unresolvable without hourly data. |
| **No transport physics** | Fire features are counts and FRP by distance band. Nothing represents whether wind actually carried a given fire's plume toward Vancouver; the fire × wind interaction terms are a crude proxy. |
| **Quantile choice is not tuned per horizon** | Q=0.80 was picked from a sweep on one objective. The tradeoff curve is monotone, so the right quantile depends on the alert cost ratio, which was never specified. |
| **No calibrated probabilities** | The gate's raw probabilities average 0.13 on actual smoke days. Isotonic calibration helps but 46 positive events across 25 years is too few to calibrate reliably. |
| **Holdout is a single split** | 2021–2024 is one test period. Model ranking under non-stationarity is inferred from that one window. |

---

## 4. Documentation defects found and fixed

| # | Issue | Status |
|---|---|---|
| 4.1 | `report_final.md` carried numbers from an earlier 10-year (3,654-row) dataset after the record was extended to 25 years (9,133 rows): lag-1 r, the smoke-day weather deltas, the lag-regression R² decomposition, the wind U-components, and the RF ablation R². Each was correct for the data it was written against. | **FIXED** (recomputed against the current dataset) |
| 4.2 | The smoke classifier was described as using "SMOTE oversampling". No SMOTE has ever existed in the code in any commit; the models use `scale_pos_weight`, as `experiment_results.md` correctly stated. | **FIXED** |
| 4.3 | `AUPRC = 0.331` was cited against a figure labelled `AP = 0.239`. | **FIXED** (both reported, §1.1) |
| 4.4 | The September 2020 peak was dated the 13th; the 163.5 µg/m³ maximum falls on the 14th. | **FIXED** |
| 4.5 | q = 0.80 was described as "the 75th percentile of next-day change" in the same section whose table lists q = 0.75 separately. | **FIXED** |
| 4.6 | A "Remaining Tasks" section tracked a course deadline and a `speaking_notes.md` file that does not exist in the repository. | **FIXED** (removed) |
| 4.7 | The dashboard described the study period as "2015 to 2025" and labelled the training split "2015-2020" — both leftovers from the 10-year era, while the code splits on 2000–2020. | **FIXED** |
| 4.8 | `requirements.txt` pinned no Streamlit version, but the app uses the `width="stretch"` API and raises `TypeError` on releases before 1.49. | **FIXED** (pinned `streamlit>=1.49`) |
| 4.9 | An unused NASA FIRMS API key was committed in `download_historical_fires.py`. The archive endpoints the script calls need no key. | **REMOVED from source** (still present in git history) |
| 4.10 | "14 Phase 2 scripts" against 13 actually present. | **FIXED** |

---

## 5. Roadmap, in order

1. **Make the classifier evaluation honest end to end** — disjoint folds, threshold selected off the
   test fold, pooled AP as the headline (§1.1–1.3). Cheap, and it turns the weakest reported number
   into a defensible one.
2. **Commit the RQ3 trend analysis as code** so the Spearman results are re-derivable (§1.6).
3. **Seed the LSTM** or report it as a distribution (§1.5).
4. **Add US Pacific Northwest fire data** — the single highest-value change available, because it
   addresses the ceiling that bounds every fire-based result (§2.3).
5. **Carry a sensor flag and de-confound the MODIS/VIIRS step** (§2.1).
6. **Move to hourly PM2.5** to resolve sub-day transport lag (§3).
7. **Add HYSPLIT back-trajectories** so transport direction is a feature rather than an inference.

---

*Verified against `scripts/`, `notebooks/`, `app/streamlit_app.py`, and recomputation from
`data/processed/merged/dataset.csv`.*
