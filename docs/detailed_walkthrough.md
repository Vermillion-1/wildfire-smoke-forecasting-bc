# Detailed Project Walkthrough

This is a learning-focused walkthrough of the Vancouver wildfire smoke project. It explains what each file does, what we computed, and what results are actually supported by the current code and notebooks.

## 1) What This Project Does

Goal: study wildfire smoke impacts on Vancouver air quality using a merged daily dataset and answer 3 research questions.

- RQ1: Can we predict next-day PM2.5 better than simple baselines?
- RQ2: What is the lag between wildfire activity and PM2.5 in Vancouver?
- RQ3: How has smoke-season risk changed from 2000 to 2025?

Current verified scope (from the active dataset and notebooks):
- Data range in `data/processed/merged/dataset.csv`: 2000-01-01 to 2025-01-01
- Rows/columns: 9,133 x 90
- Smoke threshold: PM2.5 > 25 ug/m3
- Smoke days: 46

## 2) Repository Structure (What Matters Most)

**Data pipeline:**
- `scripts/download_bc_air_quality.py`: downloads/filters Metro Vancouver PM2.5 from BC FTP, outputs daily PM2.5 files.
- `scripts/download_historical_fires.py`: downloads NASA FIRMS fire CSVs, filters BC bbox, adds distance to Vancouver, writes processed fire files.
- `scripts/download_weather.py`: calls Open-Meteo, aggregates hourly to daily weather, adds derived weather features.
- `scripts/build_dataset.py`: merges AQ + fire + weather, creates lag and calendar features, writes `dataset.csv`.

**Notebooks (Phase 1 analysis):**
- `notebooks/01_eda.ipynb`: exploratory analysis and smoke-season profiling (RQ3).
- `notebooks/02_lag_analysis.ipynb`: cross-correlation + lag analysis (RQ2).
- `notebooks/03_modeling.ipynb`: forecasting baselines and ML models (RQ1), plus holdout validation.

**Phase 2 modeling scripts (smoke-day specialisation):**
- `scripts/test_residual_model.py`: LightGBM on pm25_diff — the core residual framing experiment.
- `scripts/train_asymmetric_loss.py`: quantile regression sweep (Q=0.75–0.95) + alternative losses.
- `scripts/train_fire_subset.py`: best smoke-day model — Q=0.80 with fire-only 24 features.
- `scripts/train_smoke_detector.py`: XGBoost binary smoke-day classifier (AUPRC=0.331, 8/13 episodes).
- See `docs/experiment_results.md` for full Phase 2 documentation.

**App and docs:**
- `app/streamlit_app.py`: dashboard with 7 tabs.
- `docs/report.md`: full academic report covering both phases.
- `docs/experiment_results.md`: detailed results for all 9 Phase 2 experiments.

## 3) Data Pipeline (End-to-End)

### 3.1 PM2.5 (BC Government)

Source pattern:
- `ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/{year}/PM25.csv`

Pipeline behavior:
- Download yearly PM2.5 CSVs.
- Keep only defined Metro Vancouver station names.
- Convert to numeric and daily aggregates.
- Daily outputs include: `pm25_mean`, `pm25_median`, `pm25_max`, `pm25_min`, `pm25_std`, `n_readings`, `n_stations`.

### 3.2 Fire Data (NASA FIRMS)

Source patterns:
- VIIRS: `.../country/viirs-snpp/{year}/viirs-snpp_{year}_Canada.csv`
- MODIS: `.../country/modis/{year}/modis_{year}_Canada.csv`

Pipeline behavior:
- Filter to BC bounding box (`lat 48-60`, `lon -130 to -114`).
- Add Haversine distance to Vancouver.
- Save processed per-year files in `data/processed/fires/`.

### 3.3 Weather (Open-Meteo)

Source:
- `https://archive-api.open-meteo.com/v1/archive`

Pipeline behavior:
- Download hourly weather in date chunks.
- Aggregate to daily means/min/max/sums depending on variable.
- Derived features include:
  - `wind_u_component`, `wind_v_component`
  - `temperature_range`
  - `has_precipitation`
  - `heat_index`

### 3.4 Final Merge (`build_dataset.py`)

Verified current behavior:
- Uses latest AQ and weather processed files (important fix to avoid mixing ranges from multiple files).
- Merges on `date` with AQ as base table.
- Fills fire NaNs with 0.
- Creates PM2.5 lags: 1,2,3,7.
- Creates fire lags for all fire columns: 1,2,3.
- Adds calendar features: `month`, `day_of_year`, `is_fire_season`, `year`.
- Writes `data/processed/merged/dataset.csv`.

Current verified output:
- `shape = (9133, 90)`
- `date range = 2000-01-01 .. 2025-01-01`

## 4) Notebook 01 (EDA / RQ3)

File: `notebooks/01_eda.ipynb`

Verified outputs from executed notebook:
- Dataset shape: `(9133, 90)`
- Date range: `2000-01-01` to `2025-01-01`
- PM2.5 percentiles: p50 4.43, p75 ~6, p99 ~30
- Smoke days (`pm25 > 25`): 46
- Smoke episodes (contiguous): 14
- Mean episode duration: 3.3 days
- Max daily mean PM2.5 in smoke episodes: 163.5

Annual smoke-day counts (from notebook output):
- 2015: 2
- 2016: 0
- 2017: 13
- 2018: 8
- 2019: 0
- 2020: 8
- 2021: 2
- 2022: 6
- 2023: 4
- 2024: 0
- 2025: 0 (single day in dataset, non-smoke)

Weather means (smoke vs normal):
- Temperature: 19.53 vs 10.37 (+9.16 C), p<0.001
- Relative humidity: 75.67 vs 80.52 (-4.85), p=0.015
- Wind speed: 8.15 vs 9.61 km/h (-1.46), p=0.029
- Precipitation: 0.62 vs 5.16 mm (-4.54), p<0.001
- Pressure: 1014.88 vs 1016.57 hPa (-1.69), p=0.007

**Trend tests (Spearman rank correlations over 2000-2024):**
- Smoke days: ρ=0.480, p=0.015 (significant)
- Fire season mean PM2.5: ρ=0.202, p=0.33 (not significant)
- Worst episode PM2.5: ρ=0.648, p=0.043 (significant)
- Longest episode: ρ=0.483, p=0.015 (significant)

Three of four metrics show significant upward trends over the 25-year period.

## 5) Notebook 02 (Lag Analysis / RQ2)

File: `notebooks/02_lag_analysis.ipynb`

Verified outputs:
- Fire-season subset size: 1,220 days
- Cross-correlation peak lag by distance band (FRP vs PM2.5):
  - Close: lag 0, r = 0.1994
  - Medium: lag 0, r = 0.2605
  - Far: lag 0, r = 0.1947

Per-year lag table is variable/noisy, but aggregate means are:
- Close mean peak lag: 3.1 days
- Medium mean peak lag: 2.2 days
- Far mean peak lag: 3.5 days

Wind mediation (fire season):
- **Sign convention:** Negative U = easterly (wind from BC interior toward coast). Positive U = westerly (from Pacific).
- U-component mean: normal -0.17 vs smoke +1.22 (positive shift reflects US-origin smoke events like 2020, not interior transport)
- Mann-Whitney p-value (U): 0.1135 (not significant)
- V-component p-value: 0.0570 (marginal)

Lagged regression (in notebook):
- PM2.5 lags only: R2 = 0.7121
- PM2.5 lags + fire lags: R2 = 0.7245
- All features: R2 = 0.7375

## 6) Notebook 03 (Modeling / RQ1)

File: `notebooks/03_modeling.ipynb`

Modeling sample:
- 9,125 rows, 34 features
- Date range: 2000-01-08 to 2024-12-31
- CV: `TimeSeriesSplit(n_splits=5)`, ~1,520 test days/fold

### 6.1 Main CV results (sorted by MAE)

Best to worst (top rows):
- Persistence: MAE 1.694, RMSE 3.166, R2 0.600, smoke MAE 22.11
- Random Forest (100): MAE 1.783, R2 0.165
- Random Forest (200): MAE 1.793, R2 0.153
- LightGBM (Tuned): MAE 1.845, R2 0.146
- XGBoost (Tuned): MAE 1.848, R2 0.065

Key point:
- Persistence remains best on MAE/RMSE/R2 in this setup.

### 6.2 Feature importance and ablation

Top RF importance feature:
- `pm25_lag1` ~0.40 (dominant)

Ablation (RF 200):
- All features: MAE ~1.79, R2 ~0.15
- No fire features: MAE ~1.75, R2 ~0.20 (improves)
- Fire only: MAE ~2.50, R2 ~0.03
- Weather only: MAE ~2.50, R2 ~0.02

### 6.3 Holdout validation (2021-2024 test)

Holdout setup in notebook:
- Train: 2000-2020 (~7,600 samples)
- Test: 2021-2024 (~1,460 samples)

Holdout summary from executed notebook:
- Persistence: MAE 1.759, RMSE 3.884, R2 0.384
- RF (200): MAE 2.088, R2 -0.125
- LightGBM (Tuned): MAE 2.178, R2 -0.100
- XGBoost (Tuned): MAE 2.211, R2 -0.554

**Negative R² indicates** ML models predict worse than the test-set mean, likely due to non-stationarity (PM2.5 distribution shifted between training and test periods) and overfitting to training-period patterns.

Smoke events in holdout test set:
- 12 smoke days
- Persistence classified 7/12 correctly as smoke (>25) when thresholding predictions

## 7) Streamlit Dashboard (What It Currently Contains)

File: `app/streamlit_app.py`

The app has 7 tabs:
- Overview
- Smoke Season Trends
- Smoke Arrival Lag
- PM2.5 Forecasting
- Model Validation
- Health & Planning
- Data Explorer

Notes:
- App starts successfully in local validation.
- It uses Plotly and cached model training.
- App labels and dataset framing are aligned to the active 2000-2025 merged dataset.

## 8) What Was Corrected in This Audit

### Notebook fixes made:
- `notebooks/01_eda.ipynb`:
  - Added formal Spearman trend test cell (RQ3) over 2000-2024 (three metrics significant)
  - Added Mann-Whitney U tests to weather comparison
  - Removed `.head(10)` to show all 14 episodes
  - Excluded 2025 from annual analysis (incomplete year)
  - Fixed fire season highlighting (Jun-Sep, not Jul-Sep)
  - Cleaned up `smoke_episode_id` mutation of main DataFrame
- `notebooks/02_lag_analysis.ipynb`:
  - Fixed cross-year lag leakage — now creates lags on full time series before filtering to fire season
  - Corrected ALL wind U-component labels and interpretation (negative U = easterly/from interior)
  - Added daily resolution caveat for lag-0 finding
  - Removed unused `signal` import
  - Noted in-sample nature of lagged regression
- `notebooks/03_modeling.ipynb`:
  - Completely rewrote Key Findings section (factually wrong — claimed XGBoost beats persistence)
  - Added same-day fire feature justification
  - Fixed holdout index alignment (now uses `df_model` dates)
  - Changed misleading "7-day Rolling Mean" label to "Lag Average"
  - Removed duplicate StandardScaler import

### Code fixes made:
- `scripts/build_dataset.py`:
  - Fixed file-selection logic to avoid short-range files when multiple exports exist.
  - Now selects the widest date-range AQ/weather file.
  - Corrected stale error message reference.
- `scripts/download_historical_fires.py`:
  - Corrected return typing for `download_year`.

### Docs corrected:
- `docs/report.md` updated to match corrected notebook outputs.
- `docs/detailed_walkthrough.md` (this file) updated to match verified outputs.

Presentation notes:
- `docs/milestone_presentation_material.md` is intentionally milestone-era content and not used as the final report baseline.

## 9) Phase 2 Modeling — Smoke-Day Specialisation

Phase 1 (`03_modeling.ipynb`) confirmed persistence beats all standard ML models overall but fails on smoke days (MAE = 22.11). A second phase of nine experiments in `scripts/` specifically targets this.

### 9.1 Core Finding

**Residual framing** (predicting `pm25_diff` = tomorrow − today instead of raw PM2.5) is the single most important architectural change. It converts the autocorrelated series into a near-stationary regression problem and bakes persistence in as the implicit prior.

**Quantile regression at Q=0.80** layered on residual framing pushes the model to predict the 80th percentile of next-day change — penalising under-prediction 4× more than over-prediction. Combined with a **fire-only 24-feature set** (dropping temperature, pressure, humidity which dominate clean days), this achieves:

- Smoke-day MAE: **21.729** vs. persistence **22.353** (Δ = +0.624 µg/m³)
- Overall MAE: 2.075 (worse than persistence's 1.668 — expected tradeoff)

### 9.2 Full Experiment Results Summary

| Script | Model | Overall MAE | Smoke-day MAE | Smoke Δ | Status |
|---|---|---|---|---|---|
| `test_residual_model.py` | LightGBM MAE (residual) | 1.504 | 22.181 | +0.172 | ✓ |
| `test_refined_residual.py` | LightGBM MAE + 38 features | ~1.48 | ~22.18 | +0.17 | ✓ |
| `train_asymmetric_loss.py` | Q=0.75 | 1.852 | 21.813 | +0.540 | ✓ |
| `train_asymmetric_loss.py` | **Q=0.80 (balanced)** | **1.995** | **21.760** | **+0.593** | ✓ |
| `train_asymmetric_loss.py` | Q=0.90 | 2.521 | 21.391 | +0.962 | ✓ |
| `train_residual_correction.py` | MAE + Layer 2 (p80) | 1.509 | 22.055 | +0.298 | ✓ |
| `train_soft_blend.py` | Calib blend Q=0.80 (p²) | 1.507 | 21.965 | +0.388 | ✓ |
| `train_fire_subset.py` | **Q=0.80 fire-only 24 feat** | **2.075** | **21.729** | **+0.624** | ✓ Best |
| `train_smoke_detector.py` | XGBoost gate | AUPRC=0.331 | 8/13 episodes | — | ✓ |
| `train_asymmetric_loss.py` | Sample-weighted 20× | 1.623 | 23.274 | −0.921 | ✗ |
| `train_hurdle_model.py` | Hurdle model | 1.813 | 24.164 | −1.2% | ✗ |
| `train_lstm.py` | LSTM (PyTorch) | 2.342 | — | −27% | ✗ Negative |
| `tune_residual_model.py` | Optuna 30 trials | 1.713 | 21.880 | +0.473 | — Defaults better |

### 9.3 XGBoost Gate Episode Analysis

The gate catches 8 of 13 unique episodes. The 5 missed have structural root causes:

| Episode | Root Cause |
|---|---|
| 2005-09-12 (1 day) | Pre-2010, sparse training data — structural |
| 2017-08-01–08-10 (10 days) | Distant fires only (mean 350km), no close signal |
| 2018-08-12–08-14 (3 days) | Fast onset — fire jumped on Day 2, gate sees Day 0 |
| **2020-09-10–09-17 (8 days)** | **Oregon/California Labor Day fires. fire_count_total = 0, pm25 = 65–163. Data ceiling.** |
| 2023-08-19 (1 day) | Single pre-smoke day edge case |

The 2020 miss directly motivates HYSPLIT back-trajectory integration or US VIIRS fire data as future work.

### 9.4 Key Inferences

1. **Residual framing > any loss function.** Direct Q=0.90 without residual framing: smoke-day MAE = 39.650 (−17.3 vs persistence). With residual framing: 21.391 (+0.962).
2. **Quantile tradeoff is monotone and predictable.** Each +0.05 in alpha costs ~0.3–0.5 overall MAE and buys ~0.2–0.3 smoke-day improvement.
3. **Sample weighting backfires.** The model learns "elevated PM2.5 keeps rising" but most elevated days don't become smoke days.
4. **Weather features hurt smoke-day prediction.** They regularise the model toward normal-day behaviour, which is correct most of the time but wrong during fire events.
5. **LSTM negative.** 9K rows with 46 positive events is far too small for sequence models. Tree models dominate because the key signal (FRP × wind direction) is a multiplicative feature interaction — a single tree split.

## 10) Reproduce Verified State

```bash
source venv/bin/activate
python scripts/build_dataset.py
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --inplace --ExecutePreprocessor.kernel_name=vancouver-smoke
jupyter nbconvert --to notebook --execute notebooks/02_lag_analysis.ipynb --inplace --ExecutePreprocessor.kernel_name=vancouver-smoke
jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb --inplace --ExecutePreprocessor.kernel_name=vancouver-smoke
streamlit run app/streamlit_app.py

# Phase 2 — run from project root
python scripts/test_residual_model.py
python scripts/train_asymmetric_loss.py
python scripts/train_fire_subset.py
python scripts/train_smoke_detector.py   # requires imbalanced-learn
python scripts/train_residual_correction.py
python scripts/train_soft_blend.py
python scripts/train_hurdle_model.py
python scripts/analyze_smoke.py
python scripts/analyze_missed_episodes.py
# Optional (additional dependencies):
# python scripts/train_lstm.py           # requires torch
# python scripts/tune_residual_model.py  # requires optuna
```
