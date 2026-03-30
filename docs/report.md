# Vancouver Wildfire Smoke Predictor

**Predicting and Profiling Wildfire Smoke Impacts on Vancouver's Air Quality (2000–2025)**

CMPT 733 — Big Data Lab II
Simon Fraser University

---

## Abstract

Wildfire smoke has become a recurring public health concern in Vancouver, British Columbia, with several severe smoke events over the past two decades. This project integrates daily PM2.5 air quality measurements, NASA MODIS and VIIRS satellite fire detections, and meteorological data to investigate three research questions: (1) whether machine learning models can forecast next-day PM2.5 more accurately than naive baselines, (2) what the typical delay is between wildfire activity and elevated PM2.5 in Vancouver, and (3) how Vancouver's smoke season has changed from 2000 to 2025. We construct a merged dataset of 9,133 daily records spanning 90 features and apply time-series cross-validation to evaluate persistence, rolling-mean, linear, regularized, tree-ensemble, and gradient-boosting models. Our findings show that the persistence baseline (tomorrow equals today) remains unbeaten by any ML model (MAE = 1.69 vs. best ML MAE = 1.78), that fire-to-smoke lag peaks at zero days at daily resolution, and that smoke seasons show statistically significant upward trends in frequency, peak severity, and episode duration over 2000-2024. An interactive Streamlit dashboard accompanies this report.

---

## 1. Introduction

### 1.1 Motivation

Wildfire smoke is a growing environmental and public health challenge in western North America. Vancouver, despite being a coastal city far from most fire-prone regions, regularly experiences hazardous air quality during British Columbia's wildfire season (June–September). Major smoke events in 2017, 2018, 2020, and 2023 caused PM2.5 levels to exceed health guidelines by several multiples, prompting air quality advisories and impacting millions of residents.

Understanding and predicting these smoke events is valuable for public health planning, outdoor activity advisories, and resource allocation. This project explores whether readily available data sources — satellite fire detections, weather observations, and historical air quality — can be combined to forecast next-day PM2.5 levels and characterize smoke season patterns.

### 1.2 Research Questions

1. **PM2.5 Forecasting (RQ1):** Can simple ML models (linear regression, ridge, lasso, random forest) predict daily average PM2.5 in Vancouver one day ahead more accurately than persistence and rolling-mean baselines, using wildfire activity and meteorological features?

2. **Smoke Arrival Lag (RQ2):** What is the typical delay (in days) between a spike in BC wildfire radiative power and elevated PM2.5 in Vancouver, and how do fire distance and wind direction mediate this lag?

3. **Seasonal Risk Profiling (RQ3):** How has Vancouver's wildfire smoke season changed from 2000 to 2025 — are smoke events becoming more frequent, longer, or more intense?

---

## 2. Data Sources and Collection

### 2.1 PM2.5 Air Quality

**Source:** British Columbia Ministry of Environment FTP server (`ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/`)

Hourly PM2.5 readings from Metro Vancouver monitoring stations were retrieved for 2000–2024. The raw data comprises hourly observations across multiple stations. We aggregated to daily averages per station, then computed the Metro Vancouver daily mean across all active stations, yielding 9,133 daily records with zero missing days.

| Statistic | Value |
|---|---|
| Daily mean PM2.5 | 5.29 ug/m3 |
| Median PM2.5 | 4.43 ug/m3 |
| Max PM2.5 | 163.52 ug/m3 |
| 99th percentile | ~30 ug/m3 |
| Missing days | 0 |

### 2.2 Wildfire Data

**Source:** NASA FIRMS MODIS (2000–2011) and VIIRS (2012–2024) Active Fire Data

We downloaded MODIS fire detections (2000–2011) and VIIRS fire detections (2012–2024) for all of Canada and filtered to British Columbia. Each fire detection includes latitude, longitude, fire radiative power (FRP, in MW), confidence level, and timestamp. Fires were categorized into three distance bands from Vancouver:

- **Close:** < 200 km
- **Medium:** 200–500 km
- **Far:** 500–1,000 km

Daily aggregations include fire counts and total FRP per distance band. Lag features (1-day, 2-day, 3-day) were computed for temporal analysis.

### 2.3 Weather Data

**Source:** Open-Meteo Historical Weather API

Daily weather variables for Vancouver (49.25°N, 123.12°W) were retrieved for 2000–2025, totaling over 9,500 daily records. Variables include:

- Temperature (mean, min, max, range)
- Relative humidity (mean, min, max)
- Precipitation (sum, max)
- Wind speed and direction (mean, max, gusts)
- Mean sea-level pressure
- Derived features: U/V wind components, heat index, precipitation indicator

### 2.4 Dataset Construction

The three data sources were merged on date, producing a final dataset of **9,133 rows and 90 columns**. Additional engineered features include:

- PM2.5 lag features (1, 2, 3, and 7 days)
- Fire activity lag features (1, 2, 3 days for each distance band)
- Calendar features (month, day of year, fire season indicator)
- Smoke event flag (PM2.5 > 25 ug/m3)

The smoke threshold of 25 ug/m3 was chosen based on the BC Air Quality Health Index (AQHI) moderate risk level.

---

## 3. Methodology

### 3.1 Exploratory Data Analysis and Seasonal Risk Profiling (RQ3)

We analyzed the temporal distribution of PM2.5, identified smoke events (contiguous days above 25 ug/m3), and computed annual smoke season statistics including number of smoke days, episode count and duration, and fire season mean PM2.5. Weather conditions on smoke days were compared against normal days using descriptive statistics.

### 3.2 Cross-Correlation and Lag Analysis (RQ2)

To quantify the fire-to-smoke delay, we computed normalized cross-correlation functions between daily fire activity (FRP and fire count, by distance band) and PM2.5 during fire seasons (June–September). Cross-correlations were computed at lags from -7 to +14 days. We also performed lagged regression analysis to quantify the incremental predictive value of fire lag features beyond PM2.5 autoregression.

Wind direction analysis compared U-component distributions on smoke vs. normal days, with a two-sample t-test for statistical significance. Three major smoke episodes (August 2017, August 2018, September 2020) were examined as case studies.

### 3.3 Predictive Modeling (RQ1)

**Target variable:** Next-day PM2.5 (i.e., `pm25.shift(-1)`)

**Feature set (34 features):**

| Group | Count | Description |
|---|---|---|
| PM2.5 lags | 4 | Lag 1, 2, 3, 7 days |
| Fire same-day | 9 | Count and FRP by distance band |
| Fire lagged | 8 | Total and band-specific lags |
| Weather | 10 | Temperature, humidity, wind, pressure, precipitation |
| Calendar | 3 | Month, day of year, fire season flag |

**Baselines:**

- **Persistence:** Tomorrow's PM2.5 = today's PM2.5
- **Rolling mean:** Average of lag 1, 2, 3, and 7 values

**ML models:**

- Linear Regression
- Ridge Regression (alpha = 1.0, 10.0)
- Lasso Regression (alpha = 0.1, 1.0)
- Random Forest (100 trees: max_depth=15, min_samples_leaf=5; 200 trees: max_depth=20, min_samples_leaf=3)
- XGBoost (default and tuned)
- LightGBM (default and tuned)

**Evaluation strategy:** Expanding-window time-series cross-validation with 5 folds via scikit-learn's `TimeSeriesSplit`. This prevents data leakage by ensuring all training data precedes test data temporally. Each fold uses approximately 1,520 test days.

| Fold | Training Period | Test Period |
|---|---|---|
| 1 | 2000-01-08 to 2003-01-01 | 2003-01-02 to 2007-01-01 |
| 2 | 2000-01-08 to 2007-01-01 | 2007-01-02 to 2011-01-01 |
| 3 | 2000-01-08 to 2011-01-01 | 2011-01-02 to 2015-01-01 |
| 4 | 2000-01-08 to 2015-01-01 | 2015-01-02 to 2019-01-01 |
| 5 | 2000-01-08 to 2019-01-01 | 2019-01-02 to 2024-12-31 |

**Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared, and MAE on smoke days only (PM2.5 > 25 ug/m3). All metrics are averaged across folds.

**Feature importance** was assessed via Random Forest mean decrease in impurity (MDI) and Ridge regression coefficients. A feature ablation study systematically removed each feature group to measure its contribution.

---

## 4. Results

### 4.1 RQ3: Seasonal Risk Profiling

Over the 2000–2024 period, we identified **46 smoke event days** across **14 distinct episodes**. Smoke events are rare (0.5% of all days) and highly episodic.

**Recent Annual Smoke Day Counts (2015-2024):**

| Year | Smoke Days | Max PM2.5 | Notable Events |
|---|---|---|---|
| 2015 | 2 | 47.1 | Early July episode |
| 2016 | 0 | — | Clean year |
| 2017 | **13** | 55.5 | Longest episode (10 consecutive days, Aug 2–11) |
| 2018 | 8 | 112.1 | Severe August smoke |
| 2019 | 0 | — | Clean year |
| 2020 | 8 | **163.5** | Worst episode (Sep 11–18, US fires) |
| 2021 | 2 | 74.8 | August event |
| 2022 | 6 | 79.2 | Unusual Oct event |
| 2023 | 4 | 46.6 | August event |
| 2024 | 0 | — | Clean year |

Key observations:

- **Statistically significant upward trends.** Formal Spearman rank correlation tests over 2000–2024 reveal significant upward trends in smoke day frequency (ρ = 0.480, p = 0.015), peak severity (ρ = 0.648, p = 0.043), and episode duration (ρ = 0.483, p = 0.015). Fire season mean PM2.5 shows no significant trend (ρ = 0.202, p = 0.33). The 25-year window provides sufficient statistical power to detect these trends.
- Smoke events cluster in **August and September**, with October 2022 as an outlier.
- The **worst episode** (September 2020, peak PM2.5 = 163.5 µg/m³) occurred during a year of very low BC fire activity; the smoke originated from catastrophic Oregon/Washington wildfires, demonstrating that cross-border smoke transport is a major factor.
- Mean episode duration is **3.3 days**; the longest was 10 days (August 2017).

**Weather on smoke days vs. normal days:**

| Variable | Smoke Days | Normal Days | Difference |
|---|---|---|---|
| Temperature (°C) | 19.5 | 10.4 | +9.2 |
| Relative Humidity (%) | 75.7 | 80.5 | -4.9 |
| Precipitation (mm) | 0.62 | 5.16 | -4.54 |
| Wind Speed (km/h) | 8.15 | 9.61 | -1.46 |

Smoke days are characterized by significantly hotter, drier, and calmer conditions — consistent with the stagnant high-pressure systems that trap smoke at ground level.

### 4.2 RQ2: Smoke Arrival Lag

**Cross-correlation analysis** during fire seasons (June–September) reveals that the peak correlation between daily fire activity and PM2.5 occurs at **lag 0** (same day) across all distance bands:

| Distance Band | Peak Lag | Peak Correlation |
|---|---|---|
| Close (< 200 km) | 0 days | r = 0.199 |
| Medium (200–500 km) | 0 days | r = 0.261 |
| Far (500–1,000 km) | 0 days | r = 0.195 |

The zero-lag peak suggests that at daily resolution, smoke transport from fire to city occurs within the same calendar day, or that fire activity and PM2.5 share common meteorological drivers (e.g., hot, dry weather causes both fires and poor dispersion).

**Per-year analysis** shows more variability: mean peak lags of 3.1 days (close), 2.2 days (medium), and 3.5 days (far), but with high variance driven by small sample sizes.

**Regression analysis** confirms that PM2.5 autoregression dominates:

| Feature Set | R-squared |
|---|---|
| PM2.5 lags only | 0.712 |
| PM2.5 lags + fire lags | 0.724 |
| All features | 0.738 |

Adding fire lag features to PM2.5 lags improves R-squared by only 0.008, while weather features contribute an additional 0.018. Fire lags alone explain approximately 10% of PM2.5 variance.

**Wind direction analysis** shows that easterly winds (U-component < 0, indicating flow from BC's interior where fires burn) are slightly less common on smoke days (mean U = +1.22 m/s vs. -0.17 m/s on normal days), but the difference is not statistically significant (Mann-Whitney p = 0.114) given the small sample of 46 smoke days. The positive U shift on smoke days may reflect that severe events like September 2020 originated from US fires to the south, where southerly/westerly flow dominates, rather than from BC's interior.

### 4.3 RQ1: PM2.5 Forecasting

**Model comparison (averaged across 5 time-series CV folds):**

| Model | MAE | RMSE | R-squared | MAE (Smoke Days) |
|---|---|---|---|---|
| **Persistence baseline** | **1.694** | **3.166** | **0.600** | **22.11** |
| Random Forest (100 trees) | 1.783 | 4.842 | 0.165 | 43.60 |
| Random Forest (200 trees) | 1.793 | 4.878 | 0.153 | 43.50 |
| LightGBM (Tuned) | 1.845 | 4.898 | 0.146 | 43.89 |
| XGBoost (Tuned) | 1.848 | 5.123 | 0.065 | 45.46 |
| XGBoost | 1.848 | 4.883 | 0.151 | 42.80 |
| LightGBM | 1.864 | 4.915 | 0.140 | 43.93 |
| Lasso (alpha=0.1) | 1.858 | 4.340 | 0.329 | 34.63 |
| Ridge (alpha=10.0) | 2.003 | 4.602 | 0.246 | 32.84 |
| Ridge (alpha=1.0) | 2.004 | 4.602 | 0.246 | 32.85 |
| Linear Regression | 2.004 | 4.602 | 0.246 | 32.85 |
| Lasso (alpha=1.0) | 2.058 | 4.954 | 0.126 | 46.46 |
| Rolling Mean baseline | 2.375 | 4.786 | 0.087 | 36.06 |

**The persistence baseline outperforms all ML models** on every metric. This is a well-known phenomenon in short-horizon forecasting of autocorrelated time series: PM2.5 has a strong lag-1 autocorrelation, meaning today's value is a strong predictor of tomorrow's. The ML models, while beating the rolling-mean baseline, cannot overcome this strong autocorrelation structure.

All models struggle severely on **smoke days** (MAE 22–46 ug/m3), reflecting the fundamental challenge: smoke events are rare (46 out of 9,133 days = 0.5%), extreme, and driven by factors not fully captured in the feature set (e.g., cross-border smoke transport, plume dynamics).

**Feature importance** (Random Forest, 200 trees):

| Rank | Feature | Importance | Group |
|---|---|---|---|
| 1 | pm25_lag1 | ~0.40 | PM2.5 Lags |
| 2 | wind_v_component | ~0.05 | Weather |
| 3 | pm25_lag3 | ~0.04 | PM2.5 Lags |
| 4 | precipitation_sum | ~0.04 | Weather |
| 5 | fire_count_medium | ~0.04 | Fire Activity |

Yesterday's PM2.5 dominates at ~40% importance. Weather features collectively outweigh fire features, consistent with the lag analysis finding that local meteorology matters more than raw fire counts.

**Feature ablation study** (Random Forest, 200 trees):

| Subset | MAE | R-squared | Features |
|---|---|---|---|
| All features | ~1.79 | ~0.15 | 34 |
| No PM2.5 lags | ~1.90 | ~0.10 | 30 |
| **No fire features** | **~1.75** | **~0.20** | **17** |
| No weather | ~2.00 | ~0.10 | 24 |
| No calendar | ~1.80 | ~0.15 | 31 |
| PM2.5 lags only | ~2.00 | ~0.15 | 4 |
| Fire features only | ~2.50 | ~0.03 | 17 |
| Weather only | ~2.50 | ~0.02 | 10 |

A striking finding: **removing fire features improves model performance**. This suggests that at daily resolution, satellite fire counts add noise rather than signal for next-day PM2.5 prediction. The most parsimonious useful model combines PM2.5 lags and weather features.

---

## 5. Discussion

### 5.1 Key Findings

1. **Persistence is hard to beat.** For next-day PM2.5 forecasting, the simple heuristic "tomorrow will be like today" outperforms all tested ML models. This is consistent with the air quality forecasting literature, where persistence baselines are notoriously strong at short horizons.

2. **Fire counts are noisy predictors.** Despite the intuitive appeal of using fire detection data, satellite fire counts at daily resolution do not improve PM2.5 predictions — and actually degrade them. This may be because (a) the fire-to-smoke pathway involves complex atmospheric transport not captured by simple fire counts, (b) some smoke events originate from outside BC (e.g., the 2020 US fires), and (c) the relationship between fire intensity and smoke impact depends on meteorological conditions.

3. **Smoke seasons show significant upward trends.** Formal Spearman rank tests over 2000–2024 reveal statistically significant upward trends in smoke day frequency (ρ = 0.480, p = 0.015), peak severity (ρ = 0.648, p = 0.043), and episode duration (ρ = 0.483, p = 0.015). Fire season mean PM2.5 shows no significant trend (p = 0.33). The 25-year window provides sufficient statistical power to detect these trends that were not apparent in the shorter 10-year analysis.

4. **Cross-border smoke is a wild card.** The worst PM2.5 event (September 2020, peak 163.5 ug/m3) was driven by Oregon/Washington fires, not BC fires. Any operational forecasting system for Vancouver must account for transboundary smoke transport.

### 5.2 Limitations

- **Daily resolution** is too coarse to capture within-day smoke dynamics. Hourly modeling could reveal meaningful fire-to-smoke lags.
- **Fire data is limited to BC.** Cross-border fires (Washington, Oregon, Alberta) are significant contributors but are not included in the feature set.
- **Small sample of extreme events.** With only 46 smoke days and 14 episodes over 25 years, ML models lack sufficient training examples for extreme events.
- **No atmospheric transport modeling.** Features like HYSPLIT back-trajectories or smoke plume forecasts could substantially improve predictions.
- **Feature engineering is basic.** More sophisticated features (rolling fire intensity windows, wind-weighted fire proximity, synoptic weather patterns) could improve model performance.

### 5.3 Future Work

- Incorporate **hourly data** to capture sub-daily dynamics and meaningful lag structures.
- Add **cross-border fire data** (US VIIRS detections, Alberta fires) to account for transboundary smoke.
- Explore **sequence models** (LSTM, temporal convolutional networks) that can learn complex temporal patterns in the autocorrelated PM2.5 series.
- Integrate **HYSPLIT back-trajectory analysis** or satellite-derived smoke plume data as features.
- Develop a **smoke event classification model** (binary: smoke day or not) as a complement to the regression approach, potentially with better utility for public health advisories.
- Extend the analysis period as more years of data accumulate to better assess long-term trends.

---

## 6. Conclusion

This project demonstrates both the promise and the limitations of using satellite fire detections and weather data for urban air quality forecasting. While the integrated dataset reveals meaningful patterns — smoke events cluster in August–September, are associated with hot/dry weather, and show same-day correlation with fire activity — these patterns are insufficient for ML models to outperform a simple persistence baseline for next-day PM2.5 prediction.

The honest finding that persistence wins is itself informative: it tells us that at daily resolution with the available features, the system is dominated by short-term autocorrelation rather than external forcing. Improving upon persistence will likely require higher temporal resolution, atmospheric transport information, and cross-border fire data.

The accompanying Streamlit dashboard provides an interactive tool for exploring these findings and the underlying data.

---

## 7. References

1. BC Ministry of Environment. Air Quality Monitoring Data. FTP: `ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/`
2. NASA FIRMS. VIIRS Active Fire Data. https://firms.modaps.eosdis.nasa.gov/
3. Open-Meteo. Historical Weather API. https://open-meteo.com/
4. Larsen, A. E., et al. (2021). Impacts of fire smoke plumes on regional air quality. *Environmental Science & Technology*.
5. Yao, J., et al. (2020). Predicting wildfire smoke concentrations in British Columbia. *Journal of Exposure Science & Environmental Epidemiology*.
6. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
7. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.

---

## Appendix A: Project Structure

```
vancouver-wildfire-smoke-predictor/
├── app/
│   └── streamlit_app.py          # Interactive Streamlit dashboard
├── data/
│   ├── raw/                      # Raw downloaded data
│   └── processed/
│       └── merged/dataset.csv    # Final merged dataset (9,133 x 90)
├── docs/
│   ├── report.md                 # This report
│   └── presentation.pptx        # Slide deck
├── figures/                      # Generated visualizations (20 PNGs)
├── notebooks/
│   ├── 01_eda.ipynb              # EDA & seasonal risk profiling
│   ├── 02_lag_analysis.ipynb     # Smoke arrival lag analysis
│   └── 03_modeling.ipynb         # Predictive modeling
├── scripts/
│   ├── build_dataset.py          # Dataset construction pipeline
│   ├── download_bc_air_quality.py
│   ├── download_historical_fires.py
│   ├── download_weather.py
│   └── download_all_25years.py   # Bulk 25-year download helper
└── requirements.txt
```

## Appendix B: Key Figures

All figures are generated programmatically in the Jupyter notebooks and saved to the `figures/` directory. Key visualizations include:

- `pm25_timeseries.png` — 25-year PM2.5 time series with smoke events highlighted
- `smoke_season_trends.png` — Annual smoke days and fire season PM2.5 trends
- `cross_correlation_frp_pm25.png` — Fire radiative power vs. PM2.5 cross-correlation
- `model_comparison.png` — Model performance comparison bar chart
- `feature_importance_rf.png` — Random Forest feature importance
- `pred_vs_actual_scatter.png` — Predicted vs. actual PM2.5 scatter plot
