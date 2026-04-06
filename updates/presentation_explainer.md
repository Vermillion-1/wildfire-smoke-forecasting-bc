# Vancouver Wildfire Smoke Predictor — Complete Explainer for Presentation

> **Who this is for:** You. A beginner in statistics who needs to understand every part of this project well enough to answer tough professor questions. Read this top to bottom before your presentation.

---

## Table of Contents

1. [The Big Picture — What Is This Project?](#1-the-big-picture)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [The Three Research Questions](#3-the-three-research-questions)
4. [The Data — Where It Comes From and Why](#4-the-data)
5. [How the Data Was Built — The Pipeline](#5-the-data-pipeline)
6. [Exploratory Data Analysis (Notebook 01)](#6-eda-notebook-01)
7. [Smoke Arrival Lag Analysis (Notebook 02)](#7-lag-analysis-notebook-02)
8. [Machine Learning Models (Notebook 03)](#8-predictive-modeling-notebook-03)
9. [Key Results and Findings](#9-key-results-and-findings)
10. [The Streamlit Dashboard](#10-the-streamlit-dashboard)
11. [Statistics Concepts Explained Simply](#11-statistics-concepts-explained)
12. [Anticipated Professor Questions and Answers](#12-professor-questions-and-answers)

---

## 1. The Big Picture

**What is the project in one sentence?**

We collected 25 years of data (2000–2025) about Vancouver's air quality, nearby wildfires, and weather — then asked: *Can machine learning predict smoke? And is smoke getting worse over time?*

**What is PM2.5?**

PM2.5 stands for **Particulate Matter with diameter ≤ 2.5 micrometres**. These are tiny particles — mostly from fire smoke, car exhaust, and industrial pollution — small enough to enter your lungs and bloodstream. When wildfires burn near Vancouver, PM2.5 spikes dramatically.

- **Good air quality:** PM2.5 < 12 µg/m³
- **Unhealthy for sensitive groups:** PM2.5 > 25 µg/m³ (our "smoke day" threshold)
- **Hazardous:** PM2.5 > 150 µg/m³ (September 2020 hit **163.5 µg/m³**)

µg/m³ = micrograms per cubic metre of air.

---

## 2. The Problem We Are Solving

Every summer, wildfires in BC (and sometimes the US) fill Vancouver's air with smoke. City residents need to know:

1. **Will tomorrow be smoky?** (so they can plan outdoor activities, issue health alerts)
2. **How long after a fire starts does smoke actually reach Vancouver?**
3. **Is the smoke season getting worse year over year?**

These translate directly to our three research questions.

**Why does this matter?**
- Smoke causes respiratory illness, especially in children, elderly, and people with asthma.
- City planners and BC Ministry of Health use air quality forecasts.
- Climate change is making wildfires more common and intense — understanding trends is urgent.

---

## 3. The Three Research Questions

### RQ1 — Can ML predict tomorrow's PM2.5?

> *Can machine learning models predict next-day PM2.5 more accurately than simple baseline approaches?*

**Short answer:** No. The simplest possible forecast — "tomorrow = today" — beats every machine learning model we tried.

**Why this matters:** It teaches us something important about the nature of the data (strong day-to-day correlation) and about when ML is and isn't useful.

---

### RQ2 — How long does smoke take to arrive?

> *What is the typical delay (lag) between a wildfire in BC and elevated PM2.5 in Vancouver?*

**Short answer:** Same calendar day (lag = 0). With daily data we can't detect lags shorter than 24 hours, so all sub-day transport looks like "lag 0."

---

### RQ3 — Is smoke getting worse?

> *How has Vancouver's wildfire smoke season changed from 2000 to 2024?*

**Short answer:** Yes, and it is statistically significant. Smoke seasons are becoming more frequent, more severe, and longer.

---

## 4. The Data

We used **three independent data sources**, each measuring a different aspect of the problem.

### 4.1 Air Quality Data — What We Are Predicting

| Detail | Value |
|---|---|
| Source | BC Ministry of Environment FTP server |
| What it measures | PM2.5 (µg/m³) from 25 monitoring stations in Metro Vancouver |
| Time range | 2000–2024 |
| Original frequency | Hourly readings |
| How we used it | Averaged all stations to get one daily city-level PM2.5 value |

**Why 25 stations?** Air quality varies across the city. Averaging across stations gives a more reliable city-wide picture than relying on a single sensor (which could malfunction or be in an unusual spot).

**Why daily?** Hourly data is noisier and harder to join with fire/weather data. Daily is a standard unit for epidemiological analysis.

---

### 4.2 Fire Data — The Input We Hypothesized Would Help

| Detail | Value |
|---|---|
| Source | NASA FIRMS (Fire Information for Resource Management System) |
| Sensors | MODIS satellite (2000–2011), VIIRS satellite (2012–2024) |
| What it measures | Location, fire radiative power (FRP), and confidence for each fire detection |
| Coverage area | All of BC (bounding box: lat 48–60°N, lon −130 to −114°W) |
| How we used it | Filtered to fires within 1,000 km of Vancouver; grouped by distance band |

**What is FRP (Fire Radiative Power)?**
It is how much energy a fire is releasing — essentially a measure of fire intensity. A large, intense fire has high FRP; a small smouldering fire has low FRP. Units: megawatts (MW).

**Distance bands we created:**
- **Close:** 0–200 km (fires very near Vancouver — southern BC coast, Lower Mainland)
- **Medium:** 200–500 km (central BC interior, Okanagan)
- **Far:** 500–1,000 km (northern BC, Alberta border)

**Why distance bands?** A fire 50 km away should affect Vancouver differently than a fire 800 km away. Grouping by distance lets us test whether proximity matters.

**Why two satellites?** MODIS flew from 2000; VIIRS launched in 2012 with better resolution (~375m vs ~1km). We used the best available sensor for each time period.

---

### 4.3 Weather Data — The Hidden Mediator

| Detail | Value |
|---|---|
| Source | Open-Meteo Historical API |
| Location | Vancouver (49.28°N, 123.12°W) |
| Time range | 2000–2025 |
| Variables | Temperature, humidity, precipitation, wind speed, wind direction, pressure |
| Original frequency | Hourly |
| How we used it | Aggregated to daily (mean, max, sum) |

**Why weather matters:**
- Hot, dry, calm days trap smoke near the ground (no wind to disperse it, no rain to wash it out)
- Wind direction determines *which way* smoke travels — easterly wind blows smoke away from Vancouver; southerly/westerly wind can carry US smoke northward
- High-pressure systems cause stagnation (smoke can't escape vertically)

**Derived variables we engineered from raw weather:**
- `wind_u_component`: Eastward wind component (negative = wind blowing FROM the east)
- `wind_v_component`: Northward wind component
- `temperature_range`: Max − Min temperature that day (proxy for weather stability)
- `has_precipitation`: Binary (1 if any rain/snow, 0 otherwise)

---

### 4.4 Final Dataset

After merging all three sources, we have:

| Property | Value |
|---|---|
| Rows | 9,133 (one per day, 2000-01-01 to 2025-01-01) |
| Columns | 90 (features + target) |
| Target variable | `pm25` (daily mean PM2.5 in µg/m³) |
| Missing days | 0 (complete daily record) |

**PM2.5 statistics across 25 years:**

| Metric | Value |
|---|---|
| Mean | 5.29 µg/m³ |
| Median | 4.43 µg/m³ |
| Maximum | 163.52 µg/m³ (Sep 13, 2020) |
| Smoke days (PM2.5 > 25) | 46 out of 9,133 (0.5%) |
| Smoke episodes | 14 (groups of consecutive smoke days) |

---

## 5. The Data Pipeline

This is the sequence of code that converts raw downloaded files into the final analysis-ready dataset.

```
Raw Data (CSVs from internet)
        ↓
[download_bc_air_quality.py]   → Hourly PM2.5 → Daily city average
[download_historical_fires.py] → Fire detections → Daily counts/FRP by distance band
[download_weather.py]          → Hourly weather → Daily aggregates
        ↓
[build_dataset.py]
   1. Load air quality (base table, sets date range)
   2. Aggregate fire data by date + distance band
   3. Merge weather by date
   4. Fill fire NaNs with 0 (no fires detected = 0)
   5. Engineer lag features (PM2.5 lag 1, 2, 3, 7 days; fire lags 1, 2, 3 days)
   6. Add calendar features (month, is_fire_season, day_of_year, year)
        ↓
data/processed/merged/dataset.csv  (9,133 × 90)
```

**What is a "lag feature"?**

A lag feature is yesterday's value of a variable used as a predictor for today's target. For example:
- `pm25_lag1` = PM2.5 from 1 day ago
- `pm25_lag7` = PM2.5 from 7 days ago
- `fire_count_lag2` = number of fires detected 2 days ago

We create lags because smoke from a fire today might not show up in Vancouver's air quality until tomorrow or the day after.

**What is `is_fire_season`?**

A binary variable (1 = June/July/August/September, 0 = all other months). Wildfire risk in BC is almost entirely concentrated in these months.

---

## 6. EDA — Notebook 01

EDA = **Exploratory Data Analysis**. This is the "look at the data before modeling" step. We created plots and computed summary statistics to understand patterns, trends, and anomalies.

### 6.1 Distribution of PM2.5

The histogram of daily PM2.5 is **heavily right-skewed**:
- Most days (~80%) have PM2.5 between 0–10 µg/m³ (clean air)
- A long tail stretches to 163.5 µg/m³ (extreme smoke events)
- The median (4.43) is much lower than the mean (5.29) — this skew is classic for environmental data

**Why does skew matter?** Models that assume symmetric distributions (like ordinary linear regression) can struggle with highly skewed targets. Our models predict the *raw* PM2.5 value, not the log-transformed version, which partly explains their difficulty with extreme events.

---

### 6.2 Seasonal Patterns

| Month | Mean PM2.5 | Notes |
|---|---|---|
| Jan–Apr | ~4–5 µg/m³ | Clean winter/spring |
| May | ~4.5 µg/m³ | Fire season begins |
| Jun–Jul | ~6–7 µg/m³ | Increasing fire risk |
| **Aug–Sep** | **~9–12 µg/m³** | **Peak smoke season** |
| Oct–Dec | ~5–6 µg/m³ | Season ends |

**Why August-September?** BC wildfires peak in late summer due to:
- Accumulated heat and drought since spring
- Low humidity (plants dry out, become fuel)
- Historically, lightning storms in August ignite fires

---

### 6.3 Smoke Episodes Identified

We defined a **smoke event** as any day with PM2.5 > 25 µg/m³, and a **smoke episode** as a group of consecutive smoke days.

| Episode | Year | Start | End | Duration | Peak PM2.5 |
|---|---|---|---|---|---|
| Worst ever | 2020 | Sep 11 | Sep 18 | 8 days | **163.5 µg/m³** |
| Longest | 2017 | Aug 2 | Aug 11 | **10 days** | 55.5 µg/m³ |
| Severe | 2018 | Aug | Aug | 8 days | 112.1 µg/m³ |
| Recent | 2021 | Aug | Aug | 2 days | 74.8 µg/m³ |

Total: **14 episodes across 25 years** (about 0.56 per year, but unevenly distributed — mostly in post-2015 years).

---

### 6.4 Trend Analysis (Spearman Rank Correlation)

**What is Spearman rank correlation?**

It is a statistical test that measures whether two things tend to go up or down together over time — without assuming the relationship is perfectly linear. It ranks both variables from lowest to highest and correlates the ranks.

- ρ (rho) ranges from −1 to +1
- ρ = +1 means perfect upward trend
- ρ = 0 means no trend
- **p-value < 0.05** means the trend is statistically significant (unlikely to be random chance)

**Our trend results (2000–2024, n = 24 yearly observations):**

| What We Measured | ρ | p-value | Significant? |
|---|---|---|---|
| Smoke days per year | 0.480 | 0.015 | **Yes** |
| Peak PM2.5 per year | 0.648 | 0.043 | **Yes** |
| Longest episode per year | 0.483 | 0.015 | **Yes** |
| Fire season mean PM2.5 | 0.202 | 0.330 | No |

**Interpretation:** The number of smoke days per year, the worst single day, and the length of the longest episode are all increasing over time. But the *average* fire season PM2.5 is not significantly changing — most days are still clean; it's the worst days getting worse.

---

### 6.5 Weather Profile of Smoke Days vs. Normal Days

| Variable | Smoke Days | Normal Days | Difference | p-value |
|---|---|---|---|---|
| Temperature (°C) | 19.5 | 10.4 | **+9.1°C hotter** | < 0.001 |
| Precipitation (mm) | 0.62 | 5.16 | **−4.5 mm drier** | < 0.001 |
| Wind speed (km/h) | 8.15 | 9.61 | **−1.5 km/h calmer** | 0.029 |
| Humidity (%) | 75.7 | 80.5 | **−4.8% drier** | 0.015 |
| Pressure (hPa) | 1014.9 | 1016.6 | −1.7 hPa | 0.007 |

**What this tells us:** Smoke days have a very distinct weather signature — hot, dry, calm. This is characteristic of a **stagnant high-pressure system** where:
- Air sinks from above, warming as it compresses
- No winds to disperse pollutants horizontally
- No rain to wash particles out of the air
- Smoke from distant fires gets trapped and accumulates

---

## 7. Lag Analysis — Notebook 02

**Goal:** Find the typical delay (lag) between fire activity spiking and PM2.5 rising in Vancouver.

### 7.1 Cross-Correlation Analysis

**What is cross-correlation?**

Imagine you plot fire activity over time and PM2.5 over time. Cross-correlation asks: "If I shift the fire curve forward by 1 day, 2 days, 3 days, etc., at which shift do the two curves look most similar?"

The shift where similarity is highest = the **lag**.

**How we computed it:**
1. Subset to fire season (June–September) only — fires are irrelevant in winter
2. For each lag from −7 to +14 days, compute Pearson correlation between fire activity and PM2.5
3. Find the lag at which correlation is maximum

**Results:**

| Distance Band | Lag at Peak Correlation | Correlation at Peak |
|---|---|---|
| Close (< 200 km) | **0 days** | r = 0.199 |
| Medium (200–500 km) | **0 days** | r = 0.261 |
| Far (500–1,000 km) | **0 days** | r = 0.195 |

**What does lag = 0 mean?**

It means fire activity and PM2.5 peak on the **same calendar day**. This could mean:
1. Smoke transport from BC fires reaches Vancouver in under 24 hours (plausible for fires within 200 km)
2. Both fire activity and smoke are driven by the same weather conditions (hot, dry days cause more fires AND trap smoke)

**Important caveat:** Because our data is daily, we cannot detect any lag shorter than 24 hours. A 6-hour transport time would still appear as lag = 0.

---

### 7.2 The September 2020 Case Study — Why the Worst Event Is a Special Story

The worst smoke event (163.5 µg/m³, September 2020) was **not from BC fires at all**.

In September 2020, British Columbia had a quieter-than-average fire season. But Oregon and Washington state (USA) had catastrophic wildfires. A southerly/westerly wind pattern pushed US smoke northward into Vancouver.

This is critical because:
- Our fire data only covers BC (we missed the actual source)
- It shows that **cross-border smoke transport is a major vulnerability** in our model
- It explains why fire features sometimes don't predict smoke well

**Wind evidence:**
- On smoke days, the wind's eastward (U) component averaged +1.22 m/s (blowing from the south/west, carrying US smoke northward)
- On normal days, it averaged −0.17 m/s (typical Pacific flow)

---

### 7.3 Lagged Regression Decomposition

We ran an ordinary least squares (OLS) regression to understand how much each feature group contributes to explaining PM2.5 variance. We measured **R²** (coefficient of determination — the fraction of variance explained).

| Feature Set | R² | What Was Added |
|---|---|---|
| PM2.5 lags only (lag 1, 2, 3, 7) | 0.712 | Just yesterday's values |
| + Fire lags | 0.724 | Added fire count/FRP lags |
| + All features (weather, calendar) | 0.738 | Full feature set |

**What R² = 0.712 means:** Yesterday's PM2.5 alone explains 71.2% of the variance in today's PM2.5. That is extremely high. It tells us PM2.5 is very "sticky" — if today is smoky, tomorrow will probably also be smoky.

**Adding fire lags only adds 1.2% more explanation.** This is the key finding: fires barely matter beyond what yesterday's PM2.5 already tells us.

---

## 8. Predictive Modeling — Notebook 03

### 8.1 The Prediction Task

- **Target:** `pm25_target` = tomorrow's PM2.5 (we shift the PM2.5 column back by 1 day)
- **Features:** 34 columns selected from the 90-column dataset
- **Sample size:** 9,125 usable rows (after dropping the one NaN target at the end)

**The 34 features break down as:**

| Feature Group | Count | Examples |
|---|---|---|
| PM2.5 lags | 4 | pm25_lag1, pm25_lag2, pm25_lag3, pm25_lag7 |
| Fire same-day | 9 | fire_count_close, frp_sum_medium, fire_count_total |
| Fire lagged | 8 | fire_count_total_lag1, frp_sum_close_lag2 |
| Weather | 10 | temperature_2m_mean, wind_u_component, precipitation_sum |
| Calendar | 3 | month, day_of_year, is_fire_season |

---

### 8.2 Time Series Cross-Validation

**Why can't we use regular cross-validation (random splits)?**

In regular cross-validation, you randomly shuffle rows into train/test folds. For time series, this is **data leakage** — the model might train on rows from 2023 and test on rows from 2010. In reality, you never know the future when making predictions.

**TimeSeriesSplit (expanding window):**

We used 5 folds, each with an expanding training set:

```
Fold 1: Train [2000-2003] → Test [2003-2007]
Fold 2: Train [2000-2007] → Test [2007-2011]
Fold 3: Train [2000-2011] → Test [2011-2015]
Fold 4: Train [2000-2015] → Test [2015-2019]
Fold 5: Train [2000-2019] → Test [2019-2024]
```

Each test set is ~1,520 days (about 4 years). Results are averaged across all 5 folds.

**Why "expanding window"?** In a real-world deployment, you always have more historical data available as time passes. This mirrors how the model would actually be used.

---

### 8.3 Two Baseline Models

Before any ML, we define "dumb" baselines that ML must beat to be useful:

**Persistence Baseline:**
```
prediction = today's actual PM2.5
(i.e., assume tomorrow = today)
```

**Rolling Mean Baseline:**
```
prediction = average of PM2.5 from 1, 2, 3, and 7 days ago
```

These are not "trained" — they have zero parameters. Any model that cannot beat them is not learning anything useful.

---

### 8.4 Machine Learning Models

**Linear Models (assume straight-line relationships):**

| Model | Key Idea | What It Does Differently |
|---|---|---|
| Linear Regression | Fit a line through the data | No constraints on coefficients |
| Ridge Regression | Same, but penalises large coefficients | L2 penalty: shrinks all weights toward 0 |
| Lasso Regression | Same, but can zero out features | L1 penalty: some weights become exactly 0 (automatic feature selection) |

**Tree-Based Models (can capture non-linear relationships):**

| Model | Key Idea | Hyperparameters |
|---|---|---|
| Random Forest | Average of many decision trees | n_estimators=100 or 200, max_depth=15 or 20 |
| XGBoost | Build trees iteratively, each correcting errors of previous | n_estimators=200 or 300, learning_rate=0.1 or 0.05 |
| LightGBM | Like XGBoost but faster with histograms | Same hyperparameter range |

**Decision Tree (simplified explanation):**
A decision tree works by asking a series of yes/no questions: "Is wind speed > 5 m/s? → yes → is there rain? → no → predict PM2.5 = 8.2". A Random Forest grows 100–200 such trees with slight variations and averages their predictions.

**Hyperparameters** are settings you choose before training. For example:
- `n_estimators` = how many trees to grow (more = slower but potentially better)
- `learning_rate` = how much each new tree corrects the previous ones (lower = more conservative, often better)
- `max_depth` = how many questions each tree can ask (deeper = can learn more complex patterns but risks memorising the training data)

---

### 8.5 Evaluation Metrics

**MAE — Mean Absolute Error:**
```
MAE = average of |prediction - actual|
```
If MAE = 1.69, that means on average predictions are off by 1.69 µg/m³. Lower is better. This is the most interpretable metric.

**RMSE — Root Mean Squared Error:**
```
RMSE = sqrt(average of (prediction - actual)²)
```
Penalises large errors more than MAE does. A model that is usually good but occasionally very wrong will have high RMSE. Lower is better.

**R² — Coefficient of Determination:**
```
R² = 1 - (variance unexplained by model / total variance)
```
R² = 1.0 is a perfect model. R² = 0.0 means the model is no better than predicting the mean. R² < 0 means the model is *worse* than predicting the mean. Higher is better.

**MAE on Smoke Days:**
Because smoke days are rare (0.5% of data), we separately compute MAE only on the 46 smoke days. This shows how well models handle the extreme events that matter most for public health.

---

### 8.6 Results

**Full cross-validation results (5-fold average):**

| Model | MAE | RMSE | R² | Smoke Day MAE |
|---|---|---|---|---|
| **Persistence (baseline)** | **1.694** | **3.166** | **0.600** | 22.11 |
| Random Forest (200 trees) | 1.793 | 4.878 | 0.153 | 43.50 |
| LightGBM (Tuned) | 1.845 | 4.898 | 0.146 | 43.89 |
| XGBoost (Tuned) | 1.848 | 5.123 | 0.065 | 45.46 |
| Lasso (α=0.1) | 1.858 | 4.340 | 0.329 | 34.63 |
| Ridge (α=10) | 2.003 | 4.602 | 0.246 | 32.84 |
| Linear Regression | 2.004 | 4.602 | 0.246 | 32.85 |
| Rolling Mean (baseline) | 2.375 | 4.786 | 0.087 | 36.06 |

**The shocking result:** Persistence wins on MAE, RMSE, and R². Its R² = 0.600 is *four times* higher than the best ML model (0.153).

**Why does this happen?** Because PM2.5 is strongly autocorrelated (lag-1 correlation r ≈ 0.82). "Tomorrow = today" is already a very good bet. ML models learn noisy relationships with fire data and weather that don't add value over this simple rule.

**Why do ML models have such low R²?** R² is measured on the test sets. The models are fitting the training data somewhat, but not generalising well to future time periods — the distribution of PM2.5 shifts over time.

---

### 8.7 Feature Importance

From Random Forest (200 trees):

| Rank | Feature | Importance (%) | Group |
|---|---|---|---|
| 1 | **pm25_lag1** | ~40% | PM2.5 lags |
| 2 | wind_v_component | ~5% | Weather |
| 3 | pm25_lag3 | ~4% | PM2.5 lags |
| 4 | precipitation_sum | ~4% | Weather |
| 5 | fire_count_medium | ~4% | Fire |

Yesterday's PM2.5 alone carries 40% of all predictive power. All other 33 features together account for the remaining 60%.

---

### 8.8 Feature Ablation Study

An ablation study systematically removes each feature group to measure its contribution.

| Features Removed | MAE | R² | Change in MAE |
|---|---|---|---|
| None (all features) | 1.793 | 0.153 | — |
| Remove PM2.5 lags | 1.930 | 0.138 | +0.137 (worse) |
| **Remove fire features** | **1.750** | **0.203** | **−0.043 (better!)** |
| Remove weather | 2.218 | 0.065 | +0.425 (much worse) |
| Remove calendar | 1.802 | 0.156 | +0.009 (slightly worse) |

**The key insight:** Removing fire features *improves* the model. Fire data is adding noise, not signal. This is counterintuitive but makes sense because:
- Satellite fire counts don't capture atmospheric transport
- The worst smoke events (like September 2020) came from US fires not in our dataset
- Weather conditions already capture most of the information about when smoke stagnation occurs

---

### 8.9 Holdout Validation

To check if our models would work on genuinely unseen data, we did a final test:

- **Training:** All data from 2000–2020 (7,633 samples)
- **Testing:** All data from 2021–2024 (1,492 samples)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence | 1.759 | 3.884 | 0.384 |
| Random Forest (200) | 2.088 | 6.201 | **−0.125** |
| LightGBM (Tuned) | 2.178 | 6.341 | −0.100 |
| XGBoost (Tuned) | 2.211 | 6.908 | −0.554 |

**Negative R²** means the models perform *worse* than simply predicting the average PM2.5 every day. Why?
- The PM2.5 distribution shifted between 2000–2020 and 2021–2024
- The recent period (2021–2024) had fewer extreme events (2024 was very clean)
- Models trained on one era don't generalise perfectly to another — this is called **non-stationarity**

Persistence remains robust because it adapts automatically to current conditions.

---

## 9. Key Results and Findings

### Finding 1 — Persistence beats all ML (RQ1)
The simplest possible forecast outperforms 10 trained models. This is not a failure — it is an honest result. PM2.5 has strong day-to-day memory (r = 0.82), and no available features add enough signal to overcome this.

### Finding 2 — Fire features add noise, not signal (RQ1)
Removing satellite fire counts improves model performance. At daily resolution, fire counts don't translate cleanly into PM2.5 because atmospheric transport is too complex. Weather conditions that cause both fire and stagnation are already captured in the weather features.

### Finding 3 — Smoke arrives same day (RQ2)
Cross-correlation peaks at lag = 0 across all distance bands. With daily data, we cannot resolve finer time scales. Per-year analysis suggests 2–4 day lags are plausible but the signal is noisy.

### Finding 4 — The worst event came from the USA (RQ2, RQ3)
September 2020's record 163.5 µg/m³ event was driven by Oregon/Washington fires, not BC fires. Our model (which only uses BC fire data) completely missed this. Cross-border transport is a major gap.

### Finding 5 — Smoke seasons are significantly worsening (RQ3)
Spearman correlation with time (2000–2024):
- Smoke day frequency: ρ = 0.48, **p = 0.015**
- Peak severity: ρ = 0.65, **p = 0.043**
- Episode duration: ρ = 0.48, **p = 0.015**

All three are statistically significant at the standard threshold (α = 0.05).

### Finding 6 — Hot, dry, calm days = smoke risk (RQ3)
Smoke days are on average 9°C hotter, receive 4.5mm less rain, and have 1.5 km/h lower wind speed than non-smoke days. A simple "heat + dryness + calm" rule would capture most smoke events.

---

## 10. The Streamlit Dashboard

The dashboard (`app/streamlit_app.py`) has 7 tabs providing an interactive interface for all project results.

| Tab | What You Can Do |
|---|---|
| Overview | View 25-year PM2.5 time series, filter by year, see distribution |
| Smoke Season Trends | Annual smoke days chart, episode table, weather comparison |
| Smoke Arrival Lag | Cross-correlation plots, case studies (2017, 2018, 2020), wind analysis |
| PM2.5 Forecasting | Model comparison table, error bars, select any model to see predictions |
| Model Validation | Holdout test results (2021–2024), smoke event detection accuracy |
| Health & Planning | Calendar heatmap, extreme events table, AQI mapping |
| Data Explorer | Interactive table — filter, search, and download the full dataset |

**To run it locally:**
```bash
streamlit run app/streamlit_app.py
```

---

## 11. Statistics Concepts Explained

This section explains every statistical/ML concept used in the project in plain English.

### What is a p-value?

When we claim "smoke seasons are getting worse," the p-value tells us: "If there were actually NO trend, how likely would we be to see data at least this extreme just by chance?"

- **p = 0.015** means there is only a 1.5% chance of seeing this upward trend if there were truly no trend.
- Convention: if p < 0.05, we call the result **statistically significant**.
- If p > 0.05, we say we "fail to reject the null hypothesis" — the trend might just be random.

**Common misunderstanding:** p < 0.05 does NOT mean "we are 95% sure the trend is real." It means "the observed trend is unlikely under the null hypothesis."

---

### What is autocorrelation?

PM2.5 on Tuesday is correlated with PM2.5 on Monday. This is autocorrelation (correlation of a time series with a lagged version of itself).

Lag-1 autocorrelation r ≈ 0.82 means: if today's PM2.5 = 10, tomorrow will most likely be close to 10. This is why persistence is hard to beat.

---

### What is overfitting?

Overfitting happens when a model learns the training data *too well* — including the noise — and therefore performs poorly on new data.

Example: If the model memorises that August 13, 2017 had PM2.5 = 45.3, it will fail on any August 13 in other years when PM2.5 might be 4.

Evidence in our project: RF, XGBoost, and LightGBM achieve good CV scores (MAE ~1.8) but deteriorate on the holdout set (MAE 2.1–2.2) with negative R².

---

### What does R² = 0.600 vs. 0.153 mean intuitively?

Think of it this way: if PM2.5 varies by (say) 8 µg/m³ on average, then:
- R² = 0.600 means the model explains 60% of that variation. Only 40% is "unexplained."
- R² = 0.153 means the model explains only 15.3% — 84.7% of variation is still a mystery to the model.

---

### What is the Mann-Whitney test?

A non-parametric test for comparing two groups (e.g., smoke days vs. normal days) that does NOT assume the data is normally distributed. It ranks all values together and checks whether one group tends to have higher ranks than the other.

We used it to test whether temperature, humidity, etc. are significantly different between smoke and non-smoke days.

---

### What is Haversine distance?

A formula to calculate the great-circle distance between two points on Earth's surface given their latitude/longitude. We used it to compute the distance between each fire detection and Vancouver's centre (49.28°N, 123.12°W). Unlike simple Euclidean distance, it accounts for Earth's curvature.

---

### What is StandardScaler?

Before training linear and tree models, we standardise features so they all have mean = 0 and standard deviation = 1. This is called **feature scaling**.

```
scaled_value = (value - mean) / std_dev
```

Why? Because features on different scales (e.g., temperature in degrees vs. FRP in megawatts) can cause linear models to disproportionately weight large-scale features. Trees don't strictly need this, but it doesn't hurt.

Important: We fit the scaler *only on training data* and apply it to test data — to avoid data leakage.

---

### What is data leakage?

Data leakage happens when information from the future (or from the test set) influences the model during training. This artificially inflates performance metrics and produces models that fail in real-world deployment.

In our time-series context, we prevent leakage by:
1. Never shuffling rows (maintaining temporal order)
2. Using `TimeSeriesSplit` — test always comes after training
3. Fitting scalers only on training data
4. Target = *next-day* PM2.5, not same-day

---

## 12. Professor Questions and Answers

These are likely questions a professor could ask, with the answers you should give.

---

**Q: Why does persistence beat all ML models?**

A: Because PM2.5 has very strong lag-1 autocorrelation (r ≈ 0.82). Air quality changes gradually most of the time — if today is smoky, tomorrow will likely still be smoky. Persistence capitalises on this pattern directly. ML models try to learn complex relationships with fire and weather data, but those features don't add enough signal beyond what yesterday's PM2.5 already provides. Our feature ablation study confirms this: even removing all fire features improves the model.

---

**Q: How do you know the trends in RQ3 are statistically significant?**

A: We used the Spearman rank correlation test with n = 24 yearly observations. For smoke day frequency, we found ρ = 0.480, p = 0.015. Since p < 0.05, we reject the null hypothesis that there is no trend. The Spearman test is appropriate because: (1) yearly smoke counts are not normally distributed, and (2) we are interested in monotonic trends, not linear ones. We also checked smoke episode severity and duration, both p < 0.05.

---

**Q: Why is your sample size only 24 for the trend test?**

A: We have 25 years of data (2000–2024), giving us 24 or 25 yearly observations depending on whether 2024 is included. While n = 24 is small for detecting subtle trends, the p-values we found (0.015–0.043) suggest the trends are strong enough to detect even with this limited sample.

---

**Q: Why did you use Spearman and not Pearson correlation for trends?**

A: Spearman is more appropriate for trend analysis of annual counts because:
1. Annual smoke days are count data (non-negative integers), not normally distributed
2. Extreme years (like 2017 or 2020) would heavily influence Pearson
3. Spearman tests for *monotonic* trend (consistently going up/down), which is the ecologically relevant question

---

**Q: The worst smoke event (2020) came from US fires. Doesn't that invalidate your findings?**

A: It complicates them but does not invalidate them. The September 2020 event is documented and we discuss it explicitly as a case study. It illustrates a key limitation: our fire data only covers BC. Cross-border smoke from the US is a real and important phenomenon that our model cannot fully capture. This is explicitly listed as a limitation and a direction for future work (adding VIIRS data for the Pacific Northwest USA). It also strengthens the argument that wind direction and weather features may be more important than local fire counts for predicting smoke events.

---

**Q: How do you prevent overfitting in your time-series cross-validation?**

A: We use `TimeSeriesSplit` (5 folds, expanding window) which:
1. Never uses future data to train — test always comes strictly after training
2. Tests on ~1,520 days per fold (4 years), a large enough test set to detect overfitting
3. We also ran a separate holdout validation (train 2000–2020, test 2021–2024) which showed the models' R² goes negative, confirming that some overfitting occurred despite CV precautions

---

**Q: Why do the ML models have negative R² on the holdout?**

A: Negative R² means the model is worse than predicting the mean. This happens because of **non-stationarity**: the statistical properties of PM2.5 in 2021–2024 differ from 2000–2020. Specifically:
- 2021–2024 had fewer extreme smoke events (2024 was clean)
- Some patterns the model learned from the training period may not generalise
- The models overfitted to specific event profiles in the training data

Persistence adapts automatically because it uses only the most recent value, regardless of historical patterns.

---

**Q: Why does removing fire features improve the model?**

A: This is counterintuitive but has a clear explanation. Satellite fire counts are a **noisy proxy** for the actual physical process (smoke transport). A fire 200 km away might produce heavy smoke in Vancouver if wind is southeasterly, or no smoke at all if wind is westerly. The fire count feature doesn't tell the model about wind. Since weather features (wind components, pressure) already capture whether smoke-trapping conditions exist, fire counts add redundant and noisy information. The model is better off without the noise.

---

**Q: What is the practical value of this work if ML can't beat persistence?**

A: The project has practical value in several ways:
1. **Trend detection (RQ3)** — the Spearman trend analysis provides actionable public health information: smoke seasons are getting significantly worse
2. **Seasonal risk calendar** — we identify August–September as the highest risk months, useful for preventive health campaigns
3. **Episode characterisation** — identifying weather signatures of smoke events helps develop early warning systems
4. **Negative result value** — documenting that daily fire data does not improve forecasting saves future researchers from repeating this approach; hourly resolution or transport models (like HYSPLIT) would be the appropriate next step

---

**Q: What would you do differently to improve the model?**

A: Four main improvements:
1. **Hourly resolution data** — PM2.5 lags less than 24 hours are invisible at daily resolution; hourly data would reveal true transport times
2. **Add US fire data** — include NASA FIRMS data for Oregon, Washington, and California to capture cross-border events
3. **Add atmospheric transport model output** — tools like HYSPLIT (NOAA's trajectory model) can predict where smoke will travel based on wind fields; this would be a much stronger predictor than raw fire counts
4. **Sequence models** — LSTM (Long Short-Term Memory) neural networks are designed for time series and might capture patterns that tabular ML misses

---

**Q: Is your data missing any days?**

A: No. The final merged dataset has 9,133 consecutive days with zero missing days. However, fire features are 0 on many days (meaning no fires were detected within 1,000 km) and PM2.5 readings come from multiple stations that occasionally go offline — we handled this by averaging across all available stations on each day.

---

**Q: What is FRP and why is it better than just counting fires?**

A: Fire Radiative Power (FRP, in megawatts) measures how much energy a fire is releasing. A large, intense fire has high FRP; a small smouldering fire has low FRP. A count of 10 small fires might produce less smoke than 1 large fire. Using both fire count and FRP sum/mean gives a richer picture of total fire intensity. In our results, FRP and count had similar predictive power (both peaked at lag 0 with similar correlations).

---

**Q: How did you choose the "smoke event" threshold of 25 µg/m³?**

A: 25 µg/m³ corresponds to the BC Air Quality Index "Moderate" threshold and is a widely used health-based threshold in the North American air quality management literature. It is also consistent with Health Canada's air quality guidelines. We could have used a different threshold (e.g., 12 µg/m³ for "Good" to "Moderate" transition) but 25 provides a cleaner signal for fire smoke episodes while filtering out everyday pollution.

---

**Q: Can this model be used operationally for public health forecasting?**

A: For day-ahead PM2.5 value forecasting, persistence is currently the best approach from this analysis. Operationally, Environment Canada's air quality forecast uses numerical weather prediction models coupled with chemical transport models (AURAMS, FireWork) — these are far more sophisticated than our approach and explicitly model atmospheric chemistry. Our contribution is in the trend analysis and seasonal profiling, which provide complementary public health context. A practical application could be a simple early-warning rule: "if today is a hot (>25°C), dry (<1mm rain), calm (<8 km/h), high-fire-activity day, issue a smoke advisory for tomorrow."

---

*End of Explainer Document*

**Good luck with your presentation! The key messages to remember:**
1. Persistence wins — and that's because PM2.5 is highly autocorrelated
2. Smoke seasons are getting statistically significantly worse (p < 0.05)
3. The worst event was from US fires — cross-border transport is critical
4. Hot + dry + calm = smoke risk
5. Daily fire counts add noise, not signal — weather matters more
