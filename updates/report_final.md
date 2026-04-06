# When the Sky Turns Orange: Predicting and Profiling Wildfire Smoke Impacts on Vancouver's Air Quality (2000–2025)

**Author:** Aarish Kapila 
**Course:** CMPT 733 — Big Data Lab II, Simon Fraser University  
**Repository:** [github.com/aarishk/vancouver-wildfire-smoke-predictor](https://github.com/aarishk/vancouver-wildfire-smoke-predictor)

---

## 1. Motivation and Background

Wildfire smoke is one of the most acute and rapidly worsening air quality threats facing Pacific Northwest cities. Vancouver sits at the intersection of two fire-prone geographies: the BC Interior plateau to its northeast, and the western United States to its south. In recent summers, residents have experienced days — sometimes weeks — of hazardous air quality, with PM2.5 (fine particulate matter with diameter ≤ 2.5 µm) reaching levels associated with significant cardiovascular and respiratory harm.

The public health stakes are substantial. Short-term PM2.5 exposures above 35 µg/m³ are linked to emergency department visits for asthma and cardiac events. On September 13, 2020, Vancouver recorded a daily mean of 163.5 µg/m³ — nearly seven times the "poor air quality" threshold — as fires in Oregon and Washington sent smoke northward across the border. Yet air quality forecast tools available to the public typically provide only coarse, qualitative guidance.

Prior work has approached this problem from several directions. Numerical weather prediction (NWP) systems such as Environment Canada's FireWork model couple fire emission inventories with chemical transport models to forecast smoke concentrations. While physically principled, these systems require significant computational infrastructure and are difficult to interpret. Statistical and machine learning approaches, including LSTM networks applied to hourly PM2.5 sequences (Navares & Aznarte, 2021) and random forest models trained on meteorological reanalysis data (Chen et al., 2019), have shown promise for short-range forecasting but consistently struggle with rare, extreme smoke events due to severe class imbalance.

This project takes a data-driven approach using 25 years of observational records — air quality measurements, satellite fire detections, and meteorological data — to answer three questions that span forecasting, physical process understanding, and long-term trend detection.

---

## 2. Problem Statement

The project is organised around three research questions of increasing timescale:

**RQ1 (Forecasting):** Can machine learning models predict next-day PM2.5 more accurately than simple persistence and rolling-mean baselines? This question is challenging because PM2.5 is dominated by short-term autocorrelation, rare extreme events constitute less than 0.5% of observations, and the relationship between fire activity and urban air quality is mediated by complex atmospheric transport dynamics that are difficult to represent with daily, station-level features.

**RQ2 (Smoke Arrival Lag):** What is the typical delay between wildfire activity in British Columbia and elevated PM2.5 in Vancouver? This question is challenging because: (i) daily temporal resolution may be too coarse to resolve sub-day transport; (ii) correlations between fire activity and PM2.5 are confounded by shared meteorological drivers; and (iii) the most extreme events originate from cross-border US fires that are absent from BC-only satellite datasets.

**RQ3 (Seasonal Trends):** How has Vancouver's wildfire smoke season changed from 2000 to 2024? This requires careful choice of statistical tests given the non-normal distribution of annual smoke metrics and a sample size of only 24 yearly observations.

---

## 3. Data Science Pipeline

The pipeline consists of five stages: data collection, preprocessing and integration, feature engineering, analysis (EDA, lag analysis, modeling), and deployment.

**Stage 1 — Data Collection.** Three independent data streams are downloaded via dedicated scripts. `download_bc_air_quality.py` retrieves annual PM2.5 summary CSV files from the BC Ministry of Environment FTP server (`ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/`), handles multiple historical column-name formats across years, and filters to 25 Metro Vancouver and Lower Fraser Valley monitoring stations. `download_historical_fires.py` retrieves NASA FIRMS active fire detections for the BC bounding box (lat 48–60°N, lon −130 to −114°W), using MODIS (2000–2011) and VIIRS (2012–2024) to ensure the best available sensor is used throughout. For each fire detection, Haversine distance to Vancouver (49.28°N, 123.12°W) is computed. `download_weather.py` queries the Open-Meteo Historical API in 365-day chunks for seven hourly variables at the Vancouver grid point.

**Stage 2 — Preprocessing and Integration.** `build_dataset.py` serves as the integration layer. Air quality hourly readings are aggregated to daily city-level statistics (mean, median, max, min, standard deviation, station count). Fire detections are aggregated by calendar date into three distance bands — Close (0–200 km), Medium (200–500 km), Far (500–1,000 km) — yielding daily fire count and fire radiative power (FRP) sum per band. Weather is aggregated from hourly to daily (mean/max/min for temperature; sum for precipitation; mean for wind and pressure). All three streams are merged on the date key using a left join on the air quality base table, with fire feature NaNs filled to zero (no fires detected = no fire activity). The resulting dataset spans 2000-01-01 to 2025-01-01 with 9,133 rows and zero missing days.

**Stage 3 — Feature Engineering.** Thirty-four features are engineered from the 90-column merged dataset. PM2.5 autoregressive features (lags 1, 2, 3, 7 days) capture the strong temporal persistence of air quality. Fire activity lag features (lags 1, 2, 3 days for total count/FRP and close/medium bands) are included to model delayed smoke transport. Wind components are decomposed into orthogonal U (eastward) and V (northward) vectors using standard meteorological conventions, enabling the model to encode directional information without circular discontinuities. Calendar features (`month`, `day_of_year`, `is_fire_season`) encode seasonality.

**Stage 4 — Analysis.** Three Jupyter notebooks address the research questions sequentially. `01_eda.ipynb` performs exploratory analysis and answers RQ3 via Spearman trend tests on annual smoke metrics. `02_lag_analysis.ipynb` answers RQ2 via cross-correlation analysis and lagged linear regression decomposition. `03_modeling.ipynb` answers RQ1 by training and evaluating ten models with proper time-series cross-validation.

**Stage 5 — Deployment.** Results are exposed through an interactive Streamlit dashboard (`app/streamlit_app.py`) with seven thematic tabs and `@st.cache_data` decorators on expensive computations to ensure responsive re-rendering.

---

## 4. Methodology

**Trend Analysis (RQ3).** Annual smoke metrics (smoke days per year, peak PM2.5, longest episode duration) were computed by identifying contiguous runs of days exceeding PM2.5 > 25 µg/m³ — the BC Air Quality Index "Moderate" threshold. Spearman rank correlation (ρ) was chosen over Pearson for two reasons: annual smoke counts are non-normally distributed (many zero or near-zero years), and we are testing for monotonic rather than strictly linear trends. With n = 24 annual observations (2000–2023), a p-value threshold of α = 0.05 was applied. Mann-Whitney U tests compared weather variables between smoke and non-smoke days without assuming normality.

**Lag Analysis (RQ2).** Cross-correlation between deseasonalised fire activity (FRP sum, fire count) and PM2.5 was computed at integer lags from −7 to +14 days, restricted to fire season (June–September) to avoid dilution by winter days with no fire activity. Three distance bands were analysed independently. A lagged OLS regression decomposition was used to quantify the incremental R² contribution of PM2.5 lags, fire lags, and weather features. Wind component analysis (Mann-Whitney U on U and V components) tested whether wind direction differed systematically between smoke and non-smoke fire-season days.

**Predictive Modeling (RQ1).** The target variable is `pm25_target` = `pm25.shift(-1)`, the next calendar day's PM2.5. The feature matrix contains 34 columns. Two non-learnable baselines are evaluated: Persistence (`ŷ = pm25_today`) and Rolling Mean (`ŷ = mean(lag1, lag2, lag3, lag7)`). Five linear models are trained: OLS, Ridge (α ∈ {1.0, 10.0}), and Lasso (α ∈ {0.1, 1.0}). Four tree-based ensemble models are trained: Random Forest with `n_estimators` ∈ {100, 200}, XGBoost with default and tuned hyperparameters, and LightGBM with default and tuned hyperparameters. Tuned variants use lower learning rates (0.05), increased depth (8), and L1+L2 regularisation to reduce overfitting.

Cross-validation uses `sklearn.model_selection.TimeSeriesSplit` with five folds and an expanding training window — each test fold covers approximately four calendar years, strictly after its training period. `StandardScaler` is fit only on training data and applied to test data, preventing any information from the future from influencing feature normalisation. Evaluation metrics are MAE, RMSE, R², and smoke-day-specific MAE (computed only on the 46 days where actual PM2.5 > 25 µg/m³). A separate holdout validation trains on 2000–2020 and tests on 2021–2024 to assess temporal generalisation.

A feature ablation study systematically removes each feature group (PM2.5 lags, fire features, weather, calendar) to measure its marginal contribution independently of importance scores, which can be misleading when features are correlated.

---

## 5. Evaluation

**RQ3 Results.** All three primary smoke season metrics show statistically significant upward trends across 2000–2024: smoke day frequency (ρ = 0.480, p = 0.015), peak episode PM2.5 (ρ = 0.648, p = 0.043), and longest episode duration (ρ = 0.483, p = 0.015). Fire season mean PM2.5 is not significant (ρ = 0.202, p = 0.33), indicating that average conditions are unchanged while extremes are worsening — a pattern consistent with the intensification of tail events under climate change. The weather comparison provides physical corroboration: smoke days are 9.2°C hotter, 4.5 mm drier, and 1.5 km/h calmer than non-smoke days (all p < 0.03), consistent with the stagnant high-pressure synoptic pattern that simultaneously drives fire ignition and traps smoke near the surface.

**RQ2 Results.** Cross-correlation peaks at lag 0 across all three distance bands (r = 0.199 to 0.261). With daily resolution, this indicates smoke transport occurs within a single calendar day, though sub-day lags remain unresolvable. The lagged regression decomposition reveals that PM2.5 autoregressive features alone explain R² = 0.712 of fire-season PM2.5 variance; adding fire lags raises this only to 0.724 (+1.2%); adding all weather features reaches 0.738 (+2.6%). Fire features contribute marginally, and their effect is largely subsumed by weather. Wind component analysis shows that smoke days have a more positive U-component (eastward) than normal days (+1.22 vs −0.17 m/s), consistent with southerly/westerly flow bringing smoke from the US — as observed in the September 2020 record event.

**RQ1 Results.** The persistence baseline achieves MAE = 1.694 µg/m³ and R² = 0.600, outperforming every trained model. The best ML model, Random Forest (200 trees), achieves MAE = 1.783 and R² = 0.153. The gap in R² — 0.600 versus 0.153 — reflects the degree to which ML models overfit to training-period patterns while persistence generalises trivially. On the 2021–2024 holdout, Random Forest deteriorates to MAE = 2.088 and R² = −0.125 (worse than predicting the mean), while persistence remains robust at MAE = 1.759 and R² = 0.384. This non-stationarity is expected: the PM2.5 distribution in 2021–2024 differs from the 2000–2020 training period, and persistence adapts automatically while static models do not.

The feature ablation study provides the study's most counterintuitive finding: removing all fire features improves Random Forest performance (MAE decreases from 1.793 to 1.750; R² increases from 0.153 to 0.203). This result is robust and reproducible. It occurs because satellite fire counts, at daily resolution, carry insufficient information about atmospheric transport direction to meaningfully predict whether a given day's fires will produce smoke in Vancouver. Weather variables — particularly wind components and pressure — already encode the stagnation conditions that permit smoke accumulation, making fire counts conditionally redundant. Feature importance confirms that pm25_lag1 alone accounts for approximately 40% of Random Forest importance; wind V-component ranks second at ~5%.

---

## 6. Data Product

The data product is an interactive Streamlit dashboard (`app/streamlit_app.py`, approximately 800 lines) that makes all project findings accessible without requiring users to execute notebooks.

**Tab 1 — Overview** presents the 25-year PM2.5 time series with smoke events highlighted, a distribution histogram, and monthly seasonal patterns. A year-range slider allows temporal filtering.

**Tab 2 — Smoke Season Trends** displays the annual smoke day bar chart, a sortable smoke episode table (14 episodes across 25 years), and the weather comparison between smoke and normal days.

**Tab 3 — Smoke Arrival Lag** renders the cross-correlation figures for FRP and fire count across distance bands, an interactive episode explorer that plots PM2.5 and fire count around any selected episode, and wind direction histograms comparing smoke versus normal fire-season days.

**Tab 4 — PM2.5 Forecasting** shows the model comparison table and error bar chart, a selectbox allowing the user to choose any model and view scatter and time-series prediction plots, a feature importance chart with a slider controlling the number of top features displayed and colour-coding by feature group, and the ablation study bar chart.

**Tab 5 — Model Validation** presents the holdout evaluation (train 2000–2020, test 2021–2024) with a time-series overlay and smoke event detection accuracy.

**Tab 6 — Health & Planning** maps PM2.5 values to BC AQI categories, displays a calendar heatmap of smoke days by year and month, and provides an extreme events timeline.

**Tab 7 — Data Explorer** exposes the full 9,133 × 90 dataset with date filtering, column selection, and CSV download functionality.

Computationally expensive operations — model training, smoke episode detection — are wrapped in `@st.cache_data` to ensure the dashboard remains responsive after the first load. The dashboard runs locally with `streamlit run app/streamlit_app.py` following installation of the dependencies in `requirements.txt`.

---

## 7. Lessons Learnt

**Negative results are informative.** The finding that persistence outperforms all ML models is not a failure — it is a precise characterisation of the problem. It tells us that at daily resolution, PM2.5 forecasting is dominated by autocorrelation (r = 0.82 at lag 1), and that the features available to us do not contain sufficient additional information to overcome this. This guides future work toward hourly resolution data and physics-based transport features rather than incremental model tuning.

**Feature ablation should precede feature importance.** Importance scores (e.g., SHAP, Gini impurity) measure a feature's contribution *given all other features are present*. Ablation measures marginal contribution when a feature group is absent entirely. The fire features appeared in the top-10 importance rankings yet degraded performance when included — a contradiction only resolved by ablation.

**Cross-validation design is as important as model selection.** Standard k-fold cross-validation applied to a time series would allow the model to "see the future" during training, producing inflated performance metrics. Every model in this study would have appeared to significantly outperform persistence under a naive random split. The expanding-window TimeSeriesSplit was non-negotiable, and the gap between cross-validated and holdout performance demonstrated that even with proper CV, non-stationarity limits generalisation.

**Data scope determines result scope.** The September 2020 event — the most extreme in 25 years — was caused by fires in Oregon and Washington. A BC-only fire dataset structurally cannot explain this event. This is not a solvable problem through better modeling; it requires expanding the data scope to include Pacific Northwest US fire activity.

---

## 8. Summary

This project built a 25-year daily observational record of Vancouver's air quality, fire activity, and weather by integrating three independent data sources into a 9,133 × 90 dataset. Three research questions were addressed using statistically appropriate methods and validated with held-out data.

RQ1 found that the persistence baseline (tomorrow = today) outperforms all ten trained machine learning models on every evaluation metric, with MAE of 1.694 µg/m³ and R² of 0.600 versus the best ML result of 1.783 and 0.153. The dominant driver is PM2.5's strong lag-1 autocorrelation (r = 0.82). A feature ablation study found that satellite fire count features actively degrade ML performance, likely because daily fire counts do not encode atmospheric transport direction. Weather features contribute meaningfully, while PM2.5 autoregressive features are the single most important group.

RQ2 found that cross-correlation between fire activity and PM2.5 peaks at lag 0 across all distance bands (r = 0.199–0.261), indicating smoke transport within the same calendar day. With daily resolution, sub-day lags are unresolvable. The worst smoke event in the 25-year record (September 2020, peak 163.5 µg/m³) originated from Oregon and Washington wildfires, not BC fires — highlighting cross-border transport as a structural blind spot in BC-only analyses.

RQ3 found statistically significant upward trends in smoke day frequency (ρ = 0.480, p = 0.015), peak episode severity (ρ = 0.648, p = 0.043), and episode duration (ρ = 0.483, p = 0.015) across 2000–2024. Average fire season PM2.5 did not trend significantly — the worsening is concentrated in extremes, not the mean. Smoke days are characterised by hot (+9.2°C), dry (−4.5 mm), and calm (−1.5 km/h) conditions consistent with stagnant high-pressure meteorology, which simultaneously promotes fire ignition and prevents smoke dispersal.

The data product is an interactive Streamlit dashboard with seven tabs exposing all findings, including a full data explorer and CSV download. Future work should prioritise hourly temporal resolution to resolve true transport lags, expansion of fire data to the Pacific Northwest US, and integration of atmospheric trajectory model (HYSPLIT) outputs as physically grounded predictors.

---

*Word count: approximately 2,200 words*

---

## References

- Chen, J., et al. (2019). "A machine learning method to estimate PM2.5 concentrations across China with remote sensing, meteorological and land use information." *Science of the Total Environment*, 636, 52–60.
- Environment and Climate Change Canada. FireWork Air Quality Forecast System. Available at: weather.gc.ca.
- NASA FIRMS. Fire Information for Resource Management System. Available at: firms.modaps.eosdis.nasa.gov.
- Navares, R., & Aznarte, J. L. (2021). "Predicting air quality with deep learning LSTM: Towards comprehensive models." *Ecological Informatics*, 67, 101509.
- BC Ministry of Environment. Air Quality Monitoring Network. Available at: envistaweb.env.gov.bc.ca.
- Open-Meteo. Historical Weather API. Available at: open-meteo.com.
