# Forest Fire Prediction BC

**Predicting and Profiling Wildfire Smoke Impacts on Vancouver's Air Quality (2000–2025)**

CMPT 733: Big Data Lab II | Simon Fraser University

---

## Overview

This project integrates daily PM2.5 air quality measurements, NASA MODIS/VIIRS satellite fire detections, and meteorological data to investigate wildfire smoke impacts on Vancouver, BC. It addresses three research questions:

1. **RQ1 — Forecasting:** Can ML models predict next-day PM2.5 more accurately than naive baselines?
2. **RQ2 — Lag Analysis:** What is the delay between BC wildfire activity and elevated PM2.5 in Vancouver?
3. **RQ3 — Trend Profiling:** How has Vancouver's wildfire smoke season changed from 2000 to 2025?

Key finding: The persistence baseline (tomorrow = today) outperforms all ML models (MAE = 1.69 vs. best ML MAE = 1.78), while smoke seasons show statistically significant upward trends in frequency, peak severity, and episode duration over 2000–2024.

---

## Repository Structure

```
├── app/
│   └── streamlit_app.py          # Interactive dashboard (7 tabs)
├── data/
│   ├── raw/                      # Raw downloaded data (gitignored)
│   └── processed/
│       └── merged/dataset.csv    # Final merged dataset (9,133 x 90)
├── docs/
│   ├── report.md                 # Full academic report
│   ├── detailed_walkthrough.md   # Verified end-to-end walkthrough
│   └── presentation.pptx         # Slide deck
├── figures/                      # Generated visualizations (20 PNGs)
├── notebooks/
│   ├── 01_eda.ipynb              # EDA & seasonal risk profiling (RQ3)
│   ├── 02_lag_analysis.ipynb     # Smoke arrival lag analysis (RQ2)
│   └── 03_modeling.ipynb         # Predictive modeling (RQ1)
├── scripts/
│   ├── build_dataset.py          # Dataset construction pipeline
│   ├── download_bc_air_quality.py
│   ├── download_historical_fires.py
│   ├── download_weather.py
│   └── download_all_25years.py   # Bulk 25-year download helper
└── requirements.txt
```

---

## Data Sources

| Source | Coverage | Description |
|---|---|---|
| BC Ministry of Environment FTP | 2000–2024 | Hourly PM2.5 from Metro Vancouver stations |
| NASA FIRMS MODIS | 2000–2011 | Active fire detections (Canada) |
| NASA FIRMS VIIRS | 2012–2024 | Active fire detections (Canada) |
| Open-Meteo Historical API | 2000–2025 | Daily weather for Vancouver (49.25°N, 123.12°W) |

The final merged dataset is **9,133 rows × 90 columns**, covering 2000-01-01 to 2025-01-01 with zero missing days.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Rebuild the Dataset (optional — `dataset.csv` is included)

```bash
python scripts/download_bc_air_quality.py
python scripts/download_historical_fires.py
python scripts/download_weather.py
python scripts/build_dataset.py
```

### Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Execute Notebooks

```bash
jupyter notebook
```

Open and run `notebooks/01_eda.ipynb`, `02_lag_analysis.ipynb`, `03_modeling.ipynb` in order.

---

## Key Results

### RQ1 — PM2.5 Forecasting

| Model | MAE | R² |
|---|---|---|
| **Persistence baseline** | **1.694** | **0.600** |
| Random Forest (100 trees) | 1.783 | 0.165 |
| LightGBM (Tuned) | 1.845 | 0.146 |
| XGBoost (Tuned) | 1.848 | 0.065 |

The persistence baseline beats all ML models. PM2.5 lag-1 (~40% feature importance) dominates predictions. Removing fire features actually *improves* model performance, suggesting satellite fire counts add noise at daily resolution.

### RQ2 — Smoke Arrival Lag

Cross-correlation peaks at **lag 0** across all distance bands, indicating smoke transport from fire to city occurs within the same calendar day at daily resolution. Adding fire lag features to PM2.5 autoregression improves R² by only 0.012.

### RQ3 — Seasonal Risk Profiling

Over 2000–2024: **46 smoke days** across **14 episodes**. Spearman trend tests show statistically significant upward trends in:
- Smoke day frequency (ρ = 0.480, p = 0.015)
- Peak episode severity (ρ = 0.648, p = 0.043)
- Episode duration (ρ = 0.483, p = 0.015)

The worst event (September 2020, peak PM2.5 = 163.5 µg/m³) originated from Oregon/Washington fires, not BC — demonstrating that cross-border transport is a critical factor.

---

## Streamlit Dashboard

The interactive dashboard (`app/streamlit_app.py`) provides 7 tabs:

- **Overview** — Dataset summary and PM2.5 time series
- **Smoke Season Trends** — Annual smoke day counts and trend analysis
- **Smoke Arrival Lag** — Cross-correlation and lag visualizations
- **PM2.5 Forecasting** — Model comparison with time-series CV
- **Model Validation** — Holdout test on 2021–2024
- **Health & Planning** — Smoke event calendar and weather context
- **Data Explorer** — Interactive data table

---

## Documentation

- [`docs/report.md`](docs/report.md) — Full academic report with methodology, results, and discussion
- [`docs/detailed_walkthrough.md`](docs/detailed_walkthrough.md) — Verified end-to-end walkthrough of the pipeline and all outputs

---

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list (pandas, numpy, scikit-learn, xgboost, lightgbm, streamlit, plotly, etc.)
