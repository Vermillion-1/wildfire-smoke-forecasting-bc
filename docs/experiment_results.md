# Experiment Results & Inferences
## BC Wildfire Smoke PM2.5 Forecasting — Modeling Phase

---

## The Core Problem

Standard regression models trained to minimise MAE learn to predict near the **median** of the target distribution. For a dataset where 99.5% of days are clean and 0.5% are smoke days (PM2.5 > 25 µg/m³), the median is always clean. The model ignores spikes to keep average error low — a classically rational strategy that is operationally useless for health alerts.

**Baseline:** a persistence model (tomorrow ≈ today) is extremely hard to beat overall because PM2.5 is autocorrelated. But it fails catastrophically on smoke days — it always lags the spike by one day.

---

## Evaluation Setup

- **Dataset:** 9,125 modeling rows (2000–2024), city-level Vancouver daily dataset
- **Smoke days (PM2.5 > 25):** 46 total — 0.50% of data
- **CV method:** expanding-window, 5 folds with cutoffs 2004/2008/2012/2016/2020 (no data leakage)
- **Primary metrics:**
  - **Overall MAE** — general forecast quality
  - **Smoke-day MAE** — error specifically on PM2.5 > 25 days (operationally critical)
  - **Smoke Δ** — how much better than persistence on smoke days (positive = better)

**Persistence baseline:** Overall MAE **1.668** | Smoke-day MAE **22.353** | Normal-day MAE **1.545**

---

## Experiment 1 — Residual Framing

**Question:** Should the model predict raw PM2.5 tomorrow, or the *change* (pm25_diff = tomorrow − today)?

| Model | Smoke-day MAE | Smoke Δ |
|---|---|---|
| Direct Q=0.90 prediction | 39.650 | −17.3 |
| Residual Q=0.90 (diff + today) | 21.391 | +0.962 |

**Finding:** Predicting the change rather than the raw value is essential. The direct model ignores autocorrelation and over-predicts heavily on normal days while still under-predicting on spikes. The residual framing bakes in "tomorrow ≈ today as default" and only asks the model to predict the deviation. This alone is a more important architectural choice than any loss function.

---

## Experiment 2 — Quantile Regression Sweep

**Question:** What happens if we shift the model's objective from minimising median error to minimising a higher quantile?

LightGBM's quantile (pinball) loss with alpha > 0.5 penalises under-prediction more than over-prediction, forcing the model to predict higher values on average.

| Model | Overall MAE | Smoke-day MAE | Normal-day MAE | Smoke Δ |
|---|---|---|---|---|
| Persistence | 1.668 | 22.353 | 1.545 | — |
| LightGBM MAE (q=0.50) | 1.504 | 22.181 | 1.382 | +0.172 |
| LightGBM q=0.75 | 1.852 | 21.813 | 1.735 | +0.540 |
| **LightGBM q=0.80** | **1.995** | **21.760** | **1.878** | **+0.593** |
| LightGBM q=0.90 | 2.521 | 21.391 | 2.410 | +0.962 |
| LightGBM q=0.95 | 3.061 | 21.228 | 2.954 | +1.125 |

**Finding:** The tradeoff is perfectly monotone — higher alpha always improves smoke-day MAE and always worsens overall MAE. There is no free lunch. The relationship is near-linear: each 0.05 step in alpha costs ~0.3–0.5 overall MAE and buys ~0.2–0.3 smoke-day improvement.

**q=0.80 is the best balanced choice** — it pushes the model toward risk-averse predictions (75th percentile of next-day change) without completely sacrificing daily forecast accuracy.

---

## Experiment 3 — Alternative Loss Functions

**Question:** Can other asymmetric loss functions do better than quantile regression?

| Model | Overall MAE | Smoke-day MAE | Smoke Δ |
|---|---|---|---|
| Sample-weighted 20× (smoke days) | 1.623 | 23.274 | −0.921 |
| Custom asymmetric MAE (5× under-penalty) | 1.630 | 22.271 | +0.082 |

**Finding — Sample weighting backfires.** Upweighting smoke days and rising-smoke days (pm25 > 15) teaches the model that "whenever pm25 is elevated it's going to keep rising." Most elevated days don't become smoke days — this misfires systematically, making smoke-day predictions worse than persistence. Counter-intuitive but clear.

**Finding — Custom asymmetric MAE is equivalent to quantile but less effective.** A 5× under-prediction penalty is mathematically equivalent to quantile regression at alpha ≈ 0.83. The marginal result (+0.082 vs +0.593 for q=0.80) suggests the LightGBM native quantile objective is better calibrated than a custom gradient implementation.

---

## Experiment 4 — Multi-Layer Residual Correction (Stacking)

**Question:** Can a second "correction" model trained on Layer 1's errors during high-fire days fix the under-prediction without hurting clean-day accuracy?

Architecture:
- Layer 1: standard LightGBM MAE residual model (trained on all days)
- Layer 2: second LightGBM trained only on high-fire training days (FRP > p80 threshold), target = Layer 1's residual
- Final prediction: Layer 1 output + Layer 2 correction (applied only when fire activity is elevated)

| Configuration | Overall MAE | Smoke-day MAE | Smoke Δ | Normal Δ |
|---|---|---|---|---|
| Layer 1 alone (MAE) | 1.504 | 22.181 | +0.172 | −0.163 |
| Layer 1 + Layer 2 (p90 threshold) | 1.509 | 22.111 | +0.242 | −0.157 |
| Layer 1 + Layer 2 (p80 threshold) | 1.509 | 22.055 | **+0.298** | −0.158 |
| Q=0.75 + Layer 2 | 1.853 | 21.813 | +0.540 | same as Q=0.75 alone |

**Finding — p80 threshold is better than p90.** Including the top 20% of fire days (not just top 10%) gives Layer 2 more training examples (291 vs 146 in fold 1), producing a cleaner correction signal. Smoke Δ improves from +0.242 to +0.298.

**Finding — Stacking doesn't compound with quantile regression.** When Layer 1 is Q=0.75, Layer 2 sees mixed residuals on high-fire days (positive on smoke days, negative on non-smoke fire days). The signal averages out — the correction is near-zero. Stacking only meaningfully helps the MAE model where the residuals are consistently positive on smoke days.

**Stacking's real value:** it improves smoke-day MAE (+0.298) with almost no normal-day cost (+0.005). For a use case where overall forecast accuracy matters more than smoke-day aggressiveness, this is the right model.

---

## Experiment 5 — Soft Gate Blending

**Question:** Can the XGBoost smoke classifier's probability be used to blend the MAE model and Q=0.80 model, getting the best of both?

Architecture: `final = (1 − p_smoke) × mae_pred + p_smoke × q80_pred`

| Variant | Overall MAE | Smoke-day MAE | Smoke Δ |
|---|---|---|---|
| Raw blend (linear p) | 1.507 | 22.216 | +0.137 |
| Calibrated blend (p²) | 1.507 | 21.965 | +0.388 |
| Rank-norm blend | 1.699 | 21.778 | +0.575 |

**Gate calibration check:**
- Raw XGBoost: mean p=0.130 on actual smoke days, 0.001 on normal days
- After isotonic calibration: mean p=0.170 on smoke days

**Finding — Raw gate probabilities are too compressed.** Even on actual smoke days the gate only assigns 13% probability — the blend barely activates and defaults to the MAE model. Isotonic calibration helps (0.137 → 0.388 smoke Δ), but the probabilities are still low because 46 smoke events in 25 years is not enough to reliably calibrate a classifier.

**Finding — Rank normalization produces a near-Q=0.80 result.** Converting probabilities to their rank percentile spreads the distribution uniformly. This essentially weights each day by how unusual its fire/wind profile looks relative to all other days — smoke Δ=+0.575, nearly matching Q=0.80 alone (+0.593). A different route to the same destination.

---

## Experiment 6 — Feature Subset for Smoke-Day Focus

**Question:** Weather features (pressure, precipitation, humidity, temperature) are strong predictors on clean days. Do they add noise during fire events?

| Feature Set | Features | Overall MAE | Smoke-day MAE | Smoke Δ |
|---|---|---|---|---|
| Full set | 38 | 1.995 | 21.760 | +0.593 |
| Fire + season | 27 | 2.067 | 21.735 | +0.618 |
| **Fire-only** | **24** | **2.075** | **21.729** | **+0.624** |

Fire-only features: pm25 lags, all fire FRP/count bands (close/medium/far), wind vectors, fire-wind interaction terms.

**Finding — Dropping weather features marginally improves smoke-day MAE.** The improvement is small (+0.031) but consistent — weather variables are strong regularisers toward normal-day behaviour. During extreme fire events, atmospheric pressure and humidity are secondary to FRP and wind. Removing them lets the model weight fire signals more.

**Finding — This is the best single smoke-day model: Q=0.80, fire-only features, smoke-day MAE = 21.729, Smoke Δ = +0.624.**

---

## Experiment 7 — LSTM Deep Learning Baseline

**Question:** Can a sequence model exploit temporal dependencies the tree model ignores?

Architecture: 2-layer LSTM, hidden=64, window=7 days, 30 epochs.

| Model | Overall MAE | vs Persistence |
|---|---|---|
| Persistence | 1.842 (test set) | — |
| LSTM | 2.342 | −27% worse |

**Finding — LSTM cannot beat persistence on this dataset.** 9,125 rows is insufficient for a sequence model to generalise. With only 46 positive smoke events, the LSTM never sees enough smoke during training to learn the pattern. The temporal dependencies (lag autocorrelation) are already captured by the lag features handed to tree models — there is no sequential structure left for LSTM to exploit.

---

## Experiment 8 — XGBoost Smoke Detector (Classification)

**Question:** Can a binary classifier predict whether tomorrow will be a smoke day?

Models tested: XGBoost, LightGBM, BalancedRandomForest — all with imbalance handling via scale_pos_weight.

| Model | AUPRC | OOF Episode Recall | F2 Score |
|---|---|---|---|
| LightGBM | 0.072 | 0/13 (0%) | 0.08 |
| BalancedRandomForest | 0.210 | — | 0.34 |
| **XGBoost** | **0.331** | **8/13 (62%)** | **0.46** |

**Finding — XGBoost catches 8 of 13 unique smoke episodes** in the OOF test windows. It is the only model with meaningful precision-recall performance on this task.

**Finding — LightGBM fails completely at classification** despite outperforming XGBoost on regression. The scale_pos_weight mechanism interacts differently with LightGBM's leaf-splitting strategy — it essentially ignores the class imbalance adjustment.

---

## Experiment 9 — Gate Episode Analysis

**Why does the XGBoost gate miss 5 of 13 episodes?**

| Missed Episode | Duration | Root Cause |
|---|---|---|
| 2005-09-12 | 1 day | Pre-2010 sparse training data — structural |
| 2017-08-01 to 08-10 | 10 days | Distant fires (mean 350km), fire_count_close ≈ 0. Smoke transported from afar with no close-range signal |
| 2018-08-12 to 08-14 | 3 days | Fast-onset event — fire activity jumped on Day 2/3. Gate cannot predict Day 0 |
| **2020-09-10 to 09-17** | **8 days** | **US West Coast Labor Day fires. fire_count_total = 0–90, but pm25 = 65–163. BC fire data is completely blind to Oregon/California sources** |
| 2023-08-19 | 1 day | Single pre-smoke day (pm25=14 today, >25 tomorrow) edge case |

**Critical finding — The 2020 September miss is a data ceiling, not a model ceiling.** On 2020-09-14, `fire_count_total = 0`, `fire_frp_sum_total = 0.0`, yet `pm25 = 163 µg/m³`. The smoke came from the Oregon/California Labor Day fires — one of the worst air quality events in Pacific Northwest history. No BC fire feature can predict this. The model is not broken; our data has no visibility into out-of-province smoke transport.

This directly motivates HYSPLIT wind back-trajectory integration or US VIIRS fire data inclusion as future work.

---

## Full Results Summary

| Model | Overall MAE | Smoke-day MAE | Smoke Δ | Use Case |
|---|---|---|---|---|
| Persistence | 1.668 | 22.353 | — | Baseline |
| LightGBM MAE | 1.504 | 22.181 | +0.172 | Best overall accuracy |
| MAE + Layer 2 stack (p80) | 1.509 | 22.055 | +0.298 | Best accuracy + minimal smoke cost |
| Calib blend Q=0.80 (p²) | 1.507 | 21.965 | +0.388 | Probabilistic framing, interpretable |
| Q=0.80 full features | 1.995 | 21.760 | +0.593 | Balanced smoke/overall |
| **Q=0.80 fire-only (24 feat)** | **2.075** | **21.729** | **+0.624** | **Best smoke-day MAE** |
| Q=0.90 | 2.521 | 21.391 | +0.962 | Risk-ceiling, health alerts |
| Q=0.95 | 3.061 | 21.228 | +1.125 | Worst-case planning |
| XGBoost classifier | AUPRC=0.331 | — | 8/13 episodes | Early warning / alert |
| LSTM | 2.342 | — | −27% | Negative result |
| Sample-weighted | 1.623 | 23.274 | −0.921 | Negative result |
| Hurdle model | 1.813 | 24.164 | −1.2% | Negative result |

---

## Core Inferences

### 1. The problem is a data-ceiling problem, not a modelling problem
The single biggest performance limiter is the 2020 Labor Day episode — smoke from Oregon/California that our BC fire features cannot see. Even a perfect model with perfect BC fire data would miss this event. Adding US VIIRS fire data or HYSPLIT transport vectors would provide the biggest improvement over anything trialled here.

### 2. Residual framing is the single most important design choice
Predicting pm25_diff rather than raw pm25 is more impactful than any loss function, feature set, or architecture. It converts an autocorrelated time series into a near-stationary regression problem and bakes in the persistence baseline as the prior.

### 3. Quantile regression is the right tool for black-swan under-prediction
It has a clear theoretical justification (shift the model's objective to a higher quantile of the conditional distribution), a predictable and monotone tradeoff curve, and requires changing one number. All other asymmetric approaches (custom loss, sample weighting, stacking, blending) are more complex and either match or underperform it.

### 4. Weather features hurt smoke-day prediction
The model trained on fire signals + wind only outperforms the full 38-feature model on smoke-day MAE. Weather variables (pressure, humidity, temperature) are strong predictors on clean days but add noise during fire events where the dominant mechanism is FRP × wind transport, not meteorological conditions.

### 5. The gate misses are structurally interpretable
Not random noise — each miss has a physical explanation: out-of-province transport (2020), distant fires with no close signal (2017), fast-onset (2018), single-day lead time (2023). This means the gate's failure modes are predictable and addressable with better data, not better algorithms.

### 6. Tree models dominate because the problem is tabular and interaction-driven
The key predictive signal — fire FRP × wind direction → smoke transport — is a multiplicative interaction between two features. A single tree split captures this exactly. LSTM needs to learn it implicitly through weight products across time steps with 46 positive training examples. This is why gradient-boosted trees beat everything else here.

---

## Remaining Tasks

Modelling is complete. Remaining tasks ordered by deadline:

| Task | Priority | Deadline | Notes |
|---|---|---|---|
| Update `speaking_notes.md` (Slides 8, 9 + Q&A) | **CRITICAL** | Before April 7 | RQ1 finding needs new framing — quantile regression + XGBoost gate missing |
| Commit all untracked scripts | HIGH | Before April 7 | 10+ scripts in `scripts/` untracked |
| Update `report_final.md` (expand Section 5 RQ1 + Section 8) | HIGH | With code submission | Entire follow-up modeling phase absent from report |
| `app/streamlit_app.py` smoke alert | MEDIUM | — | Add XGBoost gate probability to app |
| `notebooks/05_regime_modeling.ipynb` | LOW | — | Present results in notebook format |
| Phase-shift visualisation (Sept 2020) | LOW | — | Plot showing quantile model detects rising trend before persistence |
