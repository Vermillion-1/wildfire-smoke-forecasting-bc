# Project TODO — BC Wildfire Smoke PM2.5 Forecasting
> **Final Presentation: April 7, 2026**
> **MPCS Innovation Prize: April 28, 2026**

---

## All Experiments — Final Results

Persistence baseline: Overall MAE **1.668** | Smoke-day MAE **22.353** | Normal-day MAE **1.545**

### Regression / Forecasting Models

| Script | Model | Overall MAE | Smoke-day MAE | Smoke Δ | Notes |
|---|---|---|---|---|---|
| `test_residual_model.py` | LightGBM MAE baseline | 1.504 | 22.181 | +0.172 ✓ | |
| `test_refined_residual.py` | LightGBM MAE + 38 features | ~1.48 | ~22.18 | +0.17 ✓ | wind-fire interactions |
| `train_asymmetric_loss.py` | LightGBM Quantile q=0.75 | 1.852 | 21.813 | +0.540 ✓ | |
| `train_asymmetric_loss.py` | **LightGBM Quantile q=0.80** | **1.995** | **21.760** | **+0.593 ✓** | **Best balanced model** |
| `train_asymmetric_loss.py` | LightGBM Quantile q=0.90 | 2.521 | 21.391 | +0.962 ✓ | Aggressive |
| `train_asymmetric_loss.py` | LightGBM Quantile q=0.95 | 3.061 | 21.228 | +1.125 ✓ | Risk ceiling |
| `train_asymmetric_loss.py` | Sample-weighted 20× | 1.623 | 23.274 | -0.921 ✗ | Backfires |
| `train_asymmetric_loss.py` | Custom Asym MAE 5× | 1.630 | 22.271 | +0.082 ✓ | Marginal |
| `train_asymmetric_loss.py` | Direct Q=0.90 (no residual) | 2.837 | 39.650 | -17.3 ✗ | **Residual framing essential** |
| `train_residual_correction.py` | MAE + Layer 2 stack (p80) | 1.509 | 22.055 | +0.298 ✓ | Best normal-day preservation |
| `train_residual_correction.py` | MAE + Layer 2 stack (p90) | 1.509 | 22.111 | +0.242 ✓ | p80 threshold is better |
| `train_residual_correction.py` | Q=0.75 + Layer 2 stack | 1.853 | 21.813 | +0.540 ✓ | L2 adds nothing on top of quantile |
| `train_soft_blend.py` | Calib blend Q=0.80 (p²) | 1.507 | 21.965 | +0.388 ✓ | Best blend variant |
| `train_soft_blend.py` | Rank-norm blend Q=0.80 | 1.699 | 21.778 | +0.575 ✓ | Near Q=0.80 result |
| `train_soft_blend.py` | Raw blend | 1.507 | 22.216 | +0.137 ✓ | Gate uncalibrated |
| `train_fire_subset.py` | **Q=0.80, fire-only 24 feat** | **2.075** | **21.729** | **+0.624 ✓** | **Best smoke-day MAE** |
| `train_fire_subset.py` | Q=0.80, fire+season 27 feat | 2.067 | 21.735 | +0.618 ✓ | |
| `train_hurdle_model.py` | Hurdle (XGB gate + Q=0.75 Stage2) | 1.813 | 24.164 | -1.2% ✗ | Stage 2 underpowered |
| `tune_residual_model.py` | Q=0.75 + Optuna 30 trials | 1.713 | 21.880 | +0.473 ✓ | Default params beat tuned |
| `train_lstm.py` | LSTM (PyTorch, 30 epochs) | 2.342 | — | -27% ✗ | Negative result |

### Classification / Detection Models

| Script | Model | AUPRC | Episode Recall | Notes |
|---|---|---|---|---|
| `train_smoke_detector.py` | **XGBoost (scale_pos_weight)** | **0.331** | **8/13 (62%)** | Best gate (OOF unique episodes) |
| `train_smoke_detector.py` | BalancedRandomForest | 0.210 | 45/52 (87%) | Lower AUPRC (overlapping fold count) |
| `train_smoke_detector.py` | LightGBM (scale_pos_weight) | 0.072 | 8/52 (15%) | Poor |

---

## Key Findings

1. **Residual framing is essential** — direct prediction with Q=0.90 gives smoke-day MAE 39.65 (vs 22.35 for persistence). Never skip this step.
2. **Q=0.80 fire-only features is the best smoke-day model** — smoke Δ=+0.624, smoke-day MAE 21.729. Dropping weather features that dominate normal days improves fire-event focus.
3. **Quantile improvement is monotone** — more aggressive alpha always improves smoke-day MAE but worsens overall MAE. Tradeoff is linear and predictable.
4. **p80 Layer 2 threshold beats p90** — more training days for Layer 2 improves correction (+0.298 vs +0.242).
5. **Layer 2 stacking adds nothing on top of quantile** — quantile model's mixed residuals on fire days wash out the correction signal.
6. **Isotonic calibration helps soft blending** — smoke Δ 0.137 → 0.388 (raw → calibrated). Gate's raw p is too compressed (0.13 mean on smoke days).
7. **Sample weighting backfires** — confuses rising-smoke days with actual smoke events.
8. **Optuna HPO hurts smoke-day MAE** when optimizing overall MAE — default params' imprecision accidentally helps smoke days.
9. **LSTM negative** — 9K rows insufficient for sequence models.

### Gate Episode Analysis (XGBoost, OOF unique episodes)
**8/13 episodes caught. 5 missed:**
| Episode | Dates | Root Cause |
|---|---|---|
| 2005-09-12 | 1 day | Pre-2010, sparse training data — structural |
| 2017-08-01 to 08-10 | 10 days | Distant fires (mean 350km), NO close activity — smoke transport from afar |
| 2018-08-12 to 08-14 | 3 days | Fast onset — fire activity jumped on Day 2; gate can't see Day 0 |
| **2020-09-10 to 09-17** | **8 days** | **US Labor Day fires (Oregon/CA) — fire_count=0 but pm25=65-163. BC fire data is blind to this. Physically unpredictable from current features.** |
| 2023-08-19 | 1 day | Single pre-smoke day (pm25=14 today, >25 tomorrow) — edge case |

**The 2020 miss directly motivates HYSPLIT wind back-trajectory integration as future work** — when smoke comes from outside BC, our fire features show zero activity.

---

## Experiment Status — ALL DONE

No further modeling experiments recommended. The design space is exhausted:
- Loss functions ✅ (MAE, quantile sweep 0.75-0.95, asymmetric, sample-weighted)
- Feature engineering ✅ (38 full, fire-only, fire+season)
- Architecture ✅ (single model, stacking, soft blending, hurdle, LSTM)
- Classification ✅ (XGBoost, BRF, LightGBM gate)
- HPO ✅ (Optuna 30 trials)

Remaining asymptotic improvement limited by data, not models:
- 2020 Labor Day episode is physically impossible to catch without US fire/transport data
- Pre-2010 sparse events are structural (not enough training examples)

---

## Rejected Approaches (do not revisit)
- **SMOGN** — synthetic oversampling violates temporal ordering
- **CNN+LSTM** — no spatial grid, single-city daily aggregate
- **Transformers / Mamba** — 9K rows insufficient
- **HMM / GAMs / DLNM / CUSUM / Monte Carlo** — complexity >> payoff
- **Deterministic offsets / spatiotemporal buffer undersampling** — spatial grid only

---

## RAM-Safe Tips
- All scripts use LightGBM + XGBoost only — safe to run without closing other apps
- Run from project root: `cd /home/aarish/Documents/Msc/CMPT733/Forest-Fire-Prediction-BC`
