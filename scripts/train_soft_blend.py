"""
Soft Gate Blending for PM2.5 Spike Forecasting
===============================================
The hard-gate hurdle model (gate fires or doesn't) suffers from false alarms:
when the XGBoost gate misfires on a normal day, the Stage 2 regressor predicts
smoke-level PM2.5, blowing up overall MAE.

Soft blending replaces the binary switch with a continuous interpolation:

    final = (1 - p_smoke) * mae_pred + p_smoke * q75_pred

where p_smoke is XGBoost's estimated probability that tomorrow is a smoke day.

On clean days (p ≈ 0): use the precise MAE model.
On rising smoke days (p ≈ 0.5): blend both, natural transition.
On confirmed smoke risk (p → 1): fully rely on the upward-biased quantile model.

No new training required — per fold we train:
  1. XGBoost classifier (same gate as train_smoke_detector.py)
  2. LightGBM MAE residual model
  3. LightGBM Q=0.75 residual model

Then blend outputs. We also test p_smoke^k scaling (k > 1 = more conservative blending).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH       = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD = 25.0
FOLD_CUTOFFS    = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]

# ---------------------------------------------------------------------------
# Features (38-feature set from test_refined_residual.py)
# ---------------------------------------------------------------------------
PM25_LAG_FEATURES = ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7"]
FIRE_FEATURES = [
    "fire_count_close", "fire_frp_sum_close",
    "fire_count_medium", "fire_frp_sum_medium",
    "fire_count_far", "fire_frp_sum_far",
    "fire_count_total", "fire_frp_sum_total",
    "fire_mean_distance_km",
]
FIRE_LAG_FEATURES = [
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_count_close_lag1", "fire_frp_sum_close_lag1",
    "fire_count_medium_lag1", "fire_frp_sum_medium_lag1",
]
WEATHER_FEATURES = [
    "temperature_2m_mean", "temperature_2m_max", "relative_humidity_2m_mean",
    "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
    "precipitation_sum", "pressure_msl_mean", "temperature_range", "has_precipitation",
]
CALENDAR_FEATURES = ["month", "day_of_year", "is_fire_season"]

# Gate uses extended feature set with episode precursors (from train_smoke_detector.py)
GATE_EXTRA = [
    "pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "fire_count_total", "fire_frp_sum_total",
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_count_close", "fire_frp_sum_close",
    "temperature_2m_mean", "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
    "pressure_msl_mean", "precipitation_sum",
    "month", "day_of_year", "is_fire_season",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")

    # Wind-fire interactions (from test_refined_residual.py)
    df["is_south_wind"] = (
        (df["wind_direction_10m_mean"] >= 135) & (df["wind_direction_10m_mean"] <= 225)
    ).astype(int)
    df["is_east_wind"] = (
        (df["wind_direction_10m_mean"] >= 45) & (df["wind_direction_10m_mean"] < 135)
    ).astype(int)
    df["fire_wind_south"] = df["fire_count_total"] * df["is_south_wind"]
    df["fire_wind_east"]  = df["fire_count_total"] * df["is_east_wind"]
    df["fire_times_u"]    = df["fire_count_total"] * df["wind_u_component"]
    df["fire_times_v"]    = df["fire_count_total"] * df["wind_v_component"]

    # Gate precursor features (from train_smoke_detector.py)
    df["pm25_3d_trend"]      = df["pm25"].diff(3)
    df["fire_frp_3d_rolling"] = df["fire_frp_sum_total"].rolling(3, min_periods=1).sum()
    df["fire_building"]      = (df["fire_count_total"] > df["fire_count_total"].shift(1)).astype(int)
    southerly = (df["wind_v_component"] > 0).astype(int)
    streak, count = [], 0
    for s in southerly:
        count = count + 1 if s else 0
        streak.append(count)
    df["wind_southerly_streak"] = streak
    smoke_mask = df["pm25"] > SMOKE_THRESHOLD
    days_since, last = [], 999
    for s in smoke_mask:
        last = 0 if s else min(last + 1, 999)
        days_since.append(last)
    df["days_since_last_smoke"] = days_since
    df["drought_proxy"] = df["precipitation_sum"].rolling(30, min_periods=7).mean().fillna(0)

    return df


NEW_FEATURES = ["fire_wind_south", "fire_wind_east", "fire_times_u", "fire_times_v"]
ALL_REG_FEATURES = (
    PM25_LAG_FEATURES + FIRE_FEATURES + FIRE_LAG_FEATURES
    + WEATHER_FEATURES + CALENDAR_FEATURES + NEW_FEATURES
)

GATE_FEATURES = GATE_EXTRA + [
    "pm25_3d_trend", "fire_frp_3d_rolling", "fire_building",
    "wind_southerly_streak", "days_since_last_smoke", "drought_proxy",
]


BASE_PARAMS = dict(
    n_estimators=500, learning_rate=0.01, max_depth=8,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)

    df["pm25_target"]     = df["pm25"].shift(-1)
    df["pm25_diff"]       = df["pm25_target"] - df["pm25"]
    df["smoke_tomorrow"]  = (df["pm25_target"] > SMOKE_THRESHOLD).astype(int)

    all_cols = list(set(ALL_REG_FEATURES + GATE_FEATURES +
                        ["pm25_target", "pm25_diff", "smoke_tomorrow", "date", "pm25"]))
    df_model = df.iloc[7:-1].dropna(subset=all_cols).reset_index(drop=True)

    X_reg  = df_model[ALL_REG_FEATURES].values
    X_gate = df_model[GATE_FEATURES].values
    y_diff    = df_model["pm25_diff"].values
    y_actual  = df_model["pm25_target"].values
    y_cls     = df_model["smoke_tomorrow"].values
    y_today   = df_model["pm25"].values
    dates     = pd.to_datetime(df_model["date"].values)

    n_pos = y_cls.sum()
    spw   = (len(y_cls) - n_pos) / max(n_pos, 1)

    smoke_mask_all = y_actual > SMOKE_THRESHOLD
    print(f"Dataset: {len(df_model)} rows | Smoke days: {smoke_mask_all.sum()} ({100*smoke_mask_all.mean():.2f}%)")
    print(f"scale_pos_weight = {spw:.1f}\n")

    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = []
    for cutoff in fold_cutoffs_dt:
        tr = np.where(dates < cutoff)[0]
        te = np.where(dates >= cutoff)[0]
        if len(tr) >= 50 and len(te) >= 50:
            folds.append((tr, te))

    # OOF arrays
    oof_mae    = np.full(len(y_actual), np.nan)
    oof_q75    = np.full(len(y_actual), np.nan)
    oof_p      = np.full(len(y_actual), np.nan)  # raw XGBoost probability
    oof_p_cal  = np.full(len(y_actual), np.nan)  # isotonic-calibrated probability
    oof_actual = np.full(len(y_actual), np.nan)
    oof_today  = np.full(len(y_actual), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_reg_tr, X_reg_te   = X_reg[train_idx],  X_reg[test_idx]
        X_gate_tr, X_gate_te = X_gate[train_idx], X_gate[test_idx]
        y_diff_tr            = y_diff[train_idx]
        y_cls_tr             = y_cls[train_idx]
        today_te             = y_today[test_idx]
        n_smoke_te           = (y_actual[test_idx] > SMOKE_THRESHOLD).sum()

        print(f"Fold {fold_i+1} | train={len(train_idx)}  test={len(test_idx)}  "
              f"smoke_days_in_test={n_smoke_te}")

        oof_actual[test_idx] = y_actual[test_idx]
        oof_today[test_idx]  = today_te

        # 1a. XGBoost gate (raw probabilities)
        gate = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        gate.fit(X_gate_tr, y_cls_tr)
        p_smoke = gate.predict_proba(X_gate_te)[:, 1]
        oof_p[test_idx] = p_smoke

        # 1b. Isotonic calibration — fit on held-out last 20% of training fold
        calib_split = int(len(train_idx) * 0.80)
        X_cal_tr = X_gate_tr[:calib_split]
        y_cal_tr = y_cls_tr[:calib_split]
        X_cal_val = X_gate_tr[calib_split:]
        y_cal_val = y_cls_tr[calib_split:]

        gate_base = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        gate_base.fit(X_cal_tr, y_cal_tr)
        cal_clf = CalibratedClassifierCV(gate_base, method="isotonic", cv="prefit")
        cal_clf.fit(X_cal_val, y_cal_val)
        p_smoke_cal = cal_clf.predict_proba(X_gate_te)[:, 1]
        oof_p_cal[test_idx] = p_smoke_cal

        # 2. LightGBM MAE residual model
        mae_model = lgb.LGBMRegressor(**BASE_PARAMS, objective="mae")
        mae_model.fit(X_reg_tr, y_diff_tr)
        mae_pred = np.maximum(today_te + mae_model.predict(X_reg_te), 0.0)
        oof_mae[test_idx] = mae_pred

        # 3. LightGBM Q=0.80 residual model (upgraded from 0.75)
        q75_model = lgb.LGBMRegressor(**BASE_PARAMS, objective="quantile", alpha=0.80)
        q75_model.fit(X_reg_tr, y_diff_tr)
        q75_pred = np.maximum(today_te + q75_model.predict(X_reg_te), 0.0)
        oof_q75[test_idx] = q75_pred

        print()

    # ---------------------------------------------------------------------------
    # Blend and evaluate
    # ---------------------------------------------------------------------------
    valid  = ~np.isnan(oof_actual)
    smoke  = valid & (oof_actual > SMOKE_THRESHOLD)
    normal = valid & (oof_actual <= SMOKE_THRESHOLD)

    persist_overall = mean_absolute_error(oof_actual[valid], oof_today[valid])
    persist_smoke   = mean_absolute_error(oof_actual[smoke], oof_today[smoke]) if smoke.sum() > 0 else float("nan")
    persist_normal  = mean_absolute_error(oof_actual[normal], oof_today[normal])

    # Test several blending strategies
    blends = {}

    # Baseline sub-models (no blending)
    blends["MAE model alone"]    = oof_mae
    blends["Q=0.75 model alone"] = oof_q75

    # Raw probability blends
    p = oof_p
    blends["Raw blend (linear p)"]  = (1 - p) * oof_mae + p * oof_q75
    p_thresh = np.where(p > 0.10, p, 0.0)
    blends["Raw blend (p>0.10)"]    = (1 - p_thresh) * oof_mae + p_thresh * oof_q75

    # Calibrated probability blends
    pc = oof_p_cal
    blends["Calib blend (linear p)"] = (1 - pc) * oof_mae + pc * oof_q75
    pc2 = pc ** 2
    blends["Calib blend (p²)"]       = (1 - pc2) * oof_mae + pc2 * oof_q75
    pc_thresh = np.where(pc > 0.10, pc, 0.0)
    blends["Calib blend (p>0.10)"]   = (1 - pc_thresh) * oof_mae + pc_thresh * oof_q75

    # Rank-normalized blend: convert p to rank percentile over valid entries only
    p_rank = np.full_like(p, np.nan)
    p_rank[valid] = rankdata(p[valid]) / valid.sum()
    blends["Rank-norm blend"]        = (1 - p_rank) * oof_mae + p_rank * oof_q75

    print("=" * 76)
    print(f"{'Model':<38} {'Overall':>8} {'Smoke-day':>10} {'Normal-day':>11}")
    print(f"{'':38} {'MAE':>8} {'MAE':>10} {'MAE':>11}")
    print("-" * 76)
    print(f"{'Persistence (baseline)':<38} {persist_overall:>8.3f} {persist_smoke:>10.3f} {persist_normal:>11.3f}")
    print("-" * 76)

    for name, preds in blends.items():
        mae_all    = mean_absolute_error(oof_actual[valid],  preds[valid])
        mae_smoke  = mean_absolute_error(oof_actual[smoke],  preds[smoke]) if smoke.sum() > 0 else float("nan")
        mae_normal = mean_absolute_error(oof_actual[normal], preds[normal])
        delta      = persist_smoke - mae_smoke
        flag = " ✓" if delta > 0 else "  "
        print(f"{name:<38} {mae_all:>8.3f} {mae_smoke:>10.3f} {mae_normal:>11.3f}"
              f"   smoke Δ={delta:+.3f}{flag}")

    print("=" * 76)
    print(f"\nSmoke days: {smoke.sum()}  |  Normal days: {normal.sum()}")

    # Show mean p_smoke on smoke days vs normal days (gate calibration check)
    print(f"\nGate calibration check:")
    print(f"  Raw XGBoost:   smoke days p={oof_p[smoke].mean():.3f}  |  normal days p={oof_p[normal].mean():.3f}")
    print(f"  Isotonic cal:  smoke days p={oof_p_cal[smoke].mean():.3f}  |  normal days p={oof_p_cal[normal].mean():.3f}")
    print(f"  Rank-norm:     smoke days p={p_rank[smoke].mean():.3f}  |  normal days p={p_rank[normal].mean():.3f}")


if __name__ == "__main__":
    main()
