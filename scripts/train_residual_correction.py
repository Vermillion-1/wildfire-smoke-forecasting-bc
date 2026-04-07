"""
Multi-Layer Residual Correction for PM2.5 Spike Forecasting
=============================================================
From: suggest_exp.md (HIGH priority, untested)

Architecture:
  Layer 1: Standard LightGBM MAE model — predicts pm25_diff for ALL days.
            This model does well on normal days but systematically under-predicts
            during high-fire events.

  Layer 2: A second LightGBM trained ONLY on high-fire training days, where the
            target is the RESIDUAL (error) left by Layer 1.
            "High-fire" = fire_frp_sum_total above the 90th percentile of the
            training fold (computed fresh per fold to avoid leakage).

  Final:    pred = layer1_pred + layer2_correction
            The correction is applied only when fire_frp_sum_total > threshold
            (same 90th-percentile threshold from the training fold).

Why this works (theory):
  Layer 1 learns the mean signal. Its errors on fire days are systematic
  (always under-predictions) and correlated with fire features. Layer 2
  learns to predict those errors from fire features, effectively adding
  an "expert" correction on days that look like fire events.

  Crucially, on clean days Layer 2 correction = 0 (not applied), so clean-day
  accuracy is fully preserved. This is unlike quantile regression which shifts
  ALL predictions upward, hurting normal-day MAE.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH           = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD     = 25.0
FIRE_PERCENTILE     = 80       # top-N% fire days get Layer 2 correction applied
FOLD_CUTOFFS        = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]

# ---------------------------------------------------------------------------
# Features — same 38-feature set as test_refined_residual.py
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return df


NEW_FEATURES = ["fire_wind_south", "fire_wind_east", "fire_times_u", "fire_times_v"]
ALL_FEATURES = (
    PM25_LAG_FEATURES + FIRE_FEATURES + FIRE_LAG_FEATURES
    + WEATHER_FEATURES + CALENDAR_FEATURES + NEW_FEATURES
)


LAYER1_PARAMS = dict(
    n_estimators=500, learning_rate=0.01, max_depth=8,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1, objective="mae",
)
LAYER2_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1, objective="mae",
)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)

    df["pm25_target"] = df["pm25"].shift(-1)
    df["pm25_diff"]   = df["pm25_target"] - df["pm25"]
    df_model = df.iloc[7:-1].dropna(subset=ALL_FEATURES + ["pm25_diff"]).reset_index(drop=True)

    X        = df_model[ALL_FEATURES].values
    y_diff   = df_model["pm25_diff"].values
    y_actual = df_model["pm25_target"].values
    y_today  = df_model["pm25"].values
    frp      = df_model["fire_frp_sum_total"].values
    dates    = pd.to_datetime(df_model["date"].values)

    smoke_mask = y_actual > SMOKE_THRESHOLD
    print(f"Dataset: {len(df_model)} rows | Smoke days: {smoke_mask.sum()} ({100*smoke_mask.mean():.2f}%)\n")

    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = []
    for cutoff in fold_cutoffs_dt:
        tr = np.where(dates < cutoff)[0]
        te = np.where(dates >= cutoff)[0]
        if len(tr) >= 50 and len(te) >= 50:
            folds.append((tr, te))

    # OOF arrays
    oof_layer1     = np.full(len(y_actual), np.nan)  # MAE Layer 1 alone
    oof_stack      = np.full(len(y_actual), np.nan)  # MAE Layer 1 + Layer 2
    oof_q75        = np.full(len(y_actual), np.nan)  # Q=0.75 Layer 1 alone
    oof_q75_stack  = np.full(len(y_actual), np.nan)  # Q=0.75 Layer 1 + Layer 2
    oof_actual     = np.full(len(y_actual), np.nan)
    oof_today      = np.full(len(y_actual), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te      = X[train_idx],      X[test_idx]
        y_diff_tr        = y_diff[train_idx]
        y_actual_te      = y_actual[test_idx]
        today_te         = y_today[test_idx]
        frp_tr, frp_te   = frp[train_idx],   frp[test_idx]
        n_smoke          = (y_actual_te > SMOKE_THRESHOLD).sum()

        oof_actual[test_idx] = y_actual_te
        oof_today[test_idx]  = today_te

        print(f"Fold {fold_i+1} | train={len(train_idx)}  test={len(test_idx)}  smoke_in_test={n_smoke}")

        # ── Layer 1: full residual model ──────────────────────────────────
        l1 = lgb.LGBMRegressor(**LAYER1_PARAMS)
        l1.fit(X_tr, y_diff_tr)

        l1_train_pred_diff = l1.predict(X_tr)
        l1_test_pred_diff  = l1.predict(X_te)

        layer1_pred_te = np.maximum(today_te + l1_test_pred_diff, 0.0)
        oof_layer1[test_idx] = layer1_pred_te

        # ── Identify high-fire days in training ───────────────────────────
        frp_threshold = np.percentile(frp_tr, FIRE_PERCENTILE)
        # Include zero-threshold case: if threshold is 0 (no fire most of the time),
        # step up to at least 1.0 so we don't train Layer 2 on all days
        frp_threshold = max(frp_threshold, 1.0)

        high_fire_tr = frp_tr > frp_threshold
        n_high_fire  = high_fire_tr.sum()

        print(f"  fire_frp threshold (p{FIRE_PERCENTILE}): {frp_threshold:.1f}  |  "
              f"high-fire training days: {n_high_fire}")

        if n_high_fire < 10:
            # Not enough high-fire days in this fold to train Layer 2 — fall back to Layer 1
            print(f"  Skipping Layer 2 (only {n_high_fire} high-fire training days)")
            oof_stack[test_idx] = layer1_pred_te
        else:
            # ── Layer 2: residual correction on high-fire days ────────────
            # Target = actual pm25_diff minus what Layer 1 predicted
            l1_residuals_tr = y_diff_tr - l1_train_pred_diff   # shape: (n_train,)
            X_hf  = X_tr[high_fire_tr]
            y_hf  = l1_residuals_tr[high_fire_tr]

            l2 = lgb.LGBMRegressor(**LAYER2_PARAMS)
            l2.fit(X_hf, y_hf)

            # Apply correction only when test-day fire activity is elevated
            high_fire_te    = frp_te > frp_threshold
            correction_te   = np.zeros(len(test_idx))
            if high_fire_te.sum() > 0:
                correction_te[high_fire_te] = l2.predict(X_te[high_fire_te])

            # Also clip correction to be non-negative (we only want upward corrections)
            correction_te = np.maximum(correction_te, 0.0)

            stack_pred_te = np.maximum(layer1_pred_te + correction_te, 0.0)
            oof_stack[test_idx] = stack_pred_te

            n_corrected = high_fire_te.sum()
            mean_corr   = correction_te[high_fire_te].mean() if n_corrected > 0 else 0.0
            print(f"  Test days receiving correction: {n_corrected}  |  "
                  f"mean correction: {mean_corr:+.2f} µg/m³")

        # ── Q=0.75 Layer 1 + its own Layer 2 correction ───────────────────
        l1_q75 = lgb.LGBMRegressor(
            **{**LAYER1_PARAMS, "objective": "quantile", "alpha": 0.75}
        )
        l1_q75.fit(X_tr, y_diff_tr)

        q75_train_pred_diff = l1_q75.predict(X_tr)
        q75_test_pred_diff  = l1_q75.predict(X_te)
        q75_pred_te = np.maximum(today_te + q75_test_pred_diff, 0.0)
        oof_q75[test_idx] = q75_pred_te

        if n_high_fire < 10:
            oof_q75_stack[test_idx] = q75_pred_te
        else:
            q75_residuals_tr = y_diff_tr - q75_train_pred_diff
            X_hf_q75 = X_tr[high_fire_tr]
            y_hf_q75 = q75_residuals_tr[high_fire_tr]

            l2_q75 = lgb.LGBMRegressor(**LAYER2_PARAMS)
            l2_q75.fit(X_hf_q75, y_hf_q75)

            high_fire_te    = frp_te > frp_threshold
            corr_q75        = np.zeros(len(test_idx))
            if high_fire_te.sum() > 0:
                corr_q75[high_fire_te] = l2_q75.predict(X_te[high_fire_te])
            corr_q75 = np.maximum(corr_q75, 0.0)

            oof_q75_stack[test_idx] = np.maximum(q75_pred_te + corr_q75, 0.0)

        print()

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    valid  = ~np.isnan(oof_actual)
    smoke  = valid & (oof_actual > SMOKE_THRESHOLD)
    normal = valid & (oof_actual <= SMOKE_THRESHOLD)

    persist_o = mean_absolute_error(oof_actual[valid],  oof_today[valid])
    persist_s = mean_absolute_error(oof_actual[smoke],  oof_today[smoke]) if smoke.sum() > 0 else float("nan")
    persist_n = mean_absolute_error(oof_actual[normal], oof_today[normal])

    def row(name, preds):
        o = mean_absolute_error(oof_actual[valid],  preds[valid])
        s = mean_absolute_error(oof_actual[smoke],  preds[smoke]) if smoke.sum() > 0 else float("nan")
        n = mean_absolute_error(oof_actual[normal], preds[normal])
        d = persist_s - s
        flag = " ✓" if d > 0 else "  "
        print(f"{name:<42} {o:>8.3f} {s:>10.3f} {n:>11.3f}   smoke Δ={d:+.3f}{flag}")

    print("=" * 80)
    print(f"{'Model':<42} {'Overall':>8} {'Smoke-day':>10} {'Normal-day':>11}")
    print(f"{'':42} {'MAE':>8} {'MAE':>10} {'MAE':>11}")
    print("-" * 80)
    print(f"{'Persistence (baseline)':<42} {persist_o:>8.3f} {persist_s:>10.3f} {persist_n:>11.3f}")
    print("-" * 80)
    row("MAE Layer 1 alone", oof_layer1)
    row("MAE Layer 1 + Layer 2 stack", oof_stack)
    row("Q=0.75 Layer 1 alone", oof_q75)
    row("Q=0.75 Layer 1 + Layer 2 stack", oof_q75_stack)
    print("=" * 80)

    # Per-fold smoke-day breakdown
    print("\nPer-fold smoke-day MAE comparison:")
    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds2 = []
    for cutoff in fold_cutoffs_dt:
        tr = np.where(dates < cutoff)[0]
        te = np.where(dates >= cutoff)[0]
        if len(tr) >= 50 and len(te) >= 50:
            folds2.append((tr, te))
    for fi, (_, te) in enumerate(folds2):
        sm = (oof_actual[te] > SMOKE_THRESHOLD)
        if sm.sum() == 0:
            continue
        per_s  = mean_absolute_error(oof_actual[te][sm], oof_today[te][sm])
        l1_s   = mean_absolute_error(oof_actual[te][sm], oof_layer1[te][sm])
        stk_s  = mean_absolute_error(oof_actual[te][sm], oof_stack[te][sm])
        q75_s  = mean_absolute_error(oof_actual[te][sm], oof_q75[te][sm])
        q75k_s = mean_absolute_error(oof_actual[te][sm], oof_q75_stack[te][sm])
        print(f"  Fold {fi+1}: n={sm.sum():3d}  Persist={per_s:.2f}  "
              f"MAE+L2={stk_s:.2f}(Δ{per_s-stk_s:+.2f})  "
              f"Q75+L2={q75k_s:.2f}(Δ{per_s-q75k_s:+.2f})")


if __name__ == "__main__":
    main()
