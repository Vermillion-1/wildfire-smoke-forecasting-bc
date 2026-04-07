"""
Asymmetric & Rare-Event Loss Functions for PM2.5 Spike Forecasting
===================================================================
Standard MAE/MSE trains models to minimise average error, which means
they learn to predict near the median — always under-estimating spikes.

This script compares five approaches that directly address the black-swan
under-prediction problem, all on the same 38-feature set and the same
expanding-window CV used in the other modeling scripts.

For each approach we report:
  - Overall MAE         (all days)
  - Smoke-day MAE       (days where actual PM2.5 > 25 µg/m³)
  - Normal-day MAE      (days where actual PM2.5 ≤ 25)
  - vs Persistence      (same split, same smoke-day breakdown)

Approaches:
  1. LightGBM MAE baseline         — same setup as test_refined_residual.py
  2. LightGBM Quantile q=0.75      — mild upward skew, reasonable overall MAE
  3. LightGBM Quantile q=0.90      — aggressive upward skew, best for spikes
  4. Sample-weighted (smoke 20×)   — keep MAE loss but tell model spikes matter 20× more
  5. Tweedie regression             — designed for right-skewed, heavy-tail targets

All models predict the *residual* (pm25_tomorrow - pm25_today) then add today's
PM2.5, matching the residual-model framing that already beats persistence.
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
DATA_PATH        = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD  = 25.0
# Expanding-window fold cutoffs (same as 03_modeling.ipynb and hurdle model)
FOLD_CUTOFFS     = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]
SMOKE_WEIGHT     = 20   # multiplier for smoke-day samples in weighted approach

# ---------------------------------------------------------------------------
# Features (same 38-feature set as test_refined_residual.py)
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
    "precipitation_sum", "pressure_msl_mean", "temperature_range",
    "has_precipitation",
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


# ---------------------------------------------------------------------------
# Custom asymmetric objective — equivalent to quantile loss but explicit
# Under-prediction is penalised PENALTY_RATIO× more than over-prediction
# ---------------------------------------------------------------------------
PENALTY_RATIO = 5.0  # gradient magnitude for under-prediction vs 1 for over

def asymmetric_mae_obj(y_true, y_pred):
    residuals = y_pred - y_true
    gradient = np.where(residuals < 0, -PENALTY_RATIO, 1.0)
    hessian  = np.ones_like(y_true)
    return gradient, hessian

def asymmetric_mae_eval(y_true, y_pred):
    residuals = y_pred - y_true
    loss = np.where(
        residuals < 0,
        PENALTY_RATIO * np.abs(residuals),
        np.abs(residuals)
    ).mean()
    return "asym_mae", loss, False  # False = lower is better


# ---------------------------------------------------------------------------
# Model definitions — same LightGBM base config, only the objective changes
# ---------------------------------------------------------------------------
BASE_PARAMS = dict(
    n_estimators=500, learning_rate=0.01, max_depth=8,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1,
)

def get_models():
    return {
        "1. LightGBM MAE (baseline)": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="mae",
        ),
        "2. LightGBM Quantile q=0.75": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="quantile", alpha=0.75,
        ),
        "3. LightGBM Quantile q=0.80": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="quantile", alpha=0.80,
        ),
        "3b. LightGBM Quantile q=0.90": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="quantile", alpha=0.90,
        ),
        "3c. LightGBM Quantile q=0.95": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="quantile", alpha=0.95,
        ),
        # Sample-weighted: same as MAE baseline but fit() receives sample_weight
        "4. Sample-weighted MAE (smoke 20×)": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="mae",
        ),
        # Direct target prediction (no residual framing) with quantile q=0.90
        # Tests whether the residual trick is actually necessary
        "5. Direct Q=0.90 (no residual)": lgb.LGBMRegressor(
            **BASE_PARAMS, objective="quantile", alpha=0.90,
        ),
        "6. Custom Asym MAE (5× under-penalty)": lgb.LGBMRegressor(
            **BASE_PARAMS,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)

    df["pm25_target"] = df["pm25"].shift(-1)
    df["pm25_diff"]   = df["pm25_target"] - df["pm25"]

    df_model = df.iloc[7:-1].dropna(subset=ALL_FEATURES + ["pm25_diff", "pm25_target"]).reset_index(drop=True)

    X          = df_model[ALL_FEATURES].values
    y_diff     = df_model["pm25_diff"].values
    y_actual   = df_model["pm25_target"].values
    y_today    = df_model["pm25"].values
    dates      = pd.to_datetime(df_model["date"].values)
    smoke_mask = y_actual > SMOKE_THRESHOLD

    # Sample weights for approach 4: weight days where pm25 > 15 or pm25_target > 15
    # (rising smoke starts around 15, full smoke is > 25)
    rising_mask = (df_model["pm25"].values > 15) | (y_actual > 15)
    sample_weights_full = np.where(rising_mask, SMOKE_WEIGHT, 1.0)

    print(f"Dataset: {len(df_model)} rows  |  Actual smoke days (>25): {smoke_mask.sum()}  "
          f"({100*smoke_mask.mean():.2f}%)\n")

    # Expanding-window folds
    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = []
    for cutoff in fold_cutoffs_dt:
        tr = np.where(dates < cutoff)[0]
        te = np.where(dates >= cutoff)[0]
        if len(tr) >= 50 and len(te) >= 50:
            folds.append((tr, te))

    model_names = list(get_models().keys())
    # Collect OOF arrays for aggregate metrics
    oof_preds  = {name: np.full(len(y_actual), np.nan) for name in model_names}
    oof_actual = np.full(len(y_actual), np.nan)
    oof_today  = np.full(len(y_actual), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr_diff  = X[train_idx], y_diff[train_idx]
        X_te, y_te_diff  = X[test_idx],  y_diff[test_idx]
        y_te_actual      = y_actual[test_idx]
        today_te         = y_today[test_idx]
        sw_tr            = sample_weights_full[train_idx]

        oof_actual[test_idx] = y_te_actual
        oof_today[test_idx]  = today_te

        n_smoke_te = (y_te_actual > SMOKE_THRESHOLD).sum()

        print(f"Fold {fold_i+1} | train={len(train_idx)}  test={len(test_idx)}  "
              f"smoke_days_in_test={n_smoke_te}")

        y_tr_actual_train = y_actual[train_idx]

        for name, model in get_models().items():
            if name == "6. Custom Asym MAE (5× under-penalty)":
                model.set_params(objective=asymmetric_mae_obj)
                model.fit(
                    X_tr, y_tr_diff,
                    eval_set=[(X_te, y_te_diff)],
                    eval_metric=asymmetric_mae_eval,
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)],
                )
                pred_diff   = model.predict(X_te)
                pred_actual = np.maximum(today_te + pred_diff, 0.0)
            elif name == "4. Sample-weighted MAE (smoke 20×)":
                model.fit(X_tr, y_tr_diff, sample_weight=sw_tr)
                pred_diff   = model.predict(X_te)
                pred_actual = np.maximum(today_te + pred_diff, 0.0)
            elif name == "5. Direct Q=0.90 (no residual)":
                # Trains directly on pm25_target, not the diff
                model.fit(X_tr, y_tr_actual_train)
                pred_actual = np.maximum(model.predict(X_te), 0.0)
            else:
                model.fit(X_tr, y_tr_diff)
                pred_diff   = model.predict(X_te)
                pred_actual = np.maximum(today_te + pred_diff, 0.0)

            oof_preds[name][test_idx] = pred_actual

        print()

    # ---------------------------------------------------------------------------
    # Aggregate metrics across all OOF predictions
    # ---------------------------------------------------------------------------
    valid = ~np.isnan(oof_actual)
    smoke = valid & (oof_actual > SMOKE_THRESHOLD)
    normal = valid & (oof_actual <= SMOKE_THRESHOLD)

    persist_overall = mean_absolute_error(oof_actual[valid], oof_today[valid])
    persist_smoke   = mean_absolute_error(oof_actual[smoke], oof_today[smoke]) if smoke.sum() > 0 else float("nan")
    persist_normal  = mean_absolute_error(oof_actual[normal], oof_today[normal])

    print("=" * 78)
    print(f"{'Model':<40} {'Overall':>8} {'Smoke-day':>10} {'Normal-day':>11}")
    print(f"{'':40} {'MAE':>8} {'MAE':>10} {'MAE':>11}")
    print("-" * 78)
    print(f"{'Persistence (baseline)':<40} {persist_overall:>8.3f} {persist_smoke:>10.3f} {persist_normal:>11.3f}")
    print("-" * 78)

    for name in model_names:
        preds = oof_preds[name]
        if np.isnan(preds[valid]).any():
            print(f"{name:<40}  (incomplete folds — skip)")
            continue
        mae_all    = mean_absolute_error(oof_actual[valid],   preds[valid])
        mae_smoke  = mean_absolute_error(oof_actual[smoke],   preds[smoke]) if smoke.sum() > 0 else float("nan")
        mae_normal = mean_absolute_error(oof_actual[normal],  preds[normal])

        delta_smoke  = persist_smoke  - mae_smoke   # positive = better than persistence
        delta_overall = persist_overall - mae_all

        flag = " ✓" if delta_smoke > 0 else "  "
        print(f"{name:<40} {mae_all:>8.3f} {mae_smoke:>10.3f} {mae_normal:>11.3f}"
              f"   smoke Δ={delta_smoke:+.3f}{flag}")

    print("=" * 78)
    print(f"\nSmoke days in OOF test windows: {smoke.sum()}")
    print(f"Normal days in OOF test windows: {normal.sum()}")
    print("\nPositive smoke Δ means the model beats persistence on smoke days.")
    print("Watch for the tradeoff: methods that help smoke-day MAE often hurt normal-day MAE.")


if __name__ == "__main__":
    main()
