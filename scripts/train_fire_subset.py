"""
Fire-Only Feature Subset for Q=0.80 Quantile Model
====================================================
Hypothesis: the 38-feature model includes weather variables (pressure,
precipitation, humidity) that are informative on clean days but add noise
during fire events. A fire-focused subset may improve smoke-day MAE.

Tests three feature sets side by side under the same Q=0.80 residual setup:
  A. Full 38 features (baseline)
  B. Fire-only: pm25 lags + all fire FRP/count + wind only (15 features)
  C. Fire + season: B + month/day_of_year/is_fire_season (18 features)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

DATA_PATH       = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD = 25.0
FOLD_CUTOFFS    = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]

# Full 38-feature set
FULL_FEATURES = [
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "fire_count_close", "fire_frp_sum_close",
    "fire_count_medium", "fire_frp_sum_medium",
    "fire_count_far", "fire_frp_sum_far",
    "fire_count_total", "fire_frp_sum_total", "fire_mean_distance_km",
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_count_close_lag1", "fire_frp_sum_close_lag1",
    "fire_count_medium_lag1", "fire_frp_sum_medium_lag1",
    "temperature_2m_mean", "temperature_2m_max", "relative_humidity_2m_mean",
    "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
    "precipitation_sum", "pressure_msl_mean", "temperature_range", "has_precipitation",
    "month", "day_of_year", "is_fire_season",
    "fire_wind_south", "fire_wind_east", "fire_times_u", "fire_times_v",
]

# Fire-only: pm25 lags + fire signals + wind vectors (no temperature, humidity, pressure, precip)
FIRE_ONLY_FEATURES = [
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "fire_count_close", "fire_frp_sum_close",
    "fire_count_medium", "fire_frp_sum_medium",
    "fire_count_far", "fire_frp_sum_far",
    "fire_count_total", "fire_frp_sum_total", "fire_mean_distance_km",
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_count_close_lag1", "fire_frp_sum_close_lag1",
    "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
    "fire_times_u", "fire_times_v",
]

# Fire + season calendar
FIRE_SEASON_FEATURES = FIRE_ONLY_FEATURES + ["month", "day_of_year", "is_fire_season"]

FEATURE_SETS = {
    "Full 38 features": FULL_FEATURES,
    "Fire-only (24 feat)": FIRE_ONLY_FEATURES,
    "Fire + season (27 feat)": FIRE_SEASON_FEATURES,
}

BASE_PARAMS = dict(
    n_estimators=500, learning_rate=0.01, max_depth=8,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    objective="quantile", alpha=0.80,
    random_state=42, verbose=-1,
)


def engineer_features(df):
    df = df.copy()
    df["is_south_wind"] = ((df["wind_direction_10m_mean"] >= 135) & (df["wind_direction_10m_mean"] <= 225)).astype(int)
    df["is_east_wind"]  = ((df["wind_direction_10m_mean"] >= 45)  & (df["wind_direction_10m_mean"] < 135)).astype(int)
    df["fire_wind_south"] = df["fire_count_total"] * df["is_south_wind"]
    df["fire_wind_east"]  = df["fire_count_total"] * df["is_east_wind"]
    df["fire_times_u"]    = df["fire_count_total"] * df["wind_u_component"]
    df["fire_times_v"]    = df["fire_count_total"] * df["wind_v_component"]
    return df


def run_cv(X, y_diff, y_actual, y_today, dates):
    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = [(np.where(dates < c)[0], np.where(dates >= c)[0])
             for c in fold_cutoffs_dt
             if (dates < c).sum() >= 50 and (dates >= c).sum() >= 50]

    oof_pred   = np.full(len(y_actual), np.nan)
    oof_actual = np.full(len(y_actual), np.nan)
    oof_today  = np.full(len(y_actual), np.nan)

    for tr, te in folds:
        m = lgb.LGBMRegressor(**BASE_PARAMS)
        m.fit(X[tr], y_diff[tr])
        oof_pred[te]   = np.maximum(y_today[te] + m.predict(X[te]), 0.0)
        oof_actual[te] = y_actual[te]
        oof_today[te]  = y_today[te]

    return oof_pred, oof_actual, oof_today


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)
    df["pm25_target"] = df["pm25"].shift(-1)
    df["pm25_diff"]   = df["pm25_target"] - df["pm25"]

    all_needed = list(set(FULL_FEATURES + FIRE_ONLY_FEATURES + FIRE_SEASON_FEATURES
                          + ["pm25_diff", "pm25_target", "pm25", "date"]))
    df_model = df.iloc[7:-1].dropna(subset=all_needed).reset_index(drop=True)

    y_diff   = df_model["pm25_diff"].values
    y_actual = df_model["pm25_target"].values
    y_today  = df_model["pm25"].values
    dates    = pd.to_datetime(df_model["date"].values)

    smoke  = y_actual > SMOKE_THRESHOLD
    print(f"Dataset: {len(df_model)} rows | Smoke days: {smoke.sum()} ({100*smoke.mean():.2f}%)\n")

    results = {}
    for name, feats in FEATURE_SETS.items():
        X = df_model[feats].values
        pred, actual, today = run_cv(X, y_diff, y_actual, y_today, dates)
        results[name] = (pred, actual, today)
        print(f"  {name}: done")

    print()
    valid = ~np.isnan(results["Full 38 features"][1])
    smoke_v  = valid & (results["Full 38 features"][1] > SMOKE_THRESHOLD)
    normal_v = valid & (results["Full 38 features"][1] <= SMOKE_THRESHOLD)

    persist_o = mean_absolute_error(results["Full 38 features"][1][valid], results["Full 38 features"][2][valid])
    persist_s = mean_absolute_error(results["Full 38 features"][1][smoke_v], results["Full 38 features"][2][smoke_v])
    persist_n = mean_absolute_error(results["Full 38 features"][1][normal_v], results["Full 38 features"][2][normal_v])

    print("=" * 76)
    print(f"{'Model':<30} {'Overall':>8} {'Smoke-day':>10} {'Normal-day':>11}  smoke Δ")
    print("-" * 76)
    print(f"{'Persistence':<30} {persist_o:>8.3f} {persist_s:>10.3f} {persist_n:>11.3f}")
    print("-" * 76)
    for name, (pred, actual, today) in results.items():
        o = mean_absolute_error(actual[valid],   pred[valid])
        s = mean_absolute_error(actual[smoke_v], pred[smoke_v])
        n = mean_absolute_error(actual[normal_v], pred[normal_v])
        d = persist_s - s
        print(f"{name:<30} {o:>8.3f} {s:>10.3f} {n:>11.3f}  {d:+.3f} {'✓' if d > 0 else '✗'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
