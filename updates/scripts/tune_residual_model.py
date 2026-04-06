"""
Optuna HPO for Quantile q=0.75 Residual Model
==============================================
Tunes LightGBM hyperparameters for the Q=0.75 residual model — our best
approach for smoke-day MAE improvement without destroying normal-day accuracy.

Optimization target: overall MAE on reconstructed pm25 (using 3-fold inner CV).
After tuning, runs a full 5-fold expanding-window evaluation with smoke-day
MAE breakdown using the best params.

Data is loaded once outside the objective to avoid redundant I/O per trial.
n_jobs=1 per model (Optuna parallelism is not used to keep RAM stable).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH       = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD = 25.0
N_TRIALS        = 30
FOLD_CUTOFFS    = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]

# ---------------------------------------------------------------------------
# Features — full 38-feature set (same as test_refined_residual.py)
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

# ---------------------------------------------------------------------------
# Load data once — passed into objective via closure
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)
    df["pm25_target"] = df["pm25"].shift(-1)
    df["pm25_diff"]   = df["pm25_target"] - df["pm25"]
    df_model = df.iloc[7:-1].dropna(subset=ALL_FEATURES + ["pm25_diff"]).reset_index(drop=True)
    X      = df_model[ALL_FEATURES].values
    y_diff = df_model["pm25_diff"].values
    y_act  = df_model["pm25_target"].values
    today  = df_model["pm25"].values
    dates  = pd.to_datetime(df_model["date"].values)
    return X, y_diff, y_act, today, dates


# ---------------------------------------------------------------------------
# Optuna objective — 3-fold inner CV on overall MAE
# ---------------------------------------------------------------------------
def make_objective(X, y_diff, y_act, today):
    tscv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = dict(
            objective="quantile",
            alpha=0.75,
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 100),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        maes = []
        for tr_idx, te_idx in tscv.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[tr_idx], y_diff[tr_idx])
            pred = np.maximum(today[te_idx] + m.predict(X[te_idx]), 0.0)
            maes.append(mean_absolute_error(y_act[te_idx], pred))
        return float(np.mean(maes))

    return objective


# ---------------------------------------------------------------------------
# Final 5-fold evaluation with best params
# ---------------------------------------------------------------------------
def final_eval(best_params, X, y_diff, y_act, today, dates):
    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = []
    for cutoff in fold_cutoffs_dt:
        tr = np.where(dates < cutoff)[0]
        te = np.where(dates >= cutoff)[0]
        if len(tr) >= 50 and len(te) >= 50:
            folds.append((tr, te))

    params = dict(
        objective="quantile", alpha=0.75,
        random_state=42, verbose=-1, n_jobs=1,
        **best_params
    )

    oof_pred   = np.full(len(y_act), np.nan)
    oof_actual = np.full(len(y_act), np.nan)
    oof_today  = np.full(len(y_act), np.nan)

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr_idx], y_diff[tr_idx])
        pred = np.maximum(today[te_idx] + m.predict(X[te_idx]), 0.0)
        oof_pred[te_idx]   = pred
        oof_actual[te_idx] = y_act[te_idx]
        oof_today[te_idx]  = today[te_idx]
        print(f"  Fold {fold_i+1}: MAE={mean_absolute_error(y_act[te_idx], pred):.4f}")

    valid  = ~np.isnan(oof_actual)
    smoke  = valid & (oof_actual > SMOKE_THRESHOLD)
    normal = valid & (oof_actual <= SMOKE_THRESHOLD)

    persist_o = mean_absolute_error(oof_actual[valid],  oof_today[valid])
    persist_s = mean_absolute_error(oof_actual[smoke],  oof_today[smoke])
    persist_n = mean_absolute_error(oof_actual[normal], oof_today[normal])
    tuned_o   = mean_absolute_error(oof_actual[valid],  oof_pred[valid])
    tuned_s   = mean_absolute_error(oof_actual[smoke],  oof_pred[smoke])
    tuned_n   = mean_absolute_error(oof_actual[normal], oof_pred[normal])

    print(f"\n{'Model':<40} {'Overall':>8} {'Smoke-day':>10} {'Normal-day':>11}")
    print("-" * 72)
    print(f"{'Persistence':<40} {persist_o:>8.3f} {persist_s:>10.3f} {persist_n:>11.3f}")
    print(f"{'Q=0.75 default params':<40} {'(see train_asymmetric_loss.py)':>32}")
    print(f"{'Q=0.75 tuned (Optuna 30 trials)':<40} {tuned_o:>8.3f} {tuned_s:>10.3f} {tuned_n:>11.3f}"
          f"   smoke Δ={persist_s-tuned_s:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    X, y_diff, y_act, today, dates = load_data()
    print(f"Dataset: {len(X)} rows\n")

    print(f"Running Optuna HPO ({N_TRIALS} trials, Q=0.75 quantile, 38 features)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(X, y_diff, y_act, today), n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nBest trial — MAE: {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nFinal 5-fold evaluation with best params:")
    final_eval(study.best_trial.params, X, y_diff, y_act, today, dates)


if __name__ == "__main__":
    main()
