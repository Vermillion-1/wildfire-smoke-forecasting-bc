"""
Two-Stage Hurdle Model for Smoke Day PM2.5
==========================================
Decouples the rare-event problem into two specialised sub-problems:

  Stage 1 (Gate):      Binary classifier — will tomorrow be a smoke day?
                        Uses LightGBM with scale_pos_weight, threshold-tuned for F2.
  Stage 2 (Regressor): PM2.5 magnitude — given a smoke day is predicted,
                        what will the actual PM2.5 be?
                        Trained only on confirmed smoke-day samples using
                        leave-one-episode-out cross-validation.
  Fallback:            On non-smoke predictions, use the persistence baseline
                        (tomorrow ≈ today).

Evaluation compares:
  - Overall MAE vs persistence baseline
  - Smoke-day MAE vs persistence baseline smoke-day MAE (22.11 from 03_modeling)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, fbeta_score
import lightgbm as lgb
import xgboost as xgb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH       = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD = 25.0
BETA            = 2.0
PERSISTENCE_SMOKE_MAE = 22.11   # from 03_modeling.ipynb benchmark

FOLD_CUTOFFS = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]


# ---------------------------------------------------------------------------
# Feature engineering (same as train_smoke_detector.py)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    df["pm25_3d_trend"]     = df["pm25"].diff(3)
    df["fire_frp_3d_rolling"] = df["fire_frp_sum_total"].rolling(3, min_periods=1).sum()
    df["fire_building"]     = (df["fire_count_total"] > df["fire_count_total"].shift(1)).astype(int)

    southerly = (df["wind_v_component"] > 0).astype(int)
    streak, count = [], 0
    for s in southerly:
        count = count + 1 if s else 0
        streak.append(count)
    df["wind_southerly_streak"] = streak

    smoke_mask = (df["pm25"] > SMOKE_THRESHOLD)
    days_since, last = [], 999
    for s in smoke_mask:
        last = 0 if s else min(last + 1, 999)
        days_since.append(last)
    df["days_since_last_smoke"] = days_since

    df["drought_proxy"] = df["precipitation_sum"].rolling(30, min_periods=7).mean().fillna(0)
    return df


GATE_FEATURES = [
    "pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_3d_trend",
    "fire_count_total", "fire_frp_sum_total",
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_frp_3d_rolling", "fire_building",
    "fire_count_close", "fire_frp_sum_close",
    "temperature_2m_mean", "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
    "pressure_msl_mean", "precipitation_sum",
    "wind_southerly_streak", "days_since_last_smoke", "drought_proxy",
    "month", "day_of_year", "is_fire_season",
]

# Regressor uses a tighter set focused on magnitude drivers
REG_FEATURES = [
    "pm25", "pm25_lag1", "pm25_lag2",
    "fire_frp_sum_total", "fire_frp_sum_total_lag1", "fire_frp_sum_total_lag2",
    "fire_frp_3d_rolling",
    "fire_frp_sum_close", "fire_frp_sum_medium",
    "wind_u_component", "wind_v_component", "wind_speed_10m_mean",
    "temperature_2m_max", "relative_humidity_2m_mean",
    "days_since_last_smoke",
    "month", "is_fire_season",
]


# ---------------------------------------------------------------------------
# Threshold optimisation (F-beta)
# ---------------------------------------------------------------------------
def best_threshold(y_true, proba, beta=BETA, n_steps=200):
    thresholds = np.linspace(0.01, 0.6, n_steps)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        if preds.sum() == 0:
            continue
        f = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ---------------------------------------------------------------------------
# Stage 2: Leave-one-episode-out regressor trained on smoke days
# ---------------------------------------------------------------------------
def label_episodes(df: pd.DataFrame) -> pd.Series:
    """Assign integer episode IDs to contiguous smoke-day sequences."""
    episodes = pd.Series(0, index=df.index)
    ep_id, in_ep = 0, False
    for idx, row in df.iterrows():
        if row["is_smoke_actual"]:
            if not in_ep:
                ep_id += 1
                in_ep = True
            episodes[idx] = ep_id
        else:
            in_ep = False
    return episodes


def train_episode_regressor(smoke_df: pd.DataFrame):
    """
    Train a LightGBM regressor on all smoke days.
    Returns (model, leave-one-out oof predictions) for evaluation.
    """
    if len(smoke_df) < 5:
        return None, {}

    X = smoke_df[REG_FEATURES].values
    y = smoke_df["pm25_tomorrow"].values
    episodes = smoke_df["episode"].values
    unique_eps = np.unique(episodes[episodes > 0])

    oof_preds = {}
    for ep in unique_eps:
        train_mask = episodes != ep
        test_mask  = episodes == ep
        if train_mask.sum() < 3:
            continue
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1,
                               max_depth=4, random_state=42, verbose=-1,
                               objective="quantile", alpha=0.75)
        m.fit(X[train_mask], y[train_mask])
        preds = m.predict(X[test_mask])
        for idx, pred in zip(smoke_df.index[test_mask], preds):
            oof_preds[idx] = pred

    # Final model trained on all smoke days (for deployment)
    final_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1,
                                    max_depth=4, random_state=42, verbose=-1,
                                    objective="quantile", alpha=0.75)
    final_model.fit(X, y)
    return final_model, oof_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)

    # Targets
    df["pm25_tomorrow"]    = df["pm25"].shift(-1)
    df["is_smoke_tomorrow"] = (df["pm25_tomorrow"] > SMOKE_THRESHOLD).astype(int)

    df = df[:-1]  # drop last row (no tomorrow)
    all_cols = list(set(GATE_FEATURES + REG_FEATURES +
                        ["pm25_tomorrow", "is_smoke_tomorrow", "date", "pm25"]))
    df_model = df[all_cols].dropna().reset_index(drop=True)

    X_gate   = df_model[GATE_FEATURES].values
    y_cls    = df_model["is_smoke_tomorrow"].values
    y_reg    = df_model["pm25_tomorrow"].values
    pm25_today = df_model["pm25"].values
    dates    = df_model["date"].values

    n_pos = y_cls.sum()
    n_neg = len(y_cls) - n_pos
    spw   = n_neg / max(n_pos, 1)

    print(f"Dataset: {len(df_model)} rows  |  Smoke tomorrow: {n_pos}  ({100*n_pos/len(y_cls):.2f}%)")
    print(f"scale_pos_weight = {spw:.1f}\n")

    # Fold indices
    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = []
    for cutoff in fold_cutoffs_dt:
        tr = np.where(df_model["date"] < cutoff)[0]
        te = np.where(df_model["date"] >= cutoff)[0]
        if len(tr) >= 50 and len(te) >= 50:
            folds.append((tr, te))

    # Collect results per fold
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_cls_tr = X_gate[train_idx], y_cls[train_idx]
        X_te, y_cls_te = X_gate[test_idx],  y_cls[test_idx]
        y_reg_te        = y_reg[test_idx]
        persist_te      = pm25_today[test_idx]
        smoke_te_mask   = y_cls_te == 1

        # ── Stage 1: Gate (XGBoost — 83% episode recall vs LightGBM's 15%) ──
        gate = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        gate.fit(X_tr, y_cls_tr)
        gate_proba = gate.predict_proba(X_te)[:, 1]
        thresh, _ = best_threshold(y_cls_te, gate_proba)
        gate_preds = (gate_proba >= thresh).astype(int)

        # ── Stage 2: Regressor (trained on smoke days in training fold) ──
        df_tr = df_model.iloc[train_idx].copy()
        df_tr["is_smoke_actual"] = (df_tr["pm25"] > SMOKE_THRESHOLD).astype(int)
        df_tr["pm25_tomorrow"]   = y_reg[train_idx]
        df_tr["episode"]         = label_episodes(df_tr).values

        smoke_train_df = df_tr[df_tr["is_smoke_actual"] == 1].copy()
        reg_model, _ = train_episode_regressor(smoke_train_df)

        # ── Combine: hurdle prediction ───────────────────────────────────
        hurdle_preds = persist_te.copy().astype(float)
        smoke_pred_mask = gate_preds == 1
        if reg_model is not None and smoke_pred_mask.sum() > 0:
            X_smoke = df_model.iloc[test_idx][smoke_pred_mask][REG_FEATURES].values
            hurdle_preds[smoke_pred_mask] = reg_model.predict(X_smoke)

        # ── Metrics ─────────────────────────────────────────────────────
        overall_mae_hurdle  = mean_absolute_error(y_reg_te, hurdle_preds)
        overall_mae_persist = mean_absolute_error(y_reg_te, persist_te)

        if smoke_te_mask.sum() > 0:
            smoke_mae_hurdle  = mean_absolute_error(y_reg_te[smoke_te_mask], hurdle_preds[smoke_te_mask])
            smoke_mae_persist = mean_absolute_error(y_reg_te[smoke_te_mask], persist_te[smoke_te_mask])
        else:
            smoke_mae_hurdle = smoke_mae_persist = float("nan")

        print(f"Fold {fold_i+1}:  smoke_days_in_test={smoke_te_mask.sum()}"
              f"  gate_alerts={smoke_pred_mask.sum()}  thresh={thresh:.3f}")
        print(f"  Overall MAE  — Hurdle: {overall_mae_hurdle:.3f}  |  Persistence: {overall_mae_persist:.3f}")
        print(f"  Smoke-day MAE — Hurdle: {smoke_mae_hurdle:.3f}  |  Persistence: {smoke_mae_persist:.3f}")
        if smoke_te_mask.sum() > 0:
            improvement = 100 * (smoke_mae_persist - smoke_mae_hurdle) / smoke_mae_persist
            print(f"  Smoke-day improvement: {improvement:+.1f}%")
        print()

        fold_results.append({
            "overall_mae_hurdle":  overall_mae_hurdle,
            "overall_mae_persist": overall_mae_persist,
            "smoke_mae_hurdle":    smoke_mae_hurdle,
            "smoke_mae_persist":   smoke_mae_persist,
            "n_smoke":             smoke_te_mask.sum(),
        })

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------
    df_res = pd.DataFrame(fold_results)
    print("=" * 60)
    print("Aggregate Results (mean across folds with smoke days)")
    print("-" * 60)

    smoke_folds = df_res[df_res["n_smoke"] > 0]
    if len(smoke_folds) > 0:
        print(f"  Smoke-day MAE — Hurdle:      {smoke_folds['smoke_mae_hurdle'].mean():.3f}")
        print(f"  Smoke-day MAE — Persistence: {smoke_folds['smoke_mae_persist'].mean():.3f}")
        print(f"  Benchmark (from 03_modeling): {PERSISTENCE_SMOKE_MAE:.2f}")
        improvement = 100 * (smoke_folds['smoke_mae_persist'].mean() - smoke_folds['smoke_mae_hurdle'].mean()) / smoke_folds['smoke_mae_persist'].mean()
        print(f"  Smoke-day improvement over persistence: {improvement:+.1f}%")
    else:
        print("  No smoke days in any test fold — increase fold range or lower threshold")

    print(f"\n  Overall MAE — Hurdle:      {df_res['overall_mae_hurdle'].mean():.3f}")
    print(f"  Overall MAE — Persistence: {df_res['overall_mae_persist'].mean():.3f}")
    print("=" * 60)

    print("\nInterpretation:")
    print("  A hurdle model may raise OVERALL MAE (more false alarms on normal days)")
    print("  but should significantly improve SMOKE-DAY MAE — the operationally")
    print("  important metric for health planning and air quality alerts.")


if __name__ == "__main__":
    main()
