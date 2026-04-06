"""
Rare Event Smoke Day Detector
==============================
Reframes the PM2.5 forecasting problem as binary classification:
  "Will tomorrow be a smoke day (PM2.5 > 25 µg/m³)?"

Key design choices for rare events (0.5% positives):
  - Evaluate with AUPRC and F2-score (not accuracy or ROC-AUC)
  - Sweep decision threshold to maximise F2 (recall weighted 2x vs precision)
  - Engineer episode-onset precursor features (trends, rolling fire, wind streaks)
  - Compare four imbalanced-learning strategies side-by-side

Outputs a precision-recall curve PNG and prints per-fold + aggregate metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    average_precision_score, fbeta_score, precision_recall_curve,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm as lgb
import xgboost as xgb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = Path("data/processed/merged/dataset.csv")
FIGURES_DIR = Path("figures")
SMOKE_THRESHOLD = 25.0          # µg/m³
N_FOLDS     = 5
BETA        = 2.0               # F-beta: recall weighted 2x vs precision

# Same expanding-window fold boundaries used in 03_modeling.ipynb
FOLD_CUTOFFS = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]


# ---------------------------------------------------------------------------
# Feature engineering: episode-onset precursors
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # PM2.5 trend: slope of pm25 over past 3 days (positive = rising)
    df["pm25_3d_trend"] = df["pm25"].diff(3)

    # Rolling 3-day fire radiative power within 1000 km
    df["fire_frp_3d_rolling"] = (
        df["fire_frp_sum_total"]
        .rolling(3, min_periods=1)
        .sum()
    )

    # Is fire activity building? (today's count > yesterday's)
    df["fire_building"] = (
        df["fire_count_total"] > df["fire_count_total"].shift(1)
    ).astype(int)

    # Consecutive days with southerly wind (v_component > 0 means wind FROM south)
    southerly = (df["wind_v_component"] > 0).astype(int)
    streak = []
    count = 0
    for s in southerly:
        count = count + 1 if s else 0
        streak.append(count)
    df["wind_southerly_streak"] = streak

    # Days since last smoke event (look backwards)
    smoke_mask = (df["pm25"] > SMOKE_THRESHOLD)
    days_since = []
    last = 999
    for s in smoke_mask:
        if s:
            last = 0
        else:
            last = min(last + 1, 999)
        days_since.append(last)
    df["days_since_last_smoke"] = days_since

    # Drought proxy: rolling 30-day precipitation deficit
    df["drought_proxy"] = (
        df["precipitation_sum"].rolling(30, min_periods=7).mean()
    ).fillna(0)

    return df


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    # PM2.5 history
    "pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    # Precursor dynamics
    "pm25_3d_trend",
    # Fire signals
    "fire_count_total", "fire_frp_sum_total",
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_frp_3d_rolling", "fire_building",
    # Close-range fire
    "fire_count_close", "fire_frp_sum_close",
    # Weather
    "temperature_2m_mean", "temperature_2m_max",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
    "pressure_msl_mean", "precipitation_sum",
    # Episode context
    "wind_southerly_streak", "days_since_last_smoke", "drought_proxy",
    # Calendar
    "month", "day_of_year", "is_fire_season",
]


# ---------------------------------------------------------------------------
# Threshold optimisation: maximise F-beta on validation predictions
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
            best_f = f
            best_t = t
    return best_t, best_f


# ---------------------------------------------------------------------------
# Episode-level recall: did we catch ≥1 alert per smoke episode?
# ---------------------------------------------------------------------------
def episode_recall(y_true, y_pred, dates):
    df_ev = pd.DataFrame({"date": dates, "true": y_true, "pred": y_pred})
    df_ev = df_ev.sort_values("date").reset_index(drop=True)

    # Label contiguous smoke days as one episode
    episodes = []
    ep_id = 0
    in_ep = False
    for _, row in df_ev.iterrows():
        if row["true"] == 1:
            if not in_ep:
                ep_id += 1
                in_ep = True
            episodes.append(ep_id)
        else:
            in_ep = False
            episodes.append(0)
    df_ev["episode"] = episodes

    n_episodes = df_ev[df_ev["episode"] > 0]["episode"].nunique()
    if n_episodes == 0:
        return float("nan"), 0, 0

    caught = df_ev[df_ev["episode"] > 0].groupby("episode")["pred"].max()
    n_caught = int(caught.sum())
    return n_caught / n_episodes, n_caught, n_episodes


# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
def get_models(scale_pos_weight: float):
    return {
        "LightGBM (scale_pos_weight)": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight, random_state=42,
            verbose=-1
        ),
        "XGBoost (scale_pos_weight)": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="logloss", verbosity=0,
        ),
        "BalancedRandomForest": BalancedRandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=2,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)

    # Binary target: is tomorrow a smoke day?
    df["smoke_tomorrow"] = (df["pm25"].shift(-1) > SMOKE_THRESHOLD).astype(int)

    # Drop last row (no tomorrow), drop rows with NaN in features
    df = df[:-1]
    all_cols = BASE_FEATURES + ["smoke_tomorrow", "date"]
    df_model = df[all_cols].dropna().reset_index(drop=True)

    X = df_model[BASE_FEATURES].values
    y = df_model["smoke_tomorrow"].values
    dates = df_model["date"].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"Dataset: {len(df_model)} rows  |  Smoke days: {n_pos}  ({100*n_pos/len(y):.2f}%)")
    print(f"scale_pos_weight = {scale_pos_weight:.1f}\n")

    # Expanding-window CV folds (same as 03_modeling.ipynb)
    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = []
    for cutoff in fold_cutoffs_dt:
        train_mask = df_model["date"] < cutoff
        test_mask  = df_model["date"] >= cutoff
        if train_mask.sum() < 50 or test_mask.sum() < 50:
            continue
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    # Final fold: train on all pre-2020, test on 2020+
    # (already included above via last cutoff = 2020-01-01)

    model_names = list(get_models(scale_pos_weight).keys())
    results = {name: {"auprc": [], "f2": [], "threshold": [],
                       "ep_recall": [], "ep_caught": [], "ep_total": []}
               for name in model_names}

    # Collect all OOF predictions for final PR curve
    oof_proba  = {name: np.full(len(y), np.nan) for name in model_names}
    oof_labels = np.full(len(y), -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        dates_te   = dates[test_idx]

        n_pos_te = y_te.sum()
        print(f"Fold {fold_i+1}: train={len(train_idx)}, test={len(test_idx)}, "
              f"smoke_days_in_test={n_pos_te}")

        oof_labels[test_idx] = y_te

        for name, model in get_models(scale_pos_weight).items():
            model.fit(X_tr, y_tr)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)[:, 1]
            else:
                proba = model.decision_function(X_te)
                proba = (proba - proba.min()) / (proba.max() - proba.min() + 1e-9)

            oof_proba[name][test_idx] = proba

            if n_pos_te == 0:
                print(f"  {name}: no smoke days in test fold — skipping metrics")
                continue

            auprc = average_precision_score(y_te, proba)
            t_opt, f2_opt = best_threshold(y_te, proba)
            preds = (proba >= t_opt).astype(int)
            ep_r, ep_caught, ep_total = episode_recall(y_te, preds, dates_te)

            results[name]["auprc"].append(auprc)
            results[name]["f2"].append(f2_opt)
            results[name]["threshold"].append(t_opt)
            results[name]["ep_recall"].append(ep_r)
            results[name]["ep_caught"].append(ep_caught)
            results[name]["ep_total"].append(ep_total)

            print(f"  {name:<35} AUPRC={auprc:.3f}  F2={f2_opt:.3f}  "
                  f"thresh={t_opt:.3f}  ep_recall={ep_caught}/{ep_total}")
        print()

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print(f"{'Model':<35} {'AUPRC':>7} {'F2':>7} {'Threshold':>10} {'EpRecall':>10}")
    print("-" * 70)
    best_model_name = None
    best_auprc = -1.0
    for name in model_names:
        r = results[name]
        if not r["auprc"]:
            continue
        mean_auprc = np.mean(r["auprc"])
        mean_f2    = np.mean(r["f2"])
        mean_thresh = np.mean(r["threshold"])
        ep_caught_total = sum(r["ep_caught"])
        ep_total_all    = sum(r["ep_total"])
        ep_recall_agg   = ep_caught_total / ep_total_all if ep_total_all else float("nan")

        print(f"{name:<35} {mean_auprc:>7.3f} {mean_f2:>7.3f} {mean_thresh:>10.3f} "
              f"  {ep_caught_total}/{ep_total_all} ({100*ep_recall_agg:.0f}%)")

        if mean_auprc > best_auprc:
            best_auprc = mean_auprc
            best_model_name = name

    print("=" * 70)
    print(f"\nBest model by AUPRC: {best_model_name}")

    # ---------------------------------------------------------------------------
    # Precision-Recall curves (aggregated OOF)
    # ---------------------------------------------------------------------------
    valid_mask = oof_labels >= 0
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_ap = oof_labels[valid_mask].mean()
    ax.axhline(baseline_ap, linestyle="--", color="gray", label=f"Random (AP={baseline_ap:.3f})")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (name, color) in zip(model_names, colors):
        proba = oof_proba[name][valid_mask]
        labels = oof_labels[valid_mask]
        nan_mask = ~np.isnan(proba)
        if nan_mask.sum() < 10:
            continue
        ap = average_precision_score(labels[nan_mask], proba[nan_mask])
        prec, rec, _ = precision_recall_curve(labels[nan_mask], proba[nan_mask])
        ax.plot(rec, prec, color=color, lw=1.8, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Smoke Day Detection\n(OOF across 5 time-series folds)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    out_path = FIGURES_DIR / "smoke_detector_pr_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPR curve saved → {out_path}")

    # ---------------------------------------------------------------------------
    # Calibration note on best model
    # ---------------------------------------------------------------------------
    print(f"\nNote: To deploy {best_model_name}, calibrate probabilities with")
    print("  CalibratedClassifierCV(model, method='isotonic', cv='prefit')")
    print("  and use the mean optimal threshold printed above as the alert cutoff.")


if __name__ == "__main__":
    main()
