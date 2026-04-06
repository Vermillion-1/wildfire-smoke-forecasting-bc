"""
Identify which smoke episodes the XGBoost gate misses.
Helps determine if missed episodes are fixable (post-2015, enough training data)
or structural (pre-2004, data too sparse to learn from).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

DATA_PATH       = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD = 25.0
FOLD_CUTOFFS    = ["2004-01-01", "2008-01-01", "2012-01-01", "2016-01-01", "2020-01-01"]

GATE_FEATURES = [
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
    "pm25_3d_trend", "fire_frp_3d_rolling", "fire_building",
    "wind_southerly_streak", "days_since_last_smoke", "drought_proxy",
]


def engineer_features(df):
    df = df.copy().sort_values("date")
    df["pm25_3d_trend"]       = df["pm25"].diff(3)
    df["fire_frp_3d_rolling"] = df["fire_frp_sum_total"].rolling(3, min_periods=1).sum()
    df["fire_building"]       = (df["fire_count_total"] > df["fire_count_total"].shift(1)).astype(int)
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


def label_episodes(smoke_series, dates):
    """Returns list of (episode_id, start_date, end_date, n_days)."""
    episodes = []
    ep_id, in_ep, start = 0, False, None
    for i, (s, d) in enumerate(zip(smoke_series, dates)):
        if s and not in_ep:
            ep_id += 1; in_ep = True; start = d
        elif not s and in_ep:
            episodes.append((ep_id, start, dates[i-1]))
            in_ep = False
    if in_ep:
        episodes.append((ep_id, start, dates[-1]))
    return episodes


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = engineer_features(df)
    df["smoke_tomorrow"] = (df["pm25"].shift(-1) > SMOKE_THRESHOLD).astype(int)

    df_model = df.iloc[:-1].dropna(subset=GATE_FEATURES + ["smoke_tomorrow"]).reset_index(drop=True)
    X    = df_model[GATE_FEATURES].values
    y    = df_model["smoke_tomorrow"].values
    dates = pd.to_datetime(df_model["date"].values)

    n_pos = y.sum()
    spw   = (len(y) - n_pos) / max(n_pos, 1)

    fold_cutoffs_dt = pd.to_datetime(FOLD_CUTOFFS)
    folds = [(np.where(dates < c)[0], np.where(dates >= c)[0])
             for c in fold_cutoffs_dt
             if (dates < c).sum() >= 50 and (dates >= c).sum() >= 50]

    # OOF predictions
    oof_proba = np.full(len(y), np.nan)
    oof_pred  = np.full(len(y), -1, dtype=int)

    for tr, te in folds:
        gate = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=spw, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        gate.fit(X[tr], y[tr])
        proba = gate.predict_proba(X[te])[:, 1]
        oof_proba[te] = proba

        # Optimal threshold from training fold (F2-optimised, ~0.043 from train_smoke_detector)
        THRESHOLD = 0.043
        oof_pred[te] = (proba >= THRESHOLD).astype(int)

    # Only evaluate on test windows
    in_test = oof_pred >= 0
    y_test       = y[in_test]
    pred_test    = oof_pred[in_test]
    dates_test   = dates[in_test]
    proba_test   = oof_proba[in_test]

    # Label episodes in test window using ACTUAL smoke (tomorrow is smoke day)
    episodes = label_episodes(y_test, dates_test)

    print(f"Total smoke episodes in OOF test windows: {len(episodes)}")
    print(f"Threshold used: 0.043\n")

    caught, missed_list = [], []
    for ep_id, start, end in episodes:
        ep_mask = (dates_test >= start) & (dates_test <= end)
        ep_preds = pred_test[ep_mask]
        ep_proba = proba_test[ep_mask]
        n_days   = ep_mask.sum()
        n_alerts = ep_preds.sum()
        max_p    = ep_proba.max()

        status = "CAUGHT" if n_alerts > 0 else "MISSED"
        row = {
            "episode": ep_id, "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"), "n_days": n_days,
            "alerts": n_alerts, "max_p": max_p, "status": status
        }
        (caught if n_alerts > 0 else missed_list).append(row)

    print(f"Caught: {len(caught)} / {len(episodes)}   Missed: {len(missed_list)}\n")

    print("── MISSED EPISODES ─────────────────────────────────────────────────")
    print(f"{'#':>3}  {'Start':>12}  {'End':>12}  {'Days':>5}  {'Max p':>7}  Notes")
    print("-" * 65)
    for r in missed_list:
        yr = int(r["start"][:4])
        note = "pre-2010 (sparse train data)" if yr < 2010 else "post-2010 — potentially fixable"
        print(f"{r['episode']:>3}  {r['start']:>12}  {r['end']:>12}  "
              f"{r['n_days']:>5}  {r['max_p']:>7.4f}  {note}")

    print()
    print("── CAUGHT EPISODES (for reference) ─────────────────────────────────")
    print(f"{'#':>3}  {'Start':>12}  {'End':>12}  {'Days':>5}  {'Alerts':>7}  {'Max p':>7}")
    print("-" * 60)
    for r in caught:
        print(f"{r['episode']:>3}  {r['start']:>12}  {r['end']:>12}  "
              f"{r['n_days']:>5}  {r['alerts']:>7}  {r['max_p']:>7.4f}")

    # Summary by era
    print("\n── Missed by era ───────────────────────────────────────────────────")
    for era_label, yr_range in [("Pre-2010", (0, 2010)), ("2010-2017", (2010, 2017)), ("2017+", (2017, 9999))]:
        era_missed  = [r for r in missed_list if yr_range[0] <= int(r["start"][:4]) < yr_range[1]]
        era_total   = [r for r in (caught + missed_list) if yr_range[0] <= int(r["start"][:4]) < yr_range[1]]
        print(f"  {era_label:>12}: {len(era_total) - len(era_missed)}/{len(era_total)} caught  "
              f"({len(era_missed)} missed)")


if __name__ == "__main__":
    main()
