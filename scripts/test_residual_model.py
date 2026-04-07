import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Define features used in original modeling
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

ALL_FEATURES = PM25_LAG_FEATURES + FIRE_FEATURES + FIRE_LAG_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES

def run_residual_model():
    DATA_PATH = Path("data/processed/merged/dataset.csv")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    
    # Target: next-day PM2.5 minus today's PM2.5
    # The innovation/change
    df["pm25_target"] = df["pm25"].shift(-1)
    df["pm25_diff"] = df["pm25_target"] - df["pm25"]
    
    df_model = df.iloc[7:-1].dropna().reset_index(drop=True)
    
    X = df_model[ALL_FEATURES]
    y_diff = df_model["pm25_diff"]
    y_actual = df_model["pm25_target"]
    y_today = df_model["pm25"] # Used as baseline persistence
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_y_true = []
    all_y_pred_total = [] # (Predicted Diff + Today's PM2.5)
    all_y_persist = []
    
    print("Running Time-Series CV on Residual Model...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_diff, y_test_diff = y_diff.iloc[train_idx], y_diff.iloc[test_idx]
        y_test_actual = y_actual.iloc[test_idx]
        y_test_today = y_today.iloc[test_idx]
        
        # Train model to predict the DIFF
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01, max_depth=-1, num_leaves=31, random_state=42, verbose=-1)
        model.fit(X_train, y_train_diff)
        
        # Predict the DIFF
        y_pred_diff = model.predict(X_test)
        
        # Predicted next-day PM2.5 = Today + Predicted Change
        y_pred_actual = y_test_today + y_pred_diff
        # Clip negative predictions as PM2.5 can't be negative
        y_pred_actual = np.maximum(y_pred_actual, 0)
        
        all_y_true.extend(y_test_actual)
        all_y_pred_total.extend(y_pred_actual)
        all_y_persist.extend(y_test_today)
        
        fold_mae = mean_absolute_error(y_test_actual, y_pred_actual)
        fold_persist_mae = mean_absolute_error(y_test_actual, y_test_today)
        print(f"Fold {fold+1}: Model MAE = {fold_mae:.4f} | Persistence MAE = {fold_persist_mae:.4f}")

    overall_mae = mean_absolute_error(all_y_true, all_y_pred_total)
    overall_persist_mae = mean_absolute_error(all_y_true, all_y_persist)
    overall_r2 = r2_score(all_y_true, all_y_pred_total)
    persist_r2 = r2_score(all_y_true, all_y_persist)

    print("\n" + "="*50)
    print(f"OVERALL RESULTS:")
    print(f"Residual Model MAE: {overall_mae:.4f}")
    print(f"Persistence MAE   : {overall_persist_mae:.4f}")
    print(f"Model Improvement : {(overall_persist_mae - overall_mae)/overall_persist_mae*100:.2f}%")
    print(f"\nModel R2 : {overall_r2:.4f}")
    print(f"Persist R2: {persist_r2:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_residual_model()
