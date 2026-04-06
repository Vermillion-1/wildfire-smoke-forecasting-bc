import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

def train_gate():
    DATA_PATH = Path("data/processed/merged/dataset.csv")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    
    # Define Anomaly (Effect-based for target, cause-based for features)
    # Target: Is tomorrow a Smoke Day (PM2.5 > 25)?
    df['is_smoke_tomorrow'] = (df['pm25'].shift(-1) > 25).astype(int)
    
    # Features group
    PM25_LAG_FEATURES = ["pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3"]
    FIRE_FEATURES = ["fire_count_total", "fire_frp_sum_total", "fire_mean_distance_km"]
    WEATHER_FEATURES = ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean", "precipitation_sum", "pressure_msl_mean"]
    
    ALL_FEATURES = PM25_LAG_FEATURES + FIRE_FEATURES + WEATHER_FEATURES
    
    # Clean data
    df_model = df.dropna(subset=['is_smoke_tomorrow'] + ALL_FEATURES).reset_index(drop=True)
    
    X = df_model[ALL_FEATURES]
    y = df_model['is_smoke_tomorrow']
    
    print(f"Dataset Size: {len(df_model)}")
    print(f"Anomaly (Smoke) Days in dataset: {y.sum()}")
    
    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Use Class Weight for imbalance (Smoke days are rare)
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        
        print(f"\nFold {fold} Results:")
        print(classification_report(y_test, preds))
        fold += 1

    # Save final model state (conceptual, usually would use joblib)
    print("\nAnomaly Gate training complete.")

if __name__ == "__main__":
    train_gate()
