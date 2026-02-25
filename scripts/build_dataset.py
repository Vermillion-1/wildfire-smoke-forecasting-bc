#!/usr/bin/env python3
"""
Build the merged modeling dataset from processed fire, air quality, and weather data.

Combines all three data sources into a single daily dataset with engineered features
for PM2.5 prediction.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --fire-distance 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "merged"

# Distance bands for fire aggregation (km)
DISTANCE_BANDS = [
    ("close", 0, 200),
    ("medium", 200, 500),
    ("far", 500, 1000),
]

# Lag days for feature engineering
PM25_LAGS = [1, 2, 3, 7]
FIRE_LAGS = [0, 1, 2, 3]


def load_air_quality() -> pd.DataFrame:
    """Load processed daily air quality data."""
    aq_dir = PROCESSED_DIR / "air_quality"
    files = sorted(aq_dir.glob("vancouver_aq_daily_*.csv"))

    if not files:
        print("No air quality files found")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Air quality: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def load_fire_data(max_distance_km: int = 1000) -> pd.DataFrame:
    """Load processed fire data and aggregate to daily features by distance band."""
    fire_dir = PROCESSED_DIR / "fires"
    files = sorted(fire_dir.glob("bc_fires_*.csv"))

    if not files:
        print("No fire data files found")
        return pd.DataFrame()

    # Load all fire detections
    dfs = []
    for f in files:
        # Skip combined multi-year files to avoid double-counting
        name = f.stem
        parts = name.split("_")
        # Pattern: bc_fires_viirs_2015 (single year) vs bc_fires_viirs_2015_2025 (range)
        if len(parts) == 4:  # single year file
            dfs.append(pd.read_csv(f))

    if not dfs:
        # Fall back to loading whatever files exist
        dfs = [pd.read_csv(f) for f in files]

    fires = pd.concat(dfs, ignore_index=True)
    fires["acq_date"] = pd.to_datetime(fires["acq_date"])
    fires = fires[fires["distance_to_vancouver_km"] <= max_distance_km].copy()

    print(f"Fire detections loaded: {len(fires):,} (within {max_distance_km}km)")

    # Aggregate by date and distance band
    daily_features = []

    for date, group in fires.groupby(fires["acq_date"].dt.date):
        row = {"date": pd.Timestamp(date)}

        for band_name, d_min, d_max in DISTANCE_BANDS:
            band = group[
                (group["distance_to_vancouver_km"] >= d_min)
                & (group["distance_to_vancouver_km"] < d_max)
            ]
            row[f"fire_count_{band_name}"] = len(band)
            row[f"fire_frp_sum_{band_name}"] = band["frp"].sum() if "frp" in band.columns else 0
            row[f"fire_frp_mean_{band_name}"] = band["frp"].mean() if "frp" in band.columns and len(band) > 0 else 0

        # Total fire features
        row["fire_count_total"] = len(group)
        if "frp" in group.columns:
            row["fire_frp_sum_total"] = group["frp"].sum()
            row["fire_frp_mean_total"] = group["frp"].mean()
        else:
            row["fire_frp_sum_total"] = 0
            row["fire_frp_mean_total"] = 0

        # Mean distance of all fires
        row["fire_mean_distance_km"] = group["distance_to_vancouver_km"].mean()

        daily_features.append(row)

    if not daily_features:
        print("No daily fire features generated")
        return pd.DataFrame()

    df = pd.DataFrame(daily_features)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Fire features: {len(df)} days with fire activity")
    return df


def load_weather() -> pd.DataFrame:
    """Load processed daily weather data."""
    weather_dir = PROCESSED_DIR / "weather"
    files = sorted(weather_dir.glob("vancouver_weather_daily_*.csv"))

    if not files:
        print("No weather files found")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Weather: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged PM2.5 and fire features."""
    df = df.sort_values("date").reset_index(drop=True)

    # Lagged PM2.5
    if "pm25" in df.columns:
        for lag in PM25_LAGS:
            df[f"pm25_lag{lag}"] = df["pm25"].shift(lag)

    # Lagged fire features
    fire_cols = [c for c in df.columns if c.startswith("fire_")]
    for col in fire_cols:
        for lag in FIRE_LAGS:
            if lag == 0:
                continue  # already have the same-day value
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_fire_season"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    df["year"] = df["date"].dt.year
    return df


def fill_fire_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN fire features with 0 (no fire activity on that day)."""
    fire_cols = [c for c in df.columns if c.startswith("fire_")]
    df[fire_cols] = df[fire_cols].fillna(0)
    return df


def build_dataset(max_distance_km: int = 1000) -> pd.DataFrame:
    """Build the full merged dataset."""
    print("=" * 60)
    print("Building Merged Dataset")
    print("=" * 60)

    # Load all data sources
    aq = load_air_quality()
    fires = load_fire_data(max_distance_km=max_distance_km)
    weather = load_weather()

    if aq.empty:
        print("\nError: Air quality data is required. Run download_air_quality.py first.")
        return pd.DataFrame()

    # Start with air quality as the base (it defines our target variable)
    df = aq.copy()

    # Merge fire data (left join: keep all AQ days, fill missing fire days with 0)
    if not fires.empty:
        df = df.merge(fires, on="date", how="left")
        df = fill_fire_zeros(df)
        print(f"After fire merge: {len(df)} rows")
    else:
        print("Warning: No fire data available")

    # Merge weather data
    if not weather.empty:
        df = df.merge(weather, on="date", how="left")
        print(f"After weather merge: {len(df)} rows")
    else:
        print("Warning: No weather data available")

    # Add engineered features
    df = add_lagged_features(df)
    df = add_calendar_features(df)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Summary
    print(f"\n{'=' * 60}")
    print("Dataset Summary")
    print("=" * 60)
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    if "pm25" in df.columns:
        print(f"\nPM2.5 stats:")
        print(f"  Mean: {df['pm25'].mean():.2f}")
        print(f"  Median: {df['pm25'].median():.2f}")
        print(f"  Max: {df['pm25'].max():.2f}")
        print(f"  Missing: {df['pm25'].isna().sum()} ({df['pm25'].isna().sum() / len(df) * 100:.1f}%)")

        # Smoke events (PM2.5 > 25 ug/m3)
        smoke_days = (df["pm25"] > 25).sum()
        print(f"  Smoke event days (>25 ug/m3): {smoke_days} ({smoke_days / len(df) * 100:.1f}%)")

    print(f"\nColumns: {list(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build merged dataset for PM2.5 prediction")
    parser.add_argument(
        "--fire-distance", type=int, default=1000,
        help="Max fire distance from Vancouver in km (default: 1000)"
    )
    args = parser.parse_args()

    df = build_dataset(max_distance_km=args.fire_distance)

    if df.empty:
        print("\nDataset is empty. Make sure all download scripts have been run first.")
        return

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
