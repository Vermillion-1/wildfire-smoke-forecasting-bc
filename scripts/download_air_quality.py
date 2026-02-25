#!/usr/bin/env python3
"""
Download air quality data for Vancouver from OpenAQ v3 API.

Usage:
    python download_air_quality.py
    python download_air_quality.py --start-date 2022-01-01 --end-date 2023-12-31
    python download_air_quality.py --parameters pm25 pm10 --radius 100
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
from tqdm import tqdm


# CONFIGURATION
OPENAQ_API_KEY = "79b5f9033fb6499dc115c3e7ba1289c73ce42d72acc1f032d72a33034dba4a18"
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# Project root and output directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "air_quality"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "air_quality"

VANCOUVER_LAT = 49.2827
VANCOUVER_LON = -123.1207
SEARCH_RADIUS_KM = 25  # Max 25km for OpenAQ API

PARAM_NAMES = ["pm25", "pm10", "o3", "no2", "so2", "co"]


def get_locations(radius_km: int = SEARCH_RADIUS_KM) -> List[dict]:
    """Get monitoring stations near Vancouver."""
    print(f"\nFinding stations within {radius_km}km of Vancouver...")

    headers = {"X-API-Key": OPENAQ_API_KEY}
    url = f"{OPENAQ_BASE_URL}/locations"
    
    params = {
        "coordinates": f"{VANCOUVER_LAT},{VANCOUVER_LON}",
        "radius": radius_km * 1000,
        "limit": 100,
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            print("No stations found")
            return []

        locations = data["results"]
        print(f"Found {len(locations)} stations")

        for loc in locations[:5]:
            name = loc.get("name", "Unknown")
            distance = loc.get("distance", 0) / 1000
            print(f"  - {name} ({distance:.1f}km)")

        return locations

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []


def get_sensor_for_parameter(location: dict, parameter: str) -> int:
    """Get sensor ID for a specific parameter from a location."""
    sensors = location.get("sensors", [])
    
    for sensor in sensors:
        param_info = sensor.get("parameter", {})
        if param_info.get("name") == parameter:
            return sensor.get("id")
    
    return None


def download_sensor_data(
    sensor_id: int, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Download measurements for a sensor."""
    headers = {"X-API-Key": OPENAQ_API_KEY}
    
    all_measurements = []
    page = 1
    
    while True:
        url = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements"
        params = {
            "datetime_from": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 1000,
            "page": page,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "results" not in data or not data["results"]:
                break

            all_measurements.extend(data["results"])
            
            # Check if there are more pages
            meta = data.get("meta", {})
            found = meta.get("found", "0")
            if isinstance(found, str):
                found = int(found.replace(">", ""))
            
            if len(all_measurements) >= found or page >= 100:
                break
                
            page += 1
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading sensor {sensor_id}: {e}")
            break

    if not all_measurements:
        return pd.DataFrame()

    # Parse measurements
    records = []
    for m in all_measurements:
        period = m.get("period", {})
        datetime_from = period.get("datetimeFrom", {})
        utc_time = datetime_from.get("utc") if isinstance(datetime_from, dict) else None
        
        records.append({
            "date": utc_time,
            "value": m.get("value"),
            "parameter": m.get("parameter", {}).get("name"),
            "sensor_id": sensor_id,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    return df


def download_all_data(
    start_date: datetime, end_date: datetime, parameters: List[str], radius_km: int
) -> pd.DataFrame:
    """Download data from all stations."""
    print(f"\n{'='*60}")
    print(f"Downloading Air Quality Data")
    print(f"{'='*60}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Parameters: {', '.join(parameters)}")

    locations = get_locations(radius_km)
    if not locations:
        return pd.DataFrame()

    all_data = []

    for location in tqdm(locations, desc="Stations"):
        loc_name = location.get("name", "Unknown")
        
        for param in parameters:
            sensor_id = get_sensor_for_parameter(location, param)
            if sensor_id:
                df = download_sensor_data(sensor_id, start_date, end_date)
                if len(df) > 0:
                    df["location_name"] = loc_name
                    all_data.append(df)

    if not all_data:
        print("\nNo data found for the specified parameters and date range.")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal measurements: {len(combined):,}")
    return combined


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily values."""
    print("\nProcessing data (daily aggregation)...")

    df["date"] = pd.to_datetime(df["date"])
    df["date_local"] = df["date"].dt.tz_convert("America/Vancouver")
    df["date_only"] = df["date_local"].dt.date

    agg = (
        df.groupby(["date_only", "parameter"])
        .agg({"value": ["mean", "min", "max", "count"]})
        .reset_index()
    )

    # Flatten column names
    agg.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in agg.columns.values]
    agg = agg.rename(
        columns={
            "date_only": "date",
            "value_mean": "mean",
            "value_min": "min",
            "value_max": "max",
            "value_count": "count",
        }
    )

    pivot = agg.pivot_table(index="date", columns="parameter", values="mean").reset_index()
    pivot.columns.name = None

    print(f"Aggregated to {len(pivot):,} days")
    return pivot


def save_data(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> Path:
    """Save raw and processed data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = OUTPUT_DIR / f"vancouver_aq_raw_{date_range}_{timestamp}.csv"
    df.to_csv(raw_file, index=False)
    print(f"\nSaved raw: {raw_file}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed = process_data(df)
    processed_file = PROCESSED_DIR / f"vancouver_aq_daily_{date_range}.csv"
    processed.to_csv(processed_file, index=False)
    print(f"Saved processed: {processed_file}")

    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    print(f"Records: {len(processed):,} days")
    print(f"Date range: {processed['date'].min()} to {processed['date'].max()}")

    for col in processed.columns:
        if col != "date" and processed[col].dtype in ["float64", "int64"]:
            print(f"\n{col.upper()}:")
            print(f"  Mean: {processed[col].mean():.2f}")
            print(f"  Min: {processed[col].min():.2f}")
            print(f"  Max: {processed[col].max():.2f}")
            print(f"  Missing: {processed[col].isna().sum()} ({processed[col].isna().sum() / len(processed) * 100:.1f}%)")

    return processed_file


def main():
    parser = argparse.ArgumentParser(description="Download Vancouver air quality data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--parameters", nargs="+", default=["pm25"], choices=PARAM_NAMES,
        help="Parameters to download (default: pm25)"
    )
    parser.add_argument("--radius", type=int, default=SEARCH_RADIUS_KM, help="Search radius in km (max 25)")
    parser.add_argument("--list-stations", action="store_true", help="List stations and exit")
    args = parser.parse_args()

    # Clamp radius to max 25km
    radius = min(args.radius, 25)

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else end_date - timedelta(days=90)

    if args.list_stations:
        get_locations(radius)
        return

    df = download_all_data(start_date, end_date, args.parameters, radius)

    if len(df) == 0:
        print("\nNo data downloaded. Try:")
        print("  1. Expanding radius (--radius 25)")
        print("  2. Changing date range")
        print("  3. Listing stations (--list-stations)")
        sys.exit(1)

    save_data(df, start_date, end_date)

    print(f"\nComplete!")
    print(f"  Raw: {OUTPUT_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
