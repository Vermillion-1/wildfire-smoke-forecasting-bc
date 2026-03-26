#!/usr/bin/env python3
"""
Download historical NASA FIRMS wildfire data for British Columbia.

Usage:
    python download_historical_fires.py --year 2023
    python download_historical_fires.py --start-year 2018 --end-year 2023
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from math import asin, cos, radians, sin, sqrt
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm


# CONFIGURATION
# =============
# NASA FIRMS API key (for future use with real-time API)
FIRMS_API_KEY = "334480ce6c3ae25635f5443f42ab39bd"

# Project root and output directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "fires"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "fires"

# Geographic boundaries
BC_BBOX = {"lat_min": 48.0, "lat_max": 60.0, "lon_min": -130.0, "lon_max": -114.0}
VANCOUVER_LAT = 49.2827
VANCOUVER_LON = -123.1207


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def download_year(year: int, sensor: str = "VIIRS") -> Optional[Path]:
    """Download fire data for a specific year and sensor."""
    # Validate year ranges
    if sensor == "VIIRS" and year < 2012:
        print("Error: VIIRS data not available before 2012")
        return None
    if sensor == "MODIS" and year < 2000:
        print("Error: MODIS data not available before 2000")
        return None

    # Build URL and filename
    sensor_slug = "viirs-snpp" if sensor == "VIIRS" else "modis"
    filename = f"{sensor_slug}_{year}_Canada.csv"
    url = f"https://firms.modaps.eosdis.nasa.gov/data/country/{sensor_slug}/{year}/{filename}"
    output_file = OUTPUT_DIR / filename

    # Skip if already downloaded
    if output_file.exists():
        print(f"✓ Already downloaded: {output_file}")
        return output_file

    # Download with progress bar
    print(f"\nDownloading {sensor} data for {year}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(output_file, "wb") as f, tqdm(
            desc=filename, total=total_size, unit="B", unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✓ Saved: {output_file}")
        return output_file

    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading: {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def filter_bc_data(csv_file: Path) -> pd.DataFrame:
    """Load CSV and filter to BC region only."""
    df = pd.read_csv(csv_file)
    print(f"  Total records: {len(df):,}")

    bc_fires = df[
        (df["latitude"] >= BC_BBOX["lat_min"])
        & (df["latitude"] <= BC_BBOX["lat_max"])
        & (df["longitude"] >= BC_BBOX["lon_min"])
        & (df["longitude"] <= BC_BBOX["lon_max"])
    ].copy()

    print(f"  Records in BC: {len(bc_fires):,}")
    return bc_fires


def add_vancouver_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance from each fire to Vancouver."""
    df["distance_to_vancouver_km"] = df.apply(
        lambda row: haversine_distance(
            row["latitude"], row["longitude"], VANCOUVER_LAT, VANCOUVER_LON
        ),
        axis=1,
    )
    return df


def process_year(year: int, sensor: str = "VIIRS") -> pd.DataFrame:
    """Download and process fire data for a single year."""
    print(f"\n{'='*60}")
    print(f"Processing {year} ({sensor})")
    print("=" * 60)

    # Download
    csv_file = download_year(year, sensor)
    if csv_file is None:
        return pd.DataFrame()

    # Filter to BC
    df = filter_bc_data(csv_file)
    if len(df) == 0:
        print("No fires found in BC region")
        return df

    # Add distance and metadata
    df = add_vancouver_distance(df)
    df["year"] = year
    df["sensor"] = sensor

    # Save processed file
    output_file = PROCESSED_DIR / f"bc_fires_{sensor.lower()}_{year}.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Processed: {output_file}")

    # Print summary
    print_summary(df, year)
    return df


def print_summary(df: pd.DataFrame, year: int):
    """Print year summary statistics."""
    print(f"\nSummary for {year}:")
    print(f"  Total fires: {len(df):,}")

    df["acq_date"] = pd.to_datetime(df["acq_date"])
    print(f"  Date range: {df['acq_date'].min().date()} to {df['acq_date'].max().date()}")

    # Peak month
    df["month"] = df["acq_date"].dt.month
    monthly_counts = df.groupby("month").size()
    peak_month = monthly_counts.idxmax()
    peak_month_name = datetime(2000, peak_month, 1).strftime("%B")
    print(f"  Peak month: {peak_month_name} ({monthly_counts[peak_month]:,} fires)")

    # Distance to Vancouver
    within_500km = (df["distance_to_vancouver_km"] <= 500).sum()
    print(f"  Within 500km of Vancouver: {within_500km:,} ({within_500km/len(df)*100:.1f}%)")
    print(f"  Mean distance: {df['distance_to_vancouver_km'].mean():.1f} km")

    # Confidence levels
    if "confidence" in df.columns:
        print(f"  Confidence:")
        for conf, count in df["confidence"].value_counts().items():
            print(f"    {conf}: {count:,} ({count/len(df)*100:.1f}%)")


def process_year_range(start_year: int, end_year: int, sensor: str = "VIIRS") -> pd.DataFrame:
    """Download and process data for a range of years."""
    all_data = []

    for year in range(start_year, end_year + 1):
        year_df = process_year(year, sensor)
        if len(year_df) > 0:
            all_data.append(year_df)

    if not all_data:
        print("\nNo data collected")
        return pd.DataFrame()

    # Combine and save
    combined = pd.concat(all_data, ignore_index=True)
    output_file = PROCESSED_DIR / f"bc_fires_{sensor.lower()}_{start_year}_{end_year}.csv"
    combined.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {start_year}-{end_year}")
    print("=" * 60)
    print(f"Total fires: {len(combined):,}")
    print(f"Date range: {combined['acq_date'].min()} to {combined['acq_date'].max()}")
    print(f"Saved: {output_file}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Download NASA FIRMS fire data for BC")
    parser.add_argument("--year", type=int, help="Download specific year")
    parser.add_argument("--start-year", type=int, help="Start year for range")
    parser.add_argument("--end-year", type=int, help="End year for range")
    parser.add_argument(
        "--sensor", choices=["VIIRS", "MODIS"], default="VIIRS",
        help="Sensor (VIIRS: 2012+, MODIS: 2000+)"
    )
    args = parser.parse_args()

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NASA FIRMS Historical Fire Data Downloader")
    print("=" * 60)

    # Download based on arguments
    if args.year:
        process_year(args.year, sensor=args.sensor)
    elif args.start_year and args.end_year:
        if args.start_year > args.end_year:
            print("Error: start-year must be <= end-year")
            sys.exit(1)
        process_year_range(args.start_year, args.end_year, sensor=args.sensor)
    else:
        # Default: last 3 years
        current_year = datetime.now().year
        start_year = current_year - 2
        print(f"\nNo year specified. Downloading {start_year}-{current_year}...")
        process_year_range(start_year, current_year, sensor=args.sensor)

    print(f"\n✓ Complete!")
    print(f"  Raw files: {OUTPUT_DIR}")
    print(f"  Processed files: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
