#!/usr/bin/env python3
"""
Download verified PM2.5 data from BC Government FTP (Annual Summary).

Source: BC Ministry of Environment - Verified Hourly Air Quality Data
FTP: ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/
Catalogue: https://catalogue.data.gov.bc.ca/dataset/77eeadf4-0c19-48bf-a47a-fa9eef01f409

Also fetches recent (last 30 days) unverified data from HTTP endpoint to
supplement years not yet validated.

Data format (CSV):
  DATE_PST, STATION_NAME, RAW_VALUE, REPORTED_VALUE, INSTRUMENT,
  UNITS, PARAMETER, EMS_ID, LATITUDE, LONGITUDE

Usage:
    python scripts/download_bc_air_quality.py
    python scripts/download_bc_air_quality.py --start-year 2020 --end-year 2024
    python scripts/download_bc_air_quality.py --recent-only
"""

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "air_quality"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "air_quality"

FTP_BASE = "ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary"
RECENT_URL = (
    "https://envistaweb.env.gov.bc.ca/aqo/csv/"
    "Hourly_Raw_Air_Data/Air_Quality/PM25.csv"
)
YEAR_TO_DATE_FTP = (
    "ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/"
    "Hourly_Raw_Air_Data/Year_to_Date/PM25.csv"
)

VANCOUVER_LAT = 49.2827
VANCOUVER_LON = -123.1207

# Metro Vancouver / Lower Fraser Valley stations we want
# (these have long-running PM2.5 records near Vancouver)
METRO_VANCOUVER_STATIONS = [
    "Burnaby Kensington Park",
    "Burnaby South",
    "North Vancouver Mahon Park",
    "North Vancouver Second Narrows",
    "Vancouver International Airport #2",
    "Vancouver Clark Drive",
    "Vancouver Robson Square",
    "Richmond South",
    "North Delta",
    "Surrey East",
    "Coquitlam Douglas College",
    "Port Moody Rocky Point Park",
    "New Westminster Sapperton Park",
    "Pitt Meadows Meadowlands School",
    "Langley Central",
    "Maple Ridge Golden Ears School",
    "Tsawwassen",
    "Horseshoe Bay",
    "Burnaby North Eton",
    "Burnaby Burmount",
    "Chilliwack Airport",
    "Abbotsford A Columbia Street",
    "Abbotsford Central",
    "Hope Airport",
    "Agassiz Centennial Park",
]


def download_ftp_file(url: str, desc: str = "", timeout: int = 600) -> bytes:
    """Download a file from FTP with retries and progress."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = urllib.request.urlopen(url, timeout=timeout)
            # Try to get content length
            length = resp.headers.get("Content-Length")
            total = int(length) if length else None

            chunks = []
            downloaded = 0
            chunk_size = 1024 * 256  # 256KB chunks

            pbar = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=desc or url.split("/")[-1],
                disable=False,
            )

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))

            pbar.close()
            resp.close()
            return b"".join(chunks)

        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def download_http_file(url: str, desc: str = "") -> bytes:
    """Download a file via HTTP (for recent data)."""
    import requests

    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0)) or None
    chunks = []
    pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc or "Download")

    for chunk in r.iter_content(chunk_size=256 * 1024):
        chunks.append(chunk)
        pbar.update(len(chunk))

    pbar.close()
    return b"".join(chunks)


def filter_metro_vancouver(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Metro Vancouver stations."""
    mask = df["STATION_NAME"].isin(METRO_VANCOUVER_STATIONS)
    filtered = df[mask].copy()
    return filtered


def process_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly PM2.5 to daily values.

    Creates a city-level daily average by averaging across all Metro
    Vancouver stations, plus keeps per-station values for reference.
    """
    df = df.copy()
    df["DATE_PST"] = pd.to_datetime(df["DATE_PST"])
    df["date"] = df["DATE_PST"].dt.date

    # Handle different column names from FTP vs HTTP sources
    if "ROUNDED_VALUE" in df.columns:
        value_col = "ROUNDED_VALUE"
    elif "REPORTED_VALUE" in df.columns:
        value_col = "REPORTED_VALUE"
    elif "RAW_VALUE" in df.columns:
        value_col = "RAW_VALUE"
    else:
        raise ValueError(f"No value column found. Columns: {list(df.columns)}")

    df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["value"])

    # City-level daily aggregation (average across all stations)
    city_daily = (
        df.groupby("date")
        .agg(
            pm25_mean=("value", "mean"),
            pm25_median=("value", "median"),
            pm25_max=("value", "max"),
            pm25_min=("value", "min"),
            pm25_std=("value", "std"),
            n_readings=("value", "count"),
            n_stations=("STATION_NAME", "nunique"),
        )
        .reset_index()
    )
    city_daily["date"] = pd.to_datetime(city_daily["date"])

    return city_daily


def download_year(year: int) -> pd.DataFrame:
    """Download and filter PM2.5 data for a single year from FTP."""
    url = f"{FTP_BASE}/{year}/PM25.csv"
    print(f"\n── Downloading {year} PM2.5 data ──")

    try:
        raw = download_ftp_file(url, desc=f"PM25 {year}")
    except Exception as e:
        print(f"  FAILED to download {year}: {e}")
        return pd.DataFrame()

    # Parse CSV
    df = pd.read_csv(io.BytesIO(raw))
    total_rows = len(df)
    print(f"  Total rows (all BC stations): {total_rows:,}")

    # Filter to Metro Vancouver
    df = filter_metro_vancouver(df)
    print(f"  Metro Vancouver rows: {len(df):,}")
    print(f"  Stations found: {df['STATION_NAME'].nunique()}")

    if len(df) > 0:
        # Save raw filtered data
        raw_file = RAW_DIR / f"bc_pm25_metro_vancouver_{year}.csv"
        df.to_csv(raw_file, index=False)
        print(f"  Saved raw: {raw_file.name}")

    return df


def download_recent() -> pd.DataFrame:
    """Download recent (last 30 days) unverified data via HTTP."""
    print("\n── Downloading recent PM2.5 data (HTTP) ──")
    try:
        raw = download_http_file(RECENT_URL, desc="Recent PM25")
        df = pd.read_csv(io.BytesIO(raw))
        total = len(df)
        df = filter_metro_vancouver(df)
        print(f"  Total rows: {total:,}, Metro Vancouver: {len(df):,}")
        return df
    except Exception as e:
        print(f"  FAILED: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Download PM2.5 data from BC Gov for Metro Vancouver"
    )
    parser.add_argument(
        "--start-year", type=int, default=2015, help="First year (default: 2015)"
    )
    parser.add_argument(
        "--end-year", type=int, default=2024, help="Last year (default: 2024)"
    )
    parser.add_argument(
        "--recent-only",
        action="store_true",
        help="Only download recent (last 30 days) data",
    )
    parser.add_argument(
        "--include-recent",
        action="store_true",
        help="Also download recent unverified data",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BC Gov PM2.5 Data Download")
    print("=" * 60)
    print(f"Stations: {len(METRO_VANCOUVER_STATIONS)} Metro Vancouver sites")

    all_dfs = []

    if args.recent_only:
        df = download_recent()
        if len(df) > 0:
            all_dfs.append(df)
    else:
        # Download verified annual data
        years = list(range(args.start_year, args.end_year + 1))
        print(f"Years: {args.start_year}–{args.end_year}")

        for year in years:
            df = download_year(year)
            if len(df) > 0:
                all_dfs.append(df)

        if args.include_recent:
            df = download_recent()
            if len(df) > 0:
                all_dfs.append(df)

    if not all_dfs:
        print("\nERROR: No data downloaded.")
        sys.exit(1)

    # Combine all data
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["DATE_PST"] = pd.to_datetime(combined["DATE_PST"])
    combined = combined.sort_values("DATE_PST").drop_duplicates(
        subset=["DATE_PST", "STATION_NAME"], keep="last"
    )
    print(f"\n{'='*60}")
    print(f"Combined: {len(combined):,} hourly readings")
    print(f"Date range: {combined['DATE_PST'].min()} to {combined['DATE_PST'].max()}")
    print(f"Stations: {combined['STATION_NAME'].nunique()}")

    # Save combined raw
    raw_combined = RAW_DIR / "bc_pm25_metro_vancouver_combined.csv"
    combined.to_csv(raw_combined, index=False)
    print(f"Saved combined raw: {raw_combined}")

    # Process to daily
    daily = process_to_daily(combined)

    date_range = f"{daily['date'].min().strftime('%Y%m%d')}_{daily['date'].max().strftime('%Y%m%d')}"
    daily_file = PROCESSED_DIR / f"vancouver_pm25_daily_{date_range}.csv"
    daily.to_csv(daily_file, index=False)
    print(f"Saved daily processed: {daily_file}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print("=" * 60)
    print(f"Total days: {len(daily):,}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"PM2.5 mean: {daily['pm25_mean'].mean():.2f} µg/m³")
    print(f"PM2.5 max:  {daily['pm25_max'].max():.2f} µg/m³")
    print(f"Days > 25 µg/m³ (smoke events): {(daily['pm25_mean'] > 25).sum()}")

    # Check coverage
    full_range = pd.date_range(daily["date"].min(), daily["date"].max())
    missing = len(full_range) - len(daily)
    print(f"Missing days: {missing} ({missing/len(full_range)*100:.1f}%)")

    # Per-year breakdown
    daily["year"] = daily["date"].dt.year
    print("\nDays per year:")
    for year, count in daily.groupby("year").size().items():
        print(f"  {year}: {count}")

    print(f"\nComplete!")


if __name__ == "__main__":
    main()
