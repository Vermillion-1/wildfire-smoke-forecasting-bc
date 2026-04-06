#!/usr/bin/env python3
"""
Download historical weather data for Vancouver from Open-Meteo API

This script fetches historical weather data (temperature, wind, humidity, etc.)
from the Open-Meteo API for Vancouver area.

Open-Meteo provides free access to historical weather data with no API key required.

Usage:
    # Download last 90 days (default)
    python scripts/download_weather.py

    # Download specific date range
    python scripts/download_weather.py --start-date 2022-01-01 --end-date 2023-12-31

    # Download specific variables
    python scripts/download_weather.py --variables temperature wind_speed --start-date 2022-01-01
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "weather"

# Open-Meteo API
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Vancouver coordinates
VANCOUVER_LAT = 49.2827
VANCOUVER_LON = -123.1207

# Available weather variables
AVAILABLE_VARIABLES = {
    "temperature_2m": "Temperature at 2m (°C)",
    "relative_humidity_2m": "Relative Humidity at 2m (%)",
    "dew_point_2m": "Dew Point at 2m (°C)",
    "precipitation": "Precipitation (mm)",
    "rain": "Rain (mm)",
    "snowfall": "Snowfall (cm)",
    "cloud_cover": "Cloud Cover (%)",
    "pressure_msl": "Sea Level Pressure (hPa)",
    "surface_pressure": "Surface Pressure (hPa)",
    "wind_speed_10m": "Wind Speed at 10m (km/h)",
    "wind_speed_100m": "Wind Speed at 100m (km/h)",
    "wind_direction_10m": "Wind Direction at 10m (°)",
    "wind_direction_100m": "Wind Direction at 100m (°)",
    "wind_gusts_10m": "Wind Gusts at 10m (km/h)",
}

# Default variables for wildfire smoke prediction
DEFAULT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
]


class OpenMeteoDownloader:
    """Download weather data from Open-Meteo API"""

    def __init__(self):
        self.base_url = OPENMETEO_BASE_URL
        self.session = requests.Session()

        # Create output directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def download_weather_data(
        self,
        start_date: datetime,
        end_date: datetime,
        variables: List[str] = None,
        latitude: float = VANCOUVER_LAT,
        longitude: float = VANCOUVER_LON,
    ) -> pd.DataFrame:
        """
        Download weather data from Open-Meteo API

        Args:
            start_date: Start date
            end_date: End date
            variables: List of variables to download
            latitude: Latitude
            longitude: Longitude

        Returns:
            DataFrame with weather data
        """
        if variables is None:
            variables = DEFAULT_VARIABLES

        print(f"\n{'=' * 70}")
        print(f"🌤️  Downloading Weather Data for Vancouver")
        print(f"{'=' * 70}")
        print(f"Location: {latitude:.4f}°N, {longitude:.4f}°W")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Variables: {len(variables)}")
        for var in variables:
            desc = AVAILABLE_VARIABLES.get(var, var)
            print(f"   - {desc}")
        print()

        # Open-Meteo has a limit on date ranges per request (typically 1 year)
        # Split into chunks if needed
        all_data = []
        current_start = start_date

        with tqdm(total=(end_date - start_date).days, desc="Downloading") as pbar:
            while current_start < end_date:
                # Download up to 1 year at a time
                current_end = min(current_start + timedelta(days=365), end_date)

                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": current_start.strftime("%Y-%m-%d"),
                    "end_date": current_end.strftime("%Y-%m-%d"),
                    "hourly": ",".join(variables),
                    "timezone": "America/Vancouver",
                }

                try:
                    response = self.session.get(
                        self.base_url, params=params, timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()

                    if "hourly" in data:
                        # Convert to DataFrame
                        hourly_data = data["hourly"]
                        df_chunk = pd.DataFrame(hourly_data)

                        # Parse time
                        df_chunk["time"] = pd.to_datetime(df_chunk["time"])

                        all_data.append(df_chunk)

                        pbar.update((current_end - current_start).days)

                    else:
                        print(f"⚠️  No data returned for {current_start.date()}")

                except requests.exceptions.RequestException as e:
                    print(f"❌ Error downloading data: {e}")
                    break

                # Move to next chunk
                current_start = current_end + timedelta(days=1)

                # Be nice to the API
                time.sleep(0.5)

        if not all_data:
            print("❌ No data downloaded")
            return pd.DataFrame()

        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates (if any overlap in chunks)
        combined_df = combined_df.drop_duplicates(subset=["time"])
        combined_df = combined_df.sort_values("time").reset_index(drop=True)

        print(f"\n✅ Downloaded {len(combined_df):,} hourly records")

        return combined_df

    def process_data(
        self, df: pd.DataFrame, temporal_resolution: str = "daily"
    ) -> pd.DataFrame:
        """
        Process and aggregate weather data

        Args:
            df: Raw weather DataFrame
            temporal_resolution: 'hourly' or 'daily'

        Returns:
            Processed DataFrame
        """
        if len(df) == 0:
            return df

        print(f"\n🔄 Processing data (resolution: {temporal_resolution})...")

        if temporal_resolution == "hourly":
            # Already hourly, just add date column
            df["date"] = df["time"].dt.date
            return df

        # Daily aggregation
        df["date"] = df["time"].dt.date

        # Group by date and aggregate
        # For most variables: use mean
        # For precipitation/rain/snow: use sum
        # For wind direction: use circular mean (approximation with mode)

        agg_dict = {}
        for col in df.columns:
            if col in ["time", "date"]:
                continue

            if "precipitation" in col or "rain" in col or "snowfall" in col:
                # Sum for precipitation
                agg_dict[col] = ["sum", "max"]
            elif "wind_direction" in col:
                # Mode for wind direction (most common direction)
                agg_dict[col] = [
                    "mean",
                    lambda x: x.mode()[0] if not x.mode().empty else x.mean(),
                ]
            elif "wind_speed" in col or "wind_gusts" in col:
                # Mean and max for wind speed
                agg_dict[col] = ["mean", "max"]
            else:
                # Mean, min, max for other variables
                agg_dict[col] = ["mean", "min", "max"]

        aggregated = df.groupby("date").agg(agg_dict).reset_index()

        # Flatten column names
        aggregated.columns = [
            "_".join(map(str, col)).strip("_") if col[1] else col[0]
            for col in aggregated.columns.values
        ]

        # Rename lambda columns
        for col in aggregated.columns:
            if "<lambda" in col:
                base_name = col.split("_<lambda")[0]
                aggregated = aggregated.rename(columns={col: f"{base_name}_mode"})

        print(f"   ✅ Aggregated to {len(aggregated):,} daily records")

        return aggregated

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived weather features useful for smoke prediction

        Args:
            df: Weather DataFrame

        Returns:
            DataFrame with added features
        """
        print("\n🔧 Adding derived features...")

        # Wind components (useful for smoke transport modeling)
        if (
            "wind_speed_10m_mean" in df.columns
            and "wind_direction_10m_mean" in df.columns
        ):
            import numpy as np

            # Convert wind direction to radians
            wind_dir_rad = np.radians(df["wind_direction_10m_mean"])

            # Calculate u (east-west) and v (north-south) components
            df["wind_u_component"] = -df["wind_speed_10m_mean"] * np.sin(wind_dir_rad)
            df["wind_v_component"] = -df["wind_speed_10m_mean"] * np.cos(wind_dir_rad)

            print("   ✅ Added wind components (u, v)")

        # Atmospheric stability indicator (temperature gradient proxy)
        if (
            "temperature_2m_mean" in df.columns
            and "temperature_2m_min" in df.columns
            and "temperature_2m_max" in df.columns
        ):
            df["temperature_range"] = (
                df["temperature_2m_max"] - df["temperature_2m_min"]
            )
            print("   ✅ Added temperature range")

        # Precipitation indicator
        if "precipitation_sum" in df.columns:
            df["has_precipitation"] = (df["precipitation_sum"] > 0).astype(int)
            print("   ✅ Added precipitation indicator")

        # Heat index approximation (for fire risk)
        if (
            "temperature_2m_mean" in df.columns
            and "relative_humidity_2m_mean" in df.columns
        ):
            # Simple heat index (works well for high temps)
            t = df["temperature_2m_mean"]
            rh = df["relative_humidity_2m_mean"]

            # Only calculate for warm days (>20°C)
            heat_index = t.copy()
            warm_mask = t > 20

            if warm_mask.any():
                t_warm = t[warm_mask]
                rh_warm = rh[warm_mask]

                # Simplified heat index formula
                hi = (
                    -8.78469475556
                    + 1.61139411 * t_warm
                    + 2.33854883889 * rh_warm
                    - 0.14611605 * t_warm * rh_warm
                    - 0.012308094 * t_warm**2
                    - 0.0164248277778 * rh_warm**2
                    + 0.002211732 * t_warm**2 * rh_warm
                    + 0.00072546 * t_warm * rh_warm**2
                    - 0.000003582 * t_warm**2 * rh_warm**2
                )

                heat_index[warm_mask] = hi

            df["heat_index"] = heat_index
            print("   ✅ Added heat index")

        return df

    def save_data(
        self, df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> Path:
        """Save processed data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        # Save raw hourly data
        raw_file = OUTPUT_DIR / f"vancouver_weather_hourly_{date_range}_{timestamp}.csv"
        df.to_csv(raw_file, index=False)
        print(f"\n💾 Raw hourly data saved: {raw_file}")

        # Save processed daily data
        daily_df = self.process_data(df, temporal_resolution="daily")
        daily_df = self.add_derived_features(daily_df)

        processed_file = PROCESSED_DIR / f"vancouver_weather_daily_{date_range}.csv"
        daily_df.to_csv(processed_file, index=False)
        print(f"💾 Processed daily data saved: {processed_file}")

        # Print summary
        self.print_summary(daily_df)

        return processed_file

    def print_summary(self, df: pd.DataFrame):
        """Print data summary statistics"""
        if len(df) == 0:
            return

        print(f"\n{'=' * 70}")
        print(f"📊 Weather Data Summary")
        print(f"{'=' * 70}")
        print(f"Records: {len(df):,} days")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print()

        # Summary for key variables
        key_vars = {
            "temperature_2m_mean": "Temperature (°C)",
            "relative_humidity_2m_mean": "Humidity (%)",
            "precipitation_sum": "Precipitation (mm)",
            "wind_speed_10m_mean": "Wind Speed (km/h)",
            "pressure_msl_mean": "Pressure (hPa)",
        }

        for var, label in key_vars.items():
            if var in df.columns:
                print(f"{label}:")
                print(f"   Mean: {df[var].mean():.2f}")
                print(f"   Min: {df[var].min():.2f}")
                print(f"   Max: {df[var].max():.2f}")
                print()


def main():
    parser = argparse.ArgumentParser(
        description="Download weather data for Vancouver from Open-Meteo"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Default: 90 days ago",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        choices=list(AVAILABLE_VARIABLES.keys()),
        help="Variables to download (default: standard set)",
    )
    parser.add_argument(
        "--list-variables",
        action="store_true",
        help="List available variables and exit",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=VANCOUVER_LAT,
        help=f"Latitude (default: {VANCOUVER_LAT})",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=VANCOUVER_LON,
        help=f"Longitude (default: {VANCOUVER_LON})",
    )

    args = parser.parse_args()

    # List variables only
    if args.list_variables:
        print("Available weather variables:")
        for var, desc in AVAILABLE_VARIABLES.items():
            print(f"  {var:30s} - {desc}")
        print(f"\nDefault variables: {', '.join(DEFAULT_VARIABLES)}")
        return

    # Parse dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=90)

    # Variables
    variables = args.variables if args.variables else DEFAULT_VARIABLES

    # Initialize downloader
    downloader = OpenMeteoDownloader()

    # Download data
    df = downloader.download_weather_data(
        start_date=start_date,
        end_date=end_date,
        variables=variables,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    if len(df) == 0:
        print("\n⚠️  No data downloaded.")
        sys.exit(1)

    # Save data
    downloader.save_data(df, start_date, end_date)

    print("\n✅ Download complete!")
    print(f"\n📁 Raw data: {OUTPUT_DIR}")
    print(f"📁 Processed data: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
