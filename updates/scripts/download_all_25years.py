#!/usr/bin/env python3
"""
Download all data for 25-year dataset (2000-2025).

This script orchestrates downloading:
1. MODIS fire data (2000-2011) + VIIRS fire data (2012-2025)
2. BC air quality data (2000-2025, where available)
3. Weather data from Open-Meteo (2000-2025)

Usage:
    python scripts/download_all_25years.py
    python scripts/download_all_25years.py --skip-fires
    python scripts/download_all_25years.py --skip-weather
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✓ Completed: {description}")
    return True


def download_fires():
    """Download fire data: MODIS (2000-2011) + VIIRS (2012-2025)."""
    print("\n" + "="*60)
    print("🔥 DOWNLOADING FIRE DATA (2000-2025)")
    print("="*60)
    
    # MODIS: 2000-2011
    success = run_command(
        [sys.executable, str(SCRIPTS_DIR / "download_historical_fires.py"),
         "--start-year", "2000", "--end-year", "2011", "--sensor", "MODIS"],
        "Downloading MODIS fire data (2000-2011)"
    )
    if not success:
        return False
    
    # VIIRS: 2012-2025
    current_year = datetime.now().year
    success = run_command(
        [sys.executable, str(SCRIPTS_DIR / "download_historical_fires.py"),
         "--start-year", "2012", "--end-year", str(current_year), "--sensor", "VIIRS"],
        f"Downloading VIIRS fire data (2012-{current_year})"
    )
    return success


def download_air_quality():
    """Download BC air quality data (2000-2025)."""
    print("\n" + "="*60)
    print("🌫️ DOWNLOADING AIR QUALITY DATA (2000-2025)")
    print("="*60)
    
    # BC Gov FTP has data from ~2000, but older years may have gaps
    current_year = datetime.now().year
    success = run_command(
        [sys.executable, str(SCRIPTS_DIR / "download_bc_air_quality.py"),
         "--start-year", "2000", "--end-year", str(current_year)],
        f"Downloading BC PM2.5 data (2000-{current_year})"
    )
    return success


def download_weather():
    """Download weather data from Open-Meteo (2000-2025)."""
    print("\n" + "="*60)
    print("🌤️ DOWNLOADING WEATHER DATA (2000-2025)")
    print("="*60)
    
    # Open-Meteo historical API (ERA5 reanalysis) goes back to 1940
    success = run_command(
        [sys.executable, str(SCRIPTS_DIR / "download_weather.py"),
         "--start-date", "2000-01-01", "--end-date", datetime.now().strftime("%Y-%m-%d")],
        "Downloading weather data (2000-present)"
    )
    return success


def build_dataset():
    """Build the merged dataset."""
    print("\n" + "="*60)
    print("📊 BUILDING MERGED DATASET")
    print("="*60)
    
    success = run_command(
        [sys.executable, str(SCRIPTS_DIR / "build_dataset.py")],
        "Building merged dataset"
    )
    return success


def main():
    parser = argparse.ArgumentParser(description="Download all data for 25-year dataset")
    parser.add_argument("--skip-fires", action="store_true", help="Skip fire data download")
    parser.add_argument("--skip-air-quality", action="store_true", help="Skip air quality download")
    parser.add_argument("--skip-weather", action="store_true", help="Skip weather download")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset build")
    args = parser.parse_args()

    print("="*60)
    print("📦 25-YEAR DATASET DOWNLOAD (2000-2025)")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Target: Fire + Air Quality + Weather data for 25 years")
    
    all_success = True
    
    if not args.skip_fires:
        if not download_fires():
            all_success = False
    else:
        print("\n⏭️ Skipping fire data download")
    
    if not args.skip_air_quality:
        if not download_air_quality():
            all_success = False
    else:
        print("\n⏭️ Skipping air quality download")
    
    if not args.skip_weather:
        if not download_weather():
            all_success = False
    else:
        print("\n⏭️ Skipping weather download")
    
    if not args.skip_build:
        if not build_dataset():
            all_success = False
    else:
        print("\n⏭️ Skipping dataset build")
    
    print("\n" + "="*60)
    if all_success:
        print("✅ ALL DOWNLOADS COMPLETE!")
    else:
        print("⚠️ Some downloads failed. Check output above.")
    print("="*60)
    
    # Print data summary
    data_dir = PROJECT_ROOT / "data"
    print(f"\nData directory: {data_dir}")
    
    for subdir in ["raw/fires", "raw/air_quality", "raw/weather", "processed/merged"]:
        path = data_dir / subdir
        if path.exists():
            files = list(path.glob("*"))
            print(f"  {subdir}: {len(files)} files")


if __name__ == "__main__":
    main()
