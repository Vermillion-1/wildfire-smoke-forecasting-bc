import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_PATH = Path("data/processed/merged/dataset.csv")
SMOKE_THRESHOLD = 25

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"Dataset shape: {df.shape}")

# Filter for fire season (usually May-Sept)
df['month'] = df['date'].dt.month
fire_season_df = df[df['month'].isin([5, 6, 7, 8, 9])]

# Define target
df['pm25_target'] = df['pm25'].shift(-1)
df = df.dropna(subset=['pm25_target'])

# Correlation matrix for PM2.5 target
fire_cols = [c for c in df.columns if 'fire' in c]
weather_cols = ['temperature_2m_mean', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'precipitation_sum', 'pressure_msl_mean']
lags = ['pm25', 'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_lag7']

corr_cols = ['pm25_target'] + lags + fire_cols[:10] + weather_cols
corr_matrix = df[corr_cols].corr()

print("\nTop Correlations with PM2.5 Target:")
print(corr_matrix['pm25_target'].sort_values(ascending=False).head(10))

# Analyze Smoke Episodes
smoke_days = df[df['pm25'] > SMOKE_THRESHOLD]
print(f"\nNumber of smoke days (PM2.5 > {SMOKE_THRESHOLD}): {len(smoke_days)}")

if not smoke_days.empty:
    print("\nAverage fire count during smoke days vs normal days:")
    print(f"Smoke Days: {smoke_days['fire_count_total'].mean():.2f}")
    print(f"Normal Days: {df[df['pm25'] <= SMOKE_THRESHOLD]['fire_count_total'].mean():.2f}")

# Wind direction analysis
# In Vancouver, wildfire smoke often comes from the interior (East/Northeast) or south (US).
# wind_direction_10m_mean: 0 or 360 is North, 90 is East, 180 is South, 270 is West.
smoke_wind = smoke_days['wind_direction_10m_mean'].value_counts(bins=8).sort_index()
print("\nWind direction distribution during smoke days (bins):")
print(smoke_wind)
