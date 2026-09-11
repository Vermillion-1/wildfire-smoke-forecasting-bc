"""
Vancouver Wildfire Smoke Predictor — Streamlit Dashboard
CMPT 733 Course Project

Analyzes wildfire smoke impacts on Vancouver's air quality (2000-2025),
addressing three research questions:

  RQ1: Can ML models predict next-day PM2.5 better than simple baselines?
  RQ2: What is the lag between BC wildfire activity and Vancouver PM2.5?
  RQ3: How has Vancouver's smoke season changed over 2000-2025?

Run with:  streamlit run app/streamlit_app.py
"""

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ─── Configuration ────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "merged" / "dataset.csv"
FIGURES_DIR = ROOT / "figures"
SMOKE_THRESHOLD = 25.0  # µg/m³ — BC AQHI moderate risk

st.set_page_config(
    page_title="Vancouver Wildfire Smoke Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Feature definitions (must match 03_modeling.ipynb) ───────────────────────

FEATURE_COLS = [
    # PM2.5 lags (4)
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    # Fire same-day (9)
    "fire_count_close", "fire_frp_sum_close",
    "fire_count_medium", "fire_frp_sum_medium",
    "fire_count_far", "fire_frp_sum_far",
    "fire_count_total", "fire_frp_sum_total", "fire_mean_distance_km",
    # Fire lagged (8)
    "fire_count_total_lag1", "fire_frp_sum_total_lag1",
    "fire_count_total_lag2", "fire_frp_sum_total_lag2",
    "fire_count_close_lag1", "fire_frp_sum_close_lag1",
    "fire_count_medium_lag1", "fire_frp_sum_medium_lag1",
    # Weather (10)
    "temperature_2m_mean", "temperature_2m_max",
    "relative_humidity_2m_mean", "wind_speed_10m_mean",
    "wind_u_component", "wind_v_component",
    "precipitation_sum", "pressure_msl_mean",
    "temperature_range", "has_precipitation",
    # Calendar (3)
    "month", "day_of_year", "is_fire_season",
]

FEATURE_GROUPS = {
    "pm25_lags": ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7"],
    "fire_same": [
        "fire_count_close", "fire_frp_sum_close",
        "fire_count_medium", "fire_frp_sum_medium",
        "fire_count_far", "fire_frp_sum_far",
        "fire_count_total", "fire_frp_sum_total", "fire_mean_distance_km",
    ],
    "fire_lag": [
        "fire_count_total_lag1", "fire_frp_sum_total_lag1",
        "fire_count_total_lag2", "fire_frp_sum_total_lag2",
        "fire_count_close_lag1", "fire_frp_sum_close_lag1",
        "fire_count_medium_lag1", "fire_frp_sum_medium_lag1",
    ],
    "weather": [
        "temperature_2m_mean", "temperature_2m_max",
        "relative_humidity_2m_mean", "wind_speed_10m_mean",
        "wind_u_component", "wind_v_component",
        "precipitation_sum", "pressure_msl_mean",
        "temperature_range", "has_precipitation",
    ],
    "calendar": ["month", "day_of_year", "is_fire_season"],
}


def get_feature_group(feat):
    """Return human-readable group name for a feature."""
    if "pm25" in feat:
        return "PM2.5 Lags"
    elif "fire" in feat:
        return "Fire Activity"
    elif feat in FEATURE_GROUPS["calendar"]:
        return "Calendar"
    else:
        return "Weather"


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month_name"] = df["date"].dt.strftime("%b")
    df["is_smoke_day"] = df["pm25"] > SMOKE_THRESHOLD
    return df


@st.cache_data
def compute_smoke_episodes(df):
    """Identify contiguous smoke episodes (PM2.5 > threshold)."""
    smoke = df["is_smoke_day"].values.astype(int)
    changes = np.diff(smoke, prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    episodes = []
    for s, e in zip(starts, ends):
        ep = df.iloc[s:e]
        episodes.append({
            "Start Date": ep["date"].iloc[0],
            "End Date": ep["date"].iloc[-1],
            "Duration (days)": len(ep),
            "Mean PM2.5": round(ep["pm25"].mean(), 1),
            "Max PM2.5": round(ep["pm25"].max(), 1),
            "Year": ep["date"].iloc[0].year,
        })
    return pd.DataFrame(episodes)


@st.cache_data
def compute_annual_stats(df):
    """Compute annual smoke-season statistics."""
    stats = []
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year]
        fire_season = ydf[ydf["is_fire_season"] == 1] if "is_fire_season" in ydf.columns else ydf
        smoke_days = int(ydf["is_smoke_day"].sum())
        stats.append({
            "Year": year,
            "Smoke Days": smoke_days,
            "Annual Mean PM2.5": round(ydf["pm25"].mean(), 2),
            "Annual Max PM2.5": round(ydf["pm25"].max(), 1),
            "Fire Season Mean PM2.5": round(fire_season["pm25"].mean(), 2) if len(fire_season) > 0 else np.nan,
        })
    return pd.DataFrame(stats)


# ─── Model Training (cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def train_and_evaluate(_df):
    """
    Train all models using expanding-window time-series CV.
    Returns results table, feature importances, and prediction DataFrames.
    """
    mdf = _df.copy()
    mdf["pm25_target"] = mdf["pm25"].shift(-1)
    mdf = mdf.dropna(subset=["pm25_target"] + FEATURE_COLS)

    X = mdf[FEATURE_COLS].values
    y = mdf["pm25_target"].values
    dates = mdf["date"].values
    pm25_today = mdf["pm25"].values
    pm25_lags = mdf[["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7"]].values

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Ridge (alpha=10.0)": Ridge(alpha=10.0),
        "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
        "Lasso (alpha=1.0)": Lasso(alpha=1.0, max_iter=10000),
        "Random Forest (100 trees)": RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
        "Random Forest (200 trees)": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=3, random_state=42, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        ),
        "XGBoost (Tuned)": xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            verbose=-1,
        ),
        "LightGBM (Tuned)": lgb.LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
    }

    all_results = {}
    predictions = {}
    rf_importance = None

    # --- Baselines ---
    baseline_fns = {
        "Persistence (today's PM2.5)": lambda: pm25_today,
        "Rolling Mean (lag avg)": lambda: pm25_lags.mean(axis=1),
    }
    for name, fn in baseline_fns.items():
        fold_metrics = []
        preds_all, actuals_all, dates_all = [], [], []
        for train_idx, test_idx in tscv.split(X):
            preds = fn()[test_idx]
            actual = y[test_idx]
            smoke_mask = actual > SMOKE_THRESHOLD
            fold_metrics.append({
                "MAE": mean_absolute_error(actual, preds),
                "RMSE": np.sqrt(mean_squared_error(actual, preds)),
                "R2": r2_score(actual, preds),
                "MAE_smoke": (mean_absolute_error(actual[smoke_mask], preds[smoke_mask])
                              if smoke_mask.sum() > 0 else np.nan),
            })
            preds_all.extend(preds)
            actuals_all.extend(actual)
            dates_all.extend(dates[test_idx])
        all_results[name] = {k: np.nanmean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        predictions[name] = pd.DataFrame({
            "date": dates_all, "actual": actuals_all, "predicted": preds_all,
        })

    # --- ML models ---
    for name, model in models.items():
        fold_metrics = []
        preds_all, actuals_all, dates_all = [], [], []
        for train_idx, test_idx in tscv.split(X):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            m = model.__class__(**model.get_params())
            m.fit(X_train, y[train_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                preds = m.predict(X_test)
            actual = y[test_idx]
            smoke_mask = actual > SMOKE_THRESHOLD

            fold_metrics.append({
                "MAE": mean_absolute_error(actual, preds),
                "RMSE": np.sqrt(mean_squared_error(actual, preds)),
                "R2": r2_score(actual, preds),
                "MAE_smoke": (mean_absolute_error(actual[smoke_mask], preds[smoke_mask])
                              if smoke_mask.sum() > 0 else np.nan),
            })
            preds_all.extend(preds)
            actuals_all.extend(actual)
            dates_all.extend(dates[test_idx])

            # Keep last fold's importance from best tree-based model
            if "XGBoost (Tuned)" in name:
                rf_importance = m.feature_importances_
            elif "200" in name and rf_importance is None:
                rf_importance = m.feature_importances_

        all_results[name] = {k: np.nanmean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        predictions[name] = pd.DataFrame({
            "date": dates_all, "actual": actuals_all, "predicted": preds_all,
        })

    # Format results
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Model"
    results_df = results_df.sort_values("MAE")

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf_importance,
    }).sort_values("Importance", ascending=False)
    importance_df["Group"] = importance_df["Feature"].apply(get_feature_group)

    return results_df, importance_df, predictions


# ─── Ablation results (hardcoded from notebook to avoid re-training 8x) ──────

ABLATION_RESULTS = pd.DataFrame([
    {"Subset": "All features", "MAE": 1.793, "RMSE": 4.878, "R2": 0.1526, "Features": 34},
    {"Subset": "No PM2.5 lags", "MAE": 1.930, "RMSE": 4.920, "R2": 0.1379, "Features": 30},
    {"Subset": "No fire features", "MAE": 1.750, "RMSE": 4.923, "R2": 0.1369, "Features": 17},
    {"Subset": "No weather", "MAE": 2.218, "RMSE": 5.125, "R2": 0.0647, "Features": 24},
    {"Subset": "No calendar", "MAE": 1.802, "RMSE": 4.867, "R2": 0.1563, "Features": 31},
    {"Subset": "PM2.5 lags only", "MAE": 2.210, "RMSE": 5.216, "R2": 0.0311, "Features": 4},
    {"Subset": "Fire features only", "MAE": 2.594, "RMSE": 5.253, "R2": 0.0174, "Features": 17},
    {"Subset": "Weather only", "MAE": 1.978, "RMSE": 5.035, "R2": 0.0973, "Features": 10},
])


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

df = load_data()
episodes_df = compute_smoke_episodes(df)
annual_stats = compute_annual_stats(df)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Vancouver Wildfire Smoke Predictor")
    st.caption("CMPT 733 — Big Data Lab II")
    st.markdown("---")
    st.markdown(
        "**Research Questions**\n"
        "1. Can ML models predict next-day PM2.5?\n"
        "2. What is the fire-to-smoke lag?\n"
        "3. How has the smoke season changed?\n"
    )
    st.markdown("---")
    st.markdown(f"**Data range:** {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    st.markdown(f"**Total days:** {len(df):,}")
    st.markdown(f"**Smoke days:** {df['is_smoke_day'].sum()} ({df['is_smoke_day'].mean()*100:.1f}%)")
    st.markdown(f"**Smoke episodes:** {len(episodes_df)}")
    st.markdown("---")
    st.markdown(
        "**Data sources**\n"
        "- PM2.5: BC Gov FTP\n"
        "- Fires: NASA FIRMS MODIS + VIIRS\n"
        "- Weather: Open-Meteo\n"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview",
    "Smoke Season Trends",
    "Smoke Arrival Lag",
    "PM2.5 Forecasting",
    "Model Validation",
    "Health & Planning",
    "Data Explorer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header(f"Air Quality Overview — Vancouver, BC ({df['date'].min():%Y}-{df['date'].max():%Y})")
    st.markdown(
        "This dashboard analyzes the impact of wildfire smoke on Vancouver's air quality "
        "using daily PM2.5 measurements, VIIRS satellite fire detections, and weather data. "
        f"A PM2.5 value above **{SMOKE_THRESHOLD:.0f} ug/m3** is classified as a smoke event day "
        "(BC AQHI moderate risk threshold)."
    )

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Days", f"{len(df):,}")
    c2.metric("Mean PM2.5", f"{df['pm25'].mean():.1f} ug/m3")
    c3.metric("Max PM2.5", f"{df['pm25'].max():.1f} ug/m3")
    c4.metric("Smoke Days", f"{df['is_smoke_day'].sum()}")
    c5.metric("Episodes", f"{len(episodes_df)}")

    st.markdown("---")

    # Interactive PM2.5 time series
    st.subheader("Daily PM2.5 Time Series")

    year_range = st.slider(
        "Select year range",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=(int(df["year"].min()), int(df["year"].max())),
        key="overview_year_range",
    )
    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=filtered["date"], y=filtered["pm25"],
        mode="lines", name="Daily PM2.5",
        line=dict(color="#1f77b4", width=0.8),
    ))
    smoke_pts = filtered[filtered["is_smoke_day"]]
    fig_ts.add_trace(go.Scatter(
        x=smoke_pts["date"], y=smoke_pts["pm25"],
        mode="markers", name="Smoke days (>25 ug/m3)",
        marker=dict(color="red", size=4),
    ))
    fig_ts.add_hline(
        y=SMOKE_THRESHOLD, line_dash="dash", line_color="red",
        annotation_text=f"Smoke threshold ({SMOKE_THRESHOLD:.0f} ug/m3)",
    )
    fig_ts.update_layout(
        yaxis_title="PM2.5 (ug/m3)", xaxis_title="Date", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_ts, width="stretch")

    # Distribution + monthly patterns
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("PM2.5 Distribution")
        fig_dist = px.histogram(
            filtered, x="pm25", nbins=100,
            labels={"pm25": "PM2.5 (ug/m3)"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_dist.add_vline(x=SMOKE_THRESHOLD, line_dash="dash", line_color="red")
        fig_dist.update_layout(
            height=350, showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_dist, width="stretch")

    with col_right:
        st.subheader("Monthly PM2.5 Patterns")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = (filtered.groupby(filtered["date"].dt.month)["pm25"]
                   .agg(["mean", "median"]).reset_index())
        monthly.columns = ["Month", "Mean", "Median"]
        monthly["Month Name"] = monthly["Month"].map(lambda m: month_names[m - 1])

        fig_mon = go.Figure()
        fig_mon.add_trace(go.Bar(
            x=monthly["Month Name"], y=monthly["Mean"],
            name="Mean", marker_color="#1f77b4",
        ))
        fig_mon.add_trace(go.Bar(
            x=monthly["Month Name"], y=monthly["Median"],
            name="Median", marker_color="#2ca02c",
        ))
        fig_mon.update_layout(
            barmode="group", height=350, yaxis_title="PM2.5 (ug/m3)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_mon, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SMOKE SEASON TRENDS (RQ3)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("RQ3: Seasonal Risk Profiling")
    st.markdown(
        "**How has Vancouver's wildfire smoke season changed from 2000 to 2024?** "
        "Are smoke events becoming more frequent, longer, or more intense?"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Annual Smoke Days")
        fig_sd = px.bar(
            annual_stats, x="Year", y="Smoke Days",
            color="Smoke Days", color_continuous_scale="YlOrRd",
        )
        fig_sd.update_layout(
            height=400, coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_sd, width="stretch")

    with col_b:
        st.subheader("Fire Season Mean PM2.5 (Jun-Sep)")
        fig_fs = px.bar(
            annual_stats, x="Year", y="Fire Season Mean PM2.5",
            color="Fire Season Mean PM2.5", color_continuous_scale="YlOrRd",
        )
        fig_fs.add_hline(y=SMOKE_THRESHOLD, line_dash="dash", line_color="red",
                         annotation_text="Smoke threshold")
        fig_fs.update_layout(
            height=400, coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_fs, width="stretch")

    # Smoke episodes table
    st.subheader("Smoke Episodes")
    if len(episodes_df) > 0:
        disp = episodes_df.copy()
        disp["Start Date"] = disp["Start Date"].dt.strftime("%Y-%m-%d")
        disp["End Date"] = disp["End Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(disp, width="stretch", hide_index=True)
    else:
        st.info("No smoke episodes found.")

    # Weather comparison
    st.subheader("Weather on Smoke Days vs. Normal Days")
    weather_vars = [
        ("temperature_2m_mean", "Temperature (C)", False),
        ("relative_humidity_2m_mean", "Relative Humidity (%)", True),
        ("wind_speed_10m_mean", "Wind Speed (km/h)", False),
        ("precipitation_sum", "Precipitation (mm)", True),
    ]
    wcols = st.columns(4)
    for i, (var, label, invert) in enumerate(weather_vars):
        with wcols[i]:
            smoke_val = df.loc[df["is_smoke_day"], var].mean()
            normal_val = df.loc[~df["is_smoke_day"], var].mean()
            delta = smoke_val - normal_val
            st.metric(
                label=label,
                value=f"{smoke_val:.1f}",
                delta=f"{delta:+.1f} vs normal",
                delta_color="inverse" if invert else "normal",
            )

    # Static EDA figures
    st.subheader("Detailed Analysis (from EDA notebook)")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        p = FIGURES_DIR / "smoke_event_calendar.png"
        if p.exists():
            st.image(str(p), caption="Smoke Event Calendar", width="stretch")
    with col_f2:
        p = FIGURES_DIR / "smoke_season_trends.png"
        if p.exists():
            st.image(str(p), caption="Smoke Season Trends", width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: SMOKE ARRIVAL LAG (RQ2)
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("RQ2: Smoke Arrival Lag Analysis")
    st.markdown(
        "**What is the typical delay between a spike in BC wildfire activity "
        "and elevated PM2.5 in Vancouver?**\n\n"
        "Key finding: Cross-correlation peaks at **lag 0** (same day) across all "
        "distance bands, suggesting smoke transport is faster than the daily resolution "
        "of this dataset. PM2.5 autoregression (yesterday's PM2.5) explains most "
        "variance (R-squared = 0.71)."
    )

    # Cross-correlation figures
    st.subheader("Cross-Correlation: Fire Activity vs. PM2.5")
    cc1, cc2 = st.columns(2)
    with cc1:
        p = FIGURES_DIR / "cross_correlation_frp_pm25.png"
        if p.exists():
            st.image(str(p), caption="Fire Radiative Power vs PM2.5", width="stretch")
    with cc2:
        p = FIGURES_DIR / "cross_correlation_count_pm25.png"
        if p.exists():
            st.image(str(p), caption="Fire Count vs PM2.5", width="stretch")

    # Episode case studies
    st.subheader("Smoke Episode Case Studies")
    episode_files = {
        "August 2017 — longest episode (10 days)": "episode_2017_aug_smoke.png",
        "August 2018 — 8 smoke days (2 episodes: Aug 13–15 and Aug 19–23)": "episode_2018_aug_smoke.png",
        "September 2020 — worst episode (peak 163.5 ug/m3)": "episode_2020_sep_smoke.png",
    }
    ep_choice = st.selectbox("Select an episode", list(episode_files.keys()))
    p = FIGURES_DIR / episode_files[ep_choice]
    if p.exists():
        st.image(str(p), width="stretch")

    # Interactive episode explorer
    st.subheader("Interactive Episode Explorer")
    if len(episodes_df) > 0:
        ep_labels = episodes_df.apply(
            lambda r: (f"{r['Start Date']:%Y-%m-%d} to {r['End Date']:%Y-%m-%d}  "
                       f"({r['Duration (days)']}d, max {r['Max PM2.5']:.0f} ug/m3)"),
            axis=1,
        ).tolist()
        ep_sel = st.selectbox("Select a smoke episode", ep_labels, key="ep_explorer")
        ep_idx = ep_labels.index(ep_sel)
        ep = episodes_df.iloc[ep_idx]

        # Show 7-day window around episode
        win_start = ep["Start Date"] - pd.Timedelta(days=7)
        win_end = ep["End Date"] + pd.Timedelta(days=7)
        win_df = df[(df["date"] >= win_start) & (df["date"] <= win_end)]

        fig_ep = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("PM2.5", "Fire Activity (total BC fire count)"),
            vertical_spacing=0.12,
        )
        fig_ep.add_trace(go.Scatter(
            x=win_df["date"], y=win_df["pm25"],
            mode="lines+markers", name="PM2.5",
            line=dict(color="#1f77b4"),
        ), row=1, col=1)
        fig_ep.add_hline(y=SMOKE_THRESHOLD, line_dash="dash", line_color="red", row=1, col=1)

        fig_ep.add_trace(go.Bar(
            x=win_df["date"], y=win_df["fire_count_total"],
            name="Fire Count", marker_color="#ff7f0e",
        ), row=2, col=1)

        # Shade episode window
        for row in [1, 2]:
            fig_ep.add_vrect(
                x0=ep["Start Date"], x1=ep["End Date"],
                fillcolor="red", opacity=0.1, line_width=0, row=row, col=1,
            )

        fig_ep.update_layout(height=500, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_ep, width="stretch")

    # Wind analysis
    st.subheader("Wind Direction: Smoke Days vs. Normal Days")
    p = FIGURES_DIR / "wind_direction_smoke_vs_normal.png"
    if p.exists():
        _, cw, _ = st.columns([1, 2, 1])
        with cw:
            st.image(str(p), width="stretch")

    st.markdown(
        "Smoke days show more **westerly** winds (from the Pacific, mean U = +1.35 vs −1.01 on normal fire-season days; "
        "Mann-Whitney p = 0.004). In this dataset's convention, positive U = westerly (from the Pacific/south), "
        "negative U = easterly (from BC's interior). The westerly shift on smoke days reflects that the worst events "
        "(e.g., September 2020) were driven by US fires to the south carried north by southerly/westerly flow, "
        "rather than by BC interior fires."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PM2.5 FORECASTING (RQ1)
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("RQ1: Next-Day PM2.5 Forecasting")
    st.markdown(
        "**Can simple ML models predict next-day PM2.5 more accurately than persistence "
        "and rolling-mean baselines?**\n\n"
        "Models are evaluated using **expanding-window time-series cross-validation** "
        "(5 folds) to prevent data leakage. 34 features span PM2.5 lags, fire activity, "
        "weather, and calendar variables."
    )

    with st.spinner("Training models (cached after first run)..."):
        results_df, importance_df, predictions = train_and_evaluate(df)

    # --- Model comparison ---
    st.subheader("Model Comparison")

    res_display = results_df.rename(columns={
        "MAE": "MAE", "RMSE": "RMSE", "R2": "R-squared", "MAE_smoke": "MAE (Smoke Days)",
    }).round(3)

    col_tbl, col_bar = st.columns([1, 1])

    with col_tbl:
        styled = (res_display.style
                  .highlight_min(subset=["MAE", "RMSE", "MAE (Smoke Days)"], color="#d4edda")
                  .highlight_max(subset=["R-squared"], color="#d4edda"))
        st.dataframe(styled, width="stretch")
        st.caption("Green = best value per metric. Sorted by MAE ascending.")

    with col_bar:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=res_display.index, y=res_display["MAE"],
            name="MAE", marker_color="#1f77b4",
        ))
        fig_comp.add_trace(go.Bar(
            x=res_display.index, y=res_display["RMSE"],
            name="RMSE", marker_color="#ff7f0e",
        ))
        fig_comp.update_layout(
            barmode="group", height=400, yaxis_title="Error (ug/m3)",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=140),
        )
        st.plotly_chart(fig_comp, width="stretch")

    # Dynamically generate finding based on best model
    best_model = results_df.index[0]  # Already sorted by MAE
    best_mae = results_df.loc[best_model, "MAE"]
    persistence_mae = results_df.loc["Persistence (today's PM2.5)", "MAE"] if "Persistence (today's PM2.5)" in results_df.index else None
    
    if persistence_mae and best_mae < persistence_mae:
        st.success(
            f"**Key finding:** **{best_model}** achieves the lowest MAE ({best_mae:.2f}), "
            f"beating the persistence baseline ({persistence_mae:.2f}). "
            "Gradient boosting models (XGBoost, LightGBM) effectively capture non-linear "
            "relationships between fire activity, weather, and next-day PM2.5."
        )
    else:
        st.info(
            "**Key finding:** The persistence baseline (tomorrow = today) remains competitive. "
            "PM2.5 is highly autocorrelated (lag-1 r = 0.80), making 'tomorrow ~ today' a strong heuristic. "
            "Advanced models provide marginal improvements but better capture extreme smoke events."
        )

    # --- Predictions vs. actuals ---
    st.subheader("Predictions vs. Actuals")

    model_names = list(predictions.keys())
    model_sel = st.selectbox("Select model", model_names, key="pred_model")
    pred_df = predictions[model_sel].copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    sc_col, ts_col = st.columns(2)

    with sc_col:
        fig_sc = px.scatter(
            pred_df, x="actual", y="predicted", opacity=0.3,
            labels={"actual": "Actual PM2.5 (ug/m3)", "predicted": "Predicted PM2.5 (ug/m3)"},
            color_discrete_sequence=["#1f77b4"],
        )
        max_v = max(pred_df["actual"].max(), pred_df["predicted"].max())
        fig_sc.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v],
            mode="lines", line=dict(dash="dash", color="red"), name="Perfect",
        ))
        fig_sc.update_layout(height=420, margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_sc, width="stretch")

    with ts_col:
        fig_pts = go.Figure()
        fig_pts.add_trace(go.Scatter(
            x=pred_df["date"], y=pred_df["actual"],
            mode="lines", name="Actual", line=dict(color="#1f77b4", width=0.8),
        ))
        fig_pts.add_trace(go.Scatter(
            x=pred_df["date"], y=pred_df["predicted"],
            mode="lines", name="Predicted", line=dict(color="#ff7f0e", width=0.8),
        ))
        fig_pts.update_layout(
            height=420, yaxis_title="PM2.5 (ug/m3)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_pts, width="stretch")

    # --- Feature importance ---
    st.subheader("Feature Importance (XGBoost Tuned)")

    top_n = st.slider("Number of features to show", 5, 34, 15, key="fi_topn")
    top_imp = importance_df.head(top_n)

    group_colors = {
        "PM2.5 Lags": "#1f77b4",
        "Fire Activity": "#ff7f0e",
        "Weather": "#2ca02c",
        "Calendar": "#d62728",
    }
    fig_imp = px.bar(
        top_imp, x="Importance", y="Feature",
        orientation="h", color="Group",
        color_discrete_map=group_colors,
    )
    fig_imp.update_layout(
        height=max(300, top_n * 28),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_imp, width="stretch")

    st.caption(
        "pm25_lag1 (yesterday's PM2.5) dominates at ~42% importance. Weather features "
        "(wind, precipitation, temperature) are collectively more important than fire "
        "features, suggesting local meteorology mediating smoke dispersion matters more "
        "than raw fire counts for next-day prediction."
    )

    # --- Feature ablation ---
    st.subheader("Feature Ablation Study")
    st.markdown(
        "How does model performance change when removing entire feature groups? "
        "This analysis helps understand which features contribute most to prediction accuracy."
    )

    abl = ABLATION_RESULTS.copy()
    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(
        x=abl["Subset"], y=abl["MAE"], name="MAE", marker_color="#1f77b4",
    ))
    fig_abl.add_trace(go.Bar(
        x=abl["Subset"], y=abl["R2"], name="R-squared", marker_color="#2ca02c",
        yaxis="y2",
    ))
    fig_abl.update_layout(
        barmode="group", height=400,
        yaxis=dict(title="MAE (ug/m3)"),
        yaxis2=dict(title="R-squared", overlaying="y", side="right", range=[0, 0.4]),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=60, t=40, b=120),
    )
    st.plotly_chart(fig_abl, width="stretch")

    st.dataframe(abl.set_index("Subset").round(3), width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: MODEL VALIDATION (Holdout Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Model Validation: Train 2000-2020 → Test 2021-2024")
    st.markdown(
        "**Comparative Analysis:** To rigorously validate our models, we perform a temporal holdout test:\n"
        "- **Training period:** 2000-2020 (all historical data, 7,664 samples)\n"
        "- **Testing period:** 2021-2024 (future prediction, 1,461 samples)\n\n"
        "This simulates a real-world scenario where we train on past data and predict future conditions."
    )

    # Holdout training function
    @st.cache_data(show_spinner=False)
    def train_holdout_models(_df):
        """Train models on 2000-2020, evaluate on 2021-2024."""
        mdf = _df.copy()
        mdf["pm25_target"] = mdf["pm25"].shift(-1)
        mdf = mdf.dropna(subset=["pm25_target"] + FEATURE_COLS)

        # Split by year
        train_mask = mdf["year"] <= 2020
        test_mask = mdf["year"] >= 2021

        X_train = mdf.loc[train_mask, FEATURE_COLS].values
        y_train = mdf.loc[train_mask, "pm25_target"].values
        X_test = mdf.loc[test_mask, FEATURE_COLS].values
        y_test = mdf.loc[test_mask, "pm25_target"].values
        dates_test = mdf.loc[test_mask, "date"].values
        pm25_today_test = mdf.loc[test_mask, "pm25"].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        holdout_results = {}
        holdout_preds = {}

        # Persistence baseline
        y_pred_persist = pm25_today_test
        holdout_results["Persistence"] = {
            "MAE": mean_absolute_error(y_test, y_pred_persist),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_persist)),
            "R2": r2_score(y_test, y_pred_persist),
        }
        holdout_preds["Persistence"] = y_pred_persist

        # ML models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
            "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1),
        }

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                y_pred = model.predict(X_test_scaled)
            holdout_results[name] = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred),
            }
            holdout_preds[name] = y_pred

        results_df = pd.DataFrame(holdout_results).T
        results_df.index.name = "Model"
        results_df = results_df.sort_values("MAE")

        return results_df, holdout_preds, y_test, dates_test, len(X_train), len(X_test)

    with st.spinner("Training models on holdout split..."):
        ho_results, ho_preds, y_test_ho, dates_test_ho, n_train, n_test = train_holdout_models(df)

    # Summary metrics
    st.subheader("Holdout Split Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", f"{n_train:,}", "2000-2020")
    col2.metric("Test Samples", f"{n_test:,}", "2021-2024")
    col3.metric("Best Model", ho_results.index[0])
    col4.metric("Best MAE", f"{ho_results.iloc[0]['MAE']:.2f} µg/m³")

    st.markdown("---")

    # Results table and chart
    st.subheader("Model Performance on Holdout Test Set")
    col_tbl, col_chart = st.columns([1, 1])

    with col_tbl:
        styled_ho = (ho_results.style
                     .highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
                     .highlight_max(subset=["R2"], color="#d4edda")
                     .format("{:.3f}"))
        st.dataframe(styled_ho, width="stretch")

    with col_chart:
        fig_ho = go.Figure()
        fig_ho.add_trace(go.Bar(x=ho_results.index, y=ho_results["MAE"], name="MAE", marker_color="#1f77b4"))
        fig_ho.add_trace(go.Bar(x=ho_results.index, y=ho_results["RMSE"], name="RMSE", marker_color="#ff7f0e"))
        fig_ho.update_layout(
            barmode="group", height=350, yaxis_title="Error (µg/m³)",
            xaxis_tickangle=-45, margin=dict(l=40, r=20, t=20, b=100),
        )
        st.plotly_chart(fig_ho, width="stretch")

    # Interpretation
    best_model_ho = ho_results.index[0]
    persist_mae = ho_results.loc["Persistence", "MAE"]
    best_mae_ho = ho_results.iloc[0]["MAE"]
    improvement = (persist_mae - best_mae_ho) / persist_mae * 100

    if best_mae_ho < persist_mae:
        st.success(
            f"**✓ {best_model_ho}** outperforms persistence baseline by **{improvement:.1f}%** on unseen data (2021-2024). "
            "This validates that the model generalizes beyond the training period."
        )
    else:
        st.info(
            f"Persistence baseline remains competitive on the holdout set. "
            "PM2.5's high autocorrelation makes 'tomorrow ≈ today' hard to beat."
        )

    st.markdown("---")

    # Predicted vs Actual visualization
    st.subheader("Predicted vs Actual: 2021-2024")

    model_choice = st.selectbox("Select model to visualize", list(ho_preds.keys()), key="ho_model")
    y_pred_ho = ho_preds[model_choice]

    col_sc, col_ts = st.columns(2)

    with col_sc:
        fig_scatter = px.scatter(
            x=y_test_ho, y=y_pred_ho, opacity=0.4,
            labels={"x": "Actual PM2.5 (µg/m³)", "y": "Predicted PM2.5 (µg/m³)"},
        )
        max_val = max(y_test_ho.max(), y_pred_ho.max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", line=dict(dash="dash", color="red"), name="Perfect",
        ))
        fig_scatter.add_hline(y=SMOKE_THRESHOLD, line_dash="dot", line_color="orange")
        fig_scatter.add_vline(x=SMOKE_THRESHOLD, line_dash="dot", line_color="orange")
        fig_scatter.update_layout(height=400, margin=dict(l=40, r=20, t=20, b=40))
        st.plotly_chart(fig_scatter, width="stretch")

    with col_ts:
        dates_plot = pd.to_datetime(dates_test_ho)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=dates_plot, y=y_test_ho, mode="lines", name="Actual", line=dict(color="#1f77b4", width=0.8)))
        fig_ts.add_trace(go.Scatter(x=dates_plot, y=y_pred_ho, mode="lines", name="Predicted", line=dict(color="#ff7f0e", width=0.8)))
        fig_ts.add_hline(y=SMOKE_THRESHOLD, line_dash="dash", line_color="red")
        fig_ts.update_layout(
            height=400, yaxis_title="PM2.5 (µg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_ts, width="stretch")

    # Smoke event analysis
    st.subheader("Smoke Event Detection Analysis")
    smoke_mask = y_test_ho > SMOKE_THRESHOLD
    n_smoke_days = smoke_mask.sum()
    predicted_as_smoke = (y_pred_ho > SMOKE_THRESHOLD)
    correctly_detected = (smoke_mask & predicted_as_smoke).sum()
    false_alarms = (~smoke_mask & predicted_as_smoke).sum()

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Actual Smoke Days", n_smoke_days)
    col_s2.metric("Correctly Detected", f"{correctly_detected} ({correctly_detected/n_smoke_days*100:.0f}%)" if n_smoke_days > 0 else "N/A")
    col_s3.metric("Missed", f"{n_smoke_days - correctly_detected}" if n_smoke_days > 0 else "N/A")
    col_s4.metric("False Alarms", false_alarms)

    # Show worst smoke days comparison
    if n_smoke_days > 0:
        st.markdown("**Top 10 Worst Smoke Days — Predicted vs Actual:**")
        smoke_df = pd.DataFrame({
            "Date": pd.to_datetime(dates_test_ho[smoke_mask]),
            "Actual PM2.5": y_test_ho[smoke_mask],
            "Predicted PM2.5": y_pred_ho[smoke_mask],
            "Error": np.abs(y_test_ho[smoke_mask] - y_pred_ho[smoke_mask]),
        }).sort_values("Actual PM2.5", ascending=False).head(10)
        smoke_df["Date"] = smoke_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(smoke_df.round(1), width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: HEALTH & PLANNING
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Health & Planning Guide")
    st.markdown(
        "**What does this app offer you?** Actionable insights to protect your health "
        "and plan outdoor activities during wildfire smoke season."
    )

    # --- Air Quality Index Categories ---
    st.subheader("Air Quality Health Categories")
    
    # Define AQI-style categories based on PM2.5
    aq_categories = pd.DataFrame([
        {"Category": "Good", "PM2.5 Range": "0 - 12",
         "Health Impact": "Air quality is satisfactory",
         "Outdoor Activity": "All activities safe"},
        {"Category": "Moderate", "PM2.5 Range": "12 - 25",
         "Health Impact": "Acceptable; sensitive individuals may experience issues",
         "Outdoor Activity": "Most activities safe"},
        {"Category": "Unhealthy (Sensitive)", "PM2.5 Range": "25 - 35",
         "Health Impact": "Sensitive groups may experience health effects",
         "Outdoor Activity": "Reduce prolonged outdoor exertion"},
        {"Category": "Unhealthy", "PM2.5 Range": "35 - 55",
         "Health Impact": "Everyone may begin to experience health effects",
         "Outdoor Activity": "Limit outdoor activities"},
        {"Category": "Very Unhealthy", "PM2.5 Range": "55 - 150",
         "Health Impact": "Health alert: everyone may experience serious effects",
         "Outdoor Activity": "Avoid outdoor activities"},
        {"Category": "Hazardous", "PM2.5 Range": "150+",
         "Health Impact": "Health emergency: entire population affected",
         "Outdoor Activity": "Stay indoors"},
    ])
    st.dataframe(aq_categories, width="stretch", hide_index=True)

    st.markdown("---")

    # --- Current Conditions Assessment ---
    st.subheader("Recent Conditions Assessment")
    
    # Get most recent data
    recent_df = df.tail(30).copy()
    latest = df.iloc[-1]
    latest_pm25 = latest["pm25"]
    latest_date = latest["date"]
    
    # Determine category
    def get_category(pm25):
        if pm25 <= 12:
            return "Good", "#28a745"
        elif pm25 <= 25:
            return "Moderate", "#ffc107"
        elif pm25 <= 35:
            return "Unhealthy (Sensitive)", "#fd7e14"
        elif pm25 <= 55:
            return "Unhealthy", "#dc3545"
        elif pm25 <= 150:
            return "Very Unhealthy", "#6f42c1"
        else:
            return "Hazardous", "#6c757d"

    cat_name, cat_color = get_category(latest_pm25)

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest PM2.5", f"{latest_pm25:.1f} µg/m³", cat_name)
    col2.metric("Date", f"{latest_date:%Y-%m-%d}")
    col3.metric("30-Day Average", f"{recent_df['pm25'].mean():.1f} µg/m³")
    
    # Activity recommendation
    if latest_pm25 <= 12:
        st.success("**Great day for outdoor activities!** Air quality is good. Enjoy running, cycling, hiking, or any outdoor exercise.")
    elif latest_pm25 <= 25:
        st.info("**Good conditions for most people.** Unusually sensitive individuals should consider reducing prolonged outdoor exertion.")
    elif latest_pm25 <= 35:
        st.warning("**Sensitive groups should take precautions.** Children, elderly, and those with respiratory conditions should limit prolonged outdoor exertion.")
    elif latest_pm25 <= 55:
        st.warning("**Everyone should limit outdoor activities.** Consider indoor exercise alternatives. Keep windows closed.")
    else:
        st.error("**Avoid outdoor activities.** Stay indoors with windows closed. Use air purifiers if available. Wear N95 mask if you must go outside.")

    st.markdown("---")

    # --- Smoke Season Calendar ---
    st.subheader("Vancouver Smoke Season Calendar")
    st.markdown(
        "Based on the available historical data, here's when smoke events typically occur in Vancouver:"
    )
    
    # Monthly smoke day frequency
    monthly_smoke = df.groupby(df["date"].dt.month).apply(
        lambda x: (x["pm25"] > SMOKE_THRESHOLD).sum() / x["date"].dt.year.nunique(),
        include_groups=False,
    ).reset_index()
    monthly_smoke.columns = ["Month", "Avg Smoke Days/Year"]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_smoke["Month Name"] = monthly_smoke["Month"].apply(lambda m: month_names[m-1])
    
    fig_calendar = go.Figure()
    colors = ["#28a745" if x < 0.5 else "#ffc107" if x < 1 else "#fd7e14" if x < 2 else "#dc3545" 
              for x in monthly_smoke["Avg Smoke Days/Year"]]
    fig_calendar.add_trace(go.Bar(
        x=monthly_smoke["Month Name"],
        y=monthly_smoke["Avg Smoke Days/Year"],
        marker_color=colors,
        text=monthly_smoke["Avg Smoke Days/Year"].round(1),
        textposition="outside",
    ))
    fig_calendar.update_layout(
        height=350, yaxis_title="Average Smoke Days per Year",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_calendar, width="stretch")
    
    # Peak months insight
    peak_month = monthly_smoke.loc[monthly_smoke["Avg Smoke Days/Year"].idxmax()]
    st.info(
        f"**Peak smoke month:** {peak_month['Month Name']} with an average of "
        f"{peak_month['Avg Smoke Days/Year']:.1f} smoke days per year. "
        "Plan outdoor events accordingly and consider backup indoor options during July-September."
    )

    st.markdown("---")

    # --- Planning Recommendations ---
    st.subheader("Planning Recommendations")

    col_plan1, col_plan2 = st.columns(2)

    with col_plan1:
        st.markdown("**Outdoor Exercise**")
        st.markdown("""
        - **Best months:** October - June (lowest smoke risk)
        - **Risky months:** July - September (peak fire season)
        - **Check daily:** PM2.5 levels before morning runs
        - **Backup plan:** Have indoor gym access during smoke events
        """)

        st.markdown("**Children & Schools**")
        st.markdown("""
        - Monitor air quality for recess/outdoor PE
        - Keep children indoors when PM2.5 > 25
        - Ensure classrooms have good air filtration
        """)

    with col_plan2:
        st.markdown("**Outdoor Events**")
        st.markdown("""
        - **Weddings/festivals:** Schedule before July or after September
        - **Camping trips:** Have alternate dates or locations ready
        - **Sports tournaments:** Indoor backup venues recommended
        """)

        st.markdown("**Home Preparation**")
        st.markdown("""
        - Invest in HEPA air purifiers (1 per 500 sq ft)
        - Stock N95/P100 masks for outdoor use
        - Seal windows and doors during smoke events
        - Create a "clean air room" in your home
        """)

    st.markdown("---")
    
    # --- Historical Context ---
    st.subheader("Historical Trend: Is It Getting Worse?")
    
    # Yearly smoke days trend
    yearly_smoke = df.groupby("year").apply(
        lambda x: pd.Series({
            "Smoke Days": (x["pm25"] > SMOKE_THRESHOLD).sum(),
            "Mean PM2.5": x["pm25"].mean(),
            "Max PM2.5": x["pm25"].max(),
        }),
        include_groups=False,
    ).reset_index()
    
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=yearly_smoke["year"], y=yearly_smoke["Smoke Days"], name="Smoke Days", marker_color="#ff7f0e"),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Scatter(x=yearly_smoke["year"], y=yearly_smoke["Mean PM2.5"], name="Annual Mean PM2.5", 
                   mode="lines+markers", line=dict(color="#1f77b4", width=2)),
        secondary_y=True
    )
    fig_trend.update_layout(
        height=400, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig_trend.update_yaxes(title_text="Smoke Days (>25 µg/m³)", secondary_y=False)
    fig_trend.update_yaxes(title_text="Mean PM2.5 (µg/m³)", secondary_y=True)
    st.plotly_chart(fig_trend, width="stretch")
    
    # Trend analysis
    recent_5yr = yearly_smoke[yearly_smoke["year"] >= 2020]["Smoke Days"].mean()
    early_5yr = yearly_smoke[yearly_smoke["year"] <= 2005]["Smoke Days"].mean()
    
    if recent_5yr > early_5yr * 1.5:
        st.warning(
            f"**Smoke events are increasing.** Recent years (2020-2025) average {recent_5yr:.1f} smoke days/year "
            f"compared to {early_5yr:.1f} in early years (2000-2005). Climate change is extending fire seasons."
        )
    else:
        st.info(
            f"**Smoke events vary year-to-year.** Recent average: {recent_5yr:.1f} smoke days/year. "
            "Major smoke events are episodic and depend on fire locations and wind patterns."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.header("Data Explorer")

    # Year filter
    all_years = sorted(df["year"].unique())
    year_filter = st.multiselect("Filter by year", all_years, default=all_years)
    explorer_df = df[df["year"].isin(year_filter)]

    # Custom scatter plot
    st.subheader("Custom Scatter Plot")

    numeric_cols = sorted([
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["year"]
    ])

    cx, cy, cc = st.columns(3)
    with cx:
        x_var = st.selectbox(
            "X-axis", numeric_cols,
            index=numeric_cols.index("pm25") if "pm25" in numeric_cols else 0,
        )
    with cy:
        default_y = "fire_count_total" if "fire_count_total" in numeric_cols else numeric_cols[1]
        y_var = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(default_y))
    with cc:
        color_options = ["None", "is_smoke_day", "year", "month", "is_fire_season"]
        color_var = st.selectbox("Color by", color_options)

    scatter_df = explorer_df.copy()
    if color_var != "None":
        scatter_df[color_var] = scatter_df[color_var].astype(str)

    fig_ex = px.scatter(
        scatter_df, x=x_var, y=y_var,
        color=color_var if color_var != "None" else None,
        opacity=0.4,
        hover_data=["date"],
    )
    fig_ex.update_layout(height=500, margin=dict(l=40, r=20, t=20, b=40))
    st.plotly_chart(fig_ex, width="stretch")

    # Correlation with PM2.5
    st.subheader("Correlation with PM2.5")

    key_vars = [
        "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
        "fire_count_close", "fire_count_medium", "fire_count_far", "fire_count_total",
        "fire_frp_sum_total",
        "temperature_2m_mean", "relative_humidity_2m_mean",
        "wind_speed_10m_mean", "wind_u_component", "wind_v_component",
        "precipitation_sum", "pressure_msl_mean",
    ]
    corr = explorer_df[["pm25"] + key_vars].corr()["pm25"].drop("pm25").sort_values(ascending=False)

    fig_corr = px.bar(
        x=corr.values, y=corr.index, orientation="h",
        labels={"x": "Correlation with PM2.5", "y": "Feature"},
        color=corr.values, color_continuous_scale="RdBu_r", range_color=[-1, 1],
    )
    fig_corr.update_layout(
        height=500, coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_corr, width="stretch")

    # Raw data table
    st.subheader("Raw Data")
    display_cols = [
        "date", "pm25", "fire_count_total", "fire_frp_sum_total",
        "temperature_2m_mean", "relative_humidity_2m_mean",
        "wind_speed_10m_mean", "precipitation_sum", "is_smoke_day",
    ]
    st.dataframe(
        explorer_df[display_cols].round(2),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download filtered data as CSV",
        explorer_df.to_csv(index=False),
        file_name="vancouver_smoke_data_filtered.csv",
        mime="text/csv",
    )
