# Milestone Presentation Material

**Vancouver Wildfire Smoke Predictor — 5-Minute Milestone Presentation**

CMPT 733 — Big Data Lab II

---

## Timing Summary

| Slide | Topic | Time |
|---|---|---|
| 1 | Title | 10s |
| 2 | The Problem | 50s |
| 3 | Three Questions | 30s |
| 4 | Data Sources | 30s |
| 5 | Progress Timeline | 60s |
| 6 | Early Findings from EDA | 30s |
| 7 | Feasibility & Risks | 45s |
| 8 | Team Roles | 20s |
| 9 | Questions | 5s |
| **Total** | | **~4 min 40s** |

~20 seconds of buffer. Comfortable under 5 minutes, well under the 6-minute penalty threshold.

---

## Slide 1: Title (~10 seconds)

### What to put on the slide

- Title: "Predicting Wildfire Smoke Impacts on Vancouver's Air Quality"
- Subtitle: CMPT 733 — Milestone Presentation
- Team member names

### What to say

> "Hi everyone, we're [team name/names]. Our project investigates wildfire smoke and air quality in Vancouver using 25 years of public data."

---

## Slide 2: The Problem (~50 seconds) — MOTIVATION [3 pts]

### What to put on the slide

- A photo of Vancouver under smoke haze (use a real one from 2017 or 2020 — Google Images has many)
- Two numbers in large font: **163 ug/m3** and **46 smoke days in 25 years**

### What to say

> "If you've lived in Vancouver during August, you've probably experienced this. The sky turns orange, you can taste the air, and you're told to stay indoors. This isn't from local pollution — it's wildfire smoke traveling hundreds of kilometers from BC's interior, and sometimes from Washington and Oregon.
>
> PM2.5 — tiny particles about 30 times thinner than a human hair — is the main pollutant during these events. Normal Vancouver air has about 5 micrograms per cubic meter. During the September 2020 smoke event, it hit 163. That's over 30 times normal, and well into the 'very unhealthy' range.
>
> The problem is: these events are hard to predict. They're rare — only 46 days in the last 25 years — but when they happen, they affect millions of people. Can we do better at anticipating them?"

---

## Slide 3: Our Three Questions (~30 seconds) — MOTIVATION [continued]

### What to put on the slide

- Three numbered questions, each one line:
  1. Can we predict tomorrow's PM2.5 using fire and weather data?
  2. How long does smoke take to travel from a wildfire to Vancouver?
  3. Is Vancouver's smoke season getting worse over time?

### What to say

> "We structured our project around three research questions. First, can simple machine learning models forecast next-day air quality better than just assuming tomorrow equals today? Second, when a fire flares up in BC's interior, how many days until the smoke reaches Vancouver? And third, looking at the 2000-2025 period, is smoke season actually getting worse, or does it just feel that way?"

---

## Slide 4: Data Sources (~30 seconds) — PROGRESS [2.5 pts]

### What to put on the slide

- Three boxes side by side:
  - **Air Quality** — BC Gov FTP, 25 Metro Vancouver stations, hourly PM2.5 -> daily averages
  - **Wildfires** — NASA satellites (MODIS + VIIRS), every fire detection in BC, distance to Vancouver computed
  - **Weather** — Open-Meteo API, temperature, humidity, wind, precipitation
- Bottom: "Merged dataset: **9,133 days x 90 features** (Jan 2000 - Jan 2025)"

### What to say

> "All our data comes from free, public sources. Air quality readings from Metro Vancouver monitoring stations via the BC government. Satellite fire detections from NASA — each one includes location and intensity. And daily weather from Open-Meteo. We built an automated pipeline that downloads, cleans, and merges all three into a single dataset: 9,133 daily records with 90 features. Zero missing days."

---

## Slide 5: What We've Done So Far (~60 seconds) — PROGRESS [continued]

### What to put on the slide

- A timeline graphic:
  - **Weeks 1-2:** Data collection pipeline (4 scripts, 3 APIs) ---- DONE
  - **Weeks 3-4:** Exploratory data analysis + seasonal profiling (RQ3) ---- DONE
  - **Weeks 5-6:** Smoke arrival lag analysis (RQ2) ---- IN PROGRESS
  - **Weeks 7-8:** Prediction modeling (RQ1) ---- UPCOMING
  - **Weeks 9-10:** Dashboard + report ---- UPCOMING
- 2-3 thumbnail figures from EDA only (e.g., pm25_timeseries.png, smoke_season_trends.png, weather_smoke_comparison.png)

### What to say

> "Here's where we stand. The data pipeline is fully built and reproducible — four scripts that anyone can run to reconstruct the dataset from scratch. Three sources, merged into 9,133 daily records with 90 features and zero missing days.
>
> We've completed our exploratory data analysis, which answers our third research question on seasonal risk. Key findings: smoke events cluster in August and September, they happen during hot, dry, calm weather, and over 2000-2024 we now detect statistically significant upward trends in smoke-day frequency, peak severity, and episode duration.
>
> We're currently in the middle of our lag analysis for research question two — investigating how long it takes for smoke from BC wildfires to reach Vancouver. Early cross-correlation results suggest the peak is at zero days at daily resolution, meaning transport may happen within the same calendar day. We're still digging into per-year patterns and wind direction effects.
>
> Prediction modeling is next."

---

## Slide 6: Early Findings from EDA (~30 seconds) — PROGRESS [continued]

### What to put on the slide

- Three key stats in large font:
  - **46 smoke days** out of 9,133 (0.5%)
  - **2020:** Worst event (163.5 ug/m3) — but from US fires, not BC
  - **PM2.5 lag-1 autocorrelation: 0.82** — today strongly predicts tomorrow
- Optional: one figure thumbnail (pm25_timeseries.png or smoke_event_calendar.png)

### What to say

> "A few highlights from the EDA. Smoke events are extremely rare — just 1.2% of days. The worst single event was September 2020, which was actually caused by fires in Oregon and Washington, not BC. That's an important finding for us because our fire data only covers BC.
>
> We also found that PM2.5 is highly autocorrelated — today's value explains about 67% of tomorrow's. This suggests that any prediction model will need to beat a very strong persistence baseline, which will be a key challenge when we get to modeling."

---

## Slide 7: Feasibility & Next Steps (~45 seconds) — FEASIBILITY [2 pts]

### What to put on the slide

- **Completed:**
  - Data pipeline (all 3 sources merged)
  - EDA and seasonal risk profiling (RQ3)
- **In progress:**
  - Smoke arrival lag analysis (RQ2) — cross-correlation, wind analysis, case studies
- **Upcoming:**
  - Prediction modeling (RQ1) — persistence baseline, linear models, random forest
  - Time-series cross-validation (not random splits — avoids data leakage)
  - Interactive Streamlit dashboard
  - Final report and presentation
- **Risks & mitigations:**

| Risk | Mitigation |
|---|---|
| ML may not beat a simple persistence baseline | Report honestly as a finding; reframe as smoke event detection |
| Only 46 smoke events in 25 years | Use time-series CV; focus on interpretability over raw accuracy |
| BC-only fire data misses cross-border smoke | Document as limitation; propose US fire data as future work |

### What to say

> "Data collection and EDA are done. Lag analysis should be wrapped up this week. Then we move into modeling — we plan to test persistence and rolling-mean baselines, linear and regularized regression, and random forest, all with time-series cross-validation to avoid leaking future data into training.
>
> One risk we're already anticipating: the high autocorrelation we found in EDA means a simple 'tomorrow equals today' baseline could be very hard to beat. If that happens, we'll report it honestly and explore reframing the problem as binary smoke event detection.
>
> Another risk is that our fire data only covers BC, but we've already seen that cross-border smoke caused the worst event. We'll document that as a limitation.
>
> We're on track — the hardest part, building the dataset, is behind us."

---

## Slide 8: Team Roles (~20 seconds) — MEMBER ROLES [0.5 pts]

### What to put on the slide

| Member | Contribution |
|---|---|
| [Member B] | [TBD] |
| [Member C] | [TBD] |
| [Member D] | [TBD] |

### What to say

> "Briefly on roles: Dhwani led the exploratory data analysis and the smoke arrival lag investigation. [Member B] handled [X]. [Member C] handled [Y]. [Member D] handled [Z]. We collaborate on analysis decisions and review each other's work."

---

## Slide 9: Thank You / Questions (~5 seconds)

### What to put on the slide

- "Thank you — Questions?"
- GitHub repo link (if public)

---

## Q&A Preparation

Likely questions and short answers to prepare for:

**"Why not use deep learning?"**
> Our dataset has 9,133 rows and only 46 smoke events. Deep learning needs far more extreme-event examples to generalize. Simple models are more appropriate and interpretable here.

**"Why daily resolution instead of hourly?"**
> The BC government provides hourly PM2.5, but the fire and weather data are most reliably available at daily aggregation. We started with daily to keep the three sources aligned. Hourly is a potential future improvement.

**"Is 25 years enough data?"**
> It is much stronger for trend analysis than a shorter window and now supports significant trend tests in several smoke metrics. For modeling, the main bottleneck remains event rarity (46 smoke days, 0.5% of all days), which limits what any model can learn about extremes.

**"What's the most surprising finding so far?"**
> That the worst PM2.5 event in our dataset (September 2020, 163.5 ug/m3) was caused by US fires, not BC fires. Our fire data only covers BC, so this is a clear gap we need to acknowledge.

**"How do you handle data leakage in time series?"**
> We use expanding-window time-series cross-validation. Training data always comes before test data chronologically. We also fit our feature scalers only on training data within each fold.

**"What if your models can't beat the persistence baseline?"**
> That's a valid and publishable finding. It tells us that at daily resolution with these features, PM2.5 is dominated by short-term autocorrelation. Beating persistence would likely require hourly data, atmospheric transport models, or cross-border fire data.
