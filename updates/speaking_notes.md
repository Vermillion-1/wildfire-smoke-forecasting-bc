# Speaking Notes — Vancouver Wildfire Smoke Predictor
> **Total time:** ~8 minutes  |  10 slides
> Read this out loud a few times before your presentation.
> The timing guides are suggestions — adjust to your natural pace.

---
## Slide 1: Title Slide  `~30 seconds`

Good afternoon everyone. My name is Aarish Khanna, and today I'm presenting our project on predicting and profiling wildfire smoke impacts on Vancouver's air quality — covering 25 years of data from 2000 to 2025.

We collected and merged three independent datasets — air quality measurements, NASA satellite fire detections, and historical weather — into a single 9,133-day record, and used it to answer three research questions about smoke prediction, smoke transport, and long-term trends.

Let me walk you through what we found.

---

## Slide 2: Why This Project?  `~60 seconds`

Every summer, residents of Vancouver wake up to orange skies and the smell of smoke. This isn't just unpleasant — it's a real public health emergency.

The pollutant we're focused on is PM2.5 — fine particulate matter smaller than 2.5 micrometres. These particles are so small they bypass your nose and throat and go directly into the deepest parts of your lungs, and from there into your bloodstream. Even short exposures can cause heart attacks, strokes, and respiratory failure in vulnerable people.

On the right you can see the air quality scale. Anything above 25 micrograms per cubic metre is considered poor — outdoor activity should be avoided. On September 13th, 2020, Vancouver recorded 163.5 — that's nearly seven times the "poor" threshold. Schools closed. People were told to stay indoors.

With climate change making fire seasons longer and more intense, we asked three questions: Can we predict smoke days in advance? How quickly does smoke travel from fire to city? And is this actually getting worse over time?

---

## Slide 3: Data Sources & Dataset  `~60 seconds`

We pulled data from three sources and merged them into one clean dataset.

First, air quality. The BC Ministry of Environment operates 25 monitoring stations across Metro Vancouver. We downloaded hourly PM2.5 readings from their FTP server for every year from 2000 to 2024, and averaged all stations to get a single daily city-level value. This becomes our target variable — what we're trying to predict.

Second, fire activity. NASA operates two satellites — MODIS from 2000, and the higher-resolution VIIRS from 2012 — that detect active fires globally. We downloaded every fire detection within 1,000 kilometres of Vancouver, calculated the exact distance from each fire to the city centre, and grouped them into three distance bands: close, medium, and far. We recorded both fire count and fire radiative power — a measure of intensity — for each band each day.

Third, weather. We pulled historical hourly weather from the Open-Meteo API and aggregated it to daily values: temperature, humidity, wind speed and direction, precipitation, and atmospheric pressure.

The final merged dataset has 9,133 rows — one per day with zero gaps — and 90 columns. From those 90 we engineered 34 features for modeling.

---

## Slide 4: Data Science Pipeline  `~45 seconds`

Here's the full pipeline in seven steps.

We start with data collection — downloading from BC government FTP, NASA FIRMS, and Open-Meteo. Step two is preprocessing: collapsing hourly readings to daily values and filtering fires to the BC bounding box.

Step three is feature engineering — the most important design choice. We created lag features so the model can see what PM2.5 and fire activity looked like one, two, three, and seven days ago. We also added calendar flags like 'is fire season' for June through September.

Steps four through six are the analysis notebooks. EDA profiles the seasonal patterns and trends. The lag analysis uses cross-correlation to find the delay between fire and smoke. Modeling trains and compares ten machine learning models.

Finally, everything feeds into a Streamlit dashboard with seven interactive tabs — which I'll show you at the end.

---

## Slide 5: RQ3: Is the Smoke Season Getting Worse?  `~75 seconds`

Let's start with Research Question 3, because I think it's the most compelling — and the most relevant for public policy.

We used Spearman rank correlation to test for trends in three metrics across 24 years of annual observations. Spearman is appropriate here because annual smoke counts are not normally distributed, and we care about monotonic trends — consistently going up — not perfectly linear ones.

For smoke day frequency, we found rho equals 0.48 with a p-value of 0.015. For peak PM2.5 severity, rho equals 0.65 with p equals 0.043. For the length of the longest smoke episode each year, rho equals 0.48, p equals 0.015. All three are statistically significant at the standard 5% threshold.

Notice that the average fire season PM2.5 was NOT significant — p equals 0.33. This is an important nuance: most days are still clean. What's changing is the extremes — they're getting worse.

The table on the right tells the story year by year. 2017 had 13 smoke days. 2018 saw a peak of 112. And 2020 shattered all records at 163.5 micrograms — and I'll tell you shortly why 2020 was especially unusual.

---

## Slide 6: RQ3: What Conditions Cause Smoke Days?  `~60 seconds`

Beyond the long-term trend, we wanted to understand what makes a day a smoke day. So we compared the 46 smoke days against the 9,087 normal days across five weather variables using the Mann-Whitney U test — a non-parametric test that doesn't assume normally distributed data.

The results are striking. Smoke days are on average 9.2 degrees Celsius hotter, receive 4.5 millimetres less rain, have winds 1.5 kilometres per hour calmer, and 4.8 percent lower humidity. Every single difference is statistically significant.

The physical mechanism is a stagnant high-pressure system, or anticyclone. When high pressure sits over the region, air sinks from above and warms as it compresses — that's why temperatures are high. The sinking air suppresses the formation of clouds and rain. Surface winds are calm. And critically, a temperature inversion forms — a layer of warm air traps cooler, polluted air near the ground, preventing the smoke from rising and dispersing.

The cruel irony is that these same conditions — hot, dry, windless — are exactly what dry out forests and allow fires to spread in the first place. Weather doesn't just transport smoke; it creates both the fire and the conditions that trap the smoke in the city.

---

## Slide 7: RQ2: How Long Does Smoke Take to Reach Vancouver?  `~60 seconds`

For Research Question 2, we used cross-correlation analysis. The idea is simple: we slide the fire activity curve forward in time — one day, two days, and so on — and measure how well it correlates with PM2.5 at each shift. The shift where correlation is highest is the lag.

We ran this for fire radiative power and fire count, separately for three distance bands, looking at lags from minus seven to plus fourteen days — during fire season only.

The result: peak correlation occurs at lag zero across all three distance bands. That means fire activity and PM2.5 peak on the same calendar day. This could mean smoke transport is sub-daily — arriving within hours — or that both fire and smoke are driven by the same weather conditions simultaneously.

An important caveat: with daily data, we cannot distinguish any lag shorter than 24 hours. A 6-hour transport time looks identical to a 0-hour transport time.

Now, about September 2020. That record event at 163.5 micrograms? BC fire activity was actually below average that year. The smoke came from catastrophic wildfires in Oregon and Washington state, carried north by a southerly wind pattern. Our model had no way to know this — our fire data only covers British Columbia. This cross-border blind spot is one of our key limitations.

---

## Slide 8: RQ1: Can ML Predict Tomorrow's PM2.5?  `~75 seconds`

And now for Research Question 1 — the one with the most nuanced answer.

We trained ten machine learning models: linear regression, Ridge, Lasso, two configurations each of Random Forest, XGBoost, and LightGBM. We used a five-fold expanding-window time series split to avoid data leakage.

**Finding A — Overall MAE:** Persistence wins. Its mean absolute error of 1.694 micrograms per cubic metre beats the best standard ML model — Random Forest at 1.783. This is because PM2.5 has extremely strong day-to-day autocorrelation — r equals 0.82. If today is clean, tomorrow will probably be clean too.

**Finding B — Smoke-day MAE:** But persistence fails catastrophically on the 46 smoke days in the dataset. Its smoke-day MAE is 22.353 — it always lags a spike by one day, which is operationally useless for health alerts. So we ran nine follow-up experiments targeting this problem.

The key insight was residual framing: instead of predicting raw PM2.5, predict the *change* — tomorrow minus today. This makes persistence the model's default prior and converts the autocorrelated series into a near-stationary regression problem. Layered on top of that, quantile regression at alpha equals 0.80 shifts the model's objective from minimising median error to minimising a higher quantile — penalising under-prediction four times more than over-prediction. Combined with a fire-only feature set of 24 variables — dropping temperature, humidity, and pressure, which dominate clean days but add noise during fire events — we achieve a smoke-day MAE of 21.729, beating persistence by 0.624 micrograms per cubic metre.

In parallel, an XGBoost binary classifier catches 8 of 13 historical smoke episodes — and the 5 it misses each have a structural physical explanation, not a modeling failure.

---

## Slide 9: RQ1: Why Does Persistence Win — And How We Beat It On Smoke Days  `~60 seconds`

This slide shows two complementary analyses.

On the left is the Random Forest feature importance ranking. Yesterday's PM2.5 alone accounts for roughly 40% of the model's predictive power. This is why persistence is so hard to beat overall — the strongest signal in the data is already yesterday's value.

On the right is the ablation study — we systematically remove each feature group. The key result highlighted in green: removing fire features actually *improves* the standard model. MAE drops by 0.043. Satellite fire counts at daily resolution are actively adding noise, because a fire count doesn't encode whether the wind is pointing toward Vancouver or away from it. Weather features — wind direction and pressure — already capture stagnation conditions, making fire counts redundant.

But here is where the story gets interesting. That ablation insight directly motivated our advanced models. If weather features regularise the model toward normal-day behavior, and fire counts are only informative during actual fire events — then for smoke-day prediction, we should strip out weather and focus on fire signals. That is exactly what our fire-only feature set does: 24 features — PM2.5 lags, fire FRP and count by distance band, and wind vectors.

Combined with residual framing and quantile regression at Q equals 0.80, the result is a smoke-day MAE of 21.729 — a 0.624 improvement over persistence. The quantile sweep confirms the tradeoff is perfectly monotone: higher alpha always buys better smoke-day MAE at the cost of overall MAE. There is no free lunch, but the tradeoff is predictable and controllable — which is exactly what you want for a public health use case where under-predicting a smoke spike is far more dangerous than over-predicting one.

---

## Slide 10: Data Product, Lessons & Future Work  `~45 seconds`

Our data product is a Streamlit dashboard with seven interactive tabs. Tabs one through three give an overview of the 25-year record, the seasonal trends, and the lag analysis with interactive case studies. Tab four lets you compare all models and drill into feature importance. Tab five shows the holdout validation on 2021 to 2024 data. Tab six maps PM2.5 values to BC Air Quality Index categories with health guidance. And tab seven is a full data explorer where you can filter any slice of the dataset and download it as CSV.

In terms of what we learned: negative results are valuable. Persistence beating ML is not a disappointment — it teaches us that PM2.5 is autocorrelation-dominated and that daily fire data isn't the right input for this forecasting task. We also learned that proper temporal cross-validation is non-negotiable for time series, and that feature ablation often reveals surprises you'd never find just from importance scores.

If we had more time, the four highest-priority directions would be: hourly data to resolve true sub-day transport lags, adding Pacific Northwest US fire data to capture cross-border events, integrating HYSPLIT atmospheric trajectory model outputs as features, and experimenting with LSTM sequence models.

Thank you — I'm happy to take questions.

---

## Handling Q&A

Common questions and quick answers:

**Q: Why does persistence beat ML?**

PM2.5 has lag-1 autocorrelation of r = 0.82 — if today is smoky, tomorrow almost certainly will be too. ML models try to learn complex fire/weather relationships, but none of that adds more signal than yesterday's value alone.

**Q: How do you know the trends are statistically significant?**

We used Spearman rank correlation on 24 annual observations. For smoke day frequency, rho = 0.48, p = 0.015. Since p < 0.05, we reject the null hypothesis that there is no trend. Spearman is appropriate because annual counts are non-normal.

**Q: Why did removing fire features improve the model?**

Fire counts don't capture atmospheric transport. A fire 200 km away might produce zero smoke in Vancouver if wind is pointing the wrong way. Weather features already encode whether stagnation conditions exist, so fire counts add noise on top.

**Q: The worst event came from US fires — doesn't that break your analysis?**

It's a key limitation we discuss explicitly. Our fire data covers only BC. The September 2020 event shows that cross-border transport is real and important. It motivates adding Pacific Northwest US fire data as future work.

**Q: Can this model be used operationally?**

For day-ahead forecasting, persistence is currently the best approach from this analysis. The practical value lies in the trend profiling and seasonal risk calendar — knowing August–September are high-risk months is actionable for public health campaigns.

**Q: Why cross-correlation and not Granger causality?**

Cross-correlation is simpler and more interpretable for a first-pass lag analysis with a small sample of fire-season days. Granger causality would be a natural extension but requires stationarity testing and additional assumptions we didn't validate here.

**Q: What is FRP and why use it alongside fire count?**

Fire Radiative Power measures how much energy a fire releases in megawatts — a proxy for intensity. Ten small fires might produce less smoke than one intense fire, so using both count and FRP gives a richer picture of total fire activity.

**Q: Does any model actually beat persistence?**

Yes — on smoke-day MAE specifically. Quantile regression at Q=0.80 with fire-only features (24 variables) achieves a smoke-day MAE of 21.729, beating persistence's 22.353 by 0.624 micrograms per cubic metre. The tradeoff is that overall MAE is worse — 2.075 versus 1.668 — because the model deliberately predicts higher values on average. That's an acceptable tradeoff for a public health alerting use case where missing a smoke spike is far more costly than a false alarm.

**Q: What is quantile regression and why use it here?**

Standard regression minimises mean absolute error, which teaches the model to predict near the median of the distribution. For a dataset where 99.5% of days are clean, the median is always "clean" — so the model never learns to predict spikes. Quantile regression at alpha=0.80 uses an asymmetric loss function that penalises under-prediction four times more than over-prediction. This forces the model to predict the 80th percentile of the next-day change distribution. The tradeoff with alpha is monotone and predictable: every 0.05 step up in alpha improves smoke-day MAE by about 0.2 to 0.3 micrograms and worsens overall MAE by about 0.3 to 0.5. One number controls the entire risk-accuracy balance.

**Q: How does the XGBoost smoke detector work and how well does it perform?**

It is a binary classifier — does tomorrow's PM2.5 exceed 25 micrograms or not? We handle the 200-to-1 class imbalance using XGBoost's scale_pos_weight parameter. The threshold is tuned to maximise F2 score, which weights recall twice as heavily as precision — appropriate when missing a smoke event is worse than a false alarm. In out-of-fold evaluation across five expanding time-series folds, it achieves an AUPRC of 0.331 versus a random baseline of about 0.005, and catches 8 of 13 unique historical smoke episodes. The 5 it misses each have a structural physical explanation: the 2020 Labor Day event came from Oregon and California fires that are invisible in BC-only satellite data; 2017 involved only distant fires with no close-range signal; 2018 was fast-onset where fire activity jumped on day 2 of the episode; 2023 was a single pre-smoke day; and 2005 falls in the pre-2010 era where training data is too sparse.

