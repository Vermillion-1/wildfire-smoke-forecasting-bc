⏺ Now I have everything I need. Let me write the final plan.

  Read 1 file (ctrl+o to expand)

⏺ Now I have a complete picture. Let me write the final plan.

⏺ Updated plan
  ⎿  File must be read first

  Read 1 file (ctrl+o to expand)

⏺ Updated plan
  ⎿  /plan to preview

⏺ User approved Claude's plan
  ⎿  Plan saved to: ~/.claude/plans/witty-floating-coral.md · /plan to edit
     Docs + TODO Sync Plan (Post-Experiment)

     Context

     All modeling experiments are complete (9 experiments, all results in experiment_results.md
     and TODO.md). However, the written deliverables (speaking_notes.md, report_final.md)
     still describe only the original RQ1 finding ("persistence beats all ML models") and are
     completely unaware of the follow-up modeling work.

     Presentation: April 7, 2026 (tomorrow) — poster format, 8 min (prof_guidelines.md:143)
     Code + Report: 15 pts — submitted with/after presentation

     The user wants: (1) eval what's pending, (2) identify easy improvements, (3) update TODO.

     ---
     Gap Analysis

     speaking_notes.md — CRITICAL (presentation tomorrow)

     Slide 8 currently says:
     ▎ "The result: persistence wins. Its mean absolute error of 1.694 micrograms beats the best
     ▎ ML model — Random Forest at 1.783."

     Slide 9 covers only the ablation finding. Q&A has no entries for quantile regression or gate.
     The entire follow-up modeling phase (9 experiments) is absent.

     New RQ1 framing (two findings):
     - Finding A: Persistence wins on overall MAE (1.694 vs LightGBM MAE 1.504 overall; for the
     original 10-model comparison Random Forest 1.783). Autocorrelation (r=0.82) explains this.
     - Finding B: Persistence fails on smoke days (smoke-day MAE 22.353). We targeted this with
     quantile regression. Best result: Q=0.80 fire-only (24 features) — smoke-day MAE 21.729,
     beats persistence by 0.624 µg/m³. XGBoost classifier catches 8/13 historical smoke episodes
     (AUPRC=0.331).

     report_final.md — HIGH (graded for technical depth)

     Section 5 (Evaluation, RQ1) concludes:
     ▎ "The persistence baseline achieves MAE = 1.694 µg/m³ and R² = 0.600, outperforming every
     ▎ trained model."

     The entire follow-up modeling phase is missing. Prof grading criterion explicitly values
     "complexity and sophistication of techniques" and "thorough assessment of approach and results."
     Adding ~300 words here is easy and materially improves the submission quality.

     TODO.md — needs pending-tasks section updated

     ---
     Action Plan

     Step 1: Update TODO.md

     Replace the "What Remains (Non-Modelling)" section at bottom with accurate priority list:

     ┌────────────────────────────────────────────────────────────────────────┬──────────┬──────────────────────┐
     │                                  Task                                  │ Priority │       Deadline       │
     ├────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────┤
     │ Update speaking_notes.md (Slides 8, 9, Q&A)                            │ CRITICAL │ Before April 7       │
     ├────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────┤
     │ Git commit all untracked scripts                                       │ HIGH     │ Before April 7       │
     ├────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────┤
     │ Update report_final.md (expand RQ1 section)                            │ HIGH     │ With code submission │
     ├────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────┤
     │ Update app/streamlit_app.py smoke alert (add XGBoost gate probability) │ MEDIUM   │ —                    │
     ├────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────┤
     │ notebooks/05_regime_modeling.ipynb                                     │ LOW      │ —                    │
     ├────────────────────────────────────────────────────────────────────────┼──────────┼──────────────────────┤
     │ Phase-shift visualization (Sept 2020)                                  │ LOW      │ —                    │
     └────────────────────────────────────────────────────────────────────────┴──────────┴──────────────────────┘

     ---
     Step 2: Update speaking_notes.md

     File: /Users/ankushsingh/Desktop/CMPT_733/Project/speaking_notes.md

     Slide 8 (~lines 97-111) — replace body with two-finding structure:

     Finding A (persistence vs standard ML, ~2 sentences — same as before but brief):
     Standard ML (Random Forest best at MAE 1.783) loses to persistence (1.694). Strong
     lag-1 autocorrelation (r=0.82) means yesterday's value is almost always a better forecast
     than any learned model.

     Finding B (smoke-day targeting — NEW):
     Persistence fails catastrophically on smoke days: it lags spikes by one day and has
     smoke-day MAE of 22.353. We ran 9 follow-up experiments targeting this. Key insight:
     predicting the CHANGE in PM2.5 (residual framing) rather than raw PM2.5 is the single
     most important architectural choice — it makes persistence the model's default prior.
     With quantile regression at Q=0.80 and fire-only features (24 features, dropping weather
     variables that dominate normal days), we achieve smoke-day MAE 21.729 — beating persistence
     by 0.624 µg/m³. A parallel XGBoost binary classifier catches 8 of 13 historical smoke
     episodes (AUPRC=0.331 vs random baseline of ~0.005).

     Slide 9 (~lines 113-124) — add one paragraph before the ablation:
     Mention residual framing + quantile regression as the key finding. "The fire-only feature
     subset finding (+0.031 smoke-day MAE) from the ablation directly motivated dropping weather
     features in the quantile model."

     Q&A section (~lines 140+) — add 3 entries:
     - Q: Does any model beat persistence?
     A: Yes, on smoke-day MAE specifically. Q=0.80 fire-only beats persistence by 0.624 µg/m³
     on the 46 smoke days. Overall MAE is worse (2.075 vs 1.668) because the model predicts
     higher values on average — a deliberate trade-off for public health alerting.
     - Q: What is quantile regression and why use it?
     A: Standard regression minimises mean error, so it learns to predict near the median —
     which for a dataset 99.5% clean days is "always clean." Quantile regression at alpha=0.80
     minimises an asymmetric loss that penalises under-prediction 4× more than over-prediction,
     forcing the model to predict the 80th percentile of the next-day change distribution. The
     tradeoff is monotone and predictable: higher alpha = better smoke-day MAE, worse overall MAE.
     - Q: How does the smoke detector work?
     A: XGBoost binary classifier with scale_pos_weight to handle 200:1 class imbalance.
     Threshold tuned to maximise F2 score (recall-weighted). OOF evaluation on 5 expanding
     folds: AUPRC=0.331 (vs 0.005 random), catches 8/13 unique smoke episodes. The 5 missed
     episodes each have a structural explanation: out-of-province smoke (2020), distant fires
     only (2017), fast-onset (2018), single-day pre-smoke (2023), pre-2010 sparse data (2005).

     ---
     Step 3: Update report_final.md

     File: /Users/ankushsingh/Desktop/CMPT_733/Project/report_final.md

     Expand Section 5 (Evaluation) RQ1 block. After the existing ablation paragraph, add:

     New subsection heading: "Addressing Smoke-Day Under-Prediction"

     Content (target ~300 words):
     1. Motivation: standard ML and persistence both have smoke-day MAE ~22 µg/m³ — operationally
     useless since a 22 µg/m³ error on a day peaking at 30-163 µg/m³ spans the entire severity range
     2. Residual framing: predicting pm25_diff (tomorrow − today) converts the autocorrelated
     series to near-stationary regression and bakes in persistence as the model's default prior
     3. Quantile regression sweep (Q=0.75 to Q=0.95): monotone tradeoff — each 0.05 alpha step
     costs ~0.3-0.5 overall MAE and gains ~0.2-0.3 smoke-day MAE improvement
     4. Fire-only feature subset (24 features, dropping temperature/humidity/pressure):
     smoke-day MAE 21.729 vs 21.760 with full features — weather features act as regularisers
     toward normal-day behavior, adding noise during fire events
     5. Best result: Q=0.80 fire-only — smoke-day MAE 21.729, beats persistence by 0.624 µg/m³,
     overall MAE 2.075 (deliberate trade-off for alerting use case)
     6. XGBoost smoke detector: AUPRC=0.331, 8/13 OOF episode recall; missed episodes have
     structural physical explanations (cross-border transport, fast-onset, sparse training data)
     7. Negative results: LSTM (27% worse — 9K rows insufficient), hurdle model (stage 2
     underpowered), sample weighting (backfires)

     Also update Section 8 (Summary) RQ1 paragraph: change "outperforming every trained model"
     to reflect the nuanced finding — persistence wins overall but quantile regression recovers
     smoke-day performance.

     ---
     Files to Modify

     1. TODO.md — replace "What Remains" section
     2. speaking_notes.md — Slide 8 body, Slide 9 add paragraph, Q&A add 3 entries
     3. report_final.md — Section 5 expand RQ1, Section 8 update RQ1 summary sentence

     Files Already Correct (no changes needed)

     - experiment_results.md — complete and accurate
     - All scripts in scripts/ — done

     Verification

     - Read back Slides 8 and 9 of speaking_notes.md — numbers must match experiment_results.md
     - Word count report_final.md — target ~2400-2500 words (from current ~2200)

⏺ Starting with TODO.md first.                                     

⏺ Update(TODO.md)                                                                                                                                                                                              
  ⎿  Error editing file
                                                                                                                                                                                                               
  Searched for 1 pattern (ctrl+o to expand)                                                                                                                                                                  
                                                                                                                                                                                                               
⏺ The "What Remains" section was in experiment_results.md, not TODO.md. Let me update experiment_results.md and then do the speaking notes and report.                                                         
                                                                                                                                                                                                             
⏺ Update(experiment_results.md)                                                                                                                                                                                
  ⎿  Added 10 lines, removed 9 lines                                                                                                                                                                           
      230                                                                                                                                                                                                      
      231  ---                                                                                                                                                                                                 
      232                                                                                                                                                                                                      
      233 -## What Remains (Non-Modelling)                                                                                                                                                                   
      233 +## Remaining Tasks                                                                                                                                                                                
      234  
      235 -Modelling is complete. Remaining tasks:                                                                                                                                                      
      235 +Modelling is complete. Remaining tasks ordered by deadline:                                                                                                                                  
      236  
      237 -| Task | Priority | Notes |                                                                                                                                                                  
      238 -|---|---|---|                                                                                                                                                                                
      239 -| `notebooks/05_regime_modeling.ipynb` | High | Present results in notebook format for submission |                                                                                          
      240 -| Commit all scripts | High | 10+ untracked scripts need committing |                                                                                                                        
      241 -| `app/streamlit_app.py` smoke alert | Medium | Add XGBoost gate probability to app |                                                                                                        
      242 -| Phase-shift visualisation (Sept 2020) | Medium | Plot showing model detects rising trend before persistence |                                                                              
      243 -| Future architecture slide | Low | HYSPLIT + US VIIRS fire data diagram |                                                                                                                   
      237 +| Task | Priority | Deadline | Notes |                                                                                                                                                       
      238 +|---|---|---|---|                                                                                                                                                                            
      239 +| Update `speaking_notes.md` (Slides 8, 9 + Q&A) | **CRITICAL** | Before April 7 | RQ1 finding needs new framing — quantile regression + XGBoost gate missing |                              
      240 +| Commit all untracked scripts | HIGH | Before April 7 | 10+ scripts in `scripts/` untracked |                                                                                               
      241 +| Update `report_final.md` (expand Section 5 RQ1 + Section 8) | HIGH | With code submission | Entire follow-up modeling phase absent from report |                                           
      242 +| `app/streamlit_app.py` smoke alert | MEDIUM | — | Add XGBoost gate probability to app |                                                                                                    
      243 +| `notebooks/05_regime_modeling.ipynb` | LOW | — | Present results in notebook format |                                                                                                      
      244 +| Phase-shift visualisation (Sept 2020) | LOW | — | Plot showing quantile model detects rising trend before persistence |                                                                    

⏺ Now update speaking_notes.md — Slide 8 first.                    

⏺ Update(speaking_notes.md)                                                                                                                                                                                    
  ⎿  Added 6 lines, removed 6 lines
       96                                                                                                                                                                                                      
       97  ## Slide 8: RQ1: Can ML Predict Tomorrow's PM2.5?  `~75 seconds`                                                                                                                                  
       98                                                                                                                                                                                                      
       99 -And now for Research Question 1 — the one with the most surprising answer.                                                                                                                        
       99 +And now for Research Question 1 — the one with the most nuanced answer.                                                                                                                             
      100                                                                                                                                                                                                    
      101 -We trained ten machine learning models: linear regression, Ridge, Lasso, two configurations each of Random Forest, XGBoost, and LightGBM. We compared them against two baselines: persistence
          - — which simply predicts tomorrow equals today — and rolling mean, which averages the last one, two, three, and seven days.                                                                  
      101 +We trained ten machine learning models: linear regression, Ridge, Lasso, two configurations each of Random Forest, XGBoost, and LightGBM. We used a five-fold expanding-window time series sp
          +lit to avoid data leakage.                                                                                                                                                                   
      102  
      103 -We used a five-fold expanding-window time series split to avoid data leakage. Each fold trains on everything up to a cutoff date and tests on the next four years — mimicking how a real fore
          -casting system would work.                                                                                                                                                                   
      103 +**Finding A — Overall MAE:** Persistence wins. Its mean absolute error of 1.694 micrograms per cubic metre beats the best standard ML model — Random Forest at 1.783. This is because PM2.5 h
          +as extremely strong day-to-day autocorrelation — r equals 0.82. If today is clean, tomorrow will probably be clean too.                                                                      
      104  
      105 -The result: persistence wins. Its mean absolute error of 1.694 micrograms beats the best ML model — Random Forest at 1.783. And its R-squared of 0.600 is almost four times higher than Rando
          -m Forest's 0.153.                                                                                                                                                                            
      105 +**Finding B — Smoke-day MAE:** But persistence fails catastrophically on the 46 smoke days in the dataset. Its smoke-day MAE is 22.353 — it always lags a spike by one day, which is operatio
          +nally useless for health alerts. So we ran nine follow-up experiments targeting this problem.                                                                                                
      106  
      107 -Why? Because PM2.5 has extremely strong day-to-day autocorrelation — r equals 0.82. If today is clean, tomorrow will probably also be clean. If today is smoky, tomorrow will probably also b
          -e smoky. Persistence exploits this perfectly without needing to learn anything.                                                                                                              
      107 +The key insight was residual framing: instead of predicting raw PM2.5, predict the *change* — tomorrow minus today. This makes persistence the model's default prior and converts the autocor
          +related series into a near-stationary regression problem. Layered on top of that, quantile regression at alpha equals 0.80 shifts the model's objective from minimising median error to minim
          +ising a higher quantile — penalising under-prediction four times more than over-prediction. Combined with a fire-only feature set of 24 variables — dropping temperature, humidity, and press
          +ure, which dominate clean days but add noise during fire events — we achieve a smoke-day MAE of 21.729, beating persistence by 0.624 micrograms per cubic metre.                             
      108  
      109 -This is not a failure of our modeling. It is an honest result that tells us something important about the data: at daily resolution, the signal in fire and weather features does not add eno
          -ugh information beyond what yesterday's PM2.5 already provides.                                                                                                                              
      109 +In parallel, an XGBoost binary classifier catches 8 of 13 historical smoke episodes — and the 5 it misses each have a structural physical explanation, not a modeling failure.               
      110  
      111  ---
      112  

⏺ Now update Slide 9:                                              

⏺ Update(speaking_notes.md)                                                                                                                                                                                    
  ⎿  Added 6 lines, removed 6 lines
      110                                                                                                                                                                                                      
      111  ---                                                                                                                                                                                               
      112                                                                                                                                                                                                      
      113 -## Slide 9: RQ1: Why Does Persistence Win? — Feature Analysis  `~60 seconds`                                                                                                                      
      113 +## Slide 9: RQ1: Why Does Persistence Win — And How We Beat It On Smoke Days  `~60 seconds`                                                                                                         
      114                                                                                                                                                                                                    
      115 -This slide shows two complementary analyses that explain exactly why.                                                                                                                        
      115 +This slide shows two complementary analyses.                                                                                                                                                 
      116  
      117 -On the left is the Random Forest feature importance ranking. The single most important feature — pm25_lag1, yesterday's PM2.5 — accounts for roughly 40% of the model's total predictive powe
          -r. All 33 other features combined account for the remaining 60%. Wind direction is second at about 5%, followed by other PM2.5 lags and precipitation.                                       
      117 +On the left is the Random Forest feature importance ranking. Yesterday's PM2.5 alone accounts for roughly 40% of the model's predictive power. This is why persistence is so hard to beat ove
          +rall — the strongest signal in the data is already yesterday's value.                                                                                                                        
      118  
      119 -On the right is the ablation study — we systematically remove each feature group and measure the impact. Removing weather hurts the model significantly — MAE goes from 1.793 to 2.218. Remov
          -ing PM2.5 lags also hurts — up to 1.930. So both weather and historical PM2.5 carry real information.                                                                                        
      119 +On the right is the ablation study — we systematically remove each feature group. The key result highlighted in green: removing fire features actually *improves* the standard model. MAE dro
          +ps by 0.043. Satellite fire counts at daily resolution are actively adding noise, because a fire count doesn't encode whether the wind is pointing toward Vancouver or away from it. Weather 
          +features — wind direction and pressure — already capture stagnation conditions, making fire counts redundant.                                                                                
      120  
      121 -But the key finding is this row highlighted in green: removing fire features actually improves the model. MAE drops by 0.043 and R-squared increases by 0.050.                               
      121 +But here is where the story gets interesting. That ablation insight directly motivated our advanced models. If weather features regularise the model toward normal-day behavior, and fire cou
          +nts are only informative during actual fire events — then for smoke-day prediction, we should strip out weather and focus on fire signals. That is exactly what our fire-only feature set doe
          +s: 24 features — PM2.5 lags, fire FRP and count by distance band, and wind vectors.                                                                                                          
      122  
      123 -This tells us that satellite fire counts at daily resolution are not just uninformative — they are actively adding noise. The reason is that a fire count doesn't capture atmospheric transpo
          -rt. A hundred fires 200 kilometres away might produce zero smoke in Vancouver if the wind is pointing the wrong way. Weather features — especially wind direction and pressure — already enco
          -de whether smoke-trapping conditions exist. Fire counts are then redundant at best, and confusing at worst.                                                                                  
      123 +Combined with residual framing and quantile regression at Q equals 0.80, the result is a smoke-day MAE of 21.729 — a 0.624 improvement over persistence. The quantile sweep confirms the trad
          +eoff is perfectly monotone: higher alpha always buys better smoke-day MAE at the cost of overall MAE. There is no free lunch, but the tradeoff is predictable and controllable — which is exa
          +ctly what you want for a public health use case where under-predicting a smoke spike is far more dangerous than over-predicting one.                                                         
      124  
      125  ---
      126  

⏺ Now add the three new Q&A entries:                               

⏺ Update(speaking_notes.md)                                                                                                                                                                                    
  ⎿  Added 12 lines
      168                                                                                                                                                                                                      
      169  Fire Radiative Power measures how much energy a fire releases in megawatts — a proxy for intensity. Ten small fires might produce less smoke than one intense fire, so using both count and F     
           RP gives a richer picture of total fire activity.                                                                                                                                                   
      170                                                                                                                                                                                                    
      171 +**Q: Does any model actually beat persistence?**                                                                                                                                                    
      172 +                                                                                                                                                                                             
      173 +Yes — on smoke-day MAE specifically. Quantile regression at Q=0.80 with fire-only features (24 variables) achieves a smoke-day MAE of 21.729, beating persistence's 22.353 by 0.624 microgram
          +s per cubic metre. The tradeoff is that overall MAE is worse — 2.075 versus 1.668 — because the model deliberately predicts higher values on average. That's an acceptable tradeoff for a pub
          +lic health alerting use case where missing a smoke spike is far more costly than a false alarm.                                                                                              
      174 +                                                                                                                                                                                             
      175 +**Q: What is quantile regression and why use it here?**                                                                                                                                      
      176 +                                                                                                                                                                                             
      177 +Standard regression minimises mean absolute error, which teaches the model to predict near the median of the distribution. For a dataset where 99.5% of days are clean, the median is always 
          +"clean" — so the model never learns to predict spikes. Quantile regression at alpha=0.80 uses an asymmetric loss function that penalises under-prediction four times more than over-predictio
          +n. This forces the model to predict the 80th percentile of the next-day change distribution. The tradeoff with alpha is monotone and predictable: every 0.05 step up in alpha improves smoke-
          +day MAE by about 0.2 to 0.3 micrograms and worsens overall MAE by about 0.3 to 0.5. One number controls the entire risk-accuracy balance.                                                    
      178 +                                                                                                                                                                                             
      179 +**Q: How does the XGBoost smoke detector work and how well does it perform?**                                                                                                                
      180 +                                                                                                                                                                                             
      181 +It is a binary classifier — does tomorrow's PM2.5 exceed 25 micrograms or not? We handle the 200-to-1 class imbalance using XGBoost's scale_pos_weight parameter. The threshold is tuned to m
          +aximise F2 score, which weights recall twice as heavily as precision — appropriate when missing a smoke event is worse than a false alarm. In out-of-fold evaluation across five expanding ti
          +me-series folds, it achieves an AUPRC of 0.331 versus a random baseline of about 0.005, and catches 8 of 13 unique historical smoke episodes. The 5 it misses each have a structural physical
          + explanation: the 2020 Labor Day event came from Oregon and California fires that are invisible in BC-only satellite data; 2017 involved only distant fires with no close-range signal; 2018 
          +was fast-onset where fire activity jumped on day 2 of the episode; 2023 was a single pre-smoke day; and 2005 falls in the pre-2010 era where training data is too sparse.                    
      182 +                                                                                                                                                                                             
  ⎿  You've hit your limit · resets 6am (America/Vancouver)
                                                                   
✻ Crunched for 6m 56s
                                                                                                                                                                                                               
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── experiment-audit-next-steps ──
❯ exit                                                                                                                                                                                                         
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+tab to cycle)                                                                                                                                                  ◐ medium · /effort  
                                                                                                                                                              Update available! Run: brew upgrade claude-code  
                                                                                                                                                                                                               
Resume this session with:                                                                                                                                                                                      
claude --resume "experiment-audit-next-steps" 