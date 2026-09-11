# Project Update Summary (Smoke-Day Modeling)

This update includes a suite of new scripts and refined modeling approaches developed to address the "Smoke-Day Under-Prediction" problem.

## Key Modeling Findings

1.  **RQ1: Persistence vs. ML Overall**
    *   **Finding:** The standard persistence baseline (tomorrow = today) achieves an overall MAE of **1.694 µg/m³**, outperforming all standard ML models (Random Forest best at 1.783).
    *   **Reason:** High day-to-day autocorrelation (r=0.80) in PM2.5 levels.

2.  **The Smoke-Day Challenge**
    *   Persistence fails catastrophically during smoke events (MAE **22.353**), as it lags the actual spikes by a full day.

3.  **Innovative Solutions Integrated:**
    *   **Residual Framing:** Instead of predicting raw PM2.5, we now predict the *change* (tomorrow minus today). This architectural choice incorporates the persistence baseline as the model's default prior.
    *   **Quantile Regression (Q=0.80):** We shifted to an asymmetric loss function that penalizes under-prediction 4x more than over-prediction. This forces the model to predict the 80th percentile of the next-day change distribution.
    *   **Fire-Only Feature Subset:** A refined set of 24 features (dropping weather variables like pressure/temp that dominate clean days but add noise during smoke events) was found to be the most effective for smoke prediction.

4.  **Results Achieved:**
    *   **Quantile Model (Q=0.80, Fire-Only):** Achieved a smoke-day MAE of **21.729**, beating persistence by **0.624 µg/m³**.
    *   **XGBoost Smoke Detector:** A parallel classifier that catches **8 out of 13** historical smoke episodes (AUPRC 0.331 vs. random baseline ~0.005).

## New Scripts and Documentation Added

*   **`scripts/`**: Training scripts for Quantile Regression (`asymmetric_loss`), XGBoost Classifier (`smoke_detector`), Anomaly Gate, and Residual Correction.
*   **`experiment_results.md`**: Detailed performance metrics for all 9 follow-up experiments.
*   **`report_final.md`**: Updated to reflect these findings.
*   **`figures/`**: Precision-Recall curves for the new smoke detection models.
