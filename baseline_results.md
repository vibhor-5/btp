# Baseline Analysis Results (SKU Backorder Dataset)

This document summarizes the outcomes of the baseline machine learning analysis for the **Can You Predict Product Backorders?** dataset.

## 1. Data Processing and Feature Engineering
A sophisticated Python script (`baseline_analysis.py`) was constructed to handle the end-to-end flow.
- **Sampling:** A 10% stratified sample of the massive `Kaggle_Training_Dataset_v2.csv` (168k rows) was utilized to provide a robust but time-efficient baseline.
- **Imputation:** Performance metrics using `-99` as placeholders were converted to `NaN`. All missing numerical values (e.g., `lead_time`, performance averages) were imputed with the column medians.
- **Encoding:** Categorical flags (Yes/No fields) were label-encoded.
- **Handling Class Imbalance:** The target variable `went_on_backorder` is severely imbalanced (0.7% positive class). We utilized **class-weighted loss functions** (`scale_pos_weight` in XGBoost and `class_weight='balanced'` in Random Forest) rather than SMOTE to effectively handle the imbalance without memory overflow.

## 2. Model Performance Evaluation

Two robust ensemble methods were trained—an **XGBoost Classifier** and a **Random Forest Classifier**. Performance metrics were evaluated on a held-out test split of 33,758 rows.

### XGBoost Results
*   **Precision (Class 1 - Backorder):** 0.08
*   **Recall (Class 1 - Backorder):** 0.70
*   **F1-Score (Class 1 - Backorder):** 0.15
*   **ROC-AUC:** 0.9300

*Observation:* While precision is low (typical for extreme imbalance where the model over-flags to be safe), XGBoost successfully captured 70% of actual backorders. The ROC-AUC of 0.93 indicates excellent overall class separation capability across thresholds.

### Random Forest Results
*   **Precision (Class 1 - Backorder):** 0.09
*   **Recall (Class 1 - Backorder):** 0.63
*   **F1-Score (Class 1 - Backorder):** 0.15
*   **ROC-AUC:** 0.9328

*Observation:* Random Forest achieved a slightly better ROC-AUC score, but captured fewer total backorders (lower recall) than XGBoost. Both models perform exceptionally well at ranking backorder risk.

## 3. Explainability (SHAP Analysis)

We successfully embedded **Explainable AI (XAI)** using the `shap.TreeExplainer` on the XGBoost model.

### Explainability Outputs Generated
1. **Global Summary Plot (`shap_summary_plot.png`):**
   * Generated to provide a high-level view of the overarching drivers of backorders.
2. **Local Explanation Card (`shap_force_plot_instance.png`):**
   * Generates a "per-SKU" visual breakdown. In the test instance evaluated, extremely low national inventory (`national_inv`) combined with low `min_bank` were heavily negative drivers, confirming the logical correlation that inventory levels dominate backorder probabilities.

## 4. Next Steps
- To improve precision (reducing false alarms), advanced threshold-tuning via the Precision-Recall curve is recommended.
- You can now use the full 1.6M rows for the final model training phase by removing the `sample_frac` limit.
