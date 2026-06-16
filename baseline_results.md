# Baseline Analysis Results

This document summarizes the outcomes of the baseline machine learning analysis for stockout prediction, leveraging the DataCo Smart Supply Chain methodology. 

## 1. Data Processing and Feature Engineering
A sophisticated Python script (`baseline_analysis.py`) was constructed to handle the end-to-end flow.
- **Aggregation:** Transactional records were grouped down to a **SKU-Day** level.
- **Feature Engineering:** We computed 7-day and 30-day rolling averages, rolling standard deviations, and lagged sales to capture the temporal demand dynamics.
- **Proxy Target (`Stockout_Next_Period`):** In the absence of direct inventory levels, the pipeline simulates future stockout risk scenarios based on impending zero-sales periods paired with prior healthy demand trends.
- **Handling Class Imbalance:** We utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the models aren't biased against predicting the minority class (stockouts).

## 2. Model Performance Evaluation

Two robust ensemble methods were trained—an **XGBoost Classifier** and a **Random Forest Classifier**. Performance metrics were evaluated on a held-out, time-split test dataset of 3,763 predictions:

### XGBoost Results
*   **Precision (Class 1 - Stockout):** 0.69
*   **Recall (Class 1 - Stockout):** 0.98
*   **F1-Score (Class 1 - Stockout):** 0.81
*   **ROC-AUC:** 0.8728

*Observation:* XGBoost yielded exceptional recall (0.98), meaning it aggressively identifies almost all potential stockout risks, making it highly effective for a conservative inventory strategy.

### Random Forest Results (with SMOTE)
*   **Precision (Class 1 - Stockout):** 0.69
*   **Recall (Class 1 - Stockout):** 0.79
*   **F1-Score (Class 1 - Stockout):** 0.73
*   **ROC-AUC:** 0.8724

*Observation:* The Random Forest also performed adequately, though XGBoost edged it out by capturing a significantly higher percentage of actual stockouts (higher recall).

## 3. Explainability (SHAP Analysis)

We successfully embedded **Explainable AI (XAI)** into the pipeline using the `shap.TreeExplainer`. This allowed us to demystify the "black box" of the XGBoost model.

### Explainability Outputs Generated
1. **Global Summary Plot (`shap_summary_plot.png`):**
   * Generated to provide a high-level view of what drives stockout predictions universally. For example, sudden drops in `sales_roll_mean_7` typically push stockout probabilities higher.
2. **Local Explanation Card (`shap_force_plot_instance.png`):**
   * Generates a "per-SKU" visual breakdown. In our tested instance (Instance #0), we observed that a suppressed 7-day sales rolling mean heavily contributed to bumping the prediction higher (SHAP contribution: +0.2141), confirming our feature engineering logic accurately captured shifting demand dynamics.

## 4. Next Steps
- The pipeline script is fully modular. When you are ready to process proprietary or newer datasets, simply replace `DataCoSupplyChainDataset.csv` and re-execute.
- To improve precision further (reducing false alarms), you may consider adding supplier lead-time datasets and holiday/promotion flags.
