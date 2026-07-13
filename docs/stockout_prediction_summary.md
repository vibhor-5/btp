# Stockout Prediction Project Summary

This document outlines the steps taken, methodology applied, and results achieved while building a baseline Transformer model to predict item stockouts based on the provided dataset.

## 1. Dataset Exploration
We started by analyzing the `train.parquet` and `eval.parquet` files located in the `data/` directory. 
- **Dataset Size & Span:** The combined dataset spans 97 unique days (from `2024-03-28` to `2024-07-02`).
- **Granularity:** Each row in the dataset represents a unique combination of a specific day (`dt`), product (`product_id`), and store (`store_id`).
- **Features:** The data includes categorical features (like `city_id`, `store_id`, `product_id`, `holiday_flag`) and continuous numeric features (like `discount`, `avg_temperature`, `avg_humidity`, `avg_wind_level`, `precpt`).
- **Target Variables:** The dataset tracks hourly stock status via `hours_stock_status` (an array of 24 integers) and provides an aggregated count of stockout hours during active store hours (6 AM to 10 PM) in `stock_hour6_22_cnt`.

## 2. Data Processing & Filtering
To ensure the baseline model trains efficiently and focuses on the most significant data, we extracted a highly varying subset of the data:
- **Identifying Top SKUs:** We calculated the variance of sales (`sale_amount`) and the variance of stockouts (`stock_hour6_22_cnt`) for all products. By multiplying these variances, we created an importance score and selected the **Top 15 most statistically important and varying SKUs**.
  - *Top 15 SKUs:* `[267, 756, 589, 96, 498, 540, 4, 300, 185, 844, 309, 186, 117, 813, 191]`
- **Time-Based Splitting:** We filtered the data down to only these 15 SKUs and applied a chronological split:
  - **Training Set (First 75 days):** 175,575 records.
  - **Testing Set (Next 15 days):** 35,115 records.
- **Exporting:** The filtered datasets were saved independently as `data/top15_train.parquet` and `data/top15_test.parquet` for reproducibility.

## 3. Problem Definition
The objective was to predict if a stockout would occur. We framed this as a **daily binary classification problem** at the **store-SKU level**:
- **Target Definition:** `target = 1` if `stock_hour6_22_cnt > 0`, else `0`.
- **Meaning:** The model predicts whether a specific SKU at a specific store will be out of stock for *at least one hour* between 6:00 AM and 10:00 PM on a given day.

## 4. Modeling Strategy: Tabular Transformer Baseline
We designed and trained a PyTorch baseline model using a Transformer architecture (`train_transformer.py`). Rather than treating the data purely as a time-series, we treated the features of a single day as a "sequence" of embeddings:
- **Categorical Features:** Handled using PyTorch `nn.Embedding` layers mapping to a uniform embedding dimension (32).
- **Numeric Features:** Handled by standardizing with `StandardScaler` and using a linear projection to match the embedding dimension.
- **Architecture:** The projected features are concatenated into a sequence, prepended with a learnable `CLS` (classification) token, and passed through a 2-layer `TransformerEncoder` with 4 attention heads.
- **Classification Head:** The output of the `CLS` token from the Transformer is passed through a small Multi-Layer Perceptron (MLP) to output the final stockout logits.

## 5. Results & Evaluation
The model was trained for 3 epochs using Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`) and the Adam optimizer. 

**Final Metrics on the 15-Day Test Set:**
- **Epoch 3 Loss:** `0.6475`
- **Test AUC (Area Under the ROC Curve):** `0.5257`
- **Test Accuracy:** `55.83%`

### Next Steps & Potential Improvements
The current baseline achieves an AUC slightly better than random guessing (0.50), indicating that while it has captured some signal, there is massive room for improvement. Potential next steps include:
1. **Feature Engineering:** Integrating historical lag features (e.g., "was it out of stock yesterday?") or rolling averages of sales.
2. **Hourly Prediction:** Modifying the model to predict the exact hour of stockout using the `hours_stock_status` array.
3. **Hyperparameter Tuning:** Increasing the complexity of the Transformer, adjusting dropout, or extending the training epochs.
4. **Time-Series Architecture:** Using models like LSTM or Temporal Convolutional Networks (TCNs) to explicitly model the chronological sequence of days for each store-SKU pair.
