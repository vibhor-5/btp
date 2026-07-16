# LSTM-Based Time-Series Stock Status Prediction Report

Author: Anurag Thakur  
Dataset: FreshRetailNet-50K  
Task: Next-day 24-hour stock-status prediction

## 1. Objective

The objective of this experiment was to move from static backorder prediction to
time-series stock-status prediction.

Earlier work on the Ali et al. / C1 backorder dataset helped us reproduce a
static ML baseline. However, that dataset did not contain a proper SKU-wise
timeline. For the next stage, we needed a dataset where stock behavior is
observed over time.

Therefore, we used FreshRetailNet-50K and trained LSTM-based models to predict:

```text
next day's 24-hour stock status
```

The model receives recent SKU-store history and predicts whether the item will
be in stock or stockout for each hour of the next day.

## 2. Dataset

Dataset used:

```text
Dingdong-Inc/FreshRetailNet-50K
```

Important columns used:

- `city_id`
- `store_id`
- `product_id`
- `dt`
- `sale_amount`
- `hours_sale`
- `stock_hour6_22_cnt`
- `hours_stock_status`
- `discount`
- `holiday_flag`
- `activity_flag`
- `precpt`
- `avg_temperature`
- `avg_humidity`
- `avg_wind_level`

Why this dataset is suitable:

- It is a true time-series dataset.
- It contains SKU-store level daily records.
- It contains hourly stock-status information.
- It allows future stock-status prediction using past sequence behavior.

## 3. Unit of Analysis

The unit of analysis was a SKU-store time series.

We created:

```text
series_id = city_id + store_id + product_id
```

This is necessary because stockout is store-specific. The same SKU may be
available in one store but stockout in another store.

## 4. Data Cleaning and Preprocessing

The following preprocessing steps were applied:

1. Converted `dt` into datetime format.
2. Removed rows with missing date, city, store, or product identifiers.
3. Removed duplicate `(series_id, dt)` rows.
4. Converted `hours_stock_status` into 24 binary hourly values.
5. Created daily summary features from hourly sales and hourly stock status.
6. Converted all model features to numeric format.
7. Filled missing numeric values with `0.0`.
8. Standardized features using only the training period.

The target was created by shifting hourly stock-status values by one day:

```text
input: previous N days
target: next day's 24 hourly stock-status values
```

## 5. Top SKU Selection

We selected the top 15 distinct SKUs statistically instead of choosing random
products.

Selection score:

```text
0.40 * total_sales
+ 0.25 * total_stockout_hours
+ 0.20 * stockout_days
+ 0.15 * sales_std
```

All components were min-max normalized before calculating the final score.

Reason for this selection:

- High-sales SKUs are more operationally important.
- SKUs with stockout history are more useful for stockout prediction.
- Sales variability helps select SKUs with non-trivial demand patterns.
- Distinct `product_id` values were enforced so that the top list was not
  dominated by the same product across multiple stores.

## 6. Train-Validation Split

Since this is a time-series task, random train-test splitting was not used.

Instead, we used a chronological split:

```text
First ~2.5 months -> training
Last 15 days      -> validation / prediction
```

This follows the real-world setting where a model is trained on past data and
used to predict future stock behavior.

## 7. How Data Was Fed to LSTM

The LSTM input was created using a sliding-window approach.

For the main setup:

```text
previous 14 days -> predict next day
```

Example:

```text
Days 1-14 -> predict Day 15
Days 2-15 -> predict Day 16
Days 3-16 -> predict Day 17
```

Each day originally had 13 features.

Input tensor shape:

```text
batch_size x 14 x 13
```

Output tensor shape:

```text
batch_size x 24
```

Each of the 24 output values represents one hourly stock-status prediction for
the next day.

## 8. Initial Standard LSTM Model

The first LSTM model used:

- hidden size: 96
- sequence length: 14 days
- input features: 13
- output size: 24
- trainable parameters: 44,952

Best validation result:

| Metric | Value |
|---|---:|
| Hour-level accuracy | 0.791 |
| Hour-level precision | 0.767 |
| Hour-level recall | 0.804 |
| Hour-level F1 | 0.785 |
| Exact 24-hour match | 0.302 |
| Mean stockout-hour count error | 4.67 hours |

Interpretation:

- The standard LSTM learned useful temporal patterns.
- It performed reasonably well at hourly prediction.
- Exact 24-hour match is stricter because all 24 hourly predictions must be
  correct for the full day to count as correct.

## 9. Motivation for LSTM Complexity Ablation

A concern was raised that if a simple RNN gives similar accuracy, then a full
LSTM may not be justified for such a short input window.

This concern is valid because:

- the sequence length is only 14 days,
- vanishing gradient may not be severe for such a short sequence,
- LSTM has more gates and parameters than a simple RNN.

Therefore, instead of defending the standard LSTM, we tested compact LSTM
variants to see whether complexity can be reduced.

## 10. Ablation Experiments

We tested three simplification directions:

1. Hidden-size reduction
2. Sequence-length reduction
3. Feature reduction

Experiment results:

| Experiment | Hidden Size | Window | Features | Params | Best F1 | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Standard LSTM | 96 | 14 | 13 | 44,952 | 0.785 | 0.804 |
| Compact LSTM | 64 | 14 | 13 | 21,784 | 0.771 | 0.775 |
| Compact LSTM | 32 | 14 | 13 | 6,808 | 0.753 | 0.811 |
| Short-window LSTM | 64 | 10 | 13 | 21,784 | 0.763 | 0.756 |
| Short-window LSTM | 64 | 7 | 13 | 21,784 | 0.776 | 0.774 |
| Feature-reduced LSTM | 64 | 14 | 6 | 19,992 | 0.788 | 0.816 |
| Feature-reduced LSTM | 32 | 14 | 6 | 5,912 | 0.790 | 0.813 |

## 11. Best Model Found

The best-performing compact LSTM was:

```text
Feature-reduced LSTM
hidden size = 32
sequence length = 14 days
features = 6
```

Performance:

- F1-score: 0.790
- Recall: 0.813
- Parameters: 5,912

Comparison with standard LSTM:

| Model | Params | F1 |
|---|---:|---:|
| Standard LSTM | 44,952 | 0.785 |
| Compact feature-reduced LSTM | 5,912 | 0.790 |

Parameter reduction:

```text
approximately 86.8%
```

## 12. Reduced Feature Set

The best compact model used only six features:

1. `sale_amount`
2. `stock_hour6_22_cnt`
3. `discount`
4. `holiday_flag`
5. `hours_sale_sum`
6. `hours_stock_status_sum`

These features capture:

- demand level,
- previous stockout behavior,
- discount effect,
- holiday effect,
- hourly sales summary,
- hourly stock-status summary.

Weather and activity-related features did not improve performance in the first
ablation, so they were removed in the compact model.

## 13. Main Findings

Key findings:

- A full-size LSTM is probably unnecessary for this short-window task.
- Reducing hidden size alone reduces parameters but also reduces F1.
- Reducing features gave the best trade-off.
- The compact feature-reduced LSTM slightly improved F1 while reducing
  parameters by about 87%.
- Therefore, the LSTM direction is defensible only as a compact LSTM, not as a
  heavy standard LSTM.

## 14. Conclusion

The final LSTM-side conclusion is:

> We should not use a standard LSTM just because LSTM is powerful. For this
> short-window stock-status prediction task, a compact feature-reduced LSTM is
> more appropriate.

The compact LSTM:

- uses fewer features,
- uses fewer hidden units,
- has far fewer parameters,
- performs slightly better than the standard LSTM.

## 15. Remaining Work

To finalize the model choice, this compact LSTM must be compared against the
RNN model under the same setup:

- same top 15 SKUs,
- same 14-day input window,
- same 15-day validation period,
- same 24-hour output target,
- same metrics,
- parameter count,
- training time,
- inference time.

Decision rule:

- If RNN gives the same performance with fewer parameters, RNN should be
  preferred.
- If compact LSTM gives better recall/F1 with acceptable complexity, compact
  LSTM is justified.

## 16. Files Related to This Work

Main scripts:

- `experiments/freshretail_lstm/src/train_top_sku_24h_lstm.py`
- `experiments/freshretail_lstm/src/run_lstm_ablation.py`

Detailed reports:

- `experiments/freshretail_lstm/reports/top_sku_24h_lstm_report.md`
- `experiments/freshretail_lstm/reports/lstm_complexity_ablation.md`
- `experiments/freshretail_lstm/reports/findings_summary.md`

Presentation:

- `outputs/freshretail_lstm_experiments.pptx`

