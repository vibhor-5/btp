# Initial C1 Replication

## Dataset verification

The exact Kaggle dataset cited by C1 was downloaded from:

`adityanarayansinha/back-order-prediction-using-ann`

- File: `data/raw/Training_Dataset_v2.csv`
- SHA-256: `77f931c197ecd4d6bcb64cd0a1bbdfe1449ac7bd3b45836ed3392536908407c4`
- Rows: 1,048,575
- Columns: 23 (22 predictors and one target)
- Non-backorders: 1,039,675
- Backorders: 8,900
- Missing `lead_time` values in the raw data: 64,518

The paper states that 791 missing values were removed and that the final
processed dataset contained 17,009 rows. Those statements cannot describe the
raw dataset. They are, however, exactly reproduced when:

1. The majority class is downsampled to 8,900 observations.
2. All 8,900 minority observations are retained.
3. The 791 remaining missing `lead_time` observations are removed.

Seed 41 recreates this arithmetic:

- Rows after downsampling: 17,800
- Missing `lead_time`: 791
- Final rows: 17,009
- Final classes: 8,648 backorders and 8,361 non-backorders

The seed is an inference and is not evidence that the authors used seed 41.
Multiple random seeds can produce 791 missing values.

## Baseline protocol

- 80/20 stratified train/test split
- Standardization fitted on training numeric data
- One-hot encoding fitted on training categorical data
- All-feature scenario includes `sku`, matching the paper's count of 22 inputs
- Five-feature scenario:
  - `national_inv`
  - `lead_time`
  - `in_transit_qty`
  - `forecast_3_month`
  - `sales_3_month`
- Fixed/default-like models; no paper-scale hyperparameter search yet

## Initial results

| Scenario | Model | Our accuracy | C1 accuracy | Our F1 | C1 F1 |
|---|---:|---:|---:|---:|---:|
| 22 predictors | Logistic regression | 0.681 | 0.68 | 0.721 | 0.68 |
| 22 predictors | Random forest | 0.899 | 0.90 | 0.903 | 0.90 |
| 22 predictors | AdaBoost | 0.857 | 0.86 | 0.863 | 0.86 |
| 22 predictors | XGBoost | 0.889 | 0.90 | 0.893 | 0.90 |
| 22 predictors | Gradient boosting | 0.876 | 0.91 | 0.880 | 0.91 |
| 5 predictors | Logistic regression | 0.664 | 0.68 | 0.728 | 0.67 |
| 5 predictors | Random forest | 0.871 | 0.88 | 0.875 | 0.88 |
| 5 predictors | AdaBoost | 0.852 | 0.86 | 0.857 | 0.86 |
| 5 predictors | XGBoost | 0.870 | 0.87 | 0.874 | 0.87 |
| 5 predictors | Gradient boosting | 0.869 | 0.87 | 0.873 | 0.87 |

The untuned baseline closely reproduces most rounded values in Table 4.
The largest accuracy difference is Gradient Boosting with all predictors
(-0.034), which is the model most likely to benefit from C1's extensive tuning.

## Environment

- Python 3.12.13
- NumPy 2.4.6
- pandas 2.3.3
- scikit-learn 1.9.0
- XGBoost 3.2.0
- macOS arm64

## Next experiment

Reconstruct the paper's randomized hyperparameter search. Because the paper's
1,000 iterations times 10-fold cross-validation is exceptionally expensive,
the search should be executed in two checkpoints:

1. A smaller search to validate parameter spaces and scoring behavior.
2. The full search for final reported replication results.

