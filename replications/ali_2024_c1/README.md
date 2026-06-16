# Ali et al. (2024) / C1 Replication

This folder contains our first replication attempt for:

> Ali, A., Jayaraman, R., Azar, E., & Maalouf, M. (2024).
> "Maximizing supply chain performance leveraging machine learning to
> anticipate customer backorders."

The goal is not to spend 100+ hours reproducing every hyperparameter-search
minute from the paper. The goal is to establish a credible baseline replication
that verifies the dataset, preprocessing logic, feature sets, model families,
and approximate reported results.

## Contents

- `src/audit_data.py` audits the raw Kaggle CSV schema and class balance.
- `src/run_baseline.py` runs the reconstructed baseline experiment.
- `reports/initial_replication.md` records the first replication checkpoint.
- `results/` contains small generated audit and baseline-result files.
- `data/raw/` is intentionally empty except for `.gitkeep`.

## Dataset

The paper cites the Kaggle dataset:

`adityanarayansinha/back-order-prediction-using-ann`

Download URL:

```bash
curl -L --fail \
  "https://www.kaggle.com/api/v1/datasets/download/adityanarayansinha/back-order-prediction-using-ann" \
  -o data/raw/back-order-prediction-using-ann.zip
unzip -o data/raw/back-order-prediction-using-ann.zip -d data/raw
```

Expected raw CSV:

- File: `data/raw/Training_Dataset_v2.csv`
- SHA-256: `77f931c197ecd4d6bcb64cd0a1bbdfe1449ac7bd3b45836ed3392536908407c4`
- Rows: `1,048,575`
- Columns: `23`
- Target: `went_on_backorder`

Raw data is not committed because it is large and externally hosted.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run:

```bash
python src/audit_data.py
python src/run_baseline.py
```

## Current checkpoint

We reconstructed the paper's final row count by:

1. Downsampling the majority class to 8,900 rows.
2. Keeping all 8,900 minority-class rows.
3. Removing missing `lead_time` rows after downsampling.

With seed `41`, this gives:

- Rows after downsampling: `17,800`
- Missing `lead_time` rows after downsampling: `791`
- Final rows: `17,009`

Seed `41` is inferred; the paper does not report a seed.

