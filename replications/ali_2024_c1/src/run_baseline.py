from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
TARGET = "went_on_backorder"
PAPER_FEATURES = [
    "national_inv",
    "lead_time",
    "in_transit_qty",
    "forecast_3_month",
    "sales_3_month",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def locate_dataset(requested: Path | None) -> Path:
    if requested:
        return requested
    candidates = sorted(RAW_DIR.glob("*.csv"), key=lambda path: path.stat().st_size)
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")
    return candidates[-1]


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    return data


def encode_target(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    encoded = values.map(mapping)
    if encoded.isna().any():
        unknown = sorted(values[encoded.isna()].unique())
        raise ValueError(f"Unknown target labels: {unknown}")
    return encoded.astype(int)


def downsample(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = data[TARGET].value_counts()
    minority_size = int(counts.min())
    sampled = [
        group.sample(n=minority_size, random_state=seed)
        for _, group in data.groupby(TARGET)
    ]
    return (
        pd.concat(sampled)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def make_pipeline(
    model: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=5_000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "adaboost": AdaBoostClassifier(n_estimators=300, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, random_state=seed
        ),
    }


def evaluate(
    data: pd.DataFrame,
    feature_names: list[str],
    scenario: str,
    seed: int,
) -> list[dict]:
    x = data[feature_names]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in feature_names if column not in numeric]
    rows = []

    for model_name, model in models(seed).items():
        pipeline = make_pipeline(model, numeric, categorical)
        fit_start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        predictions = pipeline.predict(x_test)
        predict_seconds = time.perf_counter() - predict_start

        rows.append(
            {
                "scenario": scenario,
                "model": model_name,
                "seed": seed,
                "features": len(feature_names),
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "accuracy": accuracy_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset = locate_dataset(args.data)
    data = normalize_columns(pd.read_csv(dataset, low_memory=False))
    if TARGET not in data:
        raise KeyError(f"Expected target column {TARGET!r}; found {data.columns.tolist()}")

    data[TARGET] = encode_target(data[TARGET])
    data = downsample(data, args.seed)
    missing_after_downsampling = int(data["lead_time"].isna().sum())
    data = data.dropna(subset=["lead_time"]).copy()

    excluded = {TARGET}
    all_features = [column for column in data.columns if column not in excluded]
    missing_selected = sorted(set(PAPER_FEATURES) - set(data.columns))
    if missing_selected:
        raise KeyError(f"Missing paper-selected features: {missing_selected}")

    results = []
    results.extend(evaluate(data, all_features, "all_features", args.seed))
    results.extend(evaluate(data, PAPER_FEATURES, "five_features", args.seed))

    RESULTS_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame(results)
    csv_path = RESULTS_DIR / f"baseline_seed_{args.seed}.csv"
    metadata_path = RESULTS_DIR / f"baseline_seed_{args.seed}.json"
    frame.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "seed": args.seed,
                "balanced_rows": len(data),
                "missing_lead_time_after_downsampling": missing_after_downsampling,
                "class_counts": data[TARGET].value_counts().to_dict(),
                "all_features": all_features,
                "selected_features": PAPER_FEATURES,
                "notes": [
                    "Baseline uses default/fixed model settings, not paper tuning.",
                    "Downsampling is performed before the train/test split.",
                    "Missing lead_time rows are removed after downsampling.",
                    "SKU is retained because Table 3 counts it among 22 predictors.",
                    "Scaling and encoding are fitted on training data only.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(f"\nSaved {csv_path} and {metadata_path}")


if __name__ == "__main__":
    main()
