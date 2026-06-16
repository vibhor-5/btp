from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
TARGET_CANDIDATES = ("went_on_backorder", "backorder", "target")


def find_csv_files() -> list[Path]:
    return sorted(RAW_DIR.glob("*.csv"))


def find_target(columns: list[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    return next(
        (normalized[name] for name in TARGET_CANDIDATES if name in normalized),
        None,
    )


def audit_file(path: Path) -> dict:
    data = pd.read_csv(path, low_memory=False)
    target = find_target(data.columns.tolist())
    report = {
        "file": str(path.relative_to(ROOT)),
        "rows": len(data),
        "columns": len(data.columns),
        "column_names": data.columns.tolist(),
        "missing_values": {
            key: int(value)
            for key, value in data.isna().sum().items()
            if value
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "target_column": target,
    }
    if target:
        report["target_counts"] = {
            str(key): int(value)
            for key, value in data[target].value_counts(dropna=False).items()
        }
    return report


def main() -> None:
    files = find_csv_files()
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")

    RESULTS_DIR.mkdir(exist_ok=True)
    reports = [audit_file(path) for path in files]
    output = RESULTS_DIR / "data_audit.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"\nSaved audit to {output}")


if __name__ == "__main__":
    main()

