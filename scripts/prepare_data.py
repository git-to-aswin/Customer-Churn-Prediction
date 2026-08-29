"""Build the versioned clean dataset from the raw workbook.

Usage (from the repo root, with the venv active):

    python -m scripts.prepare_data

Reads  raw_data/Telco_customer_churn.xlsx
Writes processed_data/telco_churn_clean.csv
"""

from __future__ import annotations

from src import config
from src.data import clean, load_raw, save_clean


def _summarise(raw_cols: list[str], clean_cols: list[str], rows: int) -> None:
    dropped = sorted(set(raw_cols) - set(clean_cols))
    print(f"rows           : {rows}")
    print(f"columns        : {len(raw_cols)} -> {len(clean_cols)}")
    print(f"dropped ({len(dropped):>2})    : {dropped}")


def main() -> None:
    raw = load_raw(config.RAW_DATA_PATH)
    cleaned = clean(raw)
    out_path = save_clean(cleaned, config.CLEAN_DATA_PATH)

    _summarise(list(raw.columns), list(cleaned.columns), len(cleaned))
    print(f"churn rate     : {cleaned[config.TARGET].mean():.3f}")
    print(f"written        : {out_path.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main()
