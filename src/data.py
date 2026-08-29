"""Data loading, stateless cleaning, and the train/test split.

The cleaning here is row-wise and deterministic (no fitted statistics), so it is safe
to run on the full frame before splitting. Anything that must learn from the training
data (scaling, encoding categories) lives in ``preprocess.py`` instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco workbook into a DataFrame."""
    return pd.read_excel(path)


def _coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the text ``Total Charges`` column; blank tenure-0 rows become 0.0."""
    charges = pd.to_numeric(df[config.NUMERIC_TEXT_COL], errors="coerce")
    df[config.NUMERIC_TEXT_COL] = charges.fillna(0.0)
    return df


def _collapse_structural_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fold "No internet service" / "No phone service" into plain "No"."""
    df[config.INTERNET_ADDON_COLS] = df[config.INTERNET_ADDON_COLS].replace(
        config.NO_INTERNET_VALUE, config.COLLAPSED_VALUE
    )
    df["Multiple Lines"] = df["Multiple Lines"].replace(
        config.NO_PHONE_VALUE, config.COLLAPSED_VALUE
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every stateless cleaning step and drop non-modelling columns."""
    df = df.copy()
    df = _coerce_total_charges(df)
    df = _collapse_structural_categories(df)
    return df.drop(columns=config.DROP_COLS)


def save_clean(df: pd.DataFrame, path: Path = config.CLEAN_DATA_PATH) -> Path:
    """Write the cleaned frame to CSV (versioned input for the model)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_clean(path: Path = config.CLEAN_DATA_PATH) -> pd.DataFrame:
    """Read the versioned cleaned dataset produced by ``scripts/prepare_data.py``."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m scripts.prepare_data` first."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector."""
    target = df[config.TARGET]
    features = df.drop(columns=[config.TARGET])
    return features, target


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """Stratified train/test split on the cleaned frame."""
    features, target = split_features_target(df)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )
