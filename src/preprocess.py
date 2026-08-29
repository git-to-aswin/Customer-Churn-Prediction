"""The stateful preprocessing step: a ``ColumnTransformer`` fitted on training data only.

Feeding this straight into a ``Pipeline`` guarantees every fold in cross-validation and
the final refit learn their scaling and encodings without peeking at held-out rows.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

# Unseen "No"/"Yes"-style value at inference -> encoded as -1 rather than raising.
_UNKNOWN_YESNO = -1


def _build_yesno_encoder() -> OrdinalEncoder:
    categories = [[config.COLLAPSED_VALUE, "Yes"] for _ in config.BINARY_YESNO_COLS]
    return OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=_UNKNOWN_YESNO,
    )


def build_preprocessor() -> ColumnTransformer:
    """Map raw cleaned columns to a numeric feature matrix."""
    return ColumnTransformer(
        transformers=[
            ("yesno", _build_yesno_encoder(), config.BINARY_YESNO_COLS),
            (
                "contract",
                OrdinalEncoder(categories=[config.CONTRACT_ORDER]),
                [config.CONTRACT_COL],
            ),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore", drop="if_binary", sparse_output=False
                ),
                config.NOMINAL_COLS,
            ),
            ("numeric", StandardScaler(), config.NUMERIC_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
