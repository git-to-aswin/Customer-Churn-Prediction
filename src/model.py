"""Estimator factory and the end-to-end modelling pipeline.

``build_pipeline`` chains the preprocessor with a classifier so the whole object can be
cross-validated, tuned, and persisted as one unit.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocess import build_preprocessor

LOGREG = "logreg"
HIST_GBM = "hgb"
LIGHT_GBM = "lgbm"
# HistGradientBoosting is the default: comparable to LightGBM on this dataset and ships
# with scikit-learn (no native OpenMP dependency). Use LIGHT_GBM if lightgbm is installed.
DEFAULT_ESTIMATOR = HIST_GBM

CLASSIFIER_STEP = "classifier"

# Search spaces for RandomizedSearchCV, keyed by estimator name. Keys are already
# prefixed for the pipeline step so they can be passed through unchanged.
PARAM_DISTRIBUTIONS = {
    LOGREG: {
        f"{CLASSIFIER_STEP}__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        f"{CLASSIFIER_STEP}__penalty": ["l1", "l2"],
        f"{CLASSIFIER_STEP}__solver": ["liblinear"],
    },
    LIGHT_GBM: {
        f"{CLASSIFIER_STEP}__n_estimators": [200, 400, 800],
        f"{CLASSIFIER_STEP}__learning_rate": [0.01, 0.03, 0.05, 0.1],
        f"{CLASSIFIER_STEP}__num_leaves": [15, 31, 63],
        f"{CLASSIFIER_STEP}__min_child_samples": [10, 20, 40],
        f"{CLASSIFIER_STEP}__subsample": [0.7, 0.9, 1.0],
        f"{CLASSIFIER_STEP}__colsample_bytree": [0.7, 0.9, 1.0],
    },
    HIST_GBM: {
        f"{CLASSIFIER_STEP}__max_iter": [200, 400, 800],
        f"{CLASSIFIER_STEP}__learning_rate": [0.01, 0.03, 0.05, 0.1],
        f"{CLASSIFIER_STEP}__max_leaf_nodes": [15, 31, 63],
        f"{CLASSIFIER_STEP}__min_samples_leaf": [10, 20, 40],
    },
}


def _hist_gbm(random_state: int) -> HistGradientBoostingClassifier:
    try:
        return HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state
        )
    except TypeError:  # sklearn < 1.7 has no class_weight for HGB
        return HistGradientBoostingClassifier(random_state=random_state)


def _light_gbm(random_state: int):
    from lightgbm import LGBMClassifier  # optional dependency, imported lazily

    return LGBMClassifier(
        class_weight="balanced", random_state=random_state, verbose=-1
    )


def build_estimator(name: str = DEFAULT_ESTIMATOR, random_state: int = config.RANDOM_STATE):
    """Return an unfitted classifier configured for the churn class imbalance."""
    if name == LOGREG:
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        )
    if name == HIST_GBM:
        return _hist_gbm(random_state)
    if name == LIGHT_GBM:
        return _light_gbm(random_state)
    raise ValueError(f"unknown estimator: {name!r}")


def build_pipeline(name: str = DEFAULT_ESTIMATOR, random_state: int = config.RANDOM_STATE) -> Pipeline:
    """Preprocessor + classifier as a single fit/predict unit."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (CLASSIFIER_STEP, build_estimator(name, random_state)),
        ]
    )
