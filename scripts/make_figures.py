"""Regenerate the figures used in README.md.

Usage (repo root, venv active):

    python -m scripts.make_figures

Writes PNGs to assets/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config, data
from src.evaluate import choose_threshold, plot_confusion, plot_pr_curve
from src.model import HIST_GBM, LOGREG, build_pipeline

ASSETS_DIR = config.ROOT_DIR / "assets"

# Tuned HistGradientBoosting config from notebook/02_modeling.ipynb section 5.
BEST_PARAMS = {
    "classifier__learning_rate": 0.01,
    "classifier__max_iter": 400,
    "classifier__max_leaf_nodes": 15,
    "classifier__min_samples_leaf": 40,
}


def _save(fig, name: str) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (ASSETS_DIR / name).relative_to(config.ROOT_DIR))


def figure_churn_drivers(df: pd.DataFrame) -> None:
    """Churn rate by the three strongest categoricals plus tenure buckets."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, col in zip(axes, ["Contract", "Internet Service", "Payment Method"]):
        rate = df.groupby(col)[config.TARGET].mean().sort_values()
        rate.plot.bar(ax=ax, color="#c0504d")
        ax.set_title(col)
        ax.set_ylabel("churn rate")
        ax.tick_params(axis="x", rotation=30)

    tenure_bucket = pd.cut(df["Tenure Months"], bins=[0, 6, 12, 24, 48, 72], include_lowest=True)
    df.groupby(tenure_bucket, observed=True)[config.TARGET].mean().plot(
        ax=axes[3], marker="o", color="#c0504d"
    )
    axes[3].set_title("Tenure (months)")
    axes[3].set_ylabel("churn rate")
    fig.suptitle("Churn rate by top drivers", fontsize=14)
    _save(fig, "churn_drivers.png")


def figure_correlation(df: pd.DataFrame) -> None:
    """Pearson correlation among the continuous features and the target."""
    cols = ["Tenure Months", "Monthly Charges", "Total Charges", "Latitude", "Longitude", config.TARGET]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Numeric feature correlation")
    _save(fig, "correlation.png")


def _fit_models(df: pd.DataFrame):
    x_train, x_test, y_train, y_test = data.make_train_test_split(df)
    logreg = build_pipeline(LOGREG).fit(x_train, y_train)
    hgb = build_pipeline(HIST_GBM).set_params(**BEST_PARAMS).fit(x_train, y_train)
    return logreg, hgb, x_train, x_test, y_train, y_test


def figure_model_evaluation(models) -> None:
    """PR curves for both models plus the tuned model's confusion matrix."""
    logreg, hgb, x_train, x_test, y_train, y_test = models
    threshold = choose_threshold(y_train, hgb.predict_proba(x_train)[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    plot_pr_curve(logreg, x_test, y_test, ax=axes[0])
    plot_pr_curve(hgb, x_test, y_test, ax=axes[0])
    axes[0].legend(["Logistic Regression", "HistGradientBoosting (tuned)"])
    axes[0].set_title("Precision-Recall — held-out set")
    plot_confusion(hgb, x_test, y_test, threshold, ax=axes[1])
    _save(fig, "model_evaluation.png")


def figure_permutation_importance(models) -> None:
    """Held-out permutation importance for the tuned model."""
    from sklearn.inspection import permutation_importance

    _, hgb, _, x_test, _, y_test = models
    perm = permutation_importance(
        hgb, x_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.Series(perm.importances_mean, index=x_test.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    importance.plot.barh(ax=ax, color="#4f81bd")
    ax.set_xlabel("mean PR-AUC drop when shuffled")
    ax.set_title("Permutation importance (tuned HistGradientBoosting)")
    _save(fig, "permutation_importance.png")


def main() -> None:
    sns.set_style("whitegrid")
    df = data.load_clean()

    figure_churn_drivers(df)
    figure_correlation(df)

    models = _fit_models(df)
    figure_model_evaluation(models)
    figure_permutation_importance(models)


if __name__ == "__main__":
    main()
