"""Customer churn modelling pipeline.

Modules:
    config      column groups and run constants
    data        load, stateless clean, train/test split
    preprocess  fitted ColumnTransformer (scaling + encoding)
    model       estimator factory and the full sklearn Pipeline
    evaluate    threshold selection, metrics, diagnostic plots
"""
