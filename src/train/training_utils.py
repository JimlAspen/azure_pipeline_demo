"""
Common utilities for model training.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    x_test,
    y_test,
) -> dict[str, float]:
    """
    Evaluate a trained classifier.

    Parameters
    ----------
    model
        Trained classifier.
    x_test
        Test features.
    y_test
        Test labels.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    print(y_test, predictions, probabilities)

    print(
        {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }



    )
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def log_metrics(
    metrics: dict[str, float],
) -> None:
    """
    Log evaluation metrics to MLflow.

    Parameters
    ----------
    metrics
        Dictionary of evaluation metrics.
    """
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


def log_parameters(
    parameters: dict,
) -> None:
    """
    Log model parameters to MLflow.

    Parameters
    ----------
    parameters
        Dictionary of model parameters.
    """
    for name, value in parameters.items():
        mlflow.log_param(name, value)
