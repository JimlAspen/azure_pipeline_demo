"""
Preprocessing utilities.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metadata import SCALER_FILENAME
from .schema import FEATURE_COLUMNS


def create_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )


def fit_transform(
    df: pd.DataFrame,
):
    """
    Fit preprocessing pipeline.

    Returns
    -------
    tuple
        (transformed_features, pipeline)
    """

    pipeline = create_pipeline()

    x = pipeline.fit_transform(
        df[FEATURE_COLUMNS]
    )
    return x, pipeline


def transform(
    df: pd.DataFrame,
    pipeline: Pipeline,
):
    """
    Apply preprocessing.
    """

    return pipeline.transform(
        df[FEATURE_COLUMNS]
    )


def save_pipeline(
    pipeline: Pipeline,
    output_dir: str,
) -> None:
    """
    Save preprocessing pipeline.
    """

    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output / SCALER_FILENAME,
        "wb",
    ) as f:

        pickle.dump(
            pipeline,
            f,
        )


def load_pipeline(
    model_dir: str,
) -> Pipeline:
    """
    Load preprocessing pipeline.
    """

    with open(
        Path(model_dir) / SCALER_FILENAME,
        "rb",
    ) as f:

        return pickle.load(f)