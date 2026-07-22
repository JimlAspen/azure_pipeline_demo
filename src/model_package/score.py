"""
Model scoring entry point.
"""

import pandas as pd

from .predict import load_model
from .preprocess import (
    load_preprocessor,
    transform,
)
from .validate import validate_dataframe


def score(
    df: pd.DataFrame,
    model_dir: str,
):
    """
    Score a dataframe.
    """
    validate_dataframe(
        df,
        require_target=False,
    )

    scaler = load_preprocessor(
        model_dir,
    )

    x = transform(
        df,
        scaler,
    )

    model = load_model(
        model_dir,
    )

    return model.predict_proba(x)[:, 1]