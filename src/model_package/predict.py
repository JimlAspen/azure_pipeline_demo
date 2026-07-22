"""
Prediction utilities.
"""

import pickle
from pathlib import Path


def load_model(
    model_dir: str,
):
    """
    Load trained model.
    """
    with open(
        Path(model_dir) / "model.pkl",
        "rb",
    ) as f:
        return pickle.load(f)


def predict(
    model,
    x,
):
    """
    Generate predictions.
    """
    return model.predict(x)