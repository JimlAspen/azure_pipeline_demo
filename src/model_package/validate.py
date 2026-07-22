"""
Validation utilities.
"""

from __future__ import annotations

import polars as pl

from .schema import FEATURE_COLUMNS, TARGET_COLUMN


def validate_dataframe(
    df: pl.DataFrame,
    require_target: bool = True,
) -> None:
    """
    Validate an input dataframe.

    Parameters
    ----------
    df
        Input dataframe.

    require_target
        Whether the target column is required.

    Raises
    ------
    ValueError
        If validation fails.
    """

    expected_columns = FEATURE_COLUMNS.copy()

    if require_target:
        expected_columns.append(TARGET_COLUMN)

    missing = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df.height == 0:
        raise ValueError(
            "Input dataframe is empty."
        )

    duplicated = [
        column
        for column in set(df.columns)
        if df.columns.count(column) > 1
    ]

    if duplicated:
        raise ValueError(
            f"Duplicated columns: {duplicated}"
        )

    null_count = (
        df.select(pl.all().is_null().sum())
        .to_numpy()
        .sum()
    )

    if null_count > 0:
        raise ValueError(
            "Input dataframe contains null values."
        )

    for column in FEATURE_COLUMNS:

        if not df[column].dtype.is_numeric():
            raise ValueError(
                f"{column} must be numeric."
            )

    if require_target:

        unique_targets = sorted(
            df[TARGET_COLUMN]
            .unique()
            .to_list()
        )

        if unique_targets != [0, 1]:
            raise ValueError(
                "Target column must contain only 0 and 1."
            )