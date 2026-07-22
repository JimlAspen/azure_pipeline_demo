"""
Prepare ML datasets from the Gold Delta table.

This script performs the following steps:

1. Load the Gold Delta table.
2. Validate the dataset.
3. Split the data into:
   - training dataset
   - scoring dataset
4. Remove the target column from the scoring dataset.
5. Save both datasets as Delta Lake tables.
"""

from __future__ import annotations

import argparse

import polars as pl
from sklearn.model_selection import train_test_split

from model_package.metadata import RANDOM_STATE
from model_package.schema import TARGET_COLUMN
from model_package.validate import validate_dataframe
from train.common import (
    get_storage_options,
    load_delta_table,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        required=True,
        help="Gold Delta table.",
    )

    parser.add_argument(
        "--storage_account",
        required=True,
        help="Azure Storage account.",
    )

    parser.add_argument(
        "--key_vault_url",
        required=True,
        help="Azure Key Vault URL.",
    )

    parser.add_argument(
        "--training_output_path",
        required=True,
        help="Training Delta output.",
    )

    parser.add_argument(
        "--scoring_output_path",
        required=True,
        help="Scoring Delta output.",
    )

    parser.add_argument(
        "--scoring_fraction",
        type=float,
        default=0.20,
        help="Fraction reserved for batch scoring.",
    )

    return parser.parse_args()


def write_delta(
    df: pl.DataFrame,
    output_path: str,
    storage_options: dict[str, str],
) -> None:
    """
    Write a Delta table.
    """

    df.write_delta(
        target=output_path,
        mode="overwrite",
        storage_options=storage_options,
    )


def main() -> None:
    """Run dataset preparation."""

    args = parse_args()

    storage_options = get_storage_options(
        storage_account=args.storage_account,
        key_vault_url=args.key_vault_url,
    )

    df = load_delta_table(
        input_path=args.input_path,
        storage_options=storage_options,
    )

    validate_dataframe(df)

    pdf = df.to_pandas()

    training_df, scoring_df = train_test_split(
        pdf,
        test_size=args.scoring_fraction,
        random_state=RANDOM_STATE,
        stratify=pdf[TARGET_COLUMN],
    )

    training_df = pl.from_pandas(training_df)

    scoring_df = (
        pl.from_pandas(scoring_df)
        .drop(TARGET_COLUMN)
    )

    write_delta(
        training_df,
        args.training_output_path,
        storage_options,
    )

    write_delta(
        scoring_df,
        args.scoring_output_path,
        storage_options,
    )

    print(
        f"Training rows: {training_df.height}"
    )

    print(
        f"Scoring rows: {scoring_df.height}"
    )

    print(
        "Prepare ML datasets completed successfully."
    )


if __name__ == "__main__":
    main()