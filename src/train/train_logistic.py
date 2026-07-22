"""
Train a Logistic Regression classifier.

This script performs the following steps:

1. Load the Gold Delta Lake table.
2. Validate the input data.
3. Apply preprocessing.
4. Split the dataset.
5. Train a Logistic Regression model.
6. Evaluate the model.
7. Log the experiment to MLflow.
8. Save the trained model, preprocessing pipeline, and metrics.
"""

from __future__ import annotations

import argparse

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_package.metadata import (
    LOGISTIC_EXPERIMENT,
    RANDOM_STATE,
    TEST_SIZE,
)
from model_package.preprocess import (
    fit_transform,
    transform,
    save_pipeline,
)
from model_package.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from model_package.validate import validate_dataframe
from train.common import (
    get_storage_options,
    load_delta_table,
    save_model,
    save_metrics,
)

from train.training_utils import (
    evaluate_model,
    log_metrics,
    log_parameters,
)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        required=True,
        help="Gold Delta Lake path.",
    )

    parser.add_argument(
        "--storage_account",
        required=True,
        help="Azure Storage account name.",
    )

    parser.add_argument(
        "--key_vault_url",
        required=True,
        help="Azure Key Vault URL.",
    )

    parser.add_argument(
        "--model_output_dir",
        required=True,
        help="Directory for model artifacts.",
    )

    parser.add_argument(
        "--metrics_output_dir",
        required=True,
        help="Directory for metrics.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the training workflow."""

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

    df = df.to_pandas()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    x_train, pipeline = fit_transform(train_df)
    x_test = transform(test_df, pipeline)

    y_train = train_df[TARGET_COLUMN].to_numpy()
    y_test = test_df[TARGET_COLUMN].to_numpy()

    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    model.fit(
        x_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        x_test,
        y_test,
    )
    
    parameters = {
        "model": "LogisticRegression",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_features": len(FEATURE_COLUMNS),
    }

    #mlflow.set_experiment(LOGISTIC_EXPERIMENT)

    with mlflow.start_run():
    

        log_parameters(parameters)

        log_metrics(metrics)

        save_model(
            model=model,
            filename="logistic_regression.pkl",
            output_dir=args.model_output_dir,
        )

        save_pipeline(
            pipeline=pipeline,
            output_dir=args.model_output_dir,
        )

        save_metrics(
            metrics=metrics,
            output_dir=args.metrics_output_dir,
        )

        mlflow.log_artifacts(args.metrics_output_dir)
        mlflow.log_artifacts(args.model_output_dir)
    
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
