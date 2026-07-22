"""
Common utilities for Azure ML model training.

This module provides helper functions for:
- Parsing Azure storage credentials from Azure Key Vault.
- Reading Delta Lake tables.
- Saving models and metrics.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import polars as pl
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.keyvault.secrets import SecretClient


def get_storage_options(
    storage_account: str,
    key_vault_url: str,
) -> dict[str, str]:
    """
    Retrieve Azure Storage credentials from Azure Key Vault.

    Parameters
    ----------
    storage_account : str
        Azure Storage account name.
    key_vault_url : str
        Azure Key Vault URL.

    Returns
    -------
    dict[str, str]
        Storage options for Polars Delta Lake.
    """
    credential = AzureMLOnBehalfOfCredential()

    client = SecretClient(
        vault_url=key_vault_url,
        credential=credential,
    )

    account_key = client.get_secret("MOCKDELTALAKE-KEY").value

    return {
        "azure_storage_account_name": storage_account,
        "azure_storage_account_key": account_key,
    }


def load_delta_table(
    input_path: str,
    storage_options: dict[str, str],
) -> pl.DataFrame:
    """
    Load a Delta Lake table.

    Parameters
    ----------
    input_path : str
        Delta table location.
    storage_options : dict[str, str]
        Azure Storage authentication options.

    Returns
    -------
    pl.DataFrame
        Loaded Delta table.
    """
    return pl.read_delta(
        input_path,
        storage_options=storage_options,
    )


def save_model(
    model,
    output_dir: str,
    filename: str,
) -> None:
    """
    Save a trained model.

    Parameters
    ----------
    model
        Trained scikit-learn model.
    output_dir : str
        Output directory.
    filename : str
        Model filename.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / filename, "wb") as f:
        pickle.dump(model, f)


def save_metrics(
    metrics: dict,
    output_dir: str,
) -> None:
    """
    Save evaluation metrics as JSON.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics.
    output_dir : str
        Output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)