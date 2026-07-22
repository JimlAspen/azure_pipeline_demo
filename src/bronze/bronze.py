"""Ingest a raw CSV file into the Bronze Delta Lake layer."""

import argparse
import os
from pathlib import Path

import polars as pl
import yaml
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.keyvault.secrets import SecretClient


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest a CSV file into the Bronze Delta Lake layer."
    )
    parser.add_argument(
        "--raw_csv_path",
        type=str,
        default=None,
        help="Path to the input CSV file. Overrides the value in the config file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Destination Delta Lake path. Overrides the default Bronze path.",
    )
    parser.add_argument(
        "--key_vault_url",
        type=str,
        required=True,
        help="Azure Key Vault URL.",
    )
    parser.add_argument(
        "--ready_dir",
        type=str,
        default=None,
        help="Directory for Azure ML completion marker.",
    )
    return parser.parse_args()


def load_config():
    """Load the Bronze configuration from the YAML file."""
    config_path = Path("configs/bronze.yaml")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_storage_options(storage_account, key_vault_url):
    """Retrieve Azure Storage credentials from Azure Key Vault."""
    credential = AzureMLOnBehalfOfCredential()

    client = SecretClient(
        vault_url=key_vault_url,
        credential=credential,
    )

    account_key = client.get_secret("MOCKDELTALAKE-KEY").value

    if not account_key:
        raise ValueError("Secret 'MOCKDELTALAKE-KEY' is empty.")

    return {
        "azure_storage_account_name": storage_account,
        "azure_storage_account_key": account_key,
    }

def write_ready_marker(ready_dir: str | None) -> None:
    """Create a completion marker for Azure ML pipeline dependencies.

    Parameters
    ----------
    ready_dir : str | None
        Output directory supplied by Azure ML. If None, no marker is
        written.
    """
    if ready_dir is None:
        return

    path = Path(ready_dir)
    path.mkdir(parents=True, exist_ok=True)

    (path / "done.txt").write_text(
        "Bronze stage completed successfully.\n",
        encoding="utf-8",
    )

def ingest_to_bronze():
    """Load a CSV file and write it to the Bronze Delta Lake."""
    args = parse_args()
    config = load_config()

    raw_csv_path = args.raw_csv_path or config["raw_csv_path"]

    default_destination = (
        f"abfss://{config['container']}@"
        f"{config['storage_account']}.dfs.core.windows.net/"
        f"bronze/{config['table_name']}"
    )

    destination = args.output_path or default_destination

    storage_options = get_storage_options(
        storage_account=config["storage_account"],
        key_vault_url=args.key_vault_url,
    )

    df = pl.read_csv(raw_csv_path)

    df.write_delta(
        target=destination,
        storage_options=storage_options,
        mode="overwrite",
    )

    write_ready_marker(args.ready_dir)

    print(f"Bronze table '{config['table_name']}' written successfully.")
    print(f"Destination: {destination}")


def main():
    """Run the Bronze ingestion pipeline."""
    ingest_to_bronze()


if __name__ == "__main__":
    main()