"""Transform the Bronze Delta Lake table into the Silver Delta Lake layer."""

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
        description="Transform the Bronze Delta Lake table into the Silver layer."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Input Bronze Delta Lake path. Overrides the config file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output Silver Delta Lake path. Overrides the config file.",
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
    """Load the Silver configuration from the YAML file."""
    config_path = Path("configs/silver.yaml")

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
        "Silver stage completed successfully.\n",
        encoding="utf-8",
    )

def clean_silver():
    """Read the Bronze Delta table, clean it, and write the Silver table."""
    args = parse_args()
    config = load_config()

    default_input = (
        f"abfss://{config['container']}@"
        f"{config['storage_account']}.dfs.core.windows.net/"
        f"bronze/{config['bronze_table']}"
    )

    default_output = (
        f"abfss://{config['container']}@"
        f"{config['storage_account']}.dfs.core.windows.net/"
        f"silver/{config['silver_table']}"
    )

    input_path = args.input_path or default_input
    output_path = args.output_path or default_output

    storage_options = get_storage_options(
        storage_account=config["storage_account"],
        key_vault_url=args.key_vault_url,
    )

    df = pl.read_delta(
        source=input_path,
        storage_options=storage_options,
    )

    df = (
        df.drop_nulls()
        .unique()
        .rename(
            {
                column: column.lower().replace(" ", "_")
                for column in df.columns
            }
        )
    )

    df.write_delta(
        target=output_path,
        storage_options=storage_options,
        mode="overwrite",
    )

    write_ready_marker(args.ready_dir)

    print(f"Silver table '{config['silver_table']}' written successfully.")
    print(f"Source: {input_path}")
    print(f"Destination: {output_path}")


def main():
    """Run the Silver transformation pipeline."""
    clean_silver()


if __name__ == "__main__":
    main()
