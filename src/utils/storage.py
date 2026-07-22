"""
Utility functions for constructing ADLS URLs.

These helpers keep Bronze ingestion code clean and modular.
"""


def build_raw_url(account: str, container: str, raw_path: str) -> str:
    """
    Build the HTTPS URL for accessing raw data in ADLS.

    Parameters
    ----------
    account : str
        Storage account name.
    container : str
        Container name.
    raw_path : str
        Path to the raw file inside the container.

    Returns
    -------
    str
        Fully qualified HTTPS URL for the raw file.
    """
    return f"https://{account}.blob.core.windows.net/{container}/{raw_path}"


def build_bronze_url(account: str, container: str, bronze_table: str) -> str:
    """
    Build the ABFS URL for writing Delta Lake Bronze tables.

    Parameters
    ----------
    account : str
        Storage account name.
    container : str
        Container name.
    bronze_table : str
        Name of the Bronze table.

    Returns
    -------
    str
        Fully qualified ABFS URL for the Bronze Delta table.
    """
    return (
        f"abfs://{container}@{account}.dfs.core.windows.net/"
        f"bronze/{bronze_table}"
    )
