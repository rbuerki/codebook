# pyright: reportGeneralTypeIssues=false

"""
LIST OF FUNCTIONS
-----------------

- `connect_to_db`: Open a persistent connection to DB. Returns a
    sqlalchemy engine object.
- `downcast_dtypes`: Return a copy of the dataframe with reduced
    memory usage by downcasting data formats.
- `save_df_to_parquet`: Save dataframe to parquet with options to
    add a timestamp to the file name and to keep index or not.
- `read_yaml`: Return the key-value-pairs from a YAML file, or
    a specific section of that file only.
"""

import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import sqlalchemy
from google.cloud import bigquery


logger = logging.getLogger(__name__)


def read_bq_to_df(
    query: str, dry_run: bool = False, verbose: bool = True
) -> pd.DataFrame:

    bqclient = bigquery.Client(project="dg-dp-bqondemand-dev", location="EU")

    if dry_run:
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = bqclient.query(query, job_config=job_config)
        print(
            f"This query will process {query_job.total_bytes_processed / (1024**2):,.2f} MB."
        )

    else:
        result = bqclient.query(query)
        df: pd.DataFrame = result.to_dataframe()

        if verbose:
            print(f"Created: {result.created}")
            print(f"Ended:   {result.ended}")
            print(f"Bytes:   {result.total_bytes_processed:,.0f}")

        return df


def connect_to_db(
    server: str = "BI-PRO-DB001", db_name: str = "master"
) -> sqlalchemy.engine.Engine:
    """Connect to DB and open a persistent connection. The param
    `fast_exectuemany` is active for bulk operations. Returns
    sqlalchemy engine object.
    """
    con_string = (
        f"mssql+pyodbc://@{server}/{db_name}?driver=SQL Server Native Client 11.0"
    )
    print(f"Connecting to server `{server}` and database `{db_name}`")
    return sqlalchemy.create_engine(con_string, fast_executemany=True)


def downcast_dtypes(
    df: pd.DataFrame,
    use_dtype_category: bool = False,
    category_threshold: Optional[int] = None,
    category_columns: Optional[list[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a copy of the input dataframe with reduced memory usage.
    Numeric dtypes will be downcast to the smallest possible format
    depending on the actual data. Per default there is no transformation
    to the dtype 'category', but you can change this, by

    a) either setting `use_dtype_category` to True, in which case
    columns with object dtype and less distinct values than the optional
    `category_threshold` (default value is the row count) will
    be transformed to dtype 'category'.

    b) or passing a list of column names you want to have transformed
    to dtype 'category'. In that case the parameters for option a)
    are overridden.

    Finally, one can disable the printed info output.
    """
    if verbose:
        print(
            f" Original df size before downcasting: "
            f"{df.memory_usage(deep=True).sum() / (1024**2):,.2f} MB"
        )

    df = df.copy()
    for col in [col for col in df.columns if str(df[col].dtype).startswith("int")]:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in [col for col in df.columns if str(df[col].dtype).startswith("float")]:
        df[col] = pd.to_numeric(df[col], downcast="float")

    if category_columns:
        for col in category_columns:
            df[col] = df[col].astype("category")
    else:
        if use_dtype_category:
            col_cat_threshold = category_threshold or len(df)
            for col in [col for col in df.columns if str(df[col].dtype) == "object"]:
                if df[col].nunique() < col_cat_threshold:
                    df[col] = df[col].astype("category")

    if verbose:
        print(
            f" New df size after downcasting:"
            f"{df.memory_usage(deep=True).sum() / (1024**2):,.2f} MB"
        )
    return df


def save_df_to_parquet(
    df: pd.DataFrame, path: str, add_timestamp=False, keep_index=False
):
    """Save dataframe to parquet file at given path. If folder does not
    exist, it is created. Options to add a timestamp to the filename
    (default=False) and keep the index (default=False). To retrieve, use
    `pd.read_parquet(path)`.
    """
    relpath = Path(path)
    parent = relpath.parent
    if add_timestamp:
        timestamp_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        stem = f"{relpath.stem}_{timestamp_string}"
        suffix = relpath.suffix
        relpath = Path(parent) / f"{stem}{suffix}"

    parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(relpath, index=keep_index)
    print(f"Dataframe saved to: {relpath}\n".replace("\\", "/"))


def read_yaml(file_path: Union[str, Path], section: Optional[str]) -> Any:
    """Return the key-value-pairs from a YAML file, or, if the
    optional `section` parameter is passed, only from a specific
    section of that file.
    """
    with open(file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    if not section:
        return yaml_content
    else:
        try:
            return yaml_content[section]
        except KeyError:
            logging.error(f"Section {section} not found in config file. Please check.")
            raise
