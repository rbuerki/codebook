# pyright: reportGeneralTypeIssues=false

"""
LIST OF FUNCTIONS
-----------------
- `read_bq_to_df_from_query`: Connect to BQ project and load
    results of the passed query string into a DataFrame.
- `read_bq_to_df_from_file`: Connect to BQ project and load
    results of the passed query file into a DataFrame.
- `connect_to_legacy_db`: Open a persistent connection to a SQL 
    ServerDB. Return a sqlalchemy engine object.
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
from typing import Any, List, Optional, Union

import pandas as pd
import sqlalchemy
from google.cloud import bigquery


logger = logging.getLogger(__name__)


def read_bq_to_df_from_query(
    query: str,
    project: str = "dg-dp-bqondemand-dev",
    dry_run: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a query and return results to a dataframe. (Or optionally
    perform a dry_run to check compilation and bytes to be processed.)
    The function accepts kwargs to the job_config object, e.g. to pass
    query parameters.
    """

    bqclient = bigquery.Client(project=project, location="EU")
    job_config = bigquery.QueryJobConfig(dry_run=dry_run, **kwargs)
    query_job = bqclient.query(query, job_config=job_config)
    size = (
        query_job.total_bytes_processed / (1024**3)
        if query_job.total_bytes_processed
        else 0
    )

    if dry_run:
        print(f"This query will process {size:,.2f} GB.")

    else:
        df: pd.DataFrame = query_job.to_dataframe()

        if verbose:
            print(f"Created: {query_job.created}")
            print(f"Ended:   {query_job.ended}")
            print(f"Data processed:   {size:,.2f} GB")

        return df


def read_bq_to_df_from_file(
    file_path: str,
    project: str = "dg-dp-bqondemand-dev",
    dry_run: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read a query from file and return results to a dataframe. (Or optionally
    perform a dry_run to check compilation and bytes to be processed.)
    The function accepts kwargs to the job_config object, e.g. to pass
    query parameters.
    """
    with open(file_path, "r", encoding="utf-8-sig") as file:
        query = file.read()

    df = read_bq_to_df_from_query(query, project, dry_run, verbose)
    return df


def connect_to_legacy_db(
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
    category_columns: Optional[List[str]] = None,
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
    """Save dataframe to aparquet file at given path. This works locally
    or for an existing gcs bucket. If folder does not exist, it is created.
    By default a timestamp is appended to the filename and the index is
    discarded. (Both options can be changed.)
    Note: To retrieve the file again, use `pd.read_parquet(path)`.
    """
    parent, file = path.rsplit("/", maxsplit=1)
    stem, suffix = file.rsplit(".", maxsplit=1)
    if add_timestamp:
        timestamp_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        stem = f"{stem}_{timestamp_string}"
    relpath = f"{parent}/{stem}.{suffix}"

    # Handle local folder creation if necessary
    if not str(relpath)[:2] == "gs":
        parent = Path(parent)
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
