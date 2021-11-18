"""
LIST OF FUNCTIONS
-----------------

- `connect_to_db`: Open a persistent connection to DB. Returns a
    sqlalchemy engine object.
- `save_df_to_parquet`: Safe dataframe to parquet with options to
    add a timestamp to the file name and to keep index or not.
"""


from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlalchemy


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
    print(f"Dataframe saved to: {path}\n")
