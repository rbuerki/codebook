"""
LIST OF FUNCTIONS
-----------------

Columns:
- `prettify_column_names`: Replace whitespace in column labels with
  an underscore and, by default, change to all lowercase.
- `count_dtypes`: Display a dataframe showing the count of different
  column datatypes in the passed dataframe.
- `delete_columns`: Delete selected columns permanently from the 
  passed dataframe.

Missing Values:
- `plot_nan`: Display a heatmap of the passed dataframe highlighting
  the missing values.
- `list_nan`: Display a dataframe showing the missing values with their
  respective percentage of the total values in a column.
- `handle_nan`: Apply different strategies for handling missing values
  in selected columns (simplistic approach).

Duplicates:
- list_duplicates: Display the columns containing column-wise duplicates.

Outliers:
- count_outliers_IQR_method: Detect outliers in specified columns 
  depending on specified distance from 1th / 3rd quartile. NaN ignored.
- remove_outliers_IQR_method: Remove outliers in specified columns 
  depending on specified distance from 1th / 3rd quartile. NaN ignored.

Transformations:
- apply_log: Transform values of selected columns to natural log. 
  NaN not affected by default, parameter can be changed.
- apply_log10: Transform values of selected columns to log10. 
  NaN not affected by default, parameter can be changed.
- apply_box_cox: Power transform values of selected columns with box-cox.
  NOTE: Cannot handle NaN and negvalues. Workaround to handle zero values.
- apply_yeo_j: Power transform values of selected columns with yeo-johnson.
  NOTE: Cannot handle NaN but yeo-johnson works on neg and zero values.
"""

import collections
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# COLUMNS - Names, Datatypes and Removal


def prettify_column_names(
    df: pd.DataFrame, lowercase: bool = True
) -> pd.DataFrame:
    """Replace whitespace in column labels with underscore and,
    by default, change to all lowercase.
    """
    df = df.copy()
    if lowercase:
        df.columns = (col.lower().replace(" ", "_") for col in df.columns)
    else:
        df.columns = (col.replace(" ", "_") for col in df.columns)
    return df


def count_dtypes(df: pd.DataFrame):
    """Display a dataframe showing the count of different column
    datatypes in the passed dataframe.
    """
    dtypes_dict = collections.Counter(df.dtypes.values)
    display(
        pd.DataFrame(
            data=dtypes_dict.values(),
            index=dtypes_dict.keys(),
            columns=["# cols"],
        )
    )


# TODO ... remove or refactor
# def change_dtypes(
#     df,
#     cols_to_category: Union[List[str], None] = None,
#     cols_to_object: Union[List[str], None] = None,
#     cols_to_string: Union[List[str], None] = None,
#     cols_to_integer: Union[List[str], None] = None,
#     cols_to_float: Union[List[str], None] = None,
#     cols_to_datetime: Union[List[str], None] = None,
#     datetime_pattern: str = "%Y/%m/%d",
# ) -> pd.DataFrame:
#     """Transform datatypes of selected columns in the passed
#     dataframe and return it.
#     """
#     df = df.copy()
#     dtypes_dict = {
#         tuple(cols_to_category): "category",
#         tuple(cols_to_object): str,
#         tuple(cols_to_integer): np.int64,
#         tuple(cols_to_float): np.float64,
#     }

#     for cols_list, datatype in dtypes_dict.items():
#         if cols_list is not None:
#             for col in cols_list:
#                 if col in df.columns:
#                     df[col] = df[col].astype(datatype)
#                 else:
#                     print(col + " not found")

#     # different handling for datetime columns
#     if cols_to_datetime is not None:
#         for col in cols_to_datetime:
#             if col in df.columns:
#                 df[col] = pd.to_datetime(df[col], format=datetime_pattern)
#             else:
#                 print(col + " not found")
#     return df


def delete_columns(
    df: pd.DataFrame, cols_to_delete: Iterable[str]
) -> pd.DataFrame:
    """Delete columns permanently from the passed dataframe. Note: 
    This function is structure such that more columns can be deleted
    iteratively during EDA. In the end you hold the full list of
    deleted columns.
    """
    df = df.copy()
    for col in cols_to_delete:
        try:
            df.drop(col, axis=1, inplace=True)
            print(f"Column {col} successfully deleted.")
        except KeyError:
            pass
    return df


### MISSING VALUES - Detection and handling


def plot_nan(df: pd.DataFrame, figsize: Tuple[int, int] = (14, 6), **kwargs):
    """Display a heatmap of the passed dataframe highlighting the
    missing values. Additional keyword arguments will be passed to
    the actual seaborn plot function. 
    """
    defaults = {
        "cmap": "viridis",
        "yticklabels": False,
        "cbar": False,
    }
    kwargs = {**defaults, **kwargs}
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), **kwargs)


def list_nan(df):
    """Display a dataframe showing the missing values with their
    respective percentage of the total values in a column.
    """
    if df.isnull().sum().sum() == 0:
        print("No empty cells in DataFrame.")
    else:
        total = df.isnull().sum()
        percent = round(df.isnull().sum() / len(df) * 100, 1)
        dtypes = df.dtypes

        missing_data = pd.concat(
            [total, percent, dtypes], axis=1, keys=["total", "percent", "dtype"]
        )
        missing_data = missing_data.loc[missing_data["total"] != 0].sort_values(
            ["total"], ascending=False
        )
        display(missing_data)


# TODO: continue here ...
def handle_nan(
    df,
    cols_to_impute_num=None,
    cols_to_impute_cat=None,
    cols_to_drop=None,
    drop_all_NaN=False,
):
    """Handle NaN with different strategies for selected columns. Return a
    transformed copy of the DataFrame. Note: Use with caution, as there are
    more sophisticated solutions around.

    Arguments:
    ----------
    - cols_to_impute_num: list of num columns to impute median,
        (default=None)
    - cols_to_impute_cat: list of categorical columns to impute mode,
        (default=None)
    - cols_to_drop: list of columns to drop entirely,
        (default=None)
    - drop_all_NaN: bool, if True ALL remaining rows with NaN will be removed,
        (default=False)

    Returns:
    --------
    - df_NaN: DataFrame, transformed copy of original DataFrame
    """
    df_NaN = df.copy()
    if cols_to_impute_num is not None:
        for col in cols_to_impute_num:
            if col in df_NaN.columns:
                print(
                    "{} - median value to impute: {}".format(
                        col, df_NaN[col].median()
                    )
                )
                df_NaN[col] = df_NaN[col].fillna(df_NaN[col].median())
            else:
                print(col + " not found")
    if cols_to_impute_cat is not None:
        for col in cols_to_impute_cat:
            if col in df_NaN.columns:
                print(
                    "{} - most frequent value to impute: {}".format(
                        col, df_NaN[col].value_counts().index[0]
                    )
                )
                df_NaN[col] = df_NaN[col].fillna(
                    df_NaN[col].value_counts().index[0]
                )
            else:
                print(col + " not found")
    if cols_to_drop is not None:
        for col in cols_to_drop:
            if col in df_NaN.columns:
                df_NaN.drop(col, axis=1, inplace=True)
            else:
                print(col + " not found")
    if drop_all_NaN:
        df_NaN = df_NaN.dropna(how="any")  # drop remaining rows with any NaNs

    return df_NaN


# DUPLICATES


def list_duplicates(df: pd.DataFrame):
    """Display a summary of the column-wise duplicates in the passed
    dataframe.
    """
    print("Number of column-wise duplicates per column:")
    for col in df:
        dup = df[col].loc[df[[col]].duplicated(keep=False) == 1]
        dup_nunique = dup.nunique()
        dup_full = len(dup)
        if dup_nunique > 0:
            print(
                f"- {col}: {dup_nunique} unique duplicated values "
                f"({dup_full} duplicated rows)"
            )


# OUTLIERS - Count and Removal


def count_outliers_IQR_method(
    df: pd.DataFrame, iqr_dist: Union[int, float] = 1.5
):
    """Display outlier count for numeric columns in the passed
    dataframe based on the IQR distance from the 1th / 3rd quartile
    (defaults to 1.5). NaN values are ignored for the calculations.
    """
    outlier_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in outlier_cols:
        q25, q75 = np.nanpercentile(df[col], 25), np.nanpercentile(df[col], 75)
        iqr = q75 - q25
        cut_off = iqr * iqr_dist
        lower, upper = q25 - cut_off, q75 + cut_off
        outliers = [x for x in df[col] if x < lower or x > upper]

        if len(outliers) > 0:
            outlier_pct = len(outliers) / len(df[col])
            print(
                f"\n{col}:\n"
                f" - upper cut-off value: {upper:,.2f}\n"
                f" - lower cut-off value: {lower:,.2f}\n"
                f" - Identified outliers: {len(outliers):,.0f}\n"
                f" - of total values: {outlier_pct:.1%}"
            )


def remove_outliers_IQR_method(
    df: pd.DataFrame,
    outlier_cols: Optional[List[str]] = None,
    iqr_dist: Union[int, float] = 1.5,
    return_idx_deleted: bool = False,
) -> pd.DataFrame:
    """Return a dataframe with outliers removed for selected columns.
    If no specific `outlier_cols` are specified (default), the cleaning
    is applied to all numeric columns. Outliers are removed depending
    on the desired distance from 1th and 3rd quartile (defaults to 1.5).
    NaN values are ignored.

    If `return_idx_deleted` is set to True (it defaults to Flase), then
    not only the cleaned dataframe is returned, but also the list of
    the deleted index values as the second return object.
    """
    df = df.copy()
    if outlier_cols is None:
        outlier_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    len_df_in = len(df)
    rows_to_delete = []

    for col in outlier_cols:
        q25, q75 = np.nanpercentile(df[col], 25), np.nanpercentile(df[col], 75)
        iqr = q75 - q25
        cut_off = iqr * iqr_dist
        lower, upper = q25 - cut_off, q75 + cut_off

        idx_low = df[df[col] < lower].index.tolist()
        idx_high = df[df[col] > upper].index.tolist()

        print(f"\n{col}: \nRows to remove: {len(idx_low + idx_high)}\n")
        rows_to_delete = rows_to_delete + idx_low + idx_high

    rows_to_delete = set(rows_to_delete)
    df.drop(rows_to_delete, inplace=True, axis=0)
    len_df_out = len(df)
    assert len(rows_to_delete) == (len_df_in - len_df_out)

    print(
        f"\nRows removed in total: {len_df_in - len_df_out}"
        f"\n(Percentage of original DataFrame: "
        f"{len(rows_to_delete) / len_df_in:0.1%})"
    )

    if not return_idx_deleted:
        return df
    else:
        return df, list(rows_to_delete)


# TRANSFORMATIONS


def apply_log(df, cols_to_transform=None, treat_NaN=False, rename=False):
    """Transform values of selected columns to natural log. NaN are not
    affected by default, parameter can be changed. Returns a transformed
    DataFrame, column names have "_log" appended if parameter is set.

    Arguments:
    ----------
    - df: DataFrame
    - cols_to_transform: list of columns that will have the transformation
        applied, (default is all numerical columns)
    - treat_NaN: bool, set NaN to small negative value, (default=False)
    - rename: bool, rename column with appendix, (default=False)

    Returns:
    --------
    - df: DataFrame, natural log-transformed copy of original DataFrame
    """

    df_log = df.copy()
    cols_to_transform = (
        cols_to_transform
        if cols_to_transform is not None
        else list(df_log.select_dtypes(include=["float64", "int64"]).columns)
    )

    for col in df_log[cols_to_transform]:
        if col in df_log:
            df_log[col] = df_log[col].apply(lambda x: np.log(max(x, 0.001)))
            if treat_NaN:
                df_log[col].replace(np.nan, -1, inplace=True)
        else:
            print(col + " not found")

        # Rename transformed columns
        if rename:
            df_log.rename(columns={col: col + "_log"}, inplace=True)

    return df_log


def apply_log10(df, cols_to_transform=None, treat_NaN=False, rename=False):
    """Transform values of selected columns to natural log. NaN are not
    affected by default, parameter can be changed. Returns a transformed
    DataFrame, column names have "_log10" appended if parameter is set.

    Arguments:
    ----------
    - df: DataFrame
    - cols_to_transform: list of columns that will have jy-transformation
        applied, (default is all numerical columns)
    - treat_NaN: bool, set NaN to small negative value, (default=False)
    - rename: bool, rename column with appendix, (default=False)

    Returns:
    --------
    - df: DataFrame, log10-transformed copy of original DataFrame
    """

    df_log = df.copy()
    cols_to_transform = (
        cols_to_transform
        if cols_to_transform is not None
        else list(df_log.select_dtypes(include=["float64", "int64"]).columns)
    )

    for col in df_log[cols_to_transform]:
        if col in df_log:
            df_log[col] = df_log[col].apply(lambda x: np.log10(max(x, 0.001)))
            if treat_NaN:
                df_log[col].replace(np.nan, -1, inplace=True)
        else:
            print(col + " not found")

        # Rename transformed columns
        if rename:
            df_log.rename(columns={col: col + "_log10"}, inplace=True)

    return df_log


def apply_box_cox(df, cols_to_transform=None, rename=False):
    """Transform values of selected columns with box-cox. Returns transformed
    DataFrame, column names have "_bc" appended if parameter is set.
    NOTE: Cannot handle NaN and negative values. Normally bc can works on
    positive values only, this function has a little workaround is included to
    set 0 values to 0.01.

    Arguments:
    ----------
    - df: DataFrame
    - cols_to_transform: list of columns that will have jy-transformation
        applied, (default is all numerical columns)
    - treat_NaN: bool, set NaN to small negative value, (default=False)
    - rename: bool, rename column with appendix, (default=False)

    Returns:
    --------
    - df_bc: DataFrame, box-cox-transformed copy of original DataFrame
    """

    df_bc = df.copy()
    cols_to_transfrom = (
        cols_to_transform
        if cols_to_transform is not None
        else list(df_bc.select_dtypes(include=["float64", "int64"]).columns)
    )

    for col in df_bc[cols_to_transform]:
        if col in df:
            df_bc[col] = df_bc[col].apply(lambda x: x + 0.001 if x == 0 else x)
            df_bc[col] = stats.boxcox(df_bc[col])[0]
        else:
            print(col + " not found")

        # rename transformed columns
        if rename:
            df_bc.rename(columns={col: col + "_bc"}, inplace=True)

    return df_bc


def apply_yeo_j(df, cols_to_transform=None, rename=False):
    """Transform values of selected columns with yeo-johnson. Returns transformed
    DataFrame, column names have "_yj" appended if parameter is set.
    NOTE: Cannot handle NaN but contrary to box-cox, yeo-johnson works also on
    negative and zero values.

    Arguments:
    ----------
    - df: DataFrame
    - cols_to_transform: list of columns that will have jy-transformation
        applied, (default is all numerical columns)
    - rename: bool, rename column with appendix, (default=False)

    Returns:
    --------
    - df_yj: DataFrame, yeo-johnson-transformed copy of original DataFrame
    """
    df_yj = df.copy()
    cols_to_transfrom = (
        cols_to_transform
        if cols_to_transform is not None
        else list(df_yj.select_dtypes(include=["float64", "int64"]).columns)
    )

    for col in df_yj[cols_to_transform]:
        if col in df_yj:
            df_yj[col] = stats.yeojohnson(df_yj[col])[0]
        else:
            print(col + " not found")

        # Rename transformed columns
        if rename:
            df_yj.rename(columns={col: col + "_yj"}, inplace=True)

    return df_yj
