"""
LIST OF FUNCTIONS
-----------------

Columns:
- `prettify_column_names`: Replace whitespace in column labels with
   an underscore and, by default, change to all lowercase.
- `delete_columns`: Delete selected columns permanently from the
   input dataframe.

Outliers:
- `count_outliers_IQR_method`: Detect outliers in specified columns
   depending on specified distance from 1th / 3rd quartile. NaN ignored.
- `remove_outliers_IQR_method`: Remove outliers in specified columns
   depending on specified distance from 1th / 3rd quartile. NaN ignored.
- `winsorize_values`: Return a winsorized version of the selected
   columns.

Transformations:
- `transfrom_data`: Apply the desired transformation on the selected
   columns. (Methods are log, log10, box-cox or yeo-johnson.)
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
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


def delete_columns(
    df: pd.DataFrame, cols_to_delete: Iterable[str]
) -> pd.DataFrame:
    """Delete columns permanently from the input dataframe. Note:
    This function is structured such that more and more columns can be
    added to the list and deleted iteratively during EDA. In the end
    you hold the full list of deleted columns.
    """
    df = df.copy()
    for col in cols_to_delete:
        try:
            df.drop(col, axis=1, inplace=True)
            print(f"Column {col} successfully deleted.")
        except KeyError:
            pass
    return df


def downcast_dtypes(
    df: pd.DataFrame,
    category_threshold: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a copy of the input dataframe with reduced memory usage.
    Numeric dtypes will be downcast to the smallest possible format
    depending on the actual data, object dtypes with less distinct
    values than an optional threshold (default is the rowcount) will
    be transformed to dtype 'category'.

    Limitations: Only 'object' cols are considered for conversion
    to dtype 'category'.
    """
    if verbose:
        print(
            f" Original df size before downcasting: "
            f"{df.memory_usage(deep=True).sum() / (1024**2):,.2f} MB"
        )

    df = df.copy()

    for col in df.columns:
        col_type = str(df[col].dtype)
        col_cat_threshold = category_threshold or df[col].count()
        col_unique_items = df[col].nunique()

        if col_type == "object" and col_unique_items < col_cat_threshold:
            df[col] = df[col].astype("category")
        if col_type.startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        if col_type.startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")

    if verbose:
        print(
            f" New df size after downcasting:"
            f"{df.memory_usage(deep=True).sum() / (1024**2):,.2f} MB"
        )

    return df


# OUTLIERS - Return (for an iterable), Count and Remove


def get_outlier_values_with_iqr_method(
    data: Iterable[Union[int, float]], iqr_dist: float
) -> Tuple[List[Union[int, float]], float, float]:
    """Return a list of outlier values and the lower and upper
    cut-off values for the numerical data input. The outliers are
    defined by the 'IQR-Method' for which an IQR-distance has to be
    passed as the second parameter.
    """
    q25, q75 = np.nanpercentile(data, 25), np.nanpercentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * iqr_dist
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    return outliers, lower, upper


def count_outliers_IQR_method(
    df: pd.DataFrame, iqr_dist: Union[int, float] = 1.5
):
    """Display outlier count for numeric columns in the passed
    dataframe based on the IQR distance from the 1th / 3rd quartile
    (defaults to 1.5). NaN values are ignored for the calculations.
    """
    outlier_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in outlier_cols:
        outliers, lower, upper = get_outlier_values_with_iqr_method(
            col, iqr_dist=iqr_dist
        )

        if len(outliers) > 0:
            outlier_pct = len(outliers) / len(df[col])
            print(
                f"\n{col}:\n"
                f" - effective upper cut-off value: "
                f"{min(df[col].max(), upper):,.2f}\n"
                f" - effective lower cut-off value: "
                f"{max(df[col].min(), lower):,.2f}\n"
                f" - Identified outliers: {len(outliers):,.0f}\n"
                f" - of total values: {outlier_pct:.1%}"
            )


def remove_outliers_IQR_method(
    df: pd.DataFrame,
    outlier_cols: Optional[List[str]] = None,
    iqr_dist: Union[int, float] = 1.5,
    return_idx_deleted: bool = False,
) -> (pd.DataFrame, Optional[List[str]]):
    """Return a dataframe with outliers removed for selected columns.
    If no specific `outlier_cols` are specified (default), the cleaning
    is applied to all numeric columns. Outliers are removed depending
    on the desired distance from 1th and 3rd quartile (defaults to 1.5).
    NaN values are ignored.

    If `return_idx_deleted` is set to True (it defaults to Flase),
    then not only the cleaned dataframe is returned, but also the list
    of the deleted index values as the second return object.
    """
    df = df.copy()
    if outlier_cols is None:
        outlier_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    len_df_in = len(df)
    rows_to_delete = []

    for col in outlier_cols:
        _, lower, upper = get_outlier_values_with_iqr_method(
            col, iqr_dist=iqr_dist
        )

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


def winsorize_outliers(
    df: pd.DataFrame, w_dict: Dict[str, Tuple[float, float]], **kwargs
) -> pd.DataFrame:
    """Return a winsorized version of the selected columns. Besides
    the input dataframe, a `w_dict` has to be passed, consisting
    of the column names as keys and tuples of the quantiles on each
    end to be winsorized. (Note you can substitute the floats with
    "None" when you don't want to transfrom on one side.) Additional
    kwargs can be passed to the underlying `winsorize` function
    from scipy.stats.mstats.

    Example w_dict:
        w_dict = {
            "col1": (None, 0.05),
            "col2": (0.1, 0.02),
        }

    Scipy reference here:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html
    """
    df = df.copy()
    for col, limits in w_dict.items():
        df[col] = stats.mstats.winsorize(df[col], limits=limits, **kwargs)
    return df


# TRANSFORMATIONS


def transform_data(
    df: pd.DataFrame,
    method: Optional[str] = "log",
    cols_to_transform: Optional[List[str]] = None,
    treat_nan: Optional[bool] = False,
    add_suffix: Optional[bool] = True,
) -> pd.DataFrame:
    """Apply the desired transformation on the selected columns of
    the input dataframe. Missing values are ignored (see below). YJ
    is the only option that can handle 0 or negative values, the
    other methods will shift the data accoringly if they encounter
    0 or negative values.

    Possible transformations are:
    - log (compress values more the larger they are)
    - log10 (same, but somewhat easier to interpret)
    - box-cox (optimized generalization of the log transformation)
    - yeo-johnson (same but able to handle zero and negative values)

    Note: The `treat_nan` option simply sets all nan values to -1.
    Use with caution.
    """
    df = df.copy()
    cols_to_transform = (
        cols_to_transform
        or df.select_dtypes(include=np.number).columns.tolist()
    )
    suffix_dict = {
        "log": "_log",
        "log10": "_log10",
        "box_cox": "_bc",
        "yeo_johnson": "_yj",
    }
    function_dict = {
        "log": lambda x: np.log(x),
        "log10": lambda x: np.log10(x),
    }

    for col in df[cols_to_transform]:
        shift_value = 0
        if (np.min(df[col]) <= 0) & (method != "yeo_johnson"):
            shift_value = np.min(df[col]) * -1 + 0.1
            print(
                f"Zero or negative value(s) in col {col}, "
                f"all data is shifted by {shift_value}. "
                "Alternatively use method 'yeo-johnson'."
            )
        try:
            if method in ["log", "log10"]:
                df[col] = df[col] + shift_value
                df[col] = df[col].apply(function_dict.get(method))
            elif method == "box_cox":
                df[col] = df[col] + shift_value
                df[col] = stats.boxcox(df[col])[0]
            elif method == "yeo_johnson":
                df[col] = stats.yeojohnson(df[col])[0]
            if treat_nan:
                df[col].replace(np.nan, -1, inplace=True)
            if add_suffix:
                df.rename(
                    columns={col: col + suffix_dict.get(method)}, inplace=True
                )
        except KeyError:
            print(col + " not found!")
    return df


### MISSING VALUES - Handling

# def handle_nan(
#     df,
#     cols_to_impute_num=None,
#     cols_to_impute_cat=None,
#     cols_to_drop=None,
#     drop_all_NaN=False,
# ):
#     """Handle NaN with different strategies for selected columns. Return a
#     transformed copy of the DataFrame. Note: Use with caution, as there are
#     more sophisticated solutions around.

#     Arguments:
#     ----------
#     - cols_to_impute_num: list of num columns to impute median,
#         (default=None)
#     - cols_to_impute_cat: list of categorical columns to impute mode,
#         (default=None)
#     - cols_to_drop: list of columns to drop entirely,
#         (default=None)
#     - drop_all_NaN: bool, if True ALL remaining rows with NaN will be removed,
#         (default=False)

#     Returns:
#     --------
#     - df_NaN: DataFrame, transformed copy of original DataFrame
#     """
#     df_NaN = df.copy()
#     if cols_to_impute_num is not None:
#         for col in cols_to_impute_num:
#             if col in df_NaN.columns:
#                 print(
#                     "{} - median value to impute: {}".format(
#                         col, df_NaN[col].median()
#                     )
#                 )
#                 df_NaN[col] = df_NaN[col].fillna(df_NaN[col].median())
#             else:
#                 print(col + " not found")
#     if cols_to_impute_cat is not None:
#         for col in cols_to_impute_cat:
#             if col in df_NaN.columns:
#                 print(
#                     "{} - most frequent value to impute: {}".format(
#                         col, df_NaN[col].value_counts().index[0]
#                     )
#                 )
#                 df_NaN[col] = df_NaN[col].fillna(
#                     df_NaN[col].value_counts().index[0]
#                 )
#             else:
#                 print(col + " not found")
#     if cols_to_drop is not None:
#         for col in cols_to_drop:
#             if col in df_NaN.columns:
#                 df_NaN.drop(col, axis=1, inplace=True)
#             else:
#                 print(col + " not found")
#     if drop_all_NaN:
#         df_NaN = df_NaN.dropna(how="any")

#     return df_NaN
