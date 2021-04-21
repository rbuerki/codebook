"""
LIST OF FUNCTIONS
-----------------

Dataframe Values:
- `display_distinct_values`: Return a dataframe containing the number
   of distinct values for each column of the input dataframe.
- `display_value_counts`: Display a dataframe containing the value
   counts and their respective pct for a column or a list of columns.
- `display_df_sample_transposed`: Return transposed tail of the passed
   dataframe with cols shown as rows and values for 5 instances as cols.
- `display_dtypes`: Return a dataframe showing the count of different
   datatypes for the columns in the input dataframe.

Missing Values and Duplicates:
- `display_nan`: Return a dataframe showing the missing values with
   their respective percentage of the total values in a column.
- `plot_nan`: Display a heatmap of the input dataframe, highlighting
   the missing values.
- `display_duplicates`: Print a summary of the column-wise duplicates
   in the input dataframe.

Distributions:
- `plot_distr_histograms`: Display a histogram for every numeric
   column in the input dataframe.
- `plot_distr_boxplots`: Display a boxplot for every numeric
   column in the input dataframe.
- `plot_distr_pies`: Display a pieplot for every column of dtype
  "category" (with up to 30 distinct values) in the input dataframe.
- `plot_distr_pdf_ecdf`: Display a histogram overlaid with an ECDF
   for every numeric column in the input dataframe.

Correlations:
- `plot_corr_full_heatmap`: Display a heatmap to show the correlations
   between all numeric columns in the Dataframe.
- `plot_corr_to_target_barchart`: Display a barchart for every numeric
   feature in the input dataframe to show the correlation to a
   numeric target variable.
- `plot_corr_to_target_regplots`: Display a regplot for every numeric
   feature in the input dataframe to show the correlation to a
   numeric target variable.
- `plot_corr_to_target_lineplots`: Display a lineplot for every numeric
  feature in the input dataframe with up to (by default) 100 distinct
  values to analyze the correlation to a numeric target variable.
- `plot_corr_to_target_boxplots`: Display a boxplot for every numeric
   feature column in the input dataframe to analyze the correlation to
   a target variable with few distinct values (any dtype possible).
- `plot_corr_to_target_pointplots_with_pies`: Display a pointplot
   (and corresponding piechart) for every feature with dtype "category"
   in the input dataframe to display the correlation to a numeric
   target variable.
- `plot_corr_to_target_stripplots`: Display a stripplot for each
   feature with dtype "category" in the input dataframe to analyze
   the correlation to a numeric target variable.

Cumulative Sums / Counts:
- `display_cumcurve_stats`: Return a dataframe with cumsum stats for an
  iterable of numeric values.
- plot_cumsum_curve`: Display a cumsum curve for an iterable of numeric
  values.
"""

import collections
from typing import Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.io.formats.style import Styler


# DATAFRAME VALUES


def display_distinct_values(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Return a dataframe containing the number of distinct values
    for each column of the input dataframe.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)

    n_distinct_list = [df[col].nunique() for col in df.columns]
    df_out = pd.DataFrame(
        list(zip(df.columns, n_distinct_list)),
        columns=["Column", "#_distinct_values"],
    )
    df_out.set_index("Column", drop=True, inplace=True)
    return df_out


def display_value_counts(
    df: pd.DataFrame, n_rows: Optional[int] = None, return_dict: bool = False
) -> Optional[Dict[str, pd.DataFrame]]:
    """Display a dataframe containing the value counts and their
    respective pct for each column of the input dataframe. The max
    number of values to display (ordered desc by counts) can be
    defined by the optional n_rows parameter.

    If the return_dict param is set to True (False by default)
    a dict of the dataframes is returned too.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)

    # Initialize dict for returning dataframes
    df_dict = {}

    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        prop = df[col].value_counts(dropna=False) / len(df)
        cum_prop = np.cumsum(prop)
        df_out = pd.concat(
            [counts, prop, cum_prop],
            axis=1,
            keys=["counts", "prop", "cum_prop"],
        )
        caption_str = f"{col}"

        df_dict[col] = df_out

        if n_rows is not None:
            df_out = df_out.iloc[:n_rows, :]
            caption_str = "".join([caption_str, f", top {n_rows}"])

        display(
            df_out.style.format(
                {"counts": "{:,.0f}", "prop": "{:.1%}", "cum_prop": "{:.1%}"}
            ).set_caption(caption_str)
        )
    if return_dict:
        return df_dict


def display_df_sample_transposed(
    df: pd.DataFrame,
    n_instances: int = 5,
    max_rows: int = 100,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Return a transposed sample of (by default) 5 original dataframe
    rows, so that they are shown as columns and vice versa.
    This helps to get a better overview of wide dataframes. The max
    of orignal colums to be displayed defaults to 100 and could be
    changed for really wide frames. Also a random state seed can be
    fixed for the sampling (it defaults to None).
    """
    df = df.sample(n=n_instances, random_state=random_state).copy()
    with pd.option_context("display.max_rows", max_rows):
        # TODO: Not sure if return works for very wide frames,
        # or if i have to reset to display
        return df.transpose()


def display_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe showing the count of different datatypes
    for the columns in the input dataframe.
    """
    dtypes_list = [str(val) for val in df.dtypes.values]
    dtypes_dict = collections.Counter(dtypes_list)
    return pd.DataFrame(
        data=dtypes_dict.values(), index=dtypes_dict.keys(), columns=["# cols"],
    ).sort_index()


# MISSING VALUES AND DUPLICATES


def display_nan(df: Union[pd.DataFrame, pd.Series]) -> Union[None, Styler]:
    """If there is Nan, return a dataframe styler object showing
    the missing values with their respective percentage of the total
    values in a column. Note: Empty strings qualify as NaN.

    Note: The function returns a Styler object. If you need the
    underlying dataframe, you can get it with `df.data`.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)

    df = df.copy()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    if (df == "").any().any():
        print(
            "Warning: There are empty strings in the dataframe.",
            "They are displayed as missing values.",
        )
    df = df.replace(str(""), np.nan)

    if df.isnull().sum().sum() == 0:
        print("No empty cells in DataFrame.")
        return None

    else:
        total = df.isnull().sum()
        prop = df.isnull().sum() / len(df)
        dtypes = df.dtypes

        missing_data = pd.concat(
            [total, prop, dtypes], axis=1, keys=["total", "prop", "dtype"]
        )
        missing_data = missing_data.loc[missing_data["total"] != 0].sort_values(
            ["total"], ascending=False
        )
        return missing_data.style.format({"prop": "{:0.1%}"})


def plot_nan(
    df: pd.DataFrame, figsize: Tuple[int, int] = (14, 6), **kwargs
) -> None:
    """Display a heatmap of the input dataframe, highlighting the
    missing values. Additional keyword arguments will be passed to
    the actual seaborn plot function. Note: Empty strings qualify
    as NaN.

    Attention: For large datasets this plot can be misleading. Do
    not use without calling `display_nan` function also!
    """
    df = df.copy()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    if (df == "").any().any():
        print(
            "Warning: There are empty strings in the dataframe.",
            "They are displayed as missing values.",
        )
    df = df.replace(str(""), np.nan)

    defaults = {
        "cmap": "viridis",
        "yticklabels": False,
        "cbar": False,
    }
    kwargs = {**defaults, **kwargs}
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), **kwargs)


def display_duplicates(df: Union[pd.DataFrame, pd.Series]) -> None:
    """Print a summary of the column-wise duplicates in the passed
    dataframe.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)
    dup_count = 0
    print("Number of column-wise duplicates per column:")
    for col in df:
        dup = df[col].loc[df[[col]].duplicated(keep=False) == 1]
        dup_nunique = dup.nunique()
        dup_full = len(dup)
        if dup_nunique > 0:
            print(
                f" - {col}: {dup_nunique} unique duplicated values "
                f"({dup_full} duplicated rows)"
            )
        dup_count += dup_nunique

    if dup_count == 0:
        print("... No duplicate values in columns.")


# DISTRIBUTIONS


def plot_distr_histograms(
    df: pd.DataFrame, figsize: Optional[Tuple[float, float]] = None, **kwargs
) -> None:
    """Display a histogram for every numeric column in the passed
    dataframe. If not explicitely passed, a suitable figsize is
    interfered. Additional keyword arguments will be passed to the
    actual Seaborn plot function.
    """
    num_cols = df.select_dtypes(include=np.number).columns
    defaults = {"bins": "auto", "color": "rebeccapurple", "kde": True}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = np.ceil(len(num_cols) / 4) * 3.5
        figsize = (12, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(num_cols, 1):
        plt.subplot(int(np.ceil(len(num_cols) / 4)), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.histplot(df[col].dropna(), **kwargs)
    plt.show()


def plot_distr_boxplots(
    df: pd.DataFrame, figsize: Optional[Tuple[float, float]] = None, **kwargs
) -> None:
    """Display a barchart for every numeric feature in the passed
    dataframe to show the correlation to a numeric target variable.
    If not explicitely passed, a suitable figsize isinterfered.
    Additional keyword arguments will be passed to the actual Seaborn
    plot function.
    """
    num_cols = df.select_dtypes(include=np.number).columns
    defaults = {"color": "rebeccapurple"}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = np.ceil(len(num_cols) / 4) * 3.5
        figsize = (12, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(num_cols, 1):
        plt.subplot(int(np.ceil(len(num_cols) / 4)), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.boxplot(y=col, data=df, **kwargs)
    plt.show()


def plot_distr_pies(
    df: pd.DataFrame, figsize: Optional[Tuple[float, float]] = None, **kwargs
) -> None:
    """Display a pieplot for every column of dtype "category" in the
    input dataframe that has no more than 30 distinct values. If not
    explicitely passed, a suitable figsize is interfered. Additional
    keyword arguments will be passed to the actual pandas plot
    function.
    """
    cat_cols = df.select_dtypes(include="category").columns
    defaults = {"cmap": "viridis"}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = np.ceil(len(cat_cols) / 3) * 3.5
        figsize = (12, height)
    cols_with_many_distinct_values = []
    pos = 0

    plt.figure(figsize=figsize)
    for col in cat_cols:
        if df[col].nunique() <= 30:
            pos += 1
            plt.subplot(int(np.ceil(len(cat_cols) / 3)), 3, pos)
            plt.tight_layout(w_pad=1)
            df[col].value_counts().plot(kind="pie", **kwargs)
        else:
            cols_with_many_distinct_values.append(col)
    plt.show()

    if len(cols_with_many_distinct_values) > 0:
        print("No plot (to many distinct values) for:")
        for col in cols_with_many_distinct_values:
            print(f"- {col}")


def plot_distr_pdf_ecdf(
    df: Union[pd.DataFrame, pd.core.series.Series],
    figsize: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    percentiles: Optional[Iterable[float]] = (2.5, 25, 50, 75, 97.5),
    **kwargs,
) -> None:
    """Display a histogram overlaid with an ECDF for every numeric
    column in the input dataframe. By default selected percentile
    markers are displayed on the ECDF. (Can be changed or removed.)
    If not explicitely passed, a suitable figsize is interfered.
    Additional keyword arguments will be passed to the actual Seaborn
    plot functions for the histogram and ECDF. (Note: The optional
    xlim argument works best if you plot only one series.)

    One useful kwarg is the "hue" parameter. If you use it, make sure
    to include the respective column in the dataframe, even if it is
    not numeric. The percentile markers will be displayed for an
    invisible overall ECDF curve only in this case.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    defaults = {"palette": ["rebeccapurple", "orchid"], "hue": None}
    kwargs = {**defaults, **kwargs}

    if kwargs.get("hue") is not None:
        try:
            num_cols.remove(kwargs.get("hue"))
        except ValueError:
            pass

    if figsize is None:
        height = len(num_cols) * 5
        figsize = (12, height)

    plt.subplots(nrows=len(num_cols), ncols=1, figsize=figsize)

    for pos, col in enumerate(num_cols, 1):
        plt.subplot(len(num_cols), 1, pos)

        _ = sns.histplot(data=df, x=col, alpha=0.4, **kwargs)

        plt.grid(which="major", axis="x", color="lightgray")
        plt.twinx()

        _ = sns.ecdfplot(data=df, x=col, **kwargs)

        plt.grid(which="major", axis="y", color="lightgray")

        if percentiles is not None:
            percentiles = np.array(percentiles)

            # Compute and print percentile values
            pctile_values = np.percentile(df[col].dropna(), percentiles)
            print(
                f"{col} - Percentile values: "
                f"{[round(x, 2) for x in pctile_values]}"
            )

            # Overlay percentiles as diamonds
            _ = plt.plot(
                pctile_values,
                percentiles / 100,
                marker="D",
                color="purple",
                linestyle="none",
            )

    if xlim:
        plt.xlim(xlim)
    plt.tight_layout(w_pad=1)
    plt.suptitle("PDF & ECDF\n", y=1.02, size=14)
    plt.show()


# CORRELATIONS


def plot_corr_full_heatmap(
    df: pd.DataFrame, figsize: Tuple[int, int] = (14, 10), **kwargs
) -> None:
    """Display a heatmap to show the correlations between all numeric
    columns in the Dataframe. Optional figsize and additional keyword
    arguments will be passed to the actual Seaborn plot function.
    """
    df_num = df.select_dtypes(include=np.number)
    defaults = {
        "cmap": "magma",
        "linecolor": "white",
        "linewidth": 1,
        "vmax": 1.0,
        "vmin": -1.0,
        "annot": True,
    }
    kwargs = {**defaults, **kwargs}

    plt.figure(figsize=figsize)
    sns.heatmap(df_num.corr(), **kwargs)
    plt.show()


def plot_corr_to_target_barchart(
    df: pd.DataFrame,
    target_col: str,
    figsize: Tuple[int, int] = (14, 8),
    **kwargs,
) -> None:
    """Display a barchart to show the correlations between the
    numeric features in the input dataframe and a numeric target
     variable. Optional figsize and additional keyword arguments
     will be passed to the actual pandas plot function.
    """
    df_num = df.select_dtypes(include=np.number)
    defaults = {
        "color": "rebeccapurple",
        "title": f"Correlations to Target Variable: '{target_col}'",
    }
    kwargs = {**defaults, **kwargs}
    correlations = df_num.corr()[target_col].sort_values(ascending=False)

    plt.figure(figsize=figsize)
    correlations.drop(target_col).plot.bar(**kwargs)
    plt.show()


def plot_corr_to_target_regplots(
    df: pd.DataFrame,
    target_col: str,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Display a regplot for every numeric feature in the passed
    dataframe to show the correlation to a numeric target variable.
     If not explicitely passed, a suitable figsize is interfered.
    Additional keyword arguments will be passed to the actual
    Seaborn plot function.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    try:
        num_cols.remove(target_col)
    except ValueError:
        raise ValueError(f"Found no numeric column with name {target_col}.")

    defaults = {
        "color": "rebeccapurple",
        "marker": ".",
        "line_kws": {"color": "yellow"},
    }
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = np.ceil(len(num_cols) / 2) * 3.5
        figsize = (12, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(num_cols, 1):
        plt.subplot(int(np.ceil(len(num_cols) / 2)), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.regplot(data=df, x=col, y=target_col, **kwargs)
    plt.show()


def plot_corr_to_target_lineplots(
    df: pd.DataFrame,
    target_col: str,
    value_threshold: Optional[int] = 100,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> None:
    """Display a lineplot for every numeric feature in the
    input dataframe with up to (by default) 100 distinct values
    to analyze the correlation to a numeric target variable. If not
    explicitely passed, a suitable figsize is interfered. The same
    is true for the ylim argument. Additional keyword arguments will
    be passed to the actual Seaborn plot function.

    By default this implementation uses estimator='mean' and ci=95,
    this could be changed by passing kwargs.
    """
    print(
        f"Only columns with up to {value_threshold} distinct values are shown."
    )
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if value_threshold:
        num_cols = [
            col for col in num_cols if df[col].nunique() <= value_threshold
        ]
    try:
        num_cols.remove(target_col)
    except ValueError:
        pass
    defaults = {"color": "rebeccapurple"}
    kwargs = {**defaults, **kwargs}

    if figsize is None:
        height = np.ceil(len(num_cols) / 2) * 3.5
        figsize = (12, height)
    if ylim is None:
        upper = df[target_col].max()
        lower = df[target_col].min()
        ylim = (lower, upper)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(num_cols, 1):
        plt.subplot(int(np.ceil(len(num_cols) / 2)), 2, pos)
        plt.tight_layout(w_pad=1)
        plt.ylim(*ylim)
        plt.xlabel(col)
        sns.lineplot(data=df, x=col, y=df[target_col], **kwargs)
    plt.show()


def plot_corr_to_target_boxplots(
    df: pd.DataFrame,
    target_col: str,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Display a boxplot for every numeric feature column in the
    input dataframe to analyze the correlation to a target variable
    with few distinct values (any dtype possible). If not explicitely
    passed, a suitable figsize is interfered. Additional keyword
    arguments will be passed to the actual Seaborn plot function.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    try:
        num_cols.remove(target_col)
    except ValueError:
        pass

    target_col = df[target_col].astype("category")

    defaults = {"color": "rebeccapurple"}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = np.ceil(len(num_cols) / 2) * 3.5
        figsize = (12, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(num_cols, 1):
        plt.subplot(int(np.ceil(len(num_cols) / 2)), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.boxplot(x=target_col, y=df[col], **kwargs)
        plt.xlabel(None)
    plt.show()


def plot_corr_to_target_pointplots_with_pies(
    df: pd.DataFrame,
    target_col: str,
    figsize: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Display a pointplot (and corresponding piechart) for every
    feature with dtype "category" in the input dataframe to display
    the correlation to a numeric target variable. If not explicitely
    passed, a suitable figsize is interfered. The same is true for
    the ylim tuple. No additional key word arguments allowed for this
    function. - It's complicated enough ;-).
    """
    cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
    try:
        cat_cols.remove(target_col)
    except ValueError:
        pass

    if figsize is None:
        height = np.ceil(len(cat_cols) / 2) * 5
        figsize = (12, height)
    if ylim is None:
        upper = df[target_col].max()
        lower = df[target_col].min()
        ylim = (lower, upper)

    pos = 0
    plt.figure(figsize=figsize)
    for col in cat_cols:
        pos += 1
        df_plot = (
            df[[col, target_col]]
            .groupby(col, as_index=False)
            .mean()
            .sort_values(by=target_col, ascending=False)
        )
        plt.subplot(len(cat_cols), 2, pos)
        plt.tight_layout(w_pad=1)
        plt.ylim(ylim)
        sns.pointplot(x=col, y=target_col, data=df_plot, color="rebeccapurple")
        pos += 1
        if df[col].nunique() <= 30:
            plt.subplot(len(cat_cols), 2, pos)
            df[col].value_counts().plot(kind="pie", cmap="magma")
    plt.show()


def plot_corr_to_target_stripplots(
    df: pd.DataFrame,
    target_col: str,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> None:
    """Display a stripplot for each feature with dtype "category" in
    the input dataframe to analyze the correlation to a numeric target
    variable. If not explicitely passed, a suitable figsize is
    interfered. Additional keyword arguments will be passed to the
    actual Seaborn plot function.
    """
    cat_cols = df.select_dtypes(include="category").columns

    defaults = {"palette": "rocket", "marker": "."}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = np.ceil(len(cat_cols) / 2) * 3.5
        figsize = (12, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(cat_cols, 1):
        df_plot = df[[col, target_col]]
        plt.subplot(int(np.ceil(len(cat_cols) / 2)), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.stripplot(x=col, y=target_col, data=df_plot, **kwargs)
    plt.show()


# Cumulative COUNTS / SUMS


def display_cumcurve_stats(
    iterable: Iterable[Union[int, float]], threshold_list: Iterable[float]
) -> Styler:
    """Return a dataframe styler object with stats for a cumsum
    calculation. Input variables are: an iterable of numeric values
    (must be all positive) and an iterable containing the thresholds
    you want stats for (values have to be in the range of (0, 1)).

    Note: The function returns a Styler object. If you need the
    underlying dataframe, you can get it with `df.data`.
    """
    stats_list = []
    threshold_list_sorted = sorted(threshold_list)
    iterable_checked = _check_iterable_for_cumsum(iterable)
    iterable_sorted = sorted(iterable_checked, reverse=True)
    iter_cumsum = np.cumsum(iterable_sorted)

    for threshold in threshold_list_sorted:
        thres_sum = iter_cumsum[-1] * threshold
        n_instances = sum(iter_cumsum < thres_sum) + 1
        prop_instances = round(n_instances / len(iter_cumsum), 2)
        cum_value = iter_cumsum[n_instances - 1]

        if n_instances != 1:
            lowest_value = (
                iter_cumsum[n_instances - 1] - iter_cumsum[n_instances - 2]
            )
        else:
            lowest_value = iter_cumsum[n_instances - 1]

        stats_list.append(
            {
                "value_threshold": f"{int(threshold * 100)}%",
                "total_cum_value": cum_value,
                "lowest_value_in_bin": lowest_value,
                "total_cum_count": n_instances,
                "total_cum_count_prop": prop_instances,
            }
        )
    stats_df = pd.DataFrame(stats_list).set_index("value_threshold")
    return stats_df.style.format(
        {
            "total_cum_value": "{:,.2f}",
            "lowest_value_in_bin": "{:,.2f}",
            "total_cum_count": "{:,.0f}",
            "total_cum_count_prop": "{:.1%}",
        }
    )


def plot_cumsum_curve(
    iterable: Iterable[Union[int, float]],
    threshold_list: Optional[Iterable[float]] = (0.2, 0.5, 0.8),
    figsize: Optional[Tuple[float, float]] = (12, 5),
    **kwargs,
) -> None:
    """Display a cumsum curve for an iterable of numeric values
    (must be all positive). very numeric column in the passed
    dataframe. Intercept lines are displayed for the values
    in the optional threshold_list. (You can deactivate that
    by setting it to None.) Optional figsize and additional
    keyword arguments will be passed to the actual Seaborn plot
    function.
    """
    iterable_checked = _check_iterable_for_cumsum(iterable)
    iterable_sorted = sorted(iterable_checked, reverse=True)
    iter_cumsum = np.cumsum(iterable_sorted)
    defaults = {"color": "rebeccapurple", "linewidth": 2.5}
    kwargs = {**defaults, **kwargs}

    plt.figure(figsize=figsize)
    line = sns.lineplot(data=iter_cumsum, **kwargs)

    # Make sure the xticks look good, no matter the size of the iterable
    xtick_round_value = (len(str(int(len(iter_cumsum)))) - 2) * -1
    xtick_interval = int(round(len(iter_cumsum) / 10, xtick_round_value))

    plt.xticks(range(0, len(iter_cumsum) + 10, xtick_interval))
    plt.ylabel("Cumulative Value", fontsize=12)
    plt.xlabel("Count", fontsize=12)

    title = "Cumulative Total Value Vs. Count of Instances"
    if isinstance(iterable, pd.core.series.Series):
        title = f"{iterable.name}: {title}"
    plt.title(title, fontsize=14)

    for thresh_value in threshold_list:
        thresh_sum = iter_cumsum[-1] * thresh_value
        thresh_count = sum(iter_cumsum < thresh_sum) + 1
        y_intercept = thresh_sum
        x_intercept = thresh_count
        line.hlines(
            y=thresh_sum, xmin=0, xmax=x_intercept, linewidth=0.5, color="gray"
        )
        line.vlines(
            x=thresh_count,
            ymin=0,
            ymax=y_intercept,
            linewidth=0.5,
            color="gray",
        )
        line.annotate(
            f"{thresh_value:.0%}",
            xy=(-10, y_intercept + (iter_cumsum[-1] / 100)),
            color="black",
        )

    from matplotlib.ticker import StrMethodFormatter

    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    plt.show()


def _check_iterable_for_cumsum(
    iterable: Iterable[Union[int, float]]
) -> Iterable[Union[int, float]]:
    """Raise an error if the passed iterable for the cumsum functions
    includes negative values. If there are 0 values or missing values
    print a warning. Return the iterable without the missing values.
    """
    if min(iterable) < 0:
        raise AssertionError(
            "There cannot be negative values in the input iterable."
        )
    if sum(iterable == 0) > 0:
        print(
            f"Attention: {sum(iterable == 0)} "
            f"instances with value 0 not included."
        )
    missing_values = list(filter(lambda v: v != v, iterable))
    if len(missing_values) > 0:
        iterable = list(filter(lambda v: v == v, iterable))
        print(
            f"Attention: {len(missing_values)} "
            f"instances with missing value not included."
        )
    return iterable
