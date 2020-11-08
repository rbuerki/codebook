"""
LIST OF FUNCTIONS
-----------------

Dataframe values:
- `display_distinct_values`: Return a dataframe containing the number
   of distinct values for each column of the passed dataframe.
- `display_value_counts`: Display a dataframe containing the value
   counts and their respective pct for a column or a list of columns.
- `display_tail_transposed`: Return transposed tail of the passed
   dataframe with cols shown as rows and values for 5 instances as cols.

Distributions:
- `plot_distr_histograms`: Display a histogram for every numeric
   column in the passed dataframe.
- `plot_distr_boxplots`: Display a boxplot for every numeric
   column in the passed dataframe.
- `plot_distr_pies`: Display a pieplot for every column of dtype
  "category" (with up to 30 distinct values) in the passed dataframe.

Correlations:
- `plot_corr_full_heatmap`: Display a heatmap to show the correlations
   between all numeric columns in the Dataframe.
- `plot_corr_to_target_barchart`: Display a barchart for every numeric
   feature in the passed dataframe to show the correlation to a
   numeric target variable.
- `plot_corr_to_target_regplots`: Display a regplot for every numeric
   feature in the passed dataframe to show the correlation to a
   numeric target variable.
- `plot_corr_to_target_lineplots`: Display a lineplot for every numeric
  feature in the passed dataframe with up to (by default) 100 distinct
  values to analyze the correlation to a numeric target variable.
- `plot_corr_to_target_boxplots`: Display a boxplot for every numeric
   feature column in the passed dataframe to analyze the correlation to
   a target variable with few distinct values (any dtype possible).
- `plot_corr_to_target_pointplots_with_pies`: Display a pointplot
   (and corresponding piechart) for every feature with dtype "category"
   in the passed dataframe to display the correlation to a numeric
   target variable.
- `plot_corr_to_target_stripplots`: Display a stripplot for each
   feature with dtype "category" in the passed dataframe to analyze
   the correlation to a numeric target variable.

Cumulative Sums / Counts:
- `display_cumcurve_stats`: Return a dataframe with cumsum stats for an
  iterable of numeric values.
- plot_cumsum_curve`: Display a cumsum curve for an iterable of numeric
  values.
"""

from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# DATAFRAME VALUES


def display_distinct_values(df: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Return a dataframe containing the number of distinct values
    for each column of the passed dataframe.
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
    df: pd.DataFrame, n_rows: Optional[int] = None,
):
    """Display a dataframe containing the value counts and their
    respective pct for each column of the passed dataframe. The max
    number of values to display (ordered desc by counts) can be
    defined by the optional n_rows parameter.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)

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

        if n_rows is not None:
            df_out = df_out.iloc[:n_rows, :]
            caption_str = "".join([caption_str, f", top {n_rows}"])

        display(
            df_out.style.format(
                {"counts": "{:,.0f}", "prop": "{:.1%}", "cum_prop": "{:.1%}"}
            ).set_caption(caption_str)
        )


def display_tail_transposed(
    df: pd.DataFrame, max_row: int = 100, random_state: Optional[int] = None
):
    """Return the transposed tail of the passed dataframe with the
    columns shown as rows and values for 5 sample instances as
    columns. The max number of rows can be adapted (defaults to 100).
    A random state seed can be specified (defaults to None).
    """
    df = df.sample(frac=1, random_state=random_state)
    with pd.option_context("display.max_rows", max_row):
        # TODO: Not sure if return works for very wide frames,
        # or if i have to reset to display
        return df.tail(5).transpose()


# DISTRIBUTIONS


def plot_distr_histograms(
    df: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None, **kwargs
):
    """Display a histogram for every numeric column in the passed
    dataframe. If not explicitely passed, a suitable figsize is
    interfered. Additional keyword arguments will be passed to the
    actual Seaborn plot function.
    """
    num_cols = df.select_dtypes(include=np.number).columns
    defaults = {"bins": 50, "color": "rebeccapurple", "kde": True}
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
    df: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None, **kwargs
):
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
    df: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None, **kwargs
):
    """Display a pieplot for every column of dtype "category" in the
    passed dataframe that has no more than 30 distinct values. If not
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


# CORRELATIONS


def plot_corr_full_heatmap(
    df: pd.DataFrame, figsize: Tuple[int, int] = (14, 10), **kwargs
):
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
):
    """Display a barchart to show the correlations between the
    numeric features in the passed dataframe and a numeric target
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
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs,
):
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

    defaults = {"color": "rebeccapurple", "line_kws": {"color": "yellow"}}
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
    figsize: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    **kwargs,
):
    """Display a lineplot for every numeric feature in the
    passed dataframe with up to (by default) 100 distinct values
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
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs,
):
    """Display a boxplot for every numeric feature column in the
    passed dataframe to analyze the correlation to a target variable
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
    figsize: Optional[Tuple[int, int]] = None,
    ylim: Optional[Tuple[int, int]] = None,
    **kwargs,
):
    """Display a pointplot (and corresponding piechart) for every
    feature with dtype "category" in the passed dataframe to display
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
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs,
):
    """Display a stripplot for each feature with dtype "category" in
    the passed dataframe to analyze the correlation to a numeric target
    variable. If not explicitely passed, a suitable figsize is
    interfered. Additional keyword arguments will be passed to the
    actual Seaborn plot function.
    """
    cat_cols = df.select_dtypes(include="category").columns

    defaults = {"palette": "rocket"}
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
) -> pd.DataFrame:
    """Return a dataframe with stats for a cumsum calculation. Input
    variables are: an iterable of numeric values (must be all positive)
    and an iterable containing the thresholds you want stats for
    (values have to be in the range of (0, 1)).
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
    figsize: Optional[Tuple[int, int]] = (12, 5),
    **kwargs,
):
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
    defaults = {
        "color": "rebeccapurple",
    }
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

    for thresh_value in [0.8, 0.2, 0.5]:
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
    if iterable.min() < 0:
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
