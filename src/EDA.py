""" LIST OF FUNCTIONS
    -----------------

Dataframe values:
- `display_distinct_values`: Display a dataframe containing the number
   of distinct values for each column of the passed dataframe.
- `display_value_counts_ptc`: Display a dataframe containing the value
   counts and their respective pct for a column or a list of columns.
- `display_tail_transposed`: Display transposed tail of the dataframe
   with all the orig cols as rows and values for 5 instances as columns.

# TODO replace ...
Distributions:
- plot_num_hist: Display histograms for all numerical columns in DataFrame.
- plot_num_box: Display boxplots for all numerical columns in DataFrame.
- plot_cat_pies: Display pieplots for all categorical columns in DataFrame with
  up to 30 unique values.

Correlations:
- plot_num_corrMap: Display heatmap to show correlations between all numerical
  columns in the Dataframe.
- plot_corr_bar_num_target: Display sorted barchart to show correlations between
  all numerical features and numerical target variable.
- plot_corr_regression_num_target: Display regplots to visualize correlations
  between the numerical features and numerical target variable.
- plot_corr_box_num_target: Display boxplots to show correlations between all
  numerical features and target classes.
- plot_corr_line_num_target: Display lineplots to show correlation details
  between all numerical features and target classes.
- plot_corr_strip_cat_target: Display stripplots to show correlations between
  the categorical features and numerical target variable.
- plot_corr_point_cat_target: Display pointplots (and corresponding piecharts)
  to show correlations between all categorical columns and target classes.
"""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


COLOR = "rebeccapurple"


# DATAFRAME VALUES


def display_distinct_values(df: Union[pd.DataFrame, pd.Series]):
    """Display a dataframe containing the number of distinct values
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
    display(df_out)


def display_value_counts_ptc(
    df: pd.DataFrame, n_rows: Optional[int] = None,
):
    """Display a dataframe containing the value counts and their
    respective pct for a column or a list of columns. The max
    number of values to display (ordered desc by counts) can be
    defined by the optional n_rows parameter.
    """
    if isinstance(df, pd.core.series.Series):
        df = pd.DataFrame(df)

    for col in df.columns:
        counts = df[col].value_counts()
        pct = df[col].value_counts() / len(df)
        df_out = pd.concat([counts, pct], axis=1, keys=["counts", "pct"])
        caption_str = f"{col}"

        if n_rows is not None:
            df_out = df_out.iloc[:n_rows, :]
            caption_str = f"{col}, top {n_rows}"
        display(
            df_out.style.format(
                {"counts": "{:,.0f}", "pct": "{:.1%}"}
            ).set_caption(caption_str)
        )


def display_tail_transposed(
    df: pd.DataFrame, max_row: int = 100, random_state: Optional[int] = None
):
    """Display transposed tail of the dataframe with the orig
    columns as rows and values for 5 sample instances as columns.
    The max number of rows can be adapted (defaults to 100).
    A random state seed can be specified (defaults to None).
    """
    df = df.sample(frac=1, random_state=random_state)
    with pd.option_context("display.max_rows", max_row):
        print(df.shape)
        display(df.tail(5).transpose())


# DISTRIBUTIONS


def plot_numerical_hist(
    df: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None, **kwargs
):
    """Display a histogram for every numerical column in the passed
    dataframe. A good figsize is interfered if not explicitely passed.
    Additional keyword arguments will be passed to the actual seaborn
    plot function.
    """
    df_num = df.select_dtypes(include=np.number)
    defaults = {"bins": 50, "color": COLOR, "kde": True}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = df_num.shape[1] / 4 * 4  # ;-)
        figsize = (14, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(df_num.columns, 1):
        plt.subplot(np.ceil(df_num.shape[1] / 4), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.histplot(df_num[col].dropna(), **kwargs)


def plot_numerical_box(
    df: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None, **kwargs
):
    """Display a boxplot for every numerical column in the passed
    dataframe. A good figsize is interfered if not explicitely passed.
    Additional keyword arguments will be passed to the actual seaborn
    plot function.
    """
    df_num = df.select_dtypes(include=np.number)
    defaults = {"color": COLOR}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = df_num.shape[1] / 4 * 4  # ;-)
        figsize = (14, height)

    plt.figure(figsize=figsize)
    for pos, col in enumerate(df_num.columns, 1):
        plt.subplot(np.ceil(df_num.shape[1] / 4), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.boxplot(y=col, data=df_num, **kwargs)


def plot_categorical_pies(
    df: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None, **kwargs
):
    """Display a pieplot for every categorical column in the passed
    dataframe that has no more than 30 distinct values. A good 
    figsize is interfered if not explicitely passed. Additional
    keyword arguments will be passed to the actual pandas plot 
    function.
    """
    df_cat = df.select_dtypes(include="category")
    defaults = {"cmap": "viridis"}
    kwargs = {**defaults, **kwargs}
    if figsize is None:
        height = df_cat.shape[1] / 3 * 4
        figsize = (14, height)
    cols_with_many_distinct_values = []
    pos = 0

    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        if df_cat[col].nunique() <= 30:
            pos += 1
            plt.subplot(np.ceil(df_cat.shape[1] / 3), 3, pos)
            plt.tight_layout(w_pad=1)
            df[col].value_counts().plot(kind="pie", **kwargs)
        else:
            cols_with_many_distinct_values.append(col)

    if len(cols_with_many_distinct_values) > 0:
        display(f"Not plotted: {cols_with_many_distinct_values}")


# CORRELATIONS


def plot_correlations_full_heatmap(
    df: pd.DataFrame, figsize: Tuple[int, int] = (14, 10), **kwargs
):
    """Display heatmap to show correlations between all the numerical
    columns in the Dataframe. Optional figsize and additional keyword
    arguments will be passed to the actual seaborn plot function.
    """
    df_num = df.select_dtypes(include=np.number)
    defaults = {
        "cmap": "magma",
        "linecolor": "white",
        "linewidth": 1,
        "annot": True,
    }
    kwargs = {**defaults, **kwargs}

    plt.figure(figsize=figsize)
    sns.heatmap(df_num.corr(), **kwargs)


# TODO - continue here ...
def plot_correlations_bar_num_target(
    df, target_col, figsize=(14, 6), color="rebeccapurple"
):
    """Display sorted barchart to show correlations between
    all numerical features and a numerical target variable.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(14, 6))
    - color: str (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=figsize)
    corr_target_series = df_num.corr()[target].sort_values(ascending=False)
    corr_target_series.drop(target).plot.bar(color=COLOR)
    plt.show()


def plot_corr_regression_num_target(
    df, target, figsize=(14, 16), color=("rebeccapurple", "yellow")
):
    """Display regplots to visualize correlations between the numerical
    features and numerical target variable.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(14, 16))
    - color: list of two strings (default=['rebeccapurple', 'yellow'])

    Returns:
    --------
    - None. Displays plots.
    """

    df_num = df.select_dtypes(include=["float64", "int64"]).drop(target, axis=1)
    pos = 0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos += 1
        plt.subplot(np.ceil(df_num.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.regplot(
            x=col,
            y=df[target],
            data=df_num,
            color=COLOR[0],
            line_kws={"color": color[1]},
        )


def plot_corr_box_num_target(df, target, figsize=(14, 16), color=COLOR):
    """Display lineplots to show correlation details
    between all numerical features and target classes.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(14, 16))
    - color: str (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include=["float64", "int64"])
    pos = 0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos += 1
        plt.subplot(np.ceil(df_num.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.boxplot(
            x=df[target].astype("category"), y=col, data=df_num, color=COLOR
        )


def plot_corr_line_num_target(
    df, target, figsize=(14, 16), ylim=(0, 1), color=COLOR
):
    """Display lineplots to show correlation details
    between all numerical features and target classes.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(14, 16))
    - ylim: list of two int, limits for y-axis (default=[0, 1])
    - color: str (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include=["float64", "int64"])
    pos = 0
    plt.figure(figsize=figsize)
    for col in tqdm(df_num.columns):
        pos += 1
        plt.subplot(np.ceil(df_num.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        plt.ylim(ylim)
        plt.xlabel(df[col].name)
        sns.lineplot(x=col, y=target, data=df_num, color=COLOR)


def plot_corr_strip_cat_target(df, target, figsize=(14, 32), palette="rocket"):
    """Display stripplots to show correlations between
    the categorical features and numerical target variable.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(14, 16))
    - palette: str (default='rocket')

    Returns:
    --------
    - None. Displays plot.
    """

    df_cat = df.select_dtypes(include=["category"])
    pos = 0
    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        df_plot = df[[col, target]]
        pos += 1
        plt.subplot(np.ceil(df_cat.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.stripplot(x=col, y=target, data=df_plot, palette=palette)


def plot_corr_point_cat_target(
    df, target, figsize=(14, 16), ylim=(0, 1), color=COLOR, cmap="viridis"
):
    """Display pointplots (and corresponding piecharts)
    to show correlations between all categorical columns and target classes.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(14, 16))
    - ylim: list of two int, limits for y-axis (default=[0, 1])
    - color: str (default='rebeccapurple')
    - cmap: default is 'viridis'

    Returns:
    --------
    - None. Displays plot.
    """

    df_cat = df.select_dtypes(include=["category"])
    pos = 0
    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        df_plot = (
            df[[col, target]]
            .groupby(col, as_index=False)
            .mean()
            .sort_values(by=target, ascending=False)
        )
        pos += 1
        plt.subplot(df_cat.shape[1], 2, pos)
        plt.tight_layout(w_pad=1)
        plt.ylim(ylim)
        sns.pointplot(x=col, y=target, data=df_plot, color=COLOR)
        if df[col].nunique() <= 30:
            pos += 1
            plt.subplot(df_cat.shape[1], 2, pos)
            df[col].value_counts().plot(kind="pie", cmap=cmap)
        else:
            pos += 1


# corr PairPlot numCols to numTarget - see here: https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python
# for i in range(0, len(df_num.columns), 5):
#     sns.pairplot(data=df_num,
#                 x_vars=df_num.columns[i:i+5],
#                 y_vars=['SalePrice'])
