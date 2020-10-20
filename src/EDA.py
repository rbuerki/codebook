""" LIST OF FUNCTIONS
    -----------------

Dataframe values:
- `display_distinct_values`: Display a dataframe containing the number
   of distinct values for each column of the passed dataframe.
- `display_value_counts_ptc`: Display a dataframe containing the value
   counts and their respective pct for a column or a list of columns.
- `display_tail_transposed`: Display transposed tail of the dataframe
   with all the orig cols as rows and values for 5 instances as columns.

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

from typing import Iterable, Optional, Tuple, Union

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
    df: pd.DataFrame,
    cols: Union[str, Iterable[str]],
    n_rows: Optional[int] = None,
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


def plot_num_hist(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 14),
    *kwargs,  # TODO kwargs ...
):
    """Display histograms for all numerical columns in DataFrame."""
    df_num = df.select_dtypes(include=["float64", "int64"])
    pos = 0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos += 1
        plt.subplot(np.ceil(df_num.shape[1] / 4), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.distplot(
            df_num[col].dropna(), *kwargs, bins=50, color=COLOR, kde=True
        )


def plot_num_box(df, figsize=(14, 16), color=COLOR):
    """Display boxplots for all numerical columns in DataFrame.

    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(14, 16))
    - color: string (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include=["float64", "int64"])
    pos = 0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos += 1
        plt.subplot(np.ceil(df_num.shape[1] / 4), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.boxplot(y=col, data=df_num, color=COLOR)


def plot_cat_pies(df, figsize=(14, 16), cmap="viridis"):
    """Display pieplots for all categorical columns in DataFrame with up to
    30 values.

    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(14, 16))
    - cmap: default is 'viridis'

    Returns:
    --------
    - None. Displays plot.
    """

    df_cat = df.select_dtypes(include="category")
    pos = 0
    catWithManyValues = []
    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        if df[col].nunique() <= 30:
            pos += 1
            plt.subplot(np.ceil(df_cat.shape[1] / 4), 4, pos)
            plt.tight_layout(w_pad=1)
            df[col].value_counts().plot(kind="pie", cmap=cmap)
        else:
            catWithManyValues.append(df[col].name)
    if len(catWithManyValues) > 0:
        display("Not plotted: " + str(catWithManyValues))


# CORRELATIONS


def plot_corr_map_num_all(df, figsize=(14, 16), cmap="magma"):
    """Display heatmap to show correlations between all numerical
    columns in the Dataframe.

    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(14, 16))
    - cmap: str, (default='magma')

    Returns:
    --------
    - None. Displays plot.
    """

    plt.figure(figsize=figsize)
    df_num = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(
        df_num.corr(), cmap=cmap, linecolor="white", linewidth=1, annot=True
    )


def plot_corr_bar_num_target(
    df, target, figsize=(14, 6), color="rebeccapurple"
):
    """Display sorted barchart to show correlations between
    all numerical features and numerical target variable.

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
