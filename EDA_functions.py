"""
LIST OF FUNCTIONS
-----------------

- display_tail_transposed: Display transposed tail of DataFrame with all the 
features (orig cols) as rows and values for 5 instances (orig rows) as cols.

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
- plot_corr_scatter_num_target: Display scatterplots to visualize correlations 
  between all numerical features and numerical target variable.
- plot_corr_box_num_target: Display boxplots to show correlations between all 
  numerical features and target classes.
- plot_corr_line_num_target: Display lineplots to show correlation details 
  between all numerical features and target classes.
- plot_corr_strip_cat_target: Display stripplots to show correlations between 
  the categorical features and numerical target variable.
- plot_corr_point_cat_target: Display pointplots (and corresponding piecharts) 
  to show correlations between all categorical columns and target classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
color = 'rebeccapurple'
from tqdm import tqdm


def display_tail_transposed(df, max_row=200, max_col=200):
    """Display transposed tail of DataFrame with all the features (original 
    columns) as rows and values for 5 instances (original rows) as columns.

    Arguments:
    ----------
    - df: DataFrame
    - max_row: int, max number of rows to display (default=200)
    - max_col: int, max number of columns to display (default=200)

    Returns:
    --------
    - None. Prints shape and displays transposed tail of DataFrame.

    """

    with pd.option_context("display.max_rows", max_row): 
        with pd.option_context("display.max_columns", max_col): 
            print(df.shape)
            display(df.tail().transpose())


# DISTRIBUTIONS

def plot_num_hist(df, figsize=(16, 16), bins=50, color=color, kde=True):
    """Display histograms for all numerical columns in DataFrame.
    
    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(16, 16))
    - bins: int, number of bins (default=50)
    - color: string (default='rebeccapurple')
    - kde: bool, plot of kde-line (default=True)

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include = ['float64', 'int64'])
    pos=0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos +=1
        plt.subplot(np.ceil(df_num.shape[1] / 4), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.distplot(df_num[col], bins= bins, color=color, kde=kde);


def plot_num_box(df, figsize=(16, 16), color=color):
    """Display boxplots for all numerical columns in DataFrame.

    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(16, 16))
    - color: string (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include = ['float64', 'int64'])
    pos=0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos +=1
        plt.subplot(np.ceil(df_num.shape[1] / 4), 4, pos)
        plt.tight_layout(w_pad=1)
        sns.boxplot(y=col, data=df_num, color=color);


def plot_cat_pies(df, figsize=(16, 16), cmap='viridis'):
    """Display pieplots for all categorical columns in DataFrame with up to 
    30 values.

    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(16, 16))
    - cmap: default is 'viridis'

    Returns:
    --------
    - None. Displays plot.
    """

    df_cat = df.select_dtypes(include = 'category')
    pos=0
    catWithManyValues = []
    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        if df[col].nunique() <= 30:
            pos +=1
            plt.subplot(np.ceil(df_cat.shape[1] / 4), 4, pos)
            plt.tight_layout(w_pad=1)            
            df[col].value_counts().plot(kind='pie', cmap = cmap)
        else: catWithManyValues.append(df[col].name)
    if len(catWithManyValues) > 0:
        display("Not plotted: " + str(catWithManyValues));



# CORRELATIONS
    
def plot_corr_map_num_all(df, figsize=(16, 16), cmap='magma'):
    """Display heatmap to show correlations between all numerical 
    columns in the Dataframe. 

    Arguments:
    ----------
    - df: DataFrame
    - figsize: tuple (default=(16, 16))
    - cmap: str, (default='magma')

    Returns:
    --------
    - None. Displays plot.
    """

    plt.figure(figsize=figsize)
    df_num = df.select_dtypes(include = ['int64', 'float64'])
    sns.heatmap(df_num.corr(), cmap=cmap, linecolor='white', 
                linewidth=1, annot=True);


def plot_corr_bar_num_target(df, target, figsize=(16, 6), 
                             color='rebeccapurple'):
    """Display sorted barchart to show correlations between 
    all numerical features and numerical target variable.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(16, 6))
    - color: str (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """
    
    df_num = df.select_dtypes(include = ['int64', 'float64'])

    plt.figure(figsize=figsize)
    corr_target_series = df_num.corr()[target].sort_values(ascending=False)
    corr_target_series.drop(target).plot.bar(color=color)
    plt.show();


def plot_corr_scatter_num_target(df, target, hue=False, figsize=(16, 16), 
                                 palette='rocket'):
    """Display scatterplots to visualize correlations 
    between all numerical features and numerical target variable.
    
    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - hue: str, colum label of a categorical variable (default=False)
    - figsize: tuple (default=(16, 16))
    - palette: str (default='rocket')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include = ['float64', 'int64']
                              ).drop(target, axis=1)
    pos=0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos +=1
        plt.subplot(np.ceil(df_num.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.scatterplot(x=col, y=df[target], hue=df[hue], 
                        data=df_num, palette=palette);


def plot_corr_box_num_target(df, target, figsize=(16, 16), color=color):
    """Display lineplots to show correlation details 
    between all numerical features and target classes.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(16, 16))
    - color: str (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include = ['float64', 'int64'])
    pos=0
    plt.figure(figsize=figsize)
    for col in df_num.columns:
        pos +=1
        plt.subplot(np.ceil(df_num.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)        
        sns.boxplot(x=df[target].astype('category'), y=col, 
                    data=df_num, color=color);


def plot_corr_line_num_target(df, target, figsize=(16, 16), ylim=[0, 1], 
                              color=color):
    """Display lineplots to show correlation details 
    between all numerical features and target classes.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(16, 16))
    - ylim: list of two int, limits for y-axis (default=[0, 1])
    - color: str (default='rebeccapurple')

    Returns:
    --------
    - None. Displays plot.
    """

    df_num = df.select_dtypes(include = ['float64', 'int64'])
    pos=0
    plt.figure(figsize=figsize)
    for col in tqdm(df_num.columns):
        pos +=1
        plt.subplot(np.ceil(df_num.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        plt.ylim(ylim)
        plt.xlabel(df[col].name)
        sns.lineplot(x=col, y=target, data=df_num, color=color);


def plot_corr_strip_cat_target(df, target, figsize=(16, 32), palette='rocket'):
    """Display stripplots to show correlations between 
    the categorical features and numerical target variable.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(16, 16))
    - palette: str (default='rocket')

    Returns:
    --------
    - None. Displays plot.
    """

    df_cat = df.select_dtypes(include = ['category'])
    pos=0
    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        df_plot = df[[col, target]]
        pos +=1
        plt.subplot(np.ceil(df_cat.shape[1] / 2), 2, pos)
        plt.tight_layout(w_pad=1)
        sns.stripplot(x=col, y=target, data=df_plot, palette=palette);


def plot_corr_point_cat_target(df, target, figsize=(16, 16), ylim=[0,1], 
                               color=color, cmap='viridis'):
    """Display pointplots (and corresponding piecharts) 
    to show correlations between all categorical columns and target classes.

    Arguments:
    ----------
    - df: DataFrame
    - target: str, column label of numerical target variable
    - figsize: tuple (default=(16, 16))
    - ylim: list of two int, limits for y-axis (default=[0, 1])
    - color: str (default='rebeccapurple')
    - cmap: default is 'viridis'

    Returns:
    --------
    - None. Displays plot.
    """

    df_cat = df.select_dtypes(include = ['category'])
    pos=0
    plt.figure(figsize=figsize)
    for col in df_cat.columns:
        df_plot = df[[col, target]].groupby(col, as_index=False).mean() \
                .sort_values(by=target, ascending=False)
        pos +=1
        plt.subplot(df_cat.shape[1], 2, pos)
        plt.tight_layout(w_pad=1)
        plt.ylim(ylim)
        sns.pointplot(x=col, y=target, data=df_plot,color=color)
        if df[col].nunique() <= 30:
            pos +=1
            plt.subplot(df_cat.shape[1], 2, pos)
            df[col].value_counts().plot(kind='pie', cmap = cmap)
        else: pos +=1;



# corr PairPlot numCols to numTarget - see here: https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python
# for i in range(0, len(df_num.columns), 5):
#     sns.pairplot(data=df_num,
#                 x_vars=df_num.columns[i:i+5],
#                 y_vars=['SalePrice'])
