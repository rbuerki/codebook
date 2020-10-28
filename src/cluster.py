"""
LIST OF FUNCTIONS
-----------------
# TODO

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_pca_results(df, pca, figsize=(16, 8)):
    """Create a DataFrame of the PCA results that includes dimension feature 
    weights and explained variance and visusalize the results with a bar chart

    Arguments:
    ----------
    - df: DataFrame, containing the features
    - pca: fitted sklearn PCA class object

    Returns:
    --------
    - None, plots PCA results (bar chart)
    """
    pca.fit(df)

    # Index dimensions, pca components, explained variance
    dimensions = [
        "Dimension {}".format(i) for i in range(1, len(pca.components_) + 1)
    ]
    components = pd.DataFrame(
        np.round(pca.components_, 4), columns=list(df.keys())
    )
    components.index = dimensions

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(
        np.round(ratios, 4), columns=["Explained Variance"]
    )
    variance_ratios.index = dimensions

    # Create bar plot visualization: plot feature weights as function of components
    fig, ax = plt.subplots(figsize=figsize)

    components.plot(ax=ax, kind="bar")
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    plt.legend(loc="lower right")

    # Display explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(
            i - 0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n%.4f" % (ev)
        )


#     # Return concatenated DataFrame
#     pca_results = pd.concat([variance_ratios, components], axis = 1)
#     return pca_results


def create_biplot(orig_df, reduced_df, pca, facecolors="orange"):
    """Create a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.

    Arguments:
    ----------
    - orig_df: DataFrame, before pca-transformation with column names
    - reduced_df: DataFrame, after pca-transformation (the first 2D are plotted)
    - PCA: sklearn PCA object that contains the components_ attribute

    Returns:
    --------
        - A matplotlib AxesSubplot object (for any additional customization)

    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    """

    fig, ax = plt.subplots(figsize=(14, 8))

    # Scatterplot of the reduced data
    ax.scatter(
        x=reduced_df.loc[:, "Dimension 1"],
        y=reduced_df.loc[:, "Dimension 2"],
        facecolors=facecolors,
        edgecolors="white",
        s=50,
        alpha=0.5,
    )

    # Add scaling factors to make the arrows easier to see
    arrow_size, text_pos = 1.5, 1.5

    # Add projections of the original features
    feature_vectors = pca.components_.T
    for i, v in enumerate(feature_vectors):
        ax.arrow(
            0,
            0,
            arrow_size * v[0],
            arrow_size * v[1],
            head_width=0.1,
            head_length=0.1,
            linewidth=2,
            color="red",
        )
        ax.text(
            v[0] * text_pos,
            v[1] * text_pos,
            orig_df.columns[i],
            color="black",
            ha="center",
            va="bottom",
            fontsize=16,
        )

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections", fontsize=16)

    return ax


def evaluate_kmeans(df, cluster_range, pca_dim_range=[0]):
    """Plot silhouette scores and WCSS-Inertia ('Elbow') of k-means clustering 
    for the indicated numbers of clusters. It is possible to evaluate different 
    PCA dimensionality reductions. (Default is None.)
    
    Arguments:
    ----------
    - df: DataFrame, containing the features
    - pca_dim_range: iterable, containing a list of integers indicating the
        pca_dimensions to be applied. '0' means no PCA-reduction, (Default=[0])
    - cluster_range: iterable, containing a list of integers with the desired 
        number of clusters
    
    Returns:
    --------
    - None, plots the Silhouette scores and WCSS-Inertia 'Ellbow' for each 
        number of clusters for the different PCA-dimensions.
    """

    df_sil_results_dict = {}
    df_wcss_results_dict = {}

    for n_dim in tqdm(pca_dim_range):
        if n_dim == 0:
            df_red = df
            label = "no PCA"
        else:
            pca = PCA(n_components=n_dim)
            df_red = pca.fit_transform(df)
            label = str(n_dim) + "-dim"

        sil_scores_list = []
        wcss_list = []
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(df_red)
            # Predict the cluster for each data point
            preds = kmeans.predict(df_red)
            # Calculate mean silhouette coef for the number of clusters chosen
            score = silhouette_score(df_red, preds)
            # Append to list
            sil_scores_list.append(score)

            if n_dim == 0:
                wcss_list = list(np.zeros(len(cluster_range)))
            if n_dim > 0:
                wcss = kmeans.inertia_
                wcss_list.append(wcss)

        df_sil_results_dict[label] = sil_scores_list
        df_wcss_results_dict[label] = wcss_list

    sil_df = pd.DataFrame(df_sil_results_dict)
    sil_df = sil_df.reindex(sorted(sil_df.columns), axis=1)
    sil_df.index = cluster_range

    wcss_df = pd.DataFrame(df_wcss_results_dict)
    wcss_df = wcss_df.reindex(sorted(wcss_df.columns), axis=1)
    wcss_df.index = cluster_range

    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")

    plt.subplot(2, 1, 1)
    sns.lineplot(data=sil_df, palette="viridis")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.xticks(range(cluster_range[0], cluster_range[-1] + 1, 1))

    plt.subplot(2, 1, 2)
    sns.lineplot(data=wcss_df, markers=True, palette="viridis")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.xticks(range(cluster_range[0], cluster_range[-1] + 1, 1))


def visualize_clusters(
    df, x_col, y_col, cluster_range, figsize=(16, 16), palette="rocket"
):
    """Visualize clusters on scatterplots for k-means clustering with sklearn.
    
    Arguments:
    ----------
    - df: DataFrame, containing the presumed clusters
    - x_col: string, column label for data to plot on x-axis
    - y_col: string, column label for data to plot on y-axis
    - cluster_range: list of integers, desired number of clusters
    - figsize: tuple (default=(16, 16))
    - palette: string (default='rocket')
    
    Returns:
    --------
    - None, outputs a series of scatterplots for the desired cluster range.
    """

    df = df.copy()
    position = 0
    plt.figure(figsize=figsize)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df)

        # Assign the clusters to df
        df["K_Cluster"] = kmeans.labels_
        print("\nn_clusters:", n_clusters)
        print(df["K_Cluster"].value_counts() / len(df))

        # Plot, note: explicitely set edgecolor=None for better visibility
        position += 1
        plt.subplot(np.rint((len(cluster_range) / 2) + 0.1), 2, position)
        sns.scatterplot(
            x=x_col,
            y=y_col,
            hue="K_Cluster",
            data=df,
            edgecolor=None,
            palette=palette,
            legend="full",
        )


def display_cluster_median_values(df_fit, df_orig, cluster_range):
    """Display a series of DataFrames for desired cluster range with median 
    feature values and size for each cluster.
    
    Arguments:
    ----------
    - df_fit: dataframe, preprocessed data containing the presumed clusters
    - df_orig: dataframe, original data of which median values will be displayed.
        Index has to be identical to df_fit
    - cluster_range: list of integers, desired number of clusters
    
    Returns:
    --------
    - None, displays a series of DataFrames with median values and size for 
        different clusters.
    """

    df_orig = df_orig.copy()
    assert list(df_orig.index) == list(
        df_fit.index
    ), "indices of df_fit, df_orig differ"

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(df_fit)

        # Assign the clusters to df and group
        df_orig["K_Cluster"] = kmeans.labels_
        df_display = df_orig.groupby(["K_Cluster"]).agg([np.median])
        df_size = pd.DataFrame(
            df_orig.groupby(["K_Cluster"]).agg("count").iloc[:, 0]
        )
        df_size.columns = ["k_size"]
        df_display = pd.concat([df_size, df_display], axis=1, sort=True)
        display(df_display)
