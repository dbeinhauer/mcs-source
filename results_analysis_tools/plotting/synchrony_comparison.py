"""
This module encapsulates the logic for synchrony comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_synchrony_boxplot_across_layers_full(
    df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    """
    Plots box plots of the synchrony across all layers for each time bin size variant on full dataset.

    :param df: Data to plot from.
    :param is_test: Whether we are plotting the test dataset.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    g = sns.FacetGrid(df, col="layer", col_wrap=2, height=4, sharey=True)
    g.map_dataframe(sns.boxplot, x="time_step", y="synchrony")

    # Set axis labels and titles
    g.set_axis_labels("Time bin (ms)", "Temporal Spiking Activity", size=14)
    g.set_titles(col_template="{col_name}", size=16)

    # Set tick label font size for all subplots
    for ax in g.axes.flatten():
        ax.tick_params(axis="both", labelsize=11)

    # Add a supertitle
    dataset_label = "Test" if is_test else "Train"

    g.figure.suptitle(
        f"Temporal Spiking Dynamics Distributions by Time Bin and Layer - {dataset_label} Dataset",
        fontsize=18,
    )

    # Adjust layout to make space for the title
    g.figure.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space at the top

    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_synchrony_boxplot_jitter_subset_full_comparison(
    df: pd.DataFrame,
    is_test: bool = False,
    save_fig: str = "",
    plot_subset: float = 0.01,
):
    """
     Plots box plots and jittered points of the synchrony across all layers
     for full dataset and all subset models.

    :param df: Data to plot from.
    :param is_test: Whether we are plotting the test dataset.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    :param plot_subset: Fraction of data to plot for jittered points.
    """

    # === Style Settings ===
    labelsize = 16
    titlesize = 18
    ticksize = 12
    suptitlesize = 22
    legend_titlesize = 14

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # ---- Boxplot (Left) ----
    sns.boxplot(
        data=df,
        x="layer",
        y="synchrony",
        hue="model_type",
        dodge=True,
        ax=axes[0],
        width=0.6,
        linewidth=1.5,
        fliersize=2,
    )
    axes[0].set_title("Boxplot of Temporal Spiking Dynamics", fontsize=titlesize)
    axes[0].set_xlabel("Layer", fontsize=labelsize)
    axes[0].set_ylabel("Mean Temporal Spiking Activity", fontsize=labelsize)
    axes[0].tick_params(axis="x", labelsize=ticksize, rotation=15)
    axes[0].tick_params(axis="y", labelsize=ticksize)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        ["All Model Subsets", "Full model"],
        title="Model",
        fontsize=ticksize,
        title_fontsize=legend_titlesize,
        loc="upper right",
    )

    # ---- Stripplot (Right) ----
    strip_df = df.groupby(["layer", "model_type"]).sample(
        frac=plot_subset, random_state=42
    )
    sns.stripplot(
        data=strip_df,
        x="layer",
        y="synchrony",
        hue="model_type",
        dodge=True,
        jitter=0.1,
        alpha=0.3,
        size=2.5,
        marker="o",
        ax=axes[1],
        rasterized=True,
    )
    axes[1].set_title("Jittered Temporal Spiking Dynamics Points", fontsize=titlesize)
    axes[1].set_xlabel("Layer", fontsize=labelsize)
    axes[1].set_ylabel("")  # share y-axis with boxplot
    axes[1].tick_params(axis="x", labelsize=ticksize, rotation=15)
    axes[1].tick_params(axis="y", labelsize=ticksize)

    # Legend
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles,
        ["All Model Subsets", "Full model"],
        title="Model",
        fontsize=ticksize,
        title_fontsize=legend_titlesize,
        loc="upper right",
    )

    dataset_label = "Test" if is_test else "Train"

    # Supertitle
    fig.suptitle(
        f"Temporal Spiking Dynamics Comparison: Full vs All Subset Models for {dataset_label} Dataset",
        fontsize=suptitlesize,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_fig:
        fig.figure.savefig(save_fig, format="pdf", bbox_inches="tight", dpi=150)

    else:
        plt.show()
