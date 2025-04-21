"""
This script defines functions for plotting temporal spike distribution of the dataset data.
"""

from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_temporal_spike_distribution_for_dataset_data_full(
    df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    """
    Plots temporal distribution of the spike counts overlaid for all time bin size variants on full dataset.

    :param df: Data to plot from `temporal_evolution_processor.py`.
    :param is_test: Whether we are plotting the test dataset.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """

    g = sns.FacetGrid(
        df,
        col="layer",
        col_wrap=2,  # one column = one plot per row
        height=4,  # height of each subplot in inches
        aspect=1.5,  # width = height × aspect → wider plots
        sharey=True,
    )
    g.map_dataframe(
        sns.lineplot, x="time", y="density", hue="time_step", palette="viridis"
    )
    g.add_legend(title="Time step", prop={"size": 12})  # labels
    plt.setp(g._legend.get_title(), fontsize=14)
    g._legend.set_frame_on(True)  # make sure the box is shown
    g.set_axis_labels("Time (ms)", "Normalized spike density", size=16)
    g.set_titles(col_template="{col_name}", size=18)

    dataset_label = "Test" if is_test else "Train"

    g.figure.suptitle(
        f"Temporal Spike Distributions Across Layers - {dataset_label} Dataset",
        fontsize=24,
    )
    g.figure.tight_layout(rect=[0, 0, 0.92, 0.98])

    # Add grid to each subplot and make the axis ticks larger.
    for ax in g.axes.flatten():
        ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)

    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_temporal_spiking_correlation_heatmap_for_each_bin_size_full(
    train_test_tuple: Tuple[pd.DataFrame, pd.DataFrame],
    is_test: bool = False,
    save_fig: str = "",
):
    """
    Plots the correlation matrix of the spike counts across all time steps
    across all layers for all time bin sizes for both train and test datasets on full dataset.

    :param train_test_tuple: Tuple of correlation matrices for train and test datasets.
    :param is_test: Just placeholder to work properly with plotting function.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True, gridspec_kw={"wspace": 0.15}
    )

    train_corr, test_corr = train_test_tuple

    # Train heatmap
    sns.heatmap(
        train_corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot_kws={"size": 12},
        square=True,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Train Set Correlation", fontsize=18)
    axes[0].set_xlabel("Time step (ms)", fontsize=16)
    axes[0].set_ylabel("Time step (ms)", fontsize=16)

    # Test heatmap
    sns.heatmap(
        test_corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot_kws={"size": 12},
        square=True,
        ax=axes[1],
        cbar_kws={"label": "Pearson correlation"},
    )
    colorbar = axes[1].collections[0].colorbar
    colorbar.set_label("Pearson correlation", fontsize=16)
    colorbar.ax.tick_params(labelsize=12)  # To increase the tick label font size
    axes[1].set_title("Test Set Correlation", fontsize=18)
    axes[1].set_xlabel("Time step (ms)", fontsize=16)
    axes[1].set_ylabel("")

    # Shared settings
    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)

    plt.suptitle(
        "Temporal Correlation Between Time Binnings - Train vs Test", fontsize=22
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if save_fig:
        fig.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_temporal_spike_distribution_comparison_full_subset(
    combined_df: pd.DataFrame,
    is_test: bool = False,
    save_fig: str = "",
):
    """
    Plots the comparison of the subset and full dataset temporal spike distributions
    across all layers. Plots Error bars for mean ± SD of the subset dataset and overlays
    it with the full dataset temporal dynamics.

    :param combined_df: Prepared DataFrame containing the full and subset data.
    :param is_test: Just placeholder to work properly with plotting function.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    # Create FacetGrid
    g = sns.FacetGrid(
        combined_df,
        col="layer",
        col_wrap=2,
        height=4,
        aspect=1.5,
        sharex=True,
        sharey=True,
    )

    # Plot subset: mean ± SD
    for ax, (layer_name, layer_df) in zip(
        g.axes.flatten(), combined_df.groupby("layer")
    ):
        # Split full vs subset
        full_df = layer_df[layer_df["model_type"] == "Full model"]
        subset_df = layer_df[layer_df["model_type"] == "Subset"]

        # Plot full model (solid black line)
        sns.lineplot(
            data=full_df,
            x="time",
            y="density",
            estimator="mean",
            errorbar="sd",
            color="black",
            linewidth=2,
            linestyle="--",
            ax=ax,
            label="Full model",
        )

        # Plot subset model (dashed, semi-transparent blue)
        sns.lineplot(
            data=subset_df,
            x="time",
            y="density",
            estimator="mean",
            errorbar="sd",
            color="dodgerblue",
            linewidth=1.5,
            linestyle="-",
            alpha=0.7,
            ax=ax,
            label="Subset",
        )

        ax.set_title(layer_name, fontsize=18)
        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Spike Density", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend_.remove()  # Hide subplot legends

    # Global legend
    custom_lines = [
        plt.Line2D(
            [0],
            [0],
            color="dodgerblue",
            lw=1.5,
            linestyle="--",
            alpha=0.7,
            label="Subset mean ± SD",
        ),
        plt.Line2D([0], [0], color="black", lw=1.5, label="Full model"),
    ]
    g.figure.legend(
        handles=custom_lines,
        loc="upper right",
        title="Model",
        title_fontsize=16,
        fontsize=14,
        frameon=True,
    )

    dataset_label = "Test" if is_test else "Train"
    # Supertitle
    plt.subplots_adjust(top=0.88, right=0.95)
    g.figure.suptitle(
        f"Temporal Dynamics of All Layers\n in {dataset_label} Dataset", fontsize=24
    )

    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
