"""
This script defines plotting functions for the model variant comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_types_pearson_normalized_box_plot(
    df: pd.DataFrame,
    save_fig: str = "",
):
    """
    Plots box plot of pearson and normalized cross correlations of each model
    type.

    :param df: Data to plot from.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    # Plot
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(
        data=df,
        x="model_variant",
        y="Correlation Value",
        hue="Correlation Type",
        width=0.6,  # Narrower boxes = more space between groups
        dodge=True,
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    plt.title("Comparison of Performance of Different Model Variants", size=20)
    plt.ylabel("Metric Value", size=18)
    plt.xlabel("Model Variant", size=18)
    plt.xticks(rotation=10, size=13)
    plt.legend(title="Metric", fontsize=12, title_fontsize=13)
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    else:
        plt.show()


def plot_model_variant_p_value_heatmap_cc_norm(df: pd.DataFrame, save_fig: str = ""):
    """
    Plots heatmap of p-values of model comparison test on p-value.

    :param df: P-value data to plot.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    # Custom sort order
    custom_order = [
        "simple (tanh)",
        "simple (leakytanh)",
        "dnn joint",
        "dnn separate",
        "rnn (5 steps)",
        "rnn (10 steps)",
        "syn adapt lgn (5 steps)",
    ]

    # Apply sort
    df = df.loc[custom_order, custom_order]

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(
        df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="coolwarm_r",
        cbar_kws={"label": "One-sided p-value (A > B)"},
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 12},  # Cell annotation size
    )

    # Set font sizes
    heatmap.set_title(
        "Pairwise One-Sided Paired Mann-Whitney U Test (Normalized CC: A > B)",
        fontsize=20,
    )
    heatmap.set_xlabel("Model B", fontsize=18)
    heatmap.set_ylabel("Model A", fontsize=18)

    # Set tick label sizes
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(), rotation=20, ha="right", fontsize=12
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30, fontsize=12)

    # Set colorbar label font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("One-sided p-value (A > B)", fontsize=16)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    else:
        plt.show()
