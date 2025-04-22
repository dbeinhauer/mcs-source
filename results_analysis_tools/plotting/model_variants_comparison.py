"""
This script defines plotting functions for the model variant comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_types_pearson_normalized_box_plot(
    df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    """
    Plots box plot of pearson and normalized cross correlations of each model
    type.

    :param df: Data to plot from.
    :param is_test: Placeholder to match the interface.
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
