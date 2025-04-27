"""
This script encapsulates the plotting for train-size dependency on model performance.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_dataset_size_dependency_on_norm_cc(df: pd.DataFrame, save_fig: str = ""):
    """
    Plot dependency of the normalized CC on train dataset size.

    :param df: Dataset to plot.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="train_subset_size",
        y="CC_NORM",
        alpha=0.7,
    )
    sns.lineplot(
        data=df,
        x="train_subset_size",
        y="CC_NORM",
        hue="model_variant",
        estimator="mean",
        ci="sd",
        legend=False,
    )

    plt.xlabel("Training Subset Size", fontsize=18)
    plt.ylabel("Normalized CC", fontsize=18)
    plt.title("Performance vs Training Set Size in dnn joint Model ", fontsize=20)
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
