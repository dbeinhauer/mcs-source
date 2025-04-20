"""
This module encapsulates the logic for synchrony comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_synchrony_boxplot_across_layers(
    df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    """
    Plots box plots of the synchrony across all layers for each time bin size variant.

    :param df: Data to plot from.
    :param is_test: Whether we are plotting the test dataset.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    g = sns.FacetGrid(df, col="layer", col_wrap=2, height=4, sharey=True)
    g.map_dataframe(sns.boxplot, x="time_step", y="synchrony")

    # Set axis labels and titles
    g.set_axis_labels("Time bin (ms)", "Synchrony", size=14)
    g.set_titles(col_template="{col_name}", size=16)

    # Set tick label font size for all subplots
    for ax in g.axes.flatten():
        ax.tick_params(axis="both", labelsize=11)

    # Add a supertitle
    dataset_label = "Test" if is_test else "Train"

    g.figure.suptitle(
        f"Synchrony Distributions by Time Bin and Layer - {dataset_label} Dataset",
        fontsize=18,
    )

    # Adjust layout to make space for the title
    g.figure.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space at the top

    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
