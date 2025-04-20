"""
This script defines functions for plotting temporal spike distribution of the dataset data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_temporal_spike_distribution_for_dataset_data(
    df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    """
    Plots temporal distribution of the spike counts overlaid for all time bin size variants.

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
