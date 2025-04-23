"""
This script defines plotting function for plotting the temporal synchrony across models and layers.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import nn_model.globals


def plot_multiple_models_target_prediction(
    df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    # TODO: implement
    """
    Plot all model synchrony in one large grid.
    """

    g = sns.relplot(
        data=df,
        x="time",
        y="synchrony",
        kind="line",
        hue="variant_type",
        col="model_variant",
        row="layer_name",
        facet_kws={"sharey": True, "sharex": True},
        height=3.5,
        aspect=1.5,
    )

    # Add vertical line to each subplot at stimulus change time
    def add_vline(data, color, **kwargs):
        plt.axvline(
            nn_model.globals.IMAGE_DURATION // 20 - 1,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    g.map_dataframe(add_vline)

    g.set_titles(row_template="Layer: {row_name}", col_template="Model: {col_name}")
    g.set_axis_labels("Time Step", "Fraction of Neurons Spiking")
    g._legend.set_title("Type")
    plt.tight_layout()

    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_single_model_synchrony_curves_across_layers(
    df: pd.DataFrame, model_variant: str = "", save_fig: str = ""
):
    """
    Plot single model synchrony curves for each layer.

    :param df: Dataset to plot.
    :param model_variant: Model variant label to display in the plot.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    g = sns.relplot(
        data=df,
        x="time",
        y="synchrony",
        kind="line",
        hue="variant_type",
        col="layer_name",
        col_wrap=2,
        facet_kws={"sharey": True, "sharex": True},
        height=4,
        aspect=1.5,
    )

    # Add vertical line for stimulus transition
    def add_vline(data, color, **kwargs):
        plt.axvline(
            nn_model.globals.IMAGE_DURATION // 20 - 1,
            color="gray",
            linestyle="--",
            linewidth=1.5,
        )

    g.map_dataframe(add_vline)

    # Add grid lines
    g.map(
        lambda *args, **kwargs: plt.grid(
            True, which="both", linestyle=":", linewidth=0.5, alpha=0.7
        )
    )

    # Adjust titles and axis labels
    g.set_titles(col_template="{col_name}", size=18)
    g.set_axis_labels("Time Step (20 ms)", "Fraction of Neurons Spiking", size=16)

    # Adjust legend position and styling
    g._legend.set_title("Variant Type")
    g._legend.set_bbox_to_anchor((0.98, 0.95))  # move to center right
    g._legend.set_frame_on(True)
    for text in g._legend.texts:
        text.set_fontsize(13)
    g._legend.set_title("Variant Type", prop={"size": 14})

    # Adjust tick sizes
    for ax in g.axes.flat:
        ax.tick_params(axis="both", labelsize=12)

    # Add supertitle
    g.figure.suptitle(
        f"Synchrony Dynamics Across Layers\nModel: {model_variant}", fontsize=22
    )
    g.figure.tight_layout(rect=[0, 0, 0.98, 0.96])  # Leave space at the top

    # Create a custom dashed line legend entry
    stimulus_line = mlines.Line2D(
        [],
        [],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Stimulus change",
    )
    separator = mlines.Line2D(
        [], [], color="black", linestyle="-", linewidth=0, label="────────────"
    )
    # Get current legend handles and labels
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    # Add the new dashed line to the legend
    g._legend.remove()  # Remove default legend
    g.figure.legend(
        handles + [separator, stimulus_line],
        ["Predictions", "Target", "────────────", "Stimulus change"],
        loc="upper right",
        # bbox_to_anchor=(0.98, 0.95),
        fontsize=13,
        title="Variant Type + Marker",
        title_fontsize=14,
        frameon=True,
    )

    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
