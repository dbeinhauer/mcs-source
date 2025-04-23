"""
This script defines plotting function for plotting the temporal synchrony across models and layers.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import nn_model.globals


def plot_multiple_models_teacher_forced(df: pd.DataFrame, save_fig: str = ""):
    """
    Plots synchrony curves for all TBPTT models including the teacher-forced predictions.

    :param df: Dataframe used for plotting.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """

    g = sns.relplot(
        data=df,
        x="time",
        y="synchrony",
        kind="line",
        hue="variant_type",
        row="layer_name",
        col="model_variant",
        facet_kws={"sharey": True, "sharex": True},
        height=4,
        aspect=1.2,
    )

    # Add vertical line to each subplot at stimulus change time
    def add_vline(data, color, **kwargs):
        plt.axvline(
            nn_model.globals.IMAGE_DURATION // 20,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    g.map_dataframe(add_vline)

    # Add grid lines
    g.map(
        lambda *args, **kwargs: plt.grid(
            True, which="both", linestyle=":", linewidth=0.5, alpha=0.7
        )
    )

    # Adjust titles and axis labels
    # g.set_titles(col_template="{col_name}", size=18)
    g.set_titles(row_template="{row_name}", col_template="{col_name}", size=20)
    g.set_axis_labels("Time Step (20 ms)", "Fraction of Neurons Spiking", size=17)

    # Adjust tick sizes
    for ax in g.axes.flat:
        ax.tick_params(axis="both", labelsize=13)

    # Add supertitle
    g.figure.suptitle(
        f"Synchrony Dynamics Across Layers of TBPTT Models\nwith Teacher-Forced Predictions Included",
        fontsize=24,
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

    # Get current legend handles and labels
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    # Add the new dashed line to the legend
    g._legend.remove()  # Remove default legend

    g.figure.legend(
        handles + [stimulus_line],
        ["Predictions", "Target", "Teacher-Forced Predictions", "Stimulus change"],
        loc="upper right",
        fontsize=14,
        title="Variant Type + Marker",
        title_fontsize=15,
        frameon=True,
        ncol=1,
    )

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
            nn_model.globals.IMAGE_DURATION // 20,
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


def plot_pearson_boxplot_synchrony_overall(df: pd.DataFrame, save_fig: str = ""):
    """
    Plot boxplot comparison of synchrony and overall Pearson CC for all models,
    averaged across layers.

    :param df: Melted dataframe with columns:
               ['model_variant', 'layer_name', 'subset_variant', 'Metric', 'Value']
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """

    # Step 2: Plot
    g = sns.catplot(
        data=df,
        x="model_variant",
        y="Value",
        hue="Metric",
        kind="box",
        height=6,
        aspect=1.5,
    )

    # Tick and label styling
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelrotation=20, labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

    g.set_axis_labels("Model Variant", "Pearson Correlation", fontsize=18)

    # Grid lines
    g.map(
        lambda *args, **kwargs: plt.grid(
            True, which="both", linestyle=":", linewidth=0.5, alpha=0.7
        )
    )

    # Legend styling
    g._legend.set_title("Metric", prop={"size": 13})
    for text in g._legend.texts:
        text.set_fontsize(12)
    g._legend.set_bbox_to_anchor((0.97, 0.92))
    g._legend.set_frame_on(True)

    label_map = {
        "pearson_overall": "Overall Pearson CC",
        "pearson_synchrony": "Synchrony Pearson CC",
    }

    for text in g._legend.texts:
        new_label = label_map.get(text.get_text(), text.get_text())
        text.set_text(new_label)

    # Title
    plt.suptitle("Overall vs Synchrony Pearson CC\nin Different Models", fontsize=22)

    plt.tight_layout(rect=[0, 0, 0.99, 0.99])
    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_pearson_synchrony_boxplot_layers(df: pd.DataFrame, save_fig: str = ""):
    """
    Plots boxplot comparison synchrony across layers for all models and with
    teacher-forced variant.

    :param df: Dataset to plot.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """
    g = sns.catplot(
        data=df,
        x="model_variant",
        y="pearson_synchrony",
        hue="variant_type",
        col="layer_name",
        kind="box",
        col_wrap=2,
        height=4,
        aspect=1.4,
        sharey=True,
    )

    g._legend.set_title("Prediction type:", prop={"size": 13})
    for text in g._legend.texts:
        text.set_fontsize(12)
    g._legend.set_bbox_to_anchor((0.97, 0.92))
    g._legend.set_frame_on(True)

    label_map = {
        "predictions": "Free",
        "train_like_predictions": "Teacher-Forced",
    }

    for text in g._legend.texts:
        new_label = label_map.get(text.get_text(), text.get_text())
        text.set_text(new_label)

    # === Style: Font sizes and tick rotation ===
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelrotation=25, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    # Add grid lines
    g.map(
        lambda *args, **kwargs: plt.grid(
            True, which="both", linestyle=":", linewidth=0.5, alpha=0.7
        )
    )

    g.set_axis_labels("Model Variant", "Synchrony Pearson CC", fontsize=16)
    g.set_titles("{col_name}", size=18)

    # === Suptitle ===
    plt.suptitle(
        "Comparison Synchrony Pearson CC\nBetween Free and Teacher-Forced Evaluation",
        fontsize=22,
    )

    plt.tight_layout(rect=[0, 0, 0.99, 0.99])
    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
