"""
This script defines plotting functions comparing drift of teacher-forced and free predictions.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nn_model.globals


def plot_teacher_forced_free_drift(
    df: pd.DataFrame,
    save_fig: str = "",
):
    """
    Plot drift between teacher-forced and free predictions.

    :param df: Dataset to plot.
    :param save_fig: Path where to store the figure, if `""` then do not store.
    """

    ordered_layers = ["V1_Exc_L4", "V1_Exc_L23", "V1_Inh_L4", "V1_Inh_L23"]

    df["layer_name"] = pd.Categorical(
        df["layer_name"],
        categories=ordered_layers,
        ordered=True,
    ).remove_unused_categories()

    g = sns.relplot(
        data=df,
        x="time",
        y="drift",
        hue="model_variant",
        col="layer_name",
        kind="line",
        col_wrap=2,
        height=4,
        aspect=1.5,
        facet_kws={"sharey": False, "sharex": True},
    )

    # Group the axes
    axes = g.axes.flat  # flat list of axes

    # Map layer names to axis (from your ordered categories)
    layer_to_ax = dict(zip(df["layer_name"].cat.categories, axes))

    # Function to unify y-axis limits across given layers
    def unify_ylim(layer_group):
        ymins, ymaxs = [], []
        for layer in layer_group:
            ax = layer_to_ax[layer]
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
        unified_ylim = (min(ymins), max(ymaxs))
        for layer in layer_group:
            layer_to_ax[layer].set_ylim(unified_ylim)

    # Apply for both groups
    unify_ylim(ordered_layers[0:2])
    unify_ylim(ordered_layers[2:])

    # === Axes labels and facet titles ===
    g.set_axis_labels("Time Step (20 ms)", "Drift (TF - FP)", fontsize=16)
    g.set_titles("{col_name}", size=18)

    # === Grid and tick styling ===
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # === Stimulus change marker ===
    stim_time = nn_model.globals.IMAGE_DURATION // 20
    g.map_dataframe(
        lambda data, **kws: plt.axvline(
            stim_time, linestyle="--", color="gray", linewidth=1
        )
    )

    # === Legend formatting ===
    g._legend.set_title("Model Variant", prop={"size": 13})
    for text in g._legend.texts:
        text.set_fontsize(12)

    g._legend.set_bbox_to_anchor((0.98, 0.94))  # Top right outside the plot
    g._legend.set_frame_on(True)

    # === Suptitle ===
    plt.suptitle(
        "Drift Between Free and Teacher-Forced\nPredictions per Layer", fontsize=22
    )

    plt.tight_layout(rect=[0, 0, 0.99, 0.97])  # leave space at top/right for legend
    if save_fig:
        g.figure.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
