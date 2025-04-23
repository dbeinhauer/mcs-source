import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nn_model.globals


def plot_teacher_forced_free_drift(
    df: pd.DataFrame,
    save_fig: str = "",
):
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
        facet_kws={"sharey": True, "sharex": True},
    )

    # === Axes labels and facet titles ===
    g.set_axis_labels("Time Step (20 ms)", "Drift (FR - TF)", fontsize=16)
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
