"""
This script defines plotting of the time bin spike count distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from evaluation_tools.fields.dataset_parameters import ALL_TIME_STEP_VARIANTS


def plot_dataset_variant_all_time_bins(
    normalized_df: pd.DataFrame, is_test: bool = False, save_fig: str = ""
):
    """
    Plots line plot of spike counts is each time bin for each time step size.

    :param normalized_df: Dataframe prepared for plotting.
    :param is_test: Flag whether plotting test or train dataset.
    :param save_fig: Where to store the figure. If `""` then do not store.
    """

    # Get all layers
    layers = normalized_df["layer"].unique()

    # Grid layout: choose based on number of layers
    n_cols = 2
    n_rows = int(np.ceil(len(layers) / n_cols))

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharex=True, sharey=True
    )
    axes = axes.flatten()

    # Bin centers (assumes same across all)
    bins = np.sort(normalized_df["bin"].unique())
    num_bins = len(bins)

    time_steps_sorted = sorted(ALL_TIME_STEP_VARIANTS)

    # Prepare colormap (adjust range if needed)
    cmap = cm.get_cmap("viridis", num_bins)
    time_step_colors = {ts: cmap(i) for i, ts in enumerate(time_steps_sorted)}

    # Plot each layer
    for idx, layer in enumerate(layers):
        ax = axes[idx]
        subset = normalized_df[normalized_df["layer"] == layer].copy()
        subset["time_step"] = subset["time_step"].astype(int)

        for ts in time_steps_sorted:
            ts_data = subset[subset["time_step"] == ts].sort_values("bin")
            x = ts_data["bin"]
            y = ts_data["density"]
            color = time_step_colors[ts]

            ax.plot(x, y, marker="o", alpha=0.6, color=color, label=f"{ts} ms")

        ax.set_title(
            f"Layer: {layer}",
            fontsize=18,
        )
        ax.set_xlabel(
            "Spike count",
            fontsize=16,
        )
        ax.set_ylabel(
            "Normalized bin density",
            fontsize=16,
        )

    # Remove empty subplots
    for j in range(len(layers), len(axes)):
        fig.delaxes(axes[j])

    for idx, ax in enumerate(axes):
        # Skip axes that were removed
        if not ax.has_data():
            continue

        row = idx // n_cols
        col = idx % n_cols

        # Hide Y-axis labels for non-left columns
        if col != 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        # Hide X-axis labels for non-bottom rows
        if row != n_rows - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)

    # Optional: shared legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Time step",
        bbox_to_anchor=(0.92, 0.5),
        loc="center left",
        fontsize=12,
        title_fontsize=14,
    )

    dataset_variant = "Test" if is_test else "Train"
    fig.suptitle(
        f"Normalized Histogram Distributions Across Layers - {dataset_variant} Dataset",
        fontsize=22,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 0.99])  # leave space for legend and title

    if save_fig:
        fig.savefig(save_fig, format="pdf", bbox_inches="tight")
    else:
        plt.show()
