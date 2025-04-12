"""
This source code serves for defining class used for plotting the examples.
"""

from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import nn_model.globals
from nn_model.type_variants import (
    NeuronModulePredictionFields,
    PathPlotDefaults,
    EvaluationFields,
)
from evaluation_tools.response_analyzer import ResponseAnalyzer


class ResultsPlotter:
    """
    Class that is used to plot the results of the analysis.
    """

    def __init__(self):
        pass

    @staticmethod
    def save_figure_logic(
        fig_path: str, save_fig: bool, plot_key: str, layer: str = ""
    ):
        """
        Does common logic regarding saving or showing the plot.

        :param fig_path: Path to the plot (if empty string then use default).
        :param save_fig: Flag whether we want to save the plot.
        :param plot_key: Key of the default plot path (used if path not specified).
        """
        if save_fig:
            # We want to save the plot.
            if not fig_path:
                # We have not specified the path -> use the default one.
                fig_path = nn_model.globals.DEFAULT_PLOT_PATHS[plot_key]
                if layer:
                    splitted_fig_path = fig_path.split(".")
                    fig_path = (
                        splitted_fig_path[0] + f"_{layer}." + splitted_fig_path[1]
                    )

            plt.savefig(fig_path)
        else:
            # We do not want to save the figure -> just show it.
            plt.show()

    @staticmethod
    def _create_line_plot_for_layer_dnn_responses(
        layer_dnn_responses: Dict[str, torch.Tensor], axis, label: str = ""
    ):
        """
        Creates line plot for the neuron DNN model responses.

        :param layer_dnn_responses: DNN module input and output responses.
        :param axis: Axis object where we want to plot the results.
        :param label: Label of the plot.
        """
        # Flatten the arrays for the current layer
        layer_dnn_inputs = (
            layer_dnn_responses[NeuronModulePredictionFields.INPUT.value][:, 0]
            .cpu()
            .detach()
            .numpy()
        )
        layer_dnn_outputs = (
            layer_dnn_responses[NeuronModulePredictionFields.OUTPUT.value][:, 0]
            .cpu()
            .detach()
            .numpy()
        )

        # Create a scatter plot with Seaborn
        sns.lineplot(
            x=layer_dnn_inputs, y=layer_dnn_outputs, alpha=0.5, ax=axis, label=label
        )

    @staticmethod
    def _create_dnn_responses_labels(axis):
        """
        Assigns axis labels for the DNN neuron model responses plot.
        And creates reference identity line.
        """
        # Add a reference line (y = x)
        axis.plot([-1, 1], [-1, 1], color="red", linestyle="--", label="y = x")

        # Set titles and labels
        axis.set_xlabel("dnn_inputs")
        axis.set_ylabel("dnn_outputs")
        axis.legend()

    @staticmethod
    def plot_dnn_module_responses_separate(
        dnn_responses: Dict[str, Dict[str, torch.Tensor]],
        fig_path: str = "",
        save_fig: bool = False,
    ):
        """
        Plots responses of DNN neuron module each in separate plot.

        :param dnn_responses: Inputs and outputs of the DNN module for each layer.
        :param fig_path: Path where to store the plot (if empty string -> use default path).
        :param save_fig: Flag whether we want to save the figure.
        """
        # Initialize plot grid
        num_layers = len(dnn_responses)
        num_columns = 2
        num_rows = num_layers // num_columns + (
            num_layers % num_columns
        )  # In case odd number of plots -> add additional row.
        _, axes = plt.subplots(
            num_rows, num_columns, figsize=(12, 10)
        )  # Adjust grid size if necessary

        # Flatten the subplot axes for easier iteration
        axes = axes.flatten()

        # Loop through each layer and plot.
        for layer, ax in zip(dnn_responses, axes):
            # Plot the RNN responses.
            ResultsPlotter._create_line_plot_for_layer_dnn_responses(
                dnn_responses[layer], ax
            )

            # Set titles and labels.
            ax.set_title(f"Layer {layer}")
            ResultsPlotter._create_dnn_responses_labels(ax)

        plt.tight_layout()
        ResultsPlotter.save_figure_logic(
            fig_path, save_fig, PathPlotDefaults.NEURON_MODULE_SEPARATE.value
        )

    @staticmethod
    def plot_dnn_module_responses_together(
        dnn_responses,
        fig_path: str = "",
        save_fig: bool = False,
    ):
        """
        Plot DNN neuron module responses all together.

        :param dnn_responses: Inputs and outputs of the DNN module for each layer.
        :param fig_path: Path where to store the plot (if empty string -> use default path).
        :param save_fig: Flag whether we want to save the figure.
        """
        _, axis = plt.subplots(1, 1, figsize=(12, 10))  # Adjust grid size if necessary

        for layer in dnn_responses:
            ResultsPlotter._create_line_plot_for_layer_dnn_responses(
                dnn_responses[layer], axis=axis, label=layer
            )

        # Set labels and
        axis.set_title("DNN responses")
        ResultsPlotter._create_dnn_responses_labels(axis)

        plt.tight_layout()
        ResultsPlotter.save_figure_logic(
            fig_path, save_fig, PathPlotDefaults.NEURON_MODULE_TOGETHER.value
        )

    def plot_neuron_responses_on_multiple_images(
        self, neuron_id: int, layer: str, selected_images_ids: List[int]
    ):
        """
        Plots mean neuron responses/targets per selected images.

        :param neuron_id: ID of the neuron to plot the responses for.
        :param layer: name of the layer where the selected neuron lies.
        :param selected_images_ids: list of image ids that we are interested to plot.
        """
        pass

    @staticmethod
    def compute_stimulus_blank_step() -> int:
        """
        Computes in which time step there is the stimulus/blank transition
        based on the global settings.

        :return: Returns time step in which the stimulus/blank transition happened.
        """
        return int(nn_model.globals.IMAGE_DURATION // nn_model.globals.TIME_STEP)

    @staticmethod
    def _set_mean_response_in_time_plot_variables(
        axs,
        idx: int,
        layer: str,
        y_range: Optional[Tuple[float, float]],
        stimulus_blank_border: Optional[float],
    ):
        """
        Sets all necessary parameters of the plot of mean response in time.

        :param axs: Axis object where we want to plot the results.
        :param idx: Index of the subplot where we want to plot the results.
        :param layer: Name of the layer that we want to plot.
        :param y_range: Range in y-axis to plot. If `None` then use the default.
        :param stimuli_blank_border: Time step of stimulus/blank border.
        If `None` then use default time step computed from the stimulus duration
        and time step length defined in `nn_model.globals`.
        """

        if not stimulus_blank_border:
            # Stimuli/blank border time step not provided
            # -> use default time step computed from the stimulus duration
            # and time step length defined in `nn_model.globals`.
            stimulus_blank_border = ResultsPlotter.compute_stimulus_blank_step()

        # Set line defining stimulus/blank border
        axs[idx].axvline(
            x=stimulus_blank_border,
            color="green",
            linestyle="--",
            label="Stimulus/Blank border",
        )

        # Adding titles and labels
        # axs[idx].set_title(f"Layer {layer} - Mean neural response through time")
        axs[idx].set_xlabel("Time")
        axs[idx].set_ylabel("Mean response")
        if y_range:
            # y range defined
            axs[idx].set_ylim(*y_range)  # y axis limit
        axs[idx].legend()

    @staticmethod
    def _init_mean_response_in_time_plot_object(num_layers: int) -> List:
        """
        Initializes plot object where we would plot the results of each layer.

        :param num_layers: Number of layers that we want to plot.
        :return: Returns initializes subplots list of `axs` objects.
        """

        # Create subplots for each layer.
        _, axs = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))

        # If there's only one layer, we ensure axs is a list for easier handling
        if num_layers == 1:
            axs = [axs]

        return axs

    @staticmethod
    def _create_mean_responses_plot(
        target_tensor, pred_tensor, axis, labels, colors, include_predictions=True
    ):
        target_tensor = target_tensor.detach().numpy()

        if include_predictions:
            pred_tensor = pred_tensor.detach().numpy()
            axis.plot(
                # Insert first target value (starting point of the predictions)
                np.insert(pred_tensor, 0, target_tensor[0], axis=0),
                label=labels[EvaluationFields.PREDICTIONS.value],
                color=colors[EvaluationFields.PREDICTIONS.value],
            )

        axis.plot(
            target_tensor,
            label=labels[EvaluationFields.TARGETS.value],
            color=colors[EvaluationFields.TARGETS.value],
        )

    @staticmethod
    def plot_mean_layer_data(
        data: Dict,
        include_predictions: bool,
        # identifier: str = "",
        y_range: Optional[Tuple[float, float]] = None,
        save_fig: bool = False,
        fig_path: str = "",
        stimuli_blank_border: Optional[float] = None,
    ):
        """
        Plots mean layer responses either only mean targets or also together with mean predictions.

        :param data: Data to be plotted. Either dictionary of `predictions` and `targets`
        or just layers of targets.
        :param include_predictions: Flag whether we want to include predictions in the plot creation.
        The plot should include both mean prediction and mean target in time.
        :param identifier: Identifier of the specific computed means to plot.
        Possible options are: `['prediction_mean', input_mean']`. If everything else then plot the `data`.
        :param y_range: Range in y-axis to plot. If `None` then use the default.
        :param save_fig: Flag whether we want to save the figure.
        :param fig_path: Path where we want to store the created figure.
        If empty string then use the default path.
        :param stimuli_blank_border: Time step where is the border between stimulus
        and blank stage.
        """

        # if identifier == "prediction_mean":
        #     data = self.mean_layer_responses
        # elif identifier == "input_mean":
        #     data = self.mean_input_layer_responses

        predictions = {}
        targets = data
        if include_predictions:
            predictions = data[EvaluationFields.PREDICTIONS.value]
            targets = data[EvaluationFields.TARGETS.value]

        axs = ResultsPlotter._init_mean_response_in_time_plot_object(len(targets))

        # Iterate over each layer
        for idx, layer in enumerate(targets.keys()):

            ResultsPlotter._create_mean_responses_plot(
                targets[layer],
                predictions[layer],
                axis=axs[idx],
                labels={
                    EvaluationFields.TARGETS.value: "Targets",
                    EvaluationFields.PREDICTIONS.value: "Predictions",
                },
                colors={
                    EvaluationFields.TARGETS.value: "red",
                    EvaluationFields.PREDICTIONS.value: "blue",
                },
                include_predictions=include_predictions,
            )

            axs[idx].set_title(f"Layer {layer} - Mean neural response through time")
            ResultsPlotter._set_mean_response_in_time_plot_variables(
                axs, idx, layer, y_range, stimuli_blank_border
            )

        ResultsPlotter.save_figure_logic(
            fig_path, save_fig, PathPlotDefaults.MEAN_LAYER_RESPONSES.value
        )

    @staticmethod
    def plot_mean_neuron_responses(
        data: Dict,
        mean_data: Dict,
        y_range: Optional[Tuple[float, float]] = None,
        save_fig: bool = False,
        fig_path: str = "",
        stimuli_blank_border: Optional[float] = None,
    ):

        predictions = data[EvaluationFields.PREDICTIONS.value]
        targets = data[EvaluationFields.TARGETS.value]
        mean_predictions = mean_data[EvaluationFields.PREDICTIONS.value]
        mean_targets = mean_data[EvaluationFields.TARGETS.value]
        neuron_ids_path = "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/model_size_25_subset_10.pkl"

        selected_neurons = ResponseAnalyzer.load_pickle_file(neuron_ids_path)

        for layer in targets:
            # print(selected_neurons)
            num_neurons = selected_neurons[layer].shape[0]  # predictions[layer].size(1)
            _, axs = plt.subplots(num_neurons, 1, figsize=(10, 5 * num_neurons))

            for i in range(num_neurons):
                neuron_id = selected_neurons[layer][i]

                ResultsPlotter._create_mean_responses_plot(
                    targets[layer][:, neuron_id],
                    predictions[layer][:, neuron_id],
                    axis=axs[i],
                    labels={
                        EvaluationFields.TARGETS.value: "Targets",
                        EvaluationFields.PREDICTIONS.value: "Predictions",
                    },
                    colors={
                        EvaluationFields.TARGETS.value: "red",
                        EvaluationFields.PREDICTIONS.value: "blue",
                    },
                    include_predictions=True,
                )

                pred_tensor = mean_predictions[layer].detach().numpy()
                axs[i].plot(
                    # Insert first target value (starting point of the predictions)
                    np.insert(pred_tensor, 0, mean_targets[layer][0], axis=0),
                    label="Mean Predictions",
                    color="black",
                    # linestyle="loosely dashed",
                    linestyle="dotted",
                    # linewidth=1.5,
                )

                axs[i].set_title(
                    f"Neuron {neuron_id} in layer {layer} - Mean response through time"
                )

                ResultsPlotter._set_mean_response_in_time_plot_variables(
                    axs, i, layer, y_range, stimuli_blank_border
                )

            # splitted_fig_path = fig_path.split(".")
            # layer_path = splitted_fig_path[0] + f"_{layer}." + splitted_fig_path[1]

            ResultsPlotter.save_figure_logic(
                fig_path,
                save_fig,
                PathPlotDefaults.MEAN_LAYER_RESPONSES.value,
                layer=layer,
            )
