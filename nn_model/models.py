"""
This source code contains definition of all models used in our experiments.
"""

from typing import List, Dict, Tuple

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import (
    LayerType,
    TimeStepVariant,
    ModelTypes,
    NeuronModulePredictionFields,
)
from nn_model.weights_constraints import (
    WeightTypes,
    ExcitatoryWeightConstraint,
    InhibitoryWeightConstraint,
)
from nn_model.layers import (
    ConstrainedRNNCell,
)
from nn_model.neurons import FeedForwardNeuron


class LayerConfig:
    """
    Class for storing configuration of model layer.
    """

    def __init__(
        self,
        size: int,
        layer_type: str,
        input_layers_parameters: List[Tuple[str, str]],
        neuron_model=None,
    ):
        """
        Initializes configuration based on the given parameters.
        Determines weight constraints.

        :param size: size of the layer (number of neurons).
        :param layer_type: name of the layer.
        :param input_layers_parameters: ordered list of input layer parameters of the layer.
        The parameters are in form of tuple with first value its name from `LayerType`
        and second value its time step name from `TimeStepVariant`.
        :param shared_complexity: shared complexity model(s), if no then `None`.
        """
        self.size: int = size
        self.layer_type: str = layer_type
        self.input_layers_parameters: List[Tuple[str, str]] = input_layers_parameters
        self.neuron_model = neuron_model

        # Determine weight constraints for the layer (excitatory/inhibitory).
        input_constraints = self._determine_input_constraints()
        self.constraint = self._determine_constraint(layer_type, input_constraints)

    def _determine_input_constraints(self) -> List[Dict]:
        """
        Determines input weights constraint (chooses between excitatory/inhibitory).

        :return: Returns list of dictionaries with parameters specifying
        the distribution of input weight types of each input layer.

        The format is same as the expected kwargs for `WeightConstraint` objects.
        The order of the dictionaries should be same as the order of the input layers.

        The keys in the dictionaries are:
            `part_size` (int): size of the input layer.
            `part_type` (WeightTypes value): type of the layer (exc/inh).
        """
        return [
            {
                "part_size": nn_model.globals.MODEL_SIZES[layer[0]],
                "part_type": self._get_constraint_type(layer[0]),
            }
            for layer in self.input_layers_parameters
        ]

    def _get_constraint_type(self, layer_type: str) -> str:
        """
        Determines type of the constraint that should be used for given layer.

        :param layer_type: name of the layer. Should be value from `LayerType`.
        :return: Returns identifier of constraint type (value from `WeightTypes`).
        """
        if layer_type in nn_model.globals.EXCITATORY_LAYERS:
            return WeightTypes.EXCITATORY.value
        if layer_type in nn_model.globals.INHIBITORY_LAYERS:
            return WeightTypes.INHIBITORY.value

        class WrongLayerException(Exception):
            """
            Exception class to be raised while wrong layer type chosen.
            """

        raise WrongLayerException(
            f"Wrong layer type. The type {layer_type} does not exist."
        )

    def _determine_constraint(self, layer_type: str, input_constraints: List[Dict]):
        """
        Determines weight constraints of the layer.

        :param layer_type: name of the layer. Should be value from `LayerType`,
        or different if we do not want to use the weight constraints.
        :param input_constraints: list of constraint kwargs for input layer weight constraints.
        :return: Returns appropriate `WeightConstraint` object,
        or `None` if we do not want to use the weight constraint.
        """
        if layer_type in nn_model.globals.EXCITATORY_LAYERS:
            # Excitatory neurons.
            return ExcitatoryWeightConstraint(input_constraints)
        if layer_type in nn_model.globals.INHIBITORY_LAYERS:
            # Inhibitory neurons.
            return InhibitoryWeightConstraint(input_constraints)

        # Apply no constraint.
        return None


class RNNCellModel(nn.Module):
    """
    Class defining models that use LGN outputs as input and predicts rest of the layers.
    """

    # Input layer keys (LGN).
    input_layers = [LayerType.X_ON.value, LayerType.X_OFF.value]

    # Define model architecture (output layers with its inputs).
    layers_input_parameters = {
        # V1_Exc_L4 inputs
        LayerType.V1_EXC_L4.value: [
            # X_ON_{t}
            (
                LayerType.X_ON.value,
                TimeStepVariant.CURRENT.value,
            ),
            # X_OFF_{t}
            (
                LayerType.X_OFF.value,
                TimeStepVariant.CURRENT.value,
            ),
            # V1_Inh_L4_{t-1}
            (
                LayerType.V1_INH_L4.value,
                TimeStepVariant.PREVIOUS.value,
            ),
            # V1_Exc_L23_{t-1}
            (
                LayerType.V1_EXC_L23.value,
                TimeStepVariant.PREVIOUS.value,
            ),
        ],
        # V1_Inh_L4 inputs
        LayerType.V1_INH_L4.value: [
            # X_ON_{t}
            (
                LayerType.X_ON.value,
                TimeStepVariant.CURRENT.value,
            ),
            # X_OFF_{t}
            (
                LayerType.X_OFF.value,
                TimeStepVariant.CURRENT.value,
            ),
            # V1_Exc_L4_{t-1}
            (
                LayerType.V1_EXC_L4.value,
                TimeStepVariant.PREVIOUS.value,
            ),
            # V1_Exc_L23_{t-1}
            (
                LayerType.V1_EXC_L23.value,
                TimeStepVariant.PREVIOUS.value,
            ),
        ],
        # V1_Exc_L23 inputs
        LayerType.V1_EXC_L23.value: [
            # V1_Exc_L4_{t}
            (
                LayerType.V1_EXC_L4.value,
                TimeStepVariant.CURRENT.value,
            ),
            # V1_Inh_L23_{t-1}
            (
                LayerType.V1_INH_L23.value,
                TimeStepVariant.PREVIOUS.value,
            ),
        ],
        # V1_Inh_L23 inputs
        LayerType.V1_INH_L23.value: [
            # V1_Exc_L4_{t}
            (
                LayerType.V1_EXC_L4.value,
                TimeStepVariant.CURRENT.value,
            ),
            # V1_Exc_L23_{t-1}
            (
                LayerType.V1_EXC_L23.value,
                TimeStepVariant.PREVIOUS.value,
            ),
        ],
    }

    def __init__(
        self,
        layer_sizes: Dict[str, int],
        num_hidden_time_steps: int,
        neuron_type: str,
        neuron_model_kwargs: Dict,
    ):
        """
        Initializes model parameters, sets weights constraints and creates model architecture.

        :param layer_sizes: sizes of all model layers (input included).
        :param num_hidden_time_steps: Number of hidden time steps in RNN model
        (where we do not know the targets).
        :param neuron_type: type of the neuron model used in the model
        (name from `ModelTypes`).
        :param neuron_model_kwargs: kwargs of the used neuronal models (if any).
        """
        super(RNNCellModel, self).__init__()

        # Number of hidden time steps (between individual targets) that we want to use
        # during training and evaluation (in order to learn the dynamics better).
        self.num_hidden_time_steps = num_hidden_time_steps

        # Type of the neuron used in the model.
        self.neuron_type = neuron_type

        # Kwargs to store complexity properties for various complexity types.
        self.neuron_model_kwargs = neuron_model_kwargs

        self.layer_sizes = layer_sizes  # Needed for model architecture definition

        # Layer configuration.
        self.layers_configs = self._init_layer_configs(
            layer_sizes, self._init_neuron_models()
        )

        self.return_recurrent_state = False

        # Init model.
        self._init_model_architecture()

    def switch_to_return_recurrent_state(self):
        """
        Function that is used to switch the model to return also RNN outputs.

        NOTE: Typically used for evaluation analysis.
        """
        self.return_recurrent_state = True
        for layer in self.layers.values():
            layer.switch_to_return_recurrent_state()

    def _init_simple_neuron_model(self) -> Dict:
        """
        Initializes simple neurons (`None` complexity).

        :return: Returns dictionary of layer name (`LayerType`) and `None`s.
        """
        return {layer: None for layer in RNNCellModel.layers_input_parameters}

    def _init_complex_neuron_model(
        self,
        neuron_model_kwargs: Dict,
    ) -> Dict:
        """
        Initializes complex neuron layers.

        :param layer_sizes: sizes of all model layers (input included).
        :param neuron_model_kwargs: kwargs of `SharedComplexity` object `__init__`.
        :return: Returns dictionary of layer name (`LayerType`) and appropriate shared
        complex complexity object.
        """
        return {
            layer: FeedForwardNeuron(**neuron_model_kwargs)
            for layer in RNNCellModel.layers_input_parameters
        }

    def _init_neuron_models(
        self,
    ) -> Dict:
        """
        Initializes shared complexities (neuronal models) of the model.

        :return: Returns dictionary of layer name (`LayerType`) and
        appropriate neuron model (shared complexity).
        """
        if self.neuron_type == ModelTypes.COMPLEX.value:
            # Complex complexity.
            return self._init_complex_neuron_model(
                self.neuron_model_kwargs[ModelTypes.COMPLEX.value]
            )

        # Simple neuron (no additional complexity).
        return self._init_simple_neuron_model()

    def _init_layer_configs(
        self,
        layer_sizes: Dict[str, int],
        neuron_models: Dict,
    ) -> Dict[str, LayerConfig]:
        """
        Initializes `LayerConfig` objects for all layers of the model.

        :param layer_sizes: sizes of the layers.
        :param neuron_models: shared complexities of the layers (neuron models of each layer).
        :return: Returns dictionary of layer configurations for all model layers.
        """
        return {
            layer: LayerConfig(
                layer_sizes[layer],
                layer,
                input_parameters,
                neuron_models[layer],
            )
            for layer, input_parameters in RNNCellModel.layers_input_parameters.items()
        }

    def _init_layer(self, layer: str) -> ConstrainedRNNCell:
        """
        Initializes one layer of the model.

        :param layer: layer name (`LayerType`).
        :param rnn_cell_cls: layer object variant.
        :return: Returns initializes layer object.
        """
        return ConstrainedRNNCell(
            sum(
                self.layer_sizes[layer_name]
                for layer_name, _ in RNNCellModel.layers_input_parameters[layer]
            ),
            self.layers_configs[layer].size,
            self.layers_configs[layer].constraint,
            self.layers_configs[layer].neuron_model,
        )

    def _init_model_architecture(self):
        """
        Initializes all model layers and stored them as `nn.ModuleDict` object
        under the keys from `LayerType`.
        """
        self.layers = nn.ModuleDict()

        for layer in RNNCellModel.layers_input_parameters:
            self.layers[layer] = self._init_layer(layer)

    def _init_hidden_layers(self, targets) -> Dict[str, torch.Tensor]:
        """
        Initializes hidden layers based on the model mode (training/evaluation).

        In the training mode: The hidden states are the targets from previous step.
        It should be assigned in each training step. This function just initializes
        empty dictionary object as placeholder for future usage of hidden layers in the
        training steps.

        In the evaluation mode: The hidden states are initialized once in the first
        time step with the value of target in the first time step. The rest hidden
        states are the results of the previous time step evaluation. In this function
        it creates the hidden steps for the first time step.

        :param targets: dictionary containing the targets for a neural network model.
        :return: Returns a dictionary of hidden layers moved to CUDA if in evaluation mode,
        otherwise (training mode) returns empty dictionary.
        """
        if not self.training:
            # Evaluation step. Use only time step 0 as initialization of hidden states.
            return {
                layer: hidden.to(nn_model.globals.DEVICE)
                for layer, hidden in targets.items()
            }
        # Training mode. Hidden layers are last steps from targets for each time step.
        # Assign the values in each training step (not in this function).
        return {}

    def apply_layer_neuron_complexity(
        self, layer: str, input_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies layer neuron DNN model to given input data.

        NOTE: Usually used to inspect how to model complexity module behaves.
        :param layer: Identifier of the layer to apply the complexity for.
        :param input_data: Input data tensor to apply complexity on.
        :return: Returns output of DNN module.
        """
        return self.layers[layer].apply_complexity(input_data)

    def apply_neuron_complexity_on_all_layers(
        self, start: float, end: float, step: float
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Applies neuron complexity on all layers on given range of input data.

        NOTE: This function is used mainly to inspect the behavior of the DNN neuron model.

        :param start: Start of input interval.
        :param end: End of input interval.
        :param step: Step size used to generate input interval.
        :return: Returns dictionary of input and output for each layer DNN neuron module.
        """
        input_data_range = torch.arange(start, end, step).to(nn_model.globals.DEVICE)
        input_data_range = input_data_range[:, None]

        return {
            layer: {
                NeuronModulePredictionFields.INPUT.value: input_data_range,
                NeuronModulePredictionFields.OUTPUT.value: self.apply_layer_neuron_complexity(
                    layer, input_data_range
                ),
            }
            for layer in self.layers
        }

    def _get_layer_input_tensor(
        self, current_parts: List[torch.Tensor], previous_parts: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Concatenates all input tensors (from current and previous time step)
        into one big input tensor.

        :param current_parts: list of tensors of inputs from current time step.
        :param previous_parts: list of tensors of inputs from previous time step.
        :return: Returns concatenated tensor of all input tensors.
        """
        return torch.cat(
            current_parts + previous_parts,
            dim=1,
        ).to(nn_model.globals.DEVICE)

    def _get_list_by_time_variant(
        self,
        layer_type: str,
        time_variant: str,
        values_of_given_time: Dict[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Retrieves input tensors of the given time variant from the provided list of all
        possible input tensors. The tensors are selected from the input tensors of the
        provided layer for the specified time step (previous/current).

        :param layer_type: type of the layer to obtain the inputs for. Value from `LayerType`.
        :param time_variant: time variant (previous, current) we want to obtain the values for.
        Value from `TimeStepVariant`.
        :param values_of_given_time: values for the processed time variant. In case of
        the previous time variant they are `hidden_layer`s, in case of current time values
        they are`current_time_values`).
        :return: Returns list of tensors of the
        """
        # List of all input tensors we want to retrieve.
        time_variant_list = []
        for (
            input_part_layer_name,  # Name of teh input layer
            input_part_time_variant,  # Time variant of the input layer
        ) in RNNCellModel.layers_input_parameters[layer_type]:
            # Iterate through inputs of the given layer.
            if time_variant == input_part_time_variant:
                # If the currently processed input layer belongs to given time variant
                # -> append it to the list
                time_variant_list.append(values_of_given_time[input_part_layer_name])

        return time_variant_list

    def _perform_model_time_step(
        self,
        current_time_outputs: Dict[str, torch.Tensor],
        hidden_layers: Dict[str, torch.Tensor],
        # time: int,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs model time step. It progresses the model architecture and
        computes next time step results of each layer of the model.

        :param current_time_outputs: dictionary of input layers of the model (LGN inputs).
        :param hidden_layers: dictionary of all hidden layers values of the model
        from the previous time step (in our example those are all previous outputs).
        :param time: time of the current step. Should be at least second time step.
        During the first time step we do not have necessary hidden states.
        :return: Returns dictionary of model predictions for the current time step.
        """
        # We already have LGN inputs (assign them to current time layer outputs).
        recurrent_outputs = {}
        for layer in RNNCellModel.layers_input_parameters:
            # Iterate through all output layers of the model.
            # Perform model time step for the currently processed layer.
            # NOTE: It is necessary that these layers are sorted by the
            # processing order in the model.
            current_time_outputs[layer], recurrent_outputs[layer] = self.layers[layer](
                self._get_layer_input_tensor(
                    self._get_list_by_time_variant(
                        layer,
                        TimeStepVariant.CURRENT.value,
                        current_time_outputs,
                    ),  # inputs of the layer from time (t).
                    self._get_list_by_time_variant(
                        layer,
                        TimeStepVariant.PREVIOUS.value,
                        hidden_layers,
                    ),  # inputs of the layer from time (t-1) (previous time step)
                ),
                hidden_layers[layer],  # Recurrent connection to itself from time (t-1)
            )

        return current_time_outputs, recurrent_outputs

    def _append_outputs(
        self,
        all_outputs: Dict[str, List[torch.Tensor]],
        time_step_outputs: Dict[str, torch.Tensor],
    ):
        """
        Appends outputs of each output layer to list of outputs of all time steps.

        :param all_outputs: outputs of layers of all time steps.
        :param time_step_outputs: outputs of current time step.
        """
        for layer, layer_outputs in time_step_outputs.items():
            if layer in RNNCellModel.layers_input_parameters:
                # For each output layer append output of the current time step.
                all_outputs[layer].append(layer_outputs.unsqueeze(1).cpu())

    def forward(
        self, inputs: Dict[str, torch.Tensor], hidden_states: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """
        Performs forward step of the model iterating through all time steps of the provided
        inputs and targets.

        Training forward step: It initializes hidden states as the previous time step from the
        provided targets (because of that it skips the first time step). This means that the
        model predicts only the next step during training.

        Evaluation forward step: It initializes hidden state only for time 0 as the time 0
        target values (so, we start in existing state). Other hidden states are the results
        of model predictions from the previous time step (+ LGN input from `inputs`).

        :param inputs: model inputs of size `(batch_size, num_time_steps, num_neurons)`
        or `(batch_size, num_neurons)` for train.
        :param targets: model targets - size `(batch_size, num_time_steps, num_neurons)`
        for training mode or `(batch, neurons)` for evaluation (we need only first time step).
        :return: Returns model predictions for all time steps.
        """
        # Initialize dictionaries of all model predictions.
        all_recurrent_outputs: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in RNNCellModel.layers_input_parameters
        }
        all_hidden_outputs: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in RNNCellModel.layers_input_parameters
        }

        visible_time_steps = inputs[LayerType.X_ON.value].size(
            1
        )  # Minus 1 time step because it is the initial time step
        if self.training:
            # Training mode (only one visible step)
            # We add 1 to iterate through the visible time loop correctly (we want to iterate once).
            visible_time_steps = 1 + 1

        for visible_time in range(1, visible_time_steps):
            # Prediction of the visible time steps
            # (in training only one step, in evaluation all time steps)

            # Placeholder for RNN outputs in case we would like
            # to store RNN outputs for model analysis.
            recurrent_outputs: Dict[str, torch.Tensor] = {}

            # Define input layers for the current visible time prediction.
            current_inputs = inputs  # Train mode -> only one input state
            if not self.training:
                # Evaluation mode
                # -> assign current input state
                # (we predict all visible time steps in one forward step of the model).
                current_inputs = {
                    layer: layer_input[:, visible_time, :]
                    for layer, layer_input in inputs.items()
                }
            for _ in range(self.num_hidden_time_steps):
                # Perform all hidden time steps.
                hidden_states, recurrent_outputs = self._perform_model_time_step(
                    current_inputs,
                    hidden_states,
                )

                if self.training:
                    # In train return all hidden time steps (for back-propagation through time)
                    self._append_outputs(all_hidden_outputs, hidden_states)

            if not self.training:
                # Evaluation mode -> save only predictions of the visible time steps
                self._append_outputs(all_hidden_outputs, hidden_states)

            if self.return_recurrent_state:
                # If the model is in evaluation mode
                # -> save also the results of the RNNs before neuron model.
                # Only the visible steps
                self._append_outputs(all_recurrent_outputs, recurrent_outputs)

        # Clear caches
        del inputs, hidden_states
        torch.cuda.empty_cache()

        return all_hidden_outputs, all_recurrent_outputs
