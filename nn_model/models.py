"""
This source code contains definition of all models used in our experiments.
"""

from typing import List, Dict, Tuple, Optional, Type

import torch
import torch.nn as nn

import nn_model.globals
from nn_model.type_variants import (
    LayerType,
    TimeStepVariant,
    ModelTypes,
    NeuronModulePredictionFields,
    ModelModulesFields,
)
from nn_model.layers import (
    ModelLayer,
)
from nn_model.neurons import DNNNeuron, SharedNeuronBase, RNNNeuron
from nn_model.layer_config import LayerConfig

# from nn_model.model_executer import ModelExecuter


class PrimaryVisualCortexModel(nn.Module):
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
        weight_initialization: str,
        model_modules_kwargs: Dict[str, Optional[Dict]],
    ):
        """
        Initializes model parameters, sets weights constraints and creates model architecture.

        :param layer_sizes: sizes of all model layers (input included).
        :param num_hidden_time_steps: Number of hidden time steps in RNN model
        (where we do not know the targets).
        :param neuron_type: type of the neuron model used in the model
        (name from `ModelTypes`).
        :param weight_initialization: type of weight initialization that we want to use.
        :param model_modules_kwargs: kwargs of the used neuronal models (if any) and synaptic
        adaptation models.
        """
        super(PrimaryVisualCortexModel, self).__init__()

        # Number of hidden time steps (between individual targets) that we want to use
        # during training and evaluation (in order to learn the dynamics better).
        self.num_hidden_time_steps = num_hidden_time_steps

        self.weight_initialization = weight_initialization

        # Type of the neuron used in the model.
        self.neuron_type = neuron_type

        # Kwargs defining neuron module properties and synaptic adaptation module properties.
        self.neuron_model_kwargs = model_modules_kwargs[
            ModelModulesFields.NEURON_MODULE.value
        ]
        self.synaptic_adaptation_kwargs = model_modules_kwargs[
            ModelModulesFields.SYNAPTIC_ADAPTION_MODULE.value
        ]

        self.layer_sizes = layer_sizes  # Needed for model architecture definition

        # Layer configuration.
        self.layers_configs = self._init_layer_configs(
            layer_sizes,
            self._init_neuron_models(),
            self._init_synaptic_adaptation_models(model_modules_kwargs["only_lgn"]),
        )

        # Flag whether we want to return the RNN linearity results for analysis.
        # TODO: currently broken functionality (not used).
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

    def _init_simple_neuron_model(self) -> Dict[str, Optional[SharedNeuronBase]]:
        """
        Initializes simple neurons (`None` complexity).

        :return: Returns dictionary of layer name (`LayerType`) and `None`s.
        """
        return {
            layer: None for layer in PrimaryVisualCortexModel.layers_input_parameters
        }

    def _init_complex_neuron_model(
        self,
        neuron_model_kwargs: Dict,
    ) -> Dict[str, Optional[SharedNeuronBase]]:
        """
        Initializes complex neuron layers.

        :param layer_sizes: sizes of all model layers (input included).
        :param neuron_model_kwargs: kwargs of `SharedComplexity` object `__init__`.
        :return: Returns dictionary of layer name (`LayerType`) and appropriate shared
        complex complexity object.
        """
        neuron_model: Optional[Type[SharedNeuronBase]] = None
        if self.neuron_type in nn_model.globals.DNN_MODELS:
            neuron_model = DNNNeuron
        elif self.neuron_type in nn_model.globals.RNN_MODELS:
            neuron_model = RNNNeuron

        if neuron_model is None:

            class WrongNeuronModelException(Exception):
                """
                Class used if wrong neuron model type is selected.
                """

            raise WrongNeuronModelException("Selected wrong neuron model type.")

        return {
            layer: neuron_model(**neuron_model_kwargs)
            for layer in PrimaryVisualCortexModel.layers_input_parameters
        }

    def _init_neuron_models(
        self,
    ) -> Dict[str, Optional[SharedNeuronBase]]:
        """
        Initializes shared complexities (neuronal models) of the model.

        :return: Returns dictionary of layer name (`LayerType`) and
        appropriate neuron model (shared complexity).
        """
        # if self.neuron_type in nn_model.globals.COMPLEX_MODELS:
        if self.neuron_model_kwargs is not None:
            # Complex neuron module.
            return self._init_complex_neuron_model(self.neuron_model_kwargs)

        # Simple neuron (no additional complexity).
        return self._init_simple_neuron_model()

    def _init_synaptic_adaptation_models(
        self,
        only_lgn: bool = False,
    ) -> Dict[str, Optional[Dict[str, RNNNeuron]]]:
        """
        Initializes synaptic adaptation model.
        
        :param only_lgn: Flag whether we want to apply synaptic adaptation module only on the LGN layer.

        :return: Returns synaptic adaptation models for all layers (dictionary of `None`
        if we do not want to use it).
        """
        if self.synaptic_adaptation_kwargs is not None:
            return {
                layer: {
                    input_layer: RNNNeuron(**self.synaptic_adaptation_kwargs).to(
                        nn_model.globals.DEVICE
                    ) if input_layer in {"X_ON", "X_OFF"} or not only_lgn else None # Decide whether to use synaptic adaptation only for LGN or for all layers.
                    for (
                        input_layer,
                        _,
                    ) in layer_inputs
                    + [(layer, "")]  # Add the recurrent connection.
                }
                for layer, layer_inputs in PrimaryVisualCortexModel.layers_input_parameters.items()
            }

        # We do not want to use synaptic adaptation model.
        return {
            layer: None for layer in PrimaryVisualCortexModel.layers_input_parameters
        }

    def _init_layer_configs(
        self,
        layer_sizes: Dict[str, int],
        neuron_models: Dict[str, Optional[SharedNeuronBase]],
        synaptic_adaptation_models: Dict[str, Optional[Dict[str, RNNNeuron]]],
    ) -> Dict[str, LayerConfig]:
        """
        Initializes `LayerConfig` objects for all layers of the model.

        :param layer_sizes: sizes of the layers.
        :param neuron_models: shared complexities of the layers (neuron models of each layer).
        :param synaptic_adaptation_models: synaptic adaptation models of the layers (models
        for each layer under specific key).
        :return: Returns dictionary of layer configurations for all model layers.
        """
        return {
            layer: LayerConfig(
                layer_sizes[layer],
                layer,
                input_parameters,
                neuron_models[layer],
                synaptic_adaptation_models[layer],
            )
            for layer, input_parameters in PrimaryVisualCortexModel.layers_input_parameters.items()
        }

    def _init_layer(self, layer: str) -> ModelLayer:
        """
        Initializes one layer of the model.

        :param layer: layer name (`LayerType`).
        :return: Returns initializes layer object.
        """
        return ModelLayer(
            sum(
                self.layer_sizes[layer_name]
                for layer_name, _ in PrimaryVisualCortexModel.layers_input_parameters[
                    layer
                ]
            ),
            self.layers_configs[layer].size,
            layer,
            self.layers_configs[layer].constraint,
            self.layers_configs[layer].input_constraints,
            self.weight_initialization,
            self.layers_configs[layer].neuron_model,
        )

    def _init_model_architecture(self):
        """
        Initializes all model layers and stored them as `nn.ModuleDict` object
        under the keys from `LayerType`.
        """
        self.layers = nn.ModuleDict()

        for layer in PrimaryVisualCortexModel.layers_input_parameters:
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
        synaptic_adaptation_layer_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]],
    ) -> Tuple[List[torch.Tensor], Dict[str, Optional[Tuple[torch.Tensor, ...]]]]:
        """
        Retrieves input tensors of the given time variant from the provided list of all
        possible input tensors. The tensors are selected from the input tensors of the
        provided layer for the specified time step (previous/current). After selection
        it applies synaptic adaptation model on the selected input tensors (if using it).

        :param layer_type: type of the layer to obtain the inputs for. Value from `LayerType`.
        :param time_variant: time variant (previous, current) we want to obtain the values for.
        Value from `TimeStepVariant`.
        :param values_of_given_time: values for the processed time variant. In case of
        the previous time variant they are `hidden_layer`s, in case of current time values
        they are`current_time_values`).
        :param synaptic_adaptation_layer_hidden: hidden states of the synaptic adaptation model.
        :return: Returns list of tensors of the
        """
        # List of all input tensors we want to retrieve.
        time_variant_list = []
        for (
            input_part_layer_name,  # Name of the input layer
            input_part_time_variant,  # Time variant of the input layer
        ) in PrimaryVisualCortexModel.layers_input_parameters[layer_type]:
            # Iterate through inputs of the given layer.
            if time_variant == input_part_time_variant:
                # If the currently processed input layer belongs to given time variant
                # -> optionally apply synaptic adaptation and append it to the list

                if (
                    self.layers_configs[layer_type].synaptic_activation_models
                    is not None
                ):
                    # We want to use synaptic adaptation model to the input values.
                    (
                        modified_input,
                        synaptic_adaptation_layer_hidden[input_part_layer_name],
                    ) = self.layers_configs[layer_type].apply_synaptic_adaptation(
                        input_part_layer_name,
                        values_of_given_time[input_part_layer_name],
                        synaptic_adaptation_layer_hidden[input_part_layer_name],
                    )
                    time_variant_list.append(modified_input)
                else:
                    # We do not want to use synaptic adaptation -> just pass the input value.
                    time_variant_list.append(
                        values_of_given_time[input_part_layer_name]
                    )

        return time_variant_list, synaptic_adaptation_layer_hidden

    def _get_inputs_for_layer(
        self,
        current_time_data: Dict[str, torch.Tensor],
        previous_time_data: Dict[str, torch.Tensor],
        layer: str,
        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """
        Retrieves input tensors for the given layer of the appropriate time step and
        apply the synaptic adaptation module if it is used.

        :param current_time_data: Neuron responses of the current time step.
        :param previous_time_data: Neuron responses of the previous time step.
        :param layer: Layer for which we want to obtain the inputs.
        :param synaptic_adaptation_hidden: Hidden states of the synaptic adaptation model.
        :return: Returns tuple of input tensor of the input layers of the specified layer,
        its recurrent input and hidden states of the synaptic adaptation model. The input
        from layers and recurrent input are processed by synaptic adaptation model if selected.
        """
        current_time_inputs, synaptic_adaptation_hidden[layer] = (
            self._get_list_by_time_variant(
                layer,
                TimeStepVariant.CURRENT.value,
                current_time_data,
                synaptic_adaptation_hidden[layer],
            )
        )  # Inputs of the layer from time (t).

        previous_time_inputs, synaptic_adaptation_hidden[layer] = (
            self._get_list_by_time_variant(
                layer,
                TimeStepVariant.PREVIOUS.value,
                previous_time_data,
                synaptic_adaptation_hidden[layer],
            )
        )  # inputs of the layer from time (t-1) (previous time step)

        # Concatenate all input tensors into one big input tensor (of all input layers).
        layers_input = self._get_layer_input_tensor(
            current_time_inputs, previous_time_inputs
        )

        # Recurrent connection to itself from time (t-1)
        recurrent_input = previous_time_data[layer]
        if self.layers_configs[layer].synaptic_activation_models is not None:
            # Apply synaptic adaptation module if it is used.
            (
                recurrent_input,
                synaptic_adaptation_hidden[layer][layer],
            ) = self.layers_configs[layer].apply_synaptic_adaptation(
                layer,
                previous_time_data[layer],
                synaptic_adaptation_hidden[layer][layer],
            )

        return (
            layers_input,
            recurrent_input,
            synaptic_adaptation_hidden,
        )

    def _perform_model_time_step(
        self,
        current_time_outputs: Dict[str, torch.Tensor],
        hidden_layers: Dict[str, torch.Tensor],
        neuron_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """
        Performs model time step. It progresses the model architecture and
        computes next time step results of each layer of the model.

        :param current_time_outputs: dictionary of input layers of the model (LGN inputs).
        :param hidden_layers: dictionary of all hidden layers values of the model
        from the previous time step (in our example those are all previous outputs).
        :param neuron_hidden: Neuron model hidden states (if `None` then initialize
        them using pytorch default approach).
        :param synaptic_adaptation_hidden: Hidden states of the synaptic adaptation model.
        :return: Returns tuple of dictionary of model predictions for the current time step,
        recurrent network prediction (for evaluation of neuron model functionality) and hidden
        states of neuron models and synaptic adaptation models.
        """
        # We already have LGN inputs (assign them to current time layer outputs).
        recurrent_outputs: Dict[str, Optional[Tuple[torch.Tensor, ...]]] = {}

        for layer in PrimaryVisualCortexModel.layers_input_parameters:
            # Iterate through all output layers of the model.
            # Perform model time step for the currently processed layer.
            # NOTE: It is necessary that these layers are sorted by the
            # processing order in the model.
            (
                layers_input,
                recurrent_input,
                synaptic_adaptation_hidden,
            ) = self._get_inputs_for_layer(
                current_time_outputs,
                hidden_layers,
                layer,
                synaptic_adaptation_hidden,
            )

            (
                current_time_outputs[layer],
                recurrent_outputs[layer],
                neuron_hidden[layer],
            ) = self.layers[layer](
                layers_input,
                recurrent_input.to(nn_model.globals.DEVICE),
                neuron_hidden[
                    layer
                ],  # Hidden steps of the neuron models (needed for RNN neuron models).
            )

            del layers_input, recurrent_input
            torch.cuda.empty_cache()

        return (
            current_time_outputs,
            recurrent_outputs,
            neuron_hidden,
            synaptic_adaptation_hidden,
        )

    def _append_outputs(
        self,
        all_outputs: Dict[str, List[torch.Tensor]],
        time_step_outputs,
    ):
        """
        Appends outputs of each output layer to list of outputs of all time steps.

        :param all_outputs: outputs of layersFalse of all time steps.
        :param time_step_outputs: outputs of current time step.
        """
        for layer, layer_outputs in time_step_outputs.items():
            if layer in PrimaryVisualCortexModel.layers_input_parameters:
                # For each output layer append output of the current time step.
                all_outputs[layer].append(layer_outputs.unsqueeze(1).cpu())

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        hidden_states: Dict[str, torch.Tensor],
        neuron_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ],
        evaluation_train_like_forward: bool = False,
    ) -> Tuple[
        Dict[str, List[torch.Tensor]],
        Dict[str, List[torch.Tensor]],
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """
        Performs forward step of the model iterating through all time steps of the provided
        inputs and targets.

        Training forward step: It initializes hidden states as the previous time step from the
        provided targets (because of that it skips the first time step). This means that the
        model predicts only the next step during training. The hidden states of the neuron
        and synaptic adaptation models are propagated through all time steps (backpropagation
        is performed only for the last time step though).

        Evaluation forward step: It initializes hidden state only for time 0 as the time 0
        target values (so, we start in existing state). Other hidden states are the results
        of model predictions from the previous time step (+ LGN input from `inputs`).

        :param inputs: model current time inputs of size `(batch_size, num_time_steps, num_neurons)`
        or `(batch_size, num_neurons)` for train.
        :param hidden_states: model previous time step inputs of size
        `(batch_size, num_time_steps, num_neurons)` for training mode or
        `(batch, neurons)` for evaluation (we need only first time step).
        :param neuron_hidden: Tuple of hidden states of the neurons (needed for RNN models).
        :param synaptic_adaptation_hidden: Hidden states of the synaptic adaptation model.
        :param evaluation_train_like_forward: Whether to run hidden steps resetting while 
        evaluation run (to test the performance of the model on the prediction of the only one 
        time step - same procedure as in train steps).
        :return: Returns model predictions for all specified time steps.
        """
        # Initialize dictionaries of all model predictions.
        all_recurrent_outputs: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in PrimaryVisualCortexModel.layers_input_parameters
        }
        all_hidden_outputs: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in PrimaryVisualCortexModel.layers_input_parameters
        }

        all_hidden_states = hidden_states
        visible_time_steps = inputs[LayerType.X_ON.value].size(1)
        if self.training:
            # Training mode (only one visible step)
            # We add 1 to iterate through the visible time loop correctly (we want to iterate once).
            visible_time_steps = 1 + 1

        for visible_time in range(1, visible_time_steps):
            # Prediction of the visible time steps
            # (in training only one step, in evaluation all time steps)

            # Placeholder for RNN outputs in case we would like
            # to store RNN outputs for model analysis.
            recurrent_outputs: Dict[str, Optional[Tuple[torch.Tensor, ...]]] = {}

            # Define input layers for the current visible time prediction.
            current_inputs = inputs  # Train mode -> only one input state
            if not self.training:
                # Evaluation mode (with full sequence prediction)
                # -> assign current input state
                # (we predict all visible time steps in one forward step of the model).
                current_inputs = {
                    layer: layer_input[:, visible_time, :]
                    for layer, layer_input in inputs.items()
                }
                if evaluation_train_like_forward:
                    # If evaluation train like -> set hidden state based on previous target.
                    hidden_states = {
                        layer: tensor[:, visible_time - 1, :] 
                        for layer, tensor in all_hidden_states.items()
                    }

            for _ in range(self.num_hidden_time_steps):
                # Perform all hidden time steps.
                (
                    hidden_states,
                    recurrent_outputs,
                    neuron_hidden,
                    synaptic_adaptation_hidden,
                ) = self._perform_model_time_step(
                    current_inputs,
                    hidden_states,
                    neuron_hidden,
                    synaptic_adaptation_hidden,
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
                # Only the visible

                if self.neuron_type in [
                    ModelTypes.RNN_JOINT.value,
                    ModelTypes.DNN_JOINT.value,
                ]:
                    self._append_outputs(all_recurrent_outputs, recurrent_outputs)
                    # TODO: might work for all types

        # Clear caches
        del inputs, hidden_states
        torch.cuda.empty_cache()

        return (
            all_hidden_outputs,
            all_recurrent_outputs,
            neuron_hidden,
            synaptic_adaptation_hidden,
        )
