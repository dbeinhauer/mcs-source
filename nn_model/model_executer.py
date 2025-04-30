"""
This script defines manipulation with the models and is used to execute
model training and evaluation.
"""

import re
from typing import Tuple, Dict, List, Optional, Union

import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import nn_model.globals
from nn_model.type_variants import (
    LayerType,
    ModelTypes,
    PredictionTypes,
    OptimizerTypes,
    ModelModulesFields,
)
from nn_model.dataset_loader import SparseSpikeDataset, different_times_collate_fn
from nn_model.models import (
    PrimaryVisualCortexModel,
    ModelLayer,
)
from nn_model.evaluation_metrics import NormalizedCrossCorrelation
from nn_model.evaluation_results_saver import EvaluationResultsSaver
from nn_model.logger import LoggerModel
from nn_model.dictionary_handler import DictionaryHandler


class ModelExecuter:
    """
    Class used for execution of training and evaluation steps of the models.
    """

    def __init__(self, arguments):
        """
        Initializes dataset loaders, model and evaluation steps.

        :param arguments: command line arguments.
        """
        # Basic arguments.
        self.num_epochs = arguments.num_epochs
        self.layer_sizes = nn_model.globals.MODEL_SIZES

        # Dataset initialization.
        self.num_data_workers = arguments.num_data_workers
        self.train_dataset, self.test_dataset = self._init_datasets(arguments)
        self.train_loader, self.test_loader = self._init_data_loaders(arguments)

        # Model initialization.
        self.model = self._init_model(arguments).to(device=nn_model.globals.DEVICE)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(
            arguments.optimizer_type, arguments.learning_rate
        )

        # Gradient clipping upper bound.
        self.gradient_clip = arguments.gradient_clip

        # Number of time steps to run till next optimizer step
        # (truncated backpropagation through time).
        self.num_backpropagation_time_steps = arguments.num_backpropagation_time_steps

        # Evaluation metric.
        self.evaluation_metrics = NormalizedCrossCorrelation()

        # Placeholder for the best evaluation result value.
        self.best_metric = -float("inf")

        # Object that serves for storing the paths used in model execution
        # and stores the evaluation results.
        self.evaluation_results_saver = EvaluationResultsSaver(arguments)

        # Logger of the object.
        self.logger = LoggerModel()

    @staticmethod
    def _create_shared_module_kwargs(
        module_type: str, model_type: str, arguments
    ) -> Dict:
        """
        Based on the given parameter prepares kwargs for either a neuron or
        a synaptic adaptation model creation.

        :param module_type: Type of the module to create (either neuron or synaptic adaptation).
        :param model_type: Type of the model.
        :param arguments: Command line arguments containing wanted kwargs.
        :return: Returns dictionary containing kwargs for the selected module.
        """
        # By default initialize the kwargs for the neuron module.
        num_layers = arguments.neuron_num_layers
        layer_size = arguments.neuron_layer_size
        if module_type == ModelModulesFields.SYNAPTIC_ADAPTION_MODULE.value:
            # Initialize kwargs for the synaptic adaptation module.
            layer_size = arguments.synaptic_adaptation_size
            num_layers = arguments.synaptic_adaptation_num_layers
        return {
            module_type: {
                "model_type": model_type,
                "num_layers": num_layers,
                "layer_size": layer_size,
                "residual": arguments.neuron_residual,
                "activation_function": arguments.neuron_activation_function,
                "rnn_variant": arguments.neuron_rnn_variant,
            }
        }

    @staticmethod
    def _get_neuron_model_kwargs(arguments) -> Dict[str, Optional[Dict]]:
        """
        Retrieve kwargs of the neuronal model based on the specified model.

        :param arguments: Command line arguments.
        :return: Returns dictionary with one item with key `"neuron_model"` and
        value is dictionary of kwargs for the neuronal model based on the
        specified model, `None` value if we do not want to use this module.
        """
        if arguments.model == ModelTypes.SIMPLE.value:
            # Model with simple neuron (no additional neuronal model) -> no kwargs
            return {ModelModulesFields.NEURON_MODULE.value: None}
        elif arguments.model in nn_model.globals.DNN_MODELS:
            # Feed-forward DNN model arguments.
            return ModelExecuter._create_shared_module_kwargs(
                ModelModulesFields.NEURON_MODULE.value, arguments.model, arguments
            )
        elif arguments.model in nn_model.globals.RNN_MODELS:
            # RNN model arguments (currently same as feed-forward DNNs).
            return ModelExecuter._create_shared_module_kwargs(
                ModelModulesFields.NEURON_MODULE.value, arguments.model, arguments
            )
        else:

            class NonExistingModelTypeException(Exception):
                """
                Exception class used when wrong model type is selected.
                """

            raise NonExistingModelTypeException("Non-existing model type selected.")

    @staticmethod
    def _get_synaptic_adaptation_model_kwargs(arguments) -> Dict[str, Optional[Dict]]:
        """
        Retrieves kwargs of the synaptic adaptation model.

        NOTE: We use `RNN_JOINT` neuron as a trick to use input size 1.

        :param arguments: Command line arguments.
        :return: Returns dictionary with one item with key `"synaptic_adaptation_model"` and
        value is dictionary of kwargs for the synaptic adaptation model, `None` value if we
        do not want to use this module.
        """
        if (
            arguments.model == ModelTypes.SIMPLE.value
            or not arguments.synaptic_adaptation
        ):
            # Model with simple neuron (no additional neuronal model) -> no kwargs
            return {
                ModelModulesFields.SYNAPTIC_ADAPTION_MODULE.value: None,
                "only_lgn": False,
            }

        synaptic_adaptation_kwargs = ModelExecuter._create_shared_module_kwargs(
            ModelModulesFields.SYNAPTIC_ADAPTION_MODULE.value,
            ModelTypes.RNN_JOINT.value,  # Trick, we want to use input size 1 (so we need to use joint neuron type).
            arguments,
        )
        # Add info whether we want to use synaptic adaptation module on all the layers or only on the LGN inputs.
        # Put it outside the "model type" field to not pass it to the synaptic module init (it is outside module init.).
        synaptic_adaptation_kwargs["only_lgn"] = arguments.synaptic_adaptation_only_lgn

        return synaptic_adaptation_kwargs

    @staticmethod
    def _slice_and_convert_to_float(
        data: Dict[str, torch.Tensor], slice_: Union[int, slice]
    ) -> Dict[str, torch.Tensor]:
        """
        Slices the provided data in the trials interval and converts them to float.

        NOTE: This function is used while calling `_get_data` method.

        :param data: All layers data.
        :param slice_: Slice for the trials dimension (to get rid of it in case of train step).
        :return: Returns sliced data that were converted to float.
        """
        return DictionaryHandler.apply_methods_on_dictionary(
            DictionaryHandler.apply_functions_on_dictionary(
                data, [(DictionaryHandler.slice_given_axes, ({1: slice_},))]
            ),
            [("float",)],
        )

    @staticmethod
    def _get_data(
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        test: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Converts loaded data from corresponding `DataLoader` child class
        to proper format further used in training/evaluation.

        It converts the loaded inputs and targets to dictionary of
        layer identifier and its corresponding values.
        For training dataset it removes (uses 0-th) trial dimension (not used there).
        For testing dataset it keeps the trials dimension.

        :param inputs: input slice loaded from `DataLoader` class object.
        :param targets: target slice loaded from `DataLoader` class object.
        :param test: flag whether the loaded data are from evaluation (test) set,
        otherwise `False` - train set.
        :return: Returns tuple of dictionaries of inputs and targets further used
        for either model training or evaluation. In case of training it removes the
        trials dimension (it is not used).
        """
        # Define what part of trials dimension we want to take.
        # Take `0` for train or all trials `slice(None) == :` for test.
        slice_: Union[int, slice] = slice(None) if test else 0

        inputs = {
            layer: input_data[:, slice_, :, :].float().to(nn_model.globals.DEVICE)
            for layer, input_data in inputs.items()
        }
        targets = {
            layer: output_data[
                :, slice_, :, :
            ].float()  # Do not move it to GPU as it is not always used there (only in training).
            for layer, output_data in targets.items()
        }

        return inputs, targets

    @staticmethod
    def _move_data_to_cuda(
        data_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Moves given data for all layers to CUDA.

        NOTE: This function is used in training mode while we want to use all targets
        as hidden states. We do it all at once because this operation is very time-consuming.

        :param data_dict: Dictionary of data to be moved to CUDA (for each layer).
        :return: Returns dictionary of given data moved to CUDA.
        """
        return {
            layer: data.clone().to(nn_model.globals.DEVICE)
            for layer, data in data_dict.items()
        }

    @staticmethod
    def _get_time_step_for_all_layers(
        time: int,
        dict_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieves specified time step from the provided tensor.

        NOTE: It expects that the tensors in the dictionaries have the corresponding
        values defined in the slicing dimension. It expects that there is a 3D
        tensor in the input and the time step dimension is second dimension.

        :param time: Time step we want to retrieve.
        :param dict_tensors: Dictionary of tensors we want to slice from.
        :return: Returns dictionary of the given slices of all tensors from the input
        dictionary.
        """
        # Assign previous time step from targets.
        return {layer: tensor[:, time, :] for layer, tensor in dict_tensors.items()}

    @staticmethod
    def _retrieve_current_time_step_batch_data(
        time: int,
        input_data: Dict[str, torch.Tensor],
        target_data: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieves current time slice of the provided batch data.

        :param time: Current time index.
        :param input_data: Input batch data.
        :param target_data: Output batch data.
        :return: Returns tuple of sliced input and output data.
        """
        current_inputs = ModelExecuter._get_time_step_for_all_layers(time, input_data)
        current_targets = ModelExecuter._get_time_step_for_all_layers(time, target_data)

        return current_inputs, current_targets

    @staticmethod
    def _get_train_current_time_data(
        time: int,
        input_batch: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
        all_hidden_states: Dict[str, torch.Tensor],
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Retrieves data that are needed in train step for current time step.

        We want to take current time step for inputs and targets and
        previous time step for the hidden states.

        :param time: Current time step index.
        :param input_batch: Batch of input layers data.
        :param target_batch: Batch of target layers data.
        :param all_hidden_states: All hidden states in CUDA
        (we want to take the previous time step for these).
        :return: Returns slices of the provided data for the next training time step.
        """
        # Initialize hidden states as the previous time step targets.
        current_hidden_states = ModelExecuter._get_time_step_for_all_layers(
            time - 1,
            all_hidden_states,
        )
        # Retrieve data batch data for the current time step.
        current_inputs, current_targets = (
            ModelExecuter._retrieve_current_time_step_batch_data(
                time, input_batch, target_batch
            )
        )

        return current_inputs, current_targets, current_hidden_states

    @staticmethod
    def _detach_hidden_states(
        hidden_state: Optional[Tuple[torch.Tensor, ...]],
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Detaches hidden states of the recurrent modules from the gradient graph.

        :param hidden_state: Hidden states of the module to be detached. Can be `None` in case
        we do not use hidden states (feed-forward modules).
        :return: Returns detached hidden states.
        """
        if hidden_state is None:
            # No hidden states (feed-forward modules) -> skip
            return None
        elif isinstance(hidden_state, tuple):
            # Tuple states (typically LSTM).
            return tuple(state.detach() for state in hidden_state)
        elif isinstance(hidden_state, torch.Tensor):
            # Only one hidden state (typically classical RNN, GRU).
            return hidden_state.detach()
        else:
            raise TypeError("Unsupported type for hidden_state")

    @staticmethod
    def _detach_all_hidden_states(
        neuron_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ],
    ) -> Tuple[
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """
        Detaches all hidden states of the recurrent modules using in the model.

        :param neuron_hidden: Hidden states of the neuron model.
        :param synaptic_adaptation_hidden: Hidden states of the synaptic adaptation model.
        :return: Returns tuple of detached neuron model and synaptic adaptation hidden states.
        """
        neuron_hidden = {
            layer: ModelExecuter._detach_hidden_states(hidden)
            for layer, hidden in neuron_hidden.items()
        }
        synaptic_adaptation_hidden = {
            layer: {
                input_layer: ModelExecuter._detach_hidden_states(hidden)
                for input_layer, hidden in input_hidden_states.items()
            }
            for layer, input_hidden_states in synaptic_adaptation_hidden.items()
        }

        return neuron_hidden, synaptic_adaptation_hidden

    @staticmethod
    def _init_modules_hidden_states() -> Tuple[
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """
        Initializes hidden states of the neuron and synaptic adaptation modules.

        :return: Initialized hidden states of the neuron and synaptic adaptation modules
        (all tensors are `None`).
        """
        # Hidden states of the neuron.
        neuron_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]] = {
            layer: None for layer in PrimaryVisualCortexModel.layers_input_parameters
        }

        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ] = {
            layer: {
                input_layer: None
                for input_layer, _ in layer_inputs
                + [(layer, "")]  # Add the self-recurrent connection.
            }
            for layer, layer_inputs in PrimaryVisualCortexModel.layers_input_parameters.items()
        }

        return neuron_hidden, synaptic_adaptation_hidden

    @staticmethod
    def _add_trial_predictions_to_list_of_all_predictions(
        trial_predictions: Dict[PredictionTypes, Dict[str, List[torch.Tensor]]],
        all_predictions: Dict[PredictionTypes, Dict[str, List[torch.Tensor]]],
        save_recurrent_state: bool,
    ):
        """
        Sort all the predictions for the trial and split them to corresponding
        lists of all the predictions or do not save them based on the evaluation status.

        :param trial_predictions: Predictions for the trial (list of predictions for all
        visible time steps).
        :param all_predictions: All predictions for all the trials.
        :param save_recurrent_state: Flag whether we want to also save predictions of the RNNs.
        """
        for prediction_type, layers_predictions in trial_predictions.items():
            # Iterate all predictions types (full prediction, RNN predictions...)
            for layer, layers_prediction in layers_predictions.items():
                # Iterate all layers predictions.
                current_prediction = torch.zeros(
                    0
                )  # RNN default (do not save predictions).
                if (
                    prediction_type
                    in [
                        PredictionTypes.FULL_PREDICTION,
                        PredictionTypes.TRAIN_LIKE_PREDICTION,
                    ]
                    or save_recurrent_state
                ):
                    # In case we are doing full prediction or we specified we want RNN linearity
                    # values (of layers) -> store them.
                    if prediction_type == PredictionTypes.RNN_PREDICTION:
                        # TODO: currently skipping this step as it does not work correctly.
                        continue

                    current_prediction = torch.cat(layers_prediction, dim=1)

                all_predictions[prediction_type][layer].append(current_prediction)

    @staticmethod
    def _prepare_evaluation_predictions_for_return(
        all_layers_predictions: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Takes all the predictions of given type (full predictions, rnn predictions etc.) and
        does proper modifications to convert them to proper output format. It stacks all the
        trials predictions to one tensor and permutes them to shape
        `(batch_size, num_trials, time, num_neurons)`.

        :param all_layers_predictions: All predictions for all trials in form of list.
        :return: Returns converted predictions to one big tensor of shape
        `(batch_size, num_trials, time, num_neurons)`.
        """
        # Stack all trials predictions into one torch array.
        dict_stacked_predictions = {
            key: torch.stack(value_list, dim=0)
            for key, value_list in all_layers_predictions.items()
        }
        # Reshape the prediction to shape:  `(batch_size, num_trials, time, num_neurons)`
        return {
            layer: predictions.permute(1, 0, 2, 3)
            for layer, predictions in dict_stacked_predictions.items()
        }

    def _init_datasets(
        self, arguments
    ) -> Tuple[SparseSpikeDataset, SparseSpikeDataset]:
        """
        Initializes train and test dataset.

        :param arguments: command line arguments.
        :return: Returns tuple of initialized train and test dataset.
        """
        input_layers, output_layers = DictionaryHandler.split_input_output_layers(
            self.layer_sizes
        )

        train_dataset = SparseSpikeDataset(
            arguments.train_dir,
            input_layers,
            output_layers,
            is_test=False,
            model_subset_path=arguments.subset_dir,
            dataset_subset_ratio=arguments.train_subset,  # Subset of train dataset in case of model analysis.
        )
        test_dataset = SparseSpikeDataset(
            arguments.test_dir,
            input_layers,
            output_layers,
            is_test=True,
            model_subset_path=arguments.subset_dir,
        )

        return train_dataset, test_dataset

    def _init_data_loaders(self, arguments) -> Tuple[DataLoader, DataLoader]:
        """
        Initialized train and test `DataLoader` objects.

        :return: Returns initialized train and test `Dataloader` classes.
        """
        workers_enabled = self.num_data_workers > 0
        kwargs = {
            "collate_fn": different_times_collate_fn,
            "num_workers": self.num_data_workers,  # number of workers which will supply data to GPU
            "pin_memory": workers_enabled,  # speed up data transfer to GPU
            "prefetch_factor": (
                4 if workers_enabled else None
            ),  # try to always have 4 samples ready for the GPU
            "persistent_workers": workers_enabled,  # keep the worker threads alive
        }
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=arguments.train_batch_size,
            shuffle=True,  # Shuffle the training dataset.
            **kwargs,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=nn_model.globals.TEST_BATCH_SIZE,
            shuffle=False,  # Load the test dataset always in the same order.
            **kwargs,
        )

        return train_loader, test_loader

    def _init_model(self, arguments) -> PrimaryVisualCortexModel:
        """
        Initializes the model based on the provided arguments.

        :param arguments: command line arguments containing model setup info.
        :return: Returns initializes model.
        """
        return PrimaryVisualCortexModel(
            self.layer_sizes,
            arguments.num_hidden_time_steps,
            arguments.model,
            arguments.weight_initialization,
            model_modules_kwargs={
                **ModelExecuter._get_neuron_model_kwargs(arguments),
                **ModelExecuter._get_synaptic_adaptation_model_kwargs(arguments),
            },  # Pass kwargs to generate neuron and synaptic adaptation modules.
        )

    def _init_criterion(self):
        """
        Initializes model criterion.

        :return: Returns model criterion (loss function).
        """
        return torch.nn.MSELoss()

    def _init_exc_inh_specific_learning_rate(self, learning_rate):
        """
        Splits the model parameters to two groups. One corresponding to inhibitory layer
        values and one corresponding to excitatory layers. Inhibitory layer parameters
        should use 4 times smaller learning rate than excitatory
        (because of the properties of the model)

        NOTE: This function is used in case we select `OptimizerTypes.EXC_INH_SPECIFIC`.

        :param learning_rate: Base learning rate that should be applied for excitatory
        layers (inhibitory should have learning rate 4 times smaller).
        :return: Returns group of parameters for the optimizer.
        """

        # Identify inhibitory layers to apply a 4x lower learning rate on.
        selected_params = [
            self.model.layers.V1_Exc_L23.rnn_cell.weights_ih_inh.weight,
            self.model.layers.V1_Inh_L23.rnn_cell.weights_ih_inh.weight,
            self.model.layers.V1_Inh_L23.rnn_cell.weights_hh.weight,
            self.model.layers.V1_Exc_L4.rnn_cell.weights_ih_inh.weight,
            self.model.layers.V1_Inh_L4.rnn_cell.weights_ih_inh.weight,
            self.model.layers.V1_Inh_L4.rnn_cell.weights_hh.weight,
        ]

        selected_param_ids = {id(p) for p in selected_params}

        # Create parameter groups
        param_groups = [
            # Parameters with lower learning rate (inhibitory).
            {"params": selected_params, "lr": learning_rate / 4},
            # All other parameters (excitatory).
            {
                "params": [
                    p
                    for p in self.model.parameters()
                    if id(p) not in selected_param_ids
                ],
                "lr": learning_rate,
            },
        ]

        return param_groups

    def _init_optimizer(self, optimizer_type: str, learning_rate: float):
        """
        Initializes model optimizer.

        :param learning_rate: learning rate of the optimizer.
        :return: Returns used model optimizer (Adam).
        """
        # By default same learning rate for all layers.
        param_groups = self.model.parameters()

        if optimizer_type == OptimizerTypes.EXC_INH_SPECIFIC.value:
            # Apply exc/inh specific learning rate.
            param_groups = self._init_exc_inh_specific_learning_rate(learning_rate)

        return optim.Adam(param_groups, lr=learning_rate)

    def _apply_model_constraints(self):
        """
        Applies model constraints on all model layers (excitatory/inhibitory).
        """
        for module in self.model.modules():
            if isinstance(module, ModelLayer):
                module.apply_constraints()

    def _epoch_evaluation_step(
        self,
        epoch: int,
        epoch_offset: int = 1,
        evaluation_subset_size: int = 10,
    ) -> float:
        """
        Runs continuous evaluation steps in selected training epochs on
        selected subset of test examples.

        :param epoch: current epoch number.
        :param epoch_offset: after how many epochs perform evaluation. If `-1` then never run.
        :param evaluation_subset_size: how many batches use for evaluation.
        If `-1` then all test dataset.
        :return: Average correlation across evaluated subset.
        """
        if epoch_offset != -1 and epoch % epoch_offset == 0:
            # Do continuous evaluation after this step.
            avg_correlation = self.evaluation(
                subset_for_evaluation=evaluation_subset_size, final_evaluation=False
            )
            self.model.train()
            return avg_correlation

        return 0.0

    def _update_best_model(
        self, epoch: int, continuous_evaluation_kwargs: Optional[Dict]
    ):
        """
        Runs few evaluation steps and compares the result of evaluation metric with currently
        the best ones. In case it is better, it updates the best model parameters
        with the current ones (saves them to given file).

        :param epoch: Number of current epoch. Used in case we want to run evaluation only
        on the specified epochs.
        :param continuous_evaluation_kwargs: Kwargs of continuous evaluation run.
        """
        if continuous_evaluation_kwargs is None:
            # In case no continuous evaluation kwargs were provided -> make them empty dictionary.
            continuous_evaluation_kwargs = {}

        # Perform few evaluation steps for training check in specified epochs.
        current_val_metric = self._epoch_evaluation_step(
            epoch,
            **continuous_evaluation_kwargs,
        )

        # Check if there is an improvement -> if so, update the best model.
        if current_val_metric > self.best_metric:
            self.logger.print_best_model_update(self.best_metric, current_val_metric)
            self.best_metric = current_val_metric
            torch.save(
                self.model.state_dict(),
                self.evaluation_results_saver.best_model_path,
            )

    def _model_forward_step(
        self,
        inputs: Dict[str, torch.Tensor],
        hidden_states: Dict[str, torch.Tensor],
        neuron_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ],
        evaluation_train_like_forward: bool = False,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """_summary_

        :param inputs: _description_
        :param hidden_states: _description_
        :param neuron_hidden: _description_
        :param synaptic_adaptation_hidden: _description_
        :param evaluation_train_like_forward: Whether to run hidden steps resetting while
        evaluation run (to test the performance of the model on the prediction of the only one
        time step - same procedure as in train steps).
        :return: _description_
        """
        all_predictions, _, neuron_hidden, synaptic_adaptation_hidden = self.model(
            inputs,  # input of time t
            # Hidden states based on the layer (some of them from t, some of them form t-1).
            # Time step is assigned based on model architecture during the forward function call.
            hidden_states,
            neuron_hidden,
            synaptic_adaptation_hidden,
            evaluation_train_like_forward=evaluation_train_like_forward,
        )

        # Take last time step predictions (those are the visible time steps predictions).
        predictions = self._get_time_step_for_all_layers(
            # Take time 0 because each prediction predicts data for exactly 1 time step.
            # We just want to get rid of the time dimension using this trick.
            0,
            {layer: predictions[-1] for layer, predictions in all_predictions.items()},
        )  # Take the last time step prediction (target prediction).

        return predictions, neuron_hidden, synaptic_adaptation_hidden

    def _calculate_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        total_loss = torch.zeros(1)
        for layer in predictions:
            total_loss += self.criterion(
                predictions[layer],
                targets[layer],
            )

        return total_loss

    def _optimizer_step(
        self,
        neuron_hidden: Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        synaptic_adaptation_hidden: Dict[
            str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]
        ],
    ) -> Tuple[
        Dict[str, Optional[Tuple[torch.Tensor, ...]]],
        Dict[str, Dict[str, Optional[Tuple[torch.Tensor, ...]]]],
    ]:
        """
        Performs optimizer step and all necessary operations, gradient clipping,
        detaches neuron and synaptic adaptation hidden steps and applies model
        weight constraints.

        :param neuron_hidden: Hidden states of the neuron model.
        :param synaptic_adaptation_hidden: Hidden states of the synaptic adaptation model.
        :return: Returns tuple of updated and detached neuron module hidden states
        and synaptic adaptation module hidden states.
        """
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        # Detach hidden states from the gradient graph.
        neuron_hidden, synaptic_adaptation_hidden = (
            ModelExecuter._detach_all_hidden_states(
                neuron_hidden, synaptic_adaptation_hidden
            )
        )

        # Apply weight constrains (excitatory/inhibitory) for all the layers.
        self._apply_model_constraints()

        return neuron_hidden, synaptic_adaptation_hidden

    def _train_perform_visible_time_step(
        self,
        input_batch: Dict[str, torch.Tensor],
        target_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Performs train steps for all visible time steps.

        :param input_batch: Input layers batch of data.
        :param target_batch: Output layers batch of data.
        :return: Returns average loss value through all predicted time steps.
        """
        # Determine time length of the current batch of data.
        time_length = input_batch[LayerType.X_ON.value].size(1)

        # Move all targets to CUDA - we will use them as hidden states during
        # training. Moving them all at once significantly reduces time complexity.
        all_hidden_states = ModelExecuter._move_data_to_cuda(target_batch)
        time_loss_sum = 0.0

        # Hidden states of the neuron.
        neuron_hidden, synaptic_adaptation_hidden = (
            ModelExecuter._init_modules_hidden_states()
        )
        accumulated_loss = 0

        for visible_time in range(
            1, time_length
        ):  # We skip the first time step because we do not have initial hidden values for them.
            # Get data for the current time step.
            inputs, targets, hidden_states = ModelExecuter._get_train_current_time_data(
                visible_time, input_batch, target_batch, all_hidden_states
            )

            # Perform model forward step.
            predictions, neuron_hidden, synaptic_adaptation_hidden = (
                self._model_forward_step(
                    inputs, hidden_states, neuron_hidden, synaptic_adaptation_hidden
                )
            )

            # Calculate time step loss.
            accumulated_loss += self._calculate_loss(predictions, targets)

            if (
                visible_time % self.num_backpropagation_time_steps == 0
                or visible_time == time_length - 1
            ):
                # Perform truncated backpropagation through time.
                # Perform optimizer step only after given number of time steps.

                # Backward step:
                accumulated_loss.backward()

                # Optimizer step:
                neuron_hidden, synaptic_adaptation_hidden = self._optimizer_step(
                    neuron_hidden, synaptic_adaptation_hidden
                )

                # Save loss for logs and reset counter.
                time_loss_sum += accumulated_loss.item()
                accumulated_loss = 0

                torch.cuda.empty_cache()

        del all_hidden_states
        torch.cuda.empty_cache()

        # Return average loss in each time step.
        return time_loss_sum / (time_length - 1)

    def train(
        self,
        continuous_evaluation_kwargs: Optional[Dict] = None,
        debugging_stop_index: int = -1,
    ):
        """
        Runs all model training steps.

        The training works as follows:
            1. It skips first time step (we do not have hidden states).
            2. Hidden states are targets from the previous step.
                - in the training step it makes sense (we want to learn only the next step)
                - it is possibility to have hidden time steps (we do not have targets for them)
                    - we learn "hidden" dynamics of the model
                - we do not reset hidden states of the shared modules (neuron, synaptic adaptation)
                as we do not know their real values (we want the model to learn them)
                    - we detach these states from the gradient graph after optimizer step
                        - we want to learn only based on the last time step
            3. Prediction of the model is the following (current) hidden state (output layer).

        :param continuous_evaluation_kwargs: kwargs for continuous evaluation setup
        (see `self._epoch_evaluation_step` for more information).
        :param debugging_stop_step: number of steps (batches) to be performed in debugging
        run of the training. If `-1` then train on all provided data.
        """
        # We want weights and biases library to log the training procedure.
        wandb.watch(
            self.model,
            self.criterion,
        )
        self.model.train()
        for epoch in range(self.num_epochs):
            # Iterate through all epochs.
            epoch_loss_sum = 0.0
            for i, (input_batch, target_batch) in enumerate(tqdm(self.train_loader)):
                if debugging_stop_index != -1 and i > debugging_stop_index:
                    # Train only for few batches in case of debugging.
                    break

                # Retrieve the batch of data.
                input_batch, target_batch = ModelExecuter._get_data(
                    input_batch, target_batch
                )

                # Perform optimizer steps for all visible time steps.
                avg_time_loss = self._train_perform_visible_time_step(
                    input_batch, target_batch
                )
                wandb.log({"batch_loss": avg_time_loss})
                epoch_loss_sum += avg_time_loss

            # Compute average loss for the whole epoch.
            avg_epoch_loss = epoch_loss_sum / len(self.train_loader)
            self.logger.print_epoch_loss(epoch + 1, self.num_epochs, avg_epoch_loss)
            self._update_best_model(epoch, continuous_evaluation_kwargs)

    def _load_best_model(self):
        """
        Load best model weights for final evaluation.

        NOTE: This function changes internal state of `self.model` object.
        """
        self.model.load_state_dict(
            torch.load(self.evaluation_results_saver.best_model_path, weights_only=True)
        )
        self.logger.print_best_model_evaluation(self.best_metric)

    def _get_all_trials_predictions(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_trials: int,
    ) -> Dict[PredictionTypes, Dict[str, List[torch.Tensor]]]:
        """
        Computes predictions for all trials (used for evaluation usually).

        :param inputs: dictionary of inputs for each input layer.
        Inputs of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param hidden_states: dictionary of hidden states for each output (hidden) layer
        Of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param num_trials: total number of trials in provided data.
        :return: Returns tuple of all predictions of the model (also hidden steps) and
        predictions before passing information through neuron model (RNN predictions).
        """
        # Initialize dictionaries with the keys of all output layers names.
        all_predictions: Dict[PredictionTypes, Dict[str, List[torch.Tensor]]] = {
            prediction_type: {
                layer: [] for layer in PrimaryVisualCortexModel.layers_input_parameters
            }
            for prediction_type in PredictionTypes
        }

        # Get predictions for each trial.
        for trial in range(num_trials):
            trial_inputs = {
                layer: layer_input[:, trial, :, :]
                for layer, layer_input in inputs.items()
            }
            trial_hidden = ModelExecuter._move_data_to_cuda(
                {
                    layer: layer_hidden[:, trial, 0, :].clone()
                    # Pass slices of targets in time 0 (starting hidden step,
                    # in evaluation we do not want to reset hidden states).
                    for layer, layer_hidden in targets.items()
                }
            )

            # Initialize hidden states of the neuron and synaptic adaptation modules.
            neuron_hidden, synaptic_adaptation_hidden = (
                ModelExecuter._init_modules_hidden_states()
            )

            # Get all visible time steps predictions of the model.
            trial_predictions, trial_rnn_predictions, _, _ = self.model(
                trial_inputs,
                trial_hidden,
                neuron_hidden,
                synaptic_adaptation_hidden,
                evaluation_train_like_forward=False,
            )

            # Teacher-forced evaluation results.
            neuron_hidden, synaptic_adaptation_hidden = (
                ModelExecuter._init_modules_hidden_states()
            )
            trial_all_hidden = ModelExecuter._move_data_to_cuda(
                {
                    layer: layer_input[:, trial, :, :]
                    for layer, layer_input in targets.items()
                }
            )
            trial_train_like_predictions, _, _, _ = self.model(
                trial_inputs,
                trial_all_hidden,
                neuron_hidden,
                synaptic_adaptation_hidden,
                evaluation_train_like_forward=True,
            )

            # Accumulate all trials predictions.
            ModelExecuter._add_trial_predictions_to_list_of_all_predictions(
                {
                    PredictionTypes.FULL_PREDICTION: trial_predictions,
                    PredictionTypes.TRAIN_LIKE_PREDICTION: trial_train_like_predictions,
                    PredictionTypes.RNN_PREDICTION: trial_rnn_predictions,
                },
                all_predictions,
                self.model.return_recurrent_state,
            )

        return all_predictions

    def _predict_for_evaluation(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_trials: int,
    ) -> Dict[PredictionTypes, Dict[str, torch.Tensor]]:
        """
        Performs prediction of the model for each trial. Then stacks the predictions
        into one tensors of size `(num_trials, batch_size, time, num_neurons)` and
        stores them into dictionary of predictions for each layer.

        :param inputs: Inputs for each time steps and each trial.
        Of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param targets: Targets for each time steps and each trial
        Of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param num_trials: number of trials.
        :return: Returns dictionary of model predictions for all trials.
        """
        # Get predictions for all trials.
        all_predictions = self._get_all_trials_predictions(inputs, targets, num_trials)
        torch.cuda.empty_cache()

        # Initialize dictionary of the predictions.
        return_predictions: Dict[str, Dict[str, torch.Tensor]] = {
            prediction_type: {} for prediction_type in all_predictions
        }

        for prediction_type, all_layers_predictions in all_predictions.items():
            # Iterate all prediction types
            if (
                prediction_type == PredictionTypes.RNN_PREDICTION
                and not self.model.return_recurrent_state
            ):
                # In case we do not want to save RNN predictions -> skip them
                return_predictions[prediction_type] = {}
                continue

            if prediction_type == PredictionTypes.RNN_PREDICTION:
                # TODO: currently skipping RNN linearity results storage as it is broken.
                continue

            return_predictions[prediction_type] = (
                ModelExecuter._prepare_evaluation_predictions_for_return(
                    all_layers_predictions
                )
            )

        return return_predictions

    def compute_evaluation_score(
        self, targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Computes evaluation score between vectors of all prediction
        and all target layers.

        Between two vectors that were created by concatenating all predictions and
        all targets for all output layers.

        :param targets: dictionary of targets for all layers.
        :param predictions: dictionary of predictions for all layers.
        :return: Returns tuple of evaluation score (CC_NORM) and Pearson's CC of the predictions.
        """
        # Concatenate predictions and targets across all layers.
        all_predictions = torch.cat(
            [prediction for prediction in predictions.values()],
            dim=-1,
        ).to(nn_model.globals.DEVICE)
        all_targets = torch.cat([target for target in targets.values()], dim=-1).to(
            nn_model.globals.DEVICE
        )

        # Run the calculate function once on the concatenated tensors.
        cc_norm, cc_abs = self.evaluation_metrics.calculate(
            all_predictions, all_targets
        )

        return cc_norm, cc_abs

    def evaluation(
        self,
        subset_for_evaluation: int = -1,
        print_each_step: int = 10,
        final_evaluation: bool = True,
        save_predictions: bool = False,
    ) -> float:
        """
        Performs model evaluation.

        For each example:
            1. it initializes the first hidden states with targets in time step 0
            (to start with reasonable state).
            2. After that it lets the model predict all remaining steps.
            3. After each prediction it computes the evaluation score (CC_NORM).
            4. Finally prints average evaluation score for all test examples.

        :param subset_for_evaluation: Evaluate only subset of all test batches.
        If `-1` evaluate whole test dataset.
        :param print_each_step: After how many test batches print current evaluation results.
        :param final_evaluation: Flag whether we want to run final evaluation
        (the best model on all testing dataset).
        :param save_predictions: Flag whether we want to save all model predictions and targets
        averaged through the trials.
        :return: Average cross correlation along all tried examples.
        """
        if final_evaluation:
            # We are doing final evaluation (we want to use the best model).
            self._load_best_model()

        if save_predictions:
            # We want to save the predictions for further analysis.
            self.model.switch_to_return_recurrent_state()

        self.model.eval()

        cc_norm_sum = 0.0
        cc_abs_sum = 0.0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                if subset_for_evaluation != -1 and i > subset_for_evaluation:
                    # Evaluate only subset of test data.
                    break

                inputs, targets = ModelExecuter._get_data(inputs, targets, test=True)

                # Get predictions for all the trials.
                all_predictions = self._predict_for_evaluation(
                    inputs, targets, inputs[LayerType.X_ON.value].shape[1]
                )

                if save_predictions:
                    # We want to save the predictions for further analysis.
                    self.evaluation_results_saver.save_full_evaluation(
                        i, all_predictions, targets
                    )

                cc_norm, cc_abs = self.compute_evaluation_score(
                    # Compute evaluation for all time steps except the first step (0-th).
                    {layer: target[:, :, 1:, :] for layer, target in targets.items()},
                    all_predictions[PredictionTypes.FULL_PREDICTION],
                )
                cc_norm_sum += cc_norm
                cc_abs_sum += cc_abs

                # Logging of the evaluation results.
                self.logger.wandb_batch_evaluation_logs(cc_norm, cc_abs)
                if i % print_each_step == 0:
                    self.logger.print_current_evaluation_status(
                        i + 1, cc_norm_sum, cc_abs_sum
                    )

        # Decide what was the total number of examples during evaluation.
        num_examples = subset_for_evaluation + 1
        if subset_for_evaluation == -1:
            num_examples = len(self.test_loader)

        # Average evaluation metrics calculation
        avg_cc_norm = cc_norm_sum / num_examples
        avg_cc_abs = cc_abs_sum / num_examples
        self.logger.print_final_evaluation_results(avg_cc_norm, avg_cc_abs)

        return avg_cc_norm

    def evaluate_neuron_models(
        self, start: float = -1.0, end: float = 1.0, step: float = 0.001
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Applies neuron complexity on all layers on given range of input data on the model
        with the best model performance based on the evaluation metric.

        NOTE: This function is used mainly to inspect the behavior of the DNN neuron model.

        :param start: Start of input interval.
        :param end: End of input interval.
        :param step: Step size used to generate input interval.
        :return: Returns dictionary of input and output for each layer DNN neuron module.
        """
        self._load_best_model()
        return self.model.apply_neuron_complexity_on_all_layers(start, end, step)
