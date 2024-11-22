#!/usr/bin/env python3
"""
This script defines manipulation with the models and is used to execute 
model training and evaluation.
"""

import os
import pickle
from typing import Tuple, Dict, List, Optional

import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import nn_model.globals
from nn_model.type_variants import (
    LayerType,
    ModelTypes,
    EvaluationFields,
    PredictionTypes,
)
from nn_model.dataset_loader import SparseSpikeDataset, different_times_collate_fn
from nn_model.models import (
    RNNCellModel,
    ConstrainedRNNCell,
)
from nn_model.evaluation_metrics import NormalizedCrossCorrelation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use the second GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class ModelExecuter:
    """
    Class used for execution of training and evaluation steps of the models.
    """

    # Input layer keys (LGN).
    input_layers = [LayerType.X_ON.value, LayerType.X_OFF.value]

    def __init__(self, arguments):
        """
        Initializes dataset loaders, model and evaluation steps.

        :param arguments: command line arguments.
        """
        # Basic arguments
        self.num_epochs = arguments.num_epochs
        self.layer_sizes = nn_model.globals.MODEL_SIZES

        # Dataset Init
        self.train_dataset, self.test_dataset = self._init_datasets(arguments)
        self.train_loader, self.test_loader = self._init_data_loaders()

        # Model Init
        self.model = self._init_model(arguments).to(device=nn_model.globals.device0)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(arguments.learning_rate)

        # Evaluation metric
        self.evaluation_metrics = NormalizedCrossCorrelation()

        # Placeholder for the best evaluation result value.
        self.best_metric = -float("inf")

        # Path of where the best model parameters are stored.
        self.best_model_path = self._init_model_path(arguments)
        if arguments.best_model_dir:
            # In case we specify the path of the best model -> use it
            self.best_model_path = arguments.best_model_dir

        # Directory where the results of the evaluation should be stored.
        self.full_evaluation_directory = arguments.full_evaluation_dir

        # Print experiment setup
        self._print_experiment_info(arguments)

    def _init_model_path(self, arguments) -> str:
        """
        Initializes path where to store the best model parameters.

        By default (if not specified other version in `arguments`) the path is in format:
            ```
            arguments.model_dir/ +
                model_lr-{learning_rate} +
                _{model_type} +
                _residual-{True/False} +
                _neuron_layers-{num_neuron_layers} +
                _neuron-size-{size_neuron_layer} +
                _num-hidden-time-steps-{num_hidden_time_steps} +
                .pth
            ```
            or in case `arguments.model_filename` is defined:
            `arguments.model_dir/arguments.model_filename`


        :param arguments: command line arguments
        :return: Returns the path where the best model parameters should be stored.
        """
        if not arguments.model_filename:
            # Model filename not defined -> use format:
            #       "model_lr_{learning_rate}_{model_type}_residual_{True/False}.pth"
            arguments.model_filename = "".join(
                [
                    f"model-{int(nn_model.globals.SIZE_MULTIPLIER*100)}",
                    f"_step-{nn_model.globals.TIME_STEP}",
                    f"_lr-{str(arguments.learning_rate)}",
                    f"_{arguments.model}",
                    f"_residual-{not arguments.neuron_not_residual}",
                    f"_neuron-layers-{arguments.neuron_num_layers}",
                    f"_neuron-size-{arguments.neuron_layer_size}",
                    f"_num-hidden-time-steps-{arguments.num_hidden_time_steps}",
                    ".pth",
                ]
            )

        # Path where to store the best model parameter.
        return arguments.model_dir + arguments.model_filename

    @staticmethod
    def _split_input_output_layers(
        layer_sizes,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Splits layers sizes to input and output ones.

        :return: Returns tuple of dictionaries of input and output layer sizes.
        """
        input_layers = {key: layer_sizes[key] for key in ModelExecuter.input_layers}
        output_layers = {
            key: value for key, value in layer_sizes.items() if key not in input_layers
        }

        return input_layers, output_layers

    def _init_datasets(
        self, arguments
    ) -> Tuple[SparseSpikeDataset, SparseSpikeDataset]:
        """
        Initializes train and test dataset.

        :param arguments: command line arguments.
        :return: Returns tuple of initialized train and test dataset.
        """
        input_layers, output_layers = ModelExecuter._split_input_output_layers(
            self.layer_sizes
        )

        train_dataset = SparseSpikeDataset(
            arguments.train_dir,
            input_layers,
            output_layers,
            is_test=False,
            model_subset_path=arguments.subset_dir,
        )
        test_dataset = SparseSpikeDataset(
            arguments.test_dir,
            input_layers,
            output_layers,
            is_test=True,
            model_subset_path=arguments.subset_dir,
            experiment_selection_path=arguments.experiment_selection_path,
        )

        return train_dataset, test_dataset

    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Initialized train and test `DataLoader` objects.

        :return: Returns initialized train and test `Dataloader` classes.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=nn_model.globals.train_batch_size,
            shuffle=True,  # Shuffle the training dataset
            collate_fn=different_times_collate_fn,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=nn_model.globals.test_batch_size,
            shuffle=False,  # Load the test dataset always in the same order
            collate_fn=different_times_collate_fn,
        )

        return train_loader, test_loader

    def _get_neuron_model_kwargs(self, arguments) -> Dict:
        """
        Retrieve kwargs of the neuronal model based on the specified model.

        :param arguments: command line arguments.
        :return: Returns dictionary of kwargs for the neuronal model based on the specified model.
        """
        if arguments.model == ModelTypes.SIMPLE.value:
            # Model with simple neuron (no additional neuronal model) -> no kwargs
            return {}
        if arguments.model == ModelTypes.COMPLEX.value:
            # Model with neuron that consist of small multilayer Feed-Forward NN
            # of th same layer sizes in each layer.
            return {
                ModelTypes.COMPLEX.value: {
                    "num_layers": arguments.neuron_num_layers,
                    "layer_size": arguments.neuron_layer_size,
                    "residual": not arguments.neuron_not_residual,
                }
            }

        # Wrongly defined complexity type -> treat as simple neuron.
        print("Wrong complexity, using simple complexity layer.")
        return {}

    def _init_model(self, arguments) -> RNNCellModel:
        """
        Initializes model based on the provided arguments.

        :param arguments: command line arguments containing model setup info.
        :return: Returns initializes model.
        """
        return RNNCellModel(
            self.layer_sizes,
            arguments.num_hidden_time_steps,
            arguments.model,
            neuron_model_kwargs=self._get_neuron_model_kwargs(arguments),
        )

    def _init_criterion(self):
        """
        Initializes model criterion.

        :return: Returns model criterion (loss function).
        """
        return torch.nn.MSELoss()

    def _init_optimizer(self, learning_rate: float):
        """
        Initializes model optimizer.

        :param learning_rate: learning rate of the optimizer.
        :return: Returns used model optimizer (Adam).
        """
        return optim.Adam(self.model.parameters(), lr=learning_rate)

    def _print_experiment_info(self, arguments):
        """
        Prints basic information about the experiment.

        :param arguments: command line arguments.
        """
        print(
            "\n".join(
                [
                    "NEW EXPERIMENT",
                    "---------------------------------",
                    "Running with parameters:",
                    f"Model variant: {arguments.model}",
                    f"Neuron number of layers: {arguments.neuron_num_layers}",
                    f"Neurons layer sizes: {arguments.neuron_layer_size}",
                    f"Neuron use residual connection: {not arguments.neuron_not_residual}",
                    f"Number of hidden time steps: {arguments.num_hidden_time_steps}",
                    f"Batch size: {nn_model.globals.train_batch_size}",
                    f"Learning rate: {arguments.learning_rate}",
                    f"Num epochs: {arguments.num_epochs}",
                ]
            )
        )

    def _get_data(
        self,
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
        slice_ = slice(None) if test else 0

        inputs = {
            layer: input_data[:, slice_, :, :].float().to(nn_model.globals.device0)
            for layer, input_data in inputs.items()
        }
        targets = {
            layer: output_data[
                :, slice_, :, :
            ].float()  # Do not move it to GPU as it is not always used there (only in training).
            for layer, output_data in targets.items()
        }

        return inputs, targets

    # def _compute_loss(
    #     self,
    #     predictions: Dict[str, torch.Tensor],
    #     targets: Dict[str, torch.Tensor],
    # ) -> torch.Tensor:
    #     """
    #     TODO: probably just compute the loss only for one layer itself (makes more sense)
    #     Computes model loss of all model layer predictions.

    #     :param predictions: list of model predictions for each time step (without the first one).
    #     :param targets: tensor of model targets of all time steps
    #     (we want to omit the first one in computation of the loss).
    #     :return: Loss of model predictions concatenated to one vector.
    #     """

    #     all_predictions = (
    #         torch.cat(
    #             [predictions[layer] for layer in targets.keys()],
    #             dim=1,
    #         )
    #         .float()
    #         .cpu()
    #     )
    #     all_targets = torch.cat([target.float() for target in targets.values()], dim=1)

    #     # Compute the loss for all layers at once
    #     loss = self.criterion(all_predictions, all_targets)

    #     return loss

    def _apply_model_constraints(self):
        """
        Applies model constraints on all model layers (excitatory/inhibitory).
        """
        for module in self.model.modules():
            if isinstance(module, ConstrainedRNNCell):
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
            print(
                " ".join(
                    [
                        f"Validation metric improved from {self.best_metric:.4f}",
                        f"to {current_val_metric:.4f}. Saving model...",
                    ]
                )
            )
            self.best_metric = current_val_metric
            torch.save(
                self.model.state_dict(),
                self.best_model_path,
            )

    @staticmethod
    def _move_data_to_cuda(
        data_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Moves given data for all layers to CUDA.

        NOTE: This function is used in training mode while we want to use all targets
        as hidden states. We do it all at once because this operation is very time-consuming.

        :param data_dict: Dictionary of data to be moved to CUDA (for each layer).
        :return: Returns dictionary of given data moved to CUDA.
        """
        return {
            layer: data.clone().to(nn_model.globals.device0)
            for layer, data in data_dict.items()
        }

    def _get_time_step_for_all_layers(
        self,
        time: int,
        dict_tensors: Dict[str, torch.Tensor],
        # self, tensor_slice, dict_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        TODO: maybe make this function more general.
        Retrieves only selected time step from the dictionary of data for the
        layer.

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

    def _retrieve_current_time_step_batch_data(
        self,
        time: int,
        input: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retrieves current time slice of the provided batch data.

        :param time: Current time index.
        :param input: Input batch data.
        :param target: Output batch data.
        :return: Returns tuple of sliced input and output data.
        """
        current_inputs = self._get_time_step_for_all_layers(time, input)
        current_targets = self._get_time_step_for_all_layers(time, target)

        return current_inputs, current_targets

    def _get_train_current_time_data(
        self,
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
        current_hidden_states = self._get_time_step_for_all_layers(
            time - 1,
            all_hidden_states,
        )
        # Retrieve data batch data for the current time step.
        current_inputs, current_targets = self._retrieve_current_time_step_batch_data(
            time, input_batch, target_batch
        )

        return current_inputs, current_targets, current_hidden_states

    def _predict_visible_time_step(
        self, inputs: Dict[str, torch.Tensor], hidden_states: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform model step for 1 visible time step.

        Performs model steps for hidden time steps until the next visible
        time step is reached. At the end selects the last time step prediction
        (next visible time step prediction).

        :param inputs: Inputs of the appropriate time step of the model.
        :param hidden_states: Hidden states of the appropriate time step.
        :return: Returns model prediction of the next visible time step
        (next target/last hidden prediction time step).
        """
        # Get model prediction.
        # Predict all time steps (including the hidden time steps).
        # Our target prediction is the last time step.
        all_predictions, _ = self.model(
            inputs,  # input of time t
            # Hidden states based on the layer (some of them from t,
            # some of them form t-1). The time step is assigned based
            # on the model architecture during the forward function call.
            hidden_states,
        )
        # TODO: check if problem with all_predictions

        return self._get_time_step_for_all_layers(
            # Take time 0 because each prediction predicts data for exactly 1 time step.
            # We just want to get rid of the time dimension using this trick.
            0,
            {layer: predictions[-1] for layer, predictions in all_predictions.items()},
        )  # Take the last time step prediction (target prediction).

    def _optimizer_step(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> float:
        """
        Performs optimizer step for all model layers.

        :param predictions: Predictions of the next visible time step.
        :param targets: Targets of the next visible time step.
        :return: Returns sum of losses of each layer predictions.
        """
        # Loss calculation and backward steps.
        # TODO: Need to check the correctness of this approach and
        # ideally move this to separate function.
        loss_sum = 0.0
        for layer in predictions:
            loss = self.criterion(
                predictions[layer],
                targets[layer],
            )
            loss_sum += loss.item()

            loss.backward()
            self.optimizer.step()

        return loss_sum

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
        for visible_time in range(
            1, time_length
        ):  # We skip the first time step because we do not have initial hidden values for them.
            inputs, targets, hidden_states = self._get_train_current_time_data(
                visible_time, input_batch, target_batch, all_hidden_states
            )

            # Zeroing gradient for each batch of data.
            self.optimizer.zero_grad()

            predictions = self._predict_visible_time_step(inputs, hidden_states)
            time_loss_sum += self._optimizer_step(predictions, targets)

            # Save CUDA memory. Delete not needed variables.
            del (
                hidden_states,
                inputs,
                targets,
                predictions,
            )
            torch.cuda.empty_cache()

            # Apply weight constrains (excitatory/inhibitory) for all the layers.
            self._apply_model_constraints()
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
                input_batch, target_batch = self._get_data(input_batch, target_batch)

                avg_time_loss = self._train_perform_visible_time_step(
                    input_batch, target_batch
                )
                wandb.log({"batch_loss": avg_time_loss})
                epoch_loss_sum += avg_time_loss

            avg_epoch_loss = epoch_loss_sum / len(self.train_loader)
            wandb.log(
                {
                    "epoch_loss": avg_epoch_loss,
                }
            )
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {avg_epoch_loss:.4f}"
            )
            # Run evaluation after each epoch. In case the results are the best
            # -> save model parameters.
            self._update_best_model(epoch, continuous_evaluation_kwargs)

    def _load_best_model(self):
        """
        Load best model weights for final evaluation.

        NOTE: This function changes internal state of `self.model` object.
        """
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        print(
            f"Running final evaluation on model with best CC_NORM value: {self.best_metric:.4f}"
        )

    def _add_trial_predictions_to_list_of_all_predictions(
        self,
        predictions: Dict[str, Dict[str, List[torch.Tensor]]],
        all_predictions: Dict[str, Dict[str, List[torch.Tensor]]],
    ):
        for prediction_type, layers_predictions in predictions.items():
            for layer, layers_prediction in layers_predictions.items():
                current_prediction = torch.cat(layers_prediction, dim=1)
                if (
                    prediction_type == PredictionTypes.RNN_PREDICTION.value
                    and not self.model.return_recurrent_state
                ):
                    current_prediction = torch.zeros(0)

                all_predictions[prediction_type][layer].append(current_prediction)

    def _get_all_trials_predictions(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_trials: int,
    ) -> Dict[
        str, Dict[str, List[torch.Tensor]]
    ]:  # -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
        """
        Computes predictions for all trials (used for evaluation usually).

        :param inputs: dictionary of inputs for each input layer.
        Inputs of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param hidden_states: dictionary of hidden states for each output (hidden) layer
        Of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param num_trials: total number of trials in provided data.
        :return: Returns tuple of all predictions of the model (also hidden steps) and
        predictions before passing information through neuron model.
        """
        # Initialize dictionaries with the keys of all output layers names.
        all_predictions: Dict[str, Dict[str, List[torch.Tensor]]] = {
            prediction_type.value: {
                layer: [] for layer in RNNCellModel.layers_input_parameters
            }
            for prediction_type in list(PredictionTypes)
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

            trial_predictions, trial_rnn_predictions = self.model(
                trial_inputs,
                trial_hidden,
            )

            self._add_trial_predictions_to_list_of_all_predictions(
                {
                    PredictionTypes.FULL_PREDICTION.value: trial_predictions,
                    PredictionTypes.RNN_PREDICTION.value: trial_rnn_predictions,
                },
                all_predictions,
            )

        # return dict_predictions, rnn_predictions
        return all_predictions

    def _predict_for_evaluation(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_trials: int,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
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
        # dict_predictions, rnn_predictions = self._get_all_trials_predictions(
        all_predictions = self._get_all_trials_predictions(inputs, targets, num_trials)
        torch.cuda.empty_cache()

        return_predictions: Dict[str, Dict[str, torch.Tensor]] = {
            prediction_type: {} for prediction_type in all_predictions
        }

        for prediction_type, all_layers_predictions in all_predictions.items():
            if (
                prediction_type == PredictionTypes.RNN_PREDICTION.value
                and not self.model.return_recurrent_state
            ):
                return_predictions[prediction_type] = {}
                continue

            # Stack all predictions into one torch array.
            dict_stacked_predictions = {
                key: torch.stack(value_list, dim=0)
                for key, value_list in all_layers_predictions.items()
            }
            # Reshape the prediction to shape:  `(num_trials, batch_size, time, num_neurons)`
            return_predictions[prediction_type] = {
                layer: predictions.permute(1, 0, 2, 3)
                for layer, predictions in dict_stacked_predictions.items()
            }

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
        ).to(nn_model.globals.device0)
        all_targets = torch.cat([target for target in targets.values()], dim=-1).to(
            nn_model.globals.device0
        )

        # Run the calculate function once on the concatenated tensors.
        cc_norm, cc_abs = self.evaluation_metrics.calculate(
            all_predictions, all_targets
        )

        return cc_norm, cc_abs

    def _save_predictions_batch(
        self,
        batch_index: int,
        all_predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        filename: str = "",
    ):
        """
        Saves averaged predictions and targets across the trials for the provided batch.

        :param batch_index: Index of the batch (just used for naming).
        :param predictions: Dictionary of predictions of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        :param targets: Dictionary of targets of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        :param filename: optional path of the file, if empty string then use default filename.
        Default filename:
            f"{args.full_evaluation_directory}/{args.model_filename}/batch_{batch_index}.pkl"
        """

        if not filename:
            # If the filename is not defined -> use the default one
            # Path: {full_evaluation_directory}/{model_filename}/batch_{batch_index}.pkl
            subdirectory_name = os.path.splitext(
                os.path.basename(self.best_model_path)
            )[0]
            subdirectory_path = os.path.join(
                self.full_evaluation_directory, subdirectory_name
            )
            os.makedirs(subdirectory_path, exist_ok=True)
            filename = os.path.join(subdirectory_path, f"batch_{batch_index}.pkl")

        # Save to a pickle file
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    EvaluationFields.PREDICTIONS.value: all_predictions[
                        PredictionTypes.FULL_PREDICTION.value
                    ],
                    EvaluationFields.TARGETS.value: targets,
                    EvaluationFields.RNN_PREDICTIONS.value: all_predictions[
                        PredictionTypes.RNN_PREDICTION.value
                    ],
                },
                f,
            )

    def _save_full_evaluation(
        self,
        batch_index: int,
        all_predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Saves all evaluation results together with its targets in appropriate format to
        pickle file.

        :param batch_index: ID of the batch used to determine filename (batch ID is part of it).
        :param predictions: Dictionary of predictions of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        :param targets: Dictionary of targets of the batch for all layers.
        Shape of tensor:(batch_size, time, num_neurons)
        """
        self._save_predictions_batch(
            batch_index,
            {
                prediction_type: {
                    layer: torch.mean(
                        prediction, dim=0
                    )  # Trials dimension is the first because we reshape it during prediction step.
                    for layer, prediction in predictions.items()
                }
                for prediction_type, predictions in all_predictions.items()
            },
            {
                layer: torch.mean(
                    target, dim=1
                )  # We want to skip the first time step + the trials dimension is the second.
                for layer, target in targets.items()
            },
            # {
            #     layer: torch.mean(
            #         rnn_prediction, dim=0
            #     )  # Trials dimension is the first because we reshape it during prediction step.
            #     for layer, rnn_prediction in rnn_predictions.items()
            # },
        )

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
            self._load_best_model()

        if save_predictions:
            self.model.switch_to_return_recurrent_state()

        self.model.eval()

        cc_norm_sum = 0.0
        cc_abs_sum = 0.0
        # num_examples = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                if subset_for_evaluation != -1 and i > subset_for_evaluation:
                    # Evaluate only subset of test data.
                    break

                inputs, targets = self._get_data(inputs, targets, test=True)

                # time_length = inputs[LayerType.X_ON.value].size(1)

                # for t in range(1, time_length):
                # predictions, rnn_predictions = self._predict_for_evaluation(
                all_predictions = self._predict_for_evaluation(
                    inputs, targets, inputs[LayerType.X_ON.value].shape[1]
                )

                if save_predictions:
                    self._save_full_evaluation(
                        i, all_predictions, targets
                    )  # predictions, targets, rnn_predictions)

                cc_norm, cc_abs = self.compute_evaluation_score(
                    # Compute evaluation for all time steps except the first step (0-th).
                    {layer: target[:, :, 1:, :] for layer, target in targets.items()},
                    all_predictions[PredictionTypes.FULL_PREDICTION.value],
                )
                cc_norm_sum += cc_norm
                cc_abs_sum += cc_abs

                # Where the magic happens
                wandb.log({"batch_cc_norm": cc_norm})
                wandb.log({"batch_cc_abs": cc_abs})

                if i % print_each_step == 0:
                    print(
                        "".join(
                            [
                                f"Average normalized cross correlation after step {i+1} is: ",
                                f"{cc_norm_sum/(i+1):.4f}",
                                "\n",
                                "Average Pearson's CC is: ",
                                f"{cc_abs_sum/(i+1):.4f}",
                            ]
                        )
                    )

        num_examples = subset_for_evaluation + 1
        if subset_for_evaluation == -1:
            num_examples = len(self.test_loader)

        avg_cc_norm = cc_norm_sum / num_examples
        avg_cc_abs = cc_abs_sum / num_examples
        print(f"Final average normalized cross correlation is: {avg_cc_norm:.4f}")
        print(f"Final average Pearson's CC is: {avg_cc_abs:.4f}")
        wandb.log({"CC_NORM": avg_cc_norm})
        wandb.log({"CC_ABS": avg_cc_abs})

        return avg_cc_norm

    def _prepare_selected_data_for_analysis(
        self, all_data_batches: Dict[str, List[torch.Tensor]]
    ) -> Dict:
        """
        Takes all selected data batches (subset of neurons for analysis) and
        concatenates them to one `np.array`.

        :param all_data_batches: Dictionary of list of all batches of selected data for analysis.
        :return: Returns dictionary of numpy arrays of all data selected for further
        evaluation analysis.
        """
        return {
            layer: torch.cat(all_data_batches[layer], dim=0).cpu().numpy()
            for layer in all_data_batches
        }
