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
from nn_model.type_variants import LayerType, ModelTypes
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

    # Default kwargs for continuous evaluation (evaluation after training steps).
    continuous_evaluation_kwargs = {
        "epoch_offset": 1,
        "evaluation_subset_size": 10,
    }

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

        self.best_model_path = self._init_model_path(arguments)
        self.full_evaluation_directory = arguments.full_evaluation_dir

        # Selected neurons for evaluation analysis
        self.neuron_selection = self._load_neurons_selection(
            arguments.neuron_selection_path
        )

        # Print experiment setup
        self._print_experiment_info(arguments)

    def _init_model_path(self, arguments) -> str:
        """
        Initializes path where to store the best model parameters.

        By default (if not specified other version in `arguments`) the path is in format:
            `arguments.model_dir/model_lr_{learning_rate}_{model_type}_residual_{True/False}.pth`
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
                    f"_residual-{arguments.neuron_residual}",
                    f"_neuron-layers-{arguments.neuron_num_layers}",
                    f"_neuron-size-{arguments.neuron_layer_size}",
                    f"_num-hidden-time-steps-{arguments.num_hidden_time_steps}",
                    ".pth",
                ]
            )

        # Path where to store the best model parameter.
        return arguments.model_dir + arguments.model_filename

    def _load_neurons_selection(self, neuron_selection_path: str):
        """

        :param arguments: _description_
        """

        with open(neuron_selection_path, "rb") as pickle_file:
            return pickle.load(pickle_file)

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
                    "residual": arguments.neuron_residual,
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

    def _print_experiment_info(self, argument):
        """
        Prints basic information about the experiment.

        :param argument: command line arguments.
        """
        print(
            "\n".join(
                [
                    "NEW EXPERIMENT",
                    "---------------------------------",
                    "Running with parameters:",
                    f"Model variant: {argument.model}",
                    f"Neuron number of layers: {argument.neuron_num_layers}",
                    f"Neurons layer sizes: {argument.neuron_layer_size}",
                    f"Neuron use residual connection: {argument.neuron_residual}",
                    f"Batch size: {nn_model.globals.train_batch_size}",
                    f"Learning rate: {argument.learning_rate}",
                    f"Num epochs: {argument.num_epochs}",
                ]
            )
        )

    def _get_data(
        self,
        inputs: Dict,
        targets: Dict,
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

    def _compute_loss(
        self,
        predictions: Dict[str, List[torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes model loss of all model layer predictions.

        :param predictions: list of model predictions for each time step (without the first one).
        :param targets: tensor of model targets of all time steps
        (we want to omit the first one in computation of the loss).
        :return: Loss of model predictions concatenated to one vector.
        """
        # Concatenate predictions and targets across all layers
        all_predictions = (
            torch.cat(
                [torch.cat(predictions[layer], dim=1) for layer in targets.keys()],
                dim=2,
            )
            .float()
            .cpu()
        )
        all_targets = torch.cat(
            [target[:, 1:, :].float() for target in targets.values()], dim=2
        )

        # Compute the loss for all layers at once
        loss = self.criterion(all_predictions, all_targets)

        return loss

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
            3. Prediction of the model is the following (current) hidden state (output layer).

        :param continuous_evaluation_kwargs: kwargs for continuous evaluation setup
        (see `self._epoch_evaluation_step` for more information).
        :param debugging_stop_step: number of steps (batches) to be performed in debugging
        run of the training. If `-1` then train on all provided data.
        """
        wandb.watch(
            self.model,
            self.criterion,
        )
        self.model.train()
        for epoch in range(self.num_epochs):
            loss_sum = 0.0
            for i, (input_batch, target_batch) in enumerate(tqdm(self.train_loader)):
                if debugging_stop_index != -1 and i > debugging_stop_index:
                    # Train only for few batches in case of debugging.
                    break

                # Retrieve the batch of data.
                input_batch, target_batch = self._get_data(input_batch, target_batch)
                self.optimizer.zero_grad()

                # Get model prediction.
                predictions = self.model(
                    input_batch,
                    target_batch,
                )

                del input_batch
                torch.cuda.empty_cache()

                # Compute loss of the model predictions.
                loss = self._compute_loss(predictions, target_batch)
                loss_sum += loss.item()

                del target_batch, predictions
                torch.cuda.empty_cache()

                # Perform backward step.
                loss.backward()
                self.optimizer.step()

                # Apply weight constrains (excitatory/inhibitory) for all the layers.
                self._apply_model_constraints()

                wandb.log({"batch_loss": loss.item()})

                torch.cuda.empty_cache()

            wandb.log(
                {
                    "epoch_loss": loss_sum / len(self.train_loader),
                }
            )
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Average Loss: {loss_sum/len(self.train_loader):.4f}"
            )
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

    def _get_all_trials_predictions(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_trials: int,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Computes predictions for all trials (used for evaluation usually).

        :param inputs: dictionary of inputs for each input layer.
        Inputs of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param hidden_states: dictionary of hidden states for each output (hidden) layer
        Of shape: `(batch_size, num_trials, num_time_steps, num_neurons)`
        :param num_trials: total number of trials in provided data.
        :return: Returns dictionary of lists of predictions for all trials with key layer name.
        """
        dict_predictions = {}

        # Get predictions for each trial.
        for trial in range(num_trials):
            trial_inputs = {
                layer: layer_input[:, trial, :, :]
                for layer, layer_input in inputs.items()
            }
            trial_hidden = {
                layer: layer_hidden[:, trial, 0, :].clone()
                # Pass slices of targets in time 0 (starting hidden step,
                # in evaluation we do not want to reset hidden states).
                for layer, layer_hidden in targets.items()
            }
            trial_predictions = self.model(
                trial_inputs,
                trial_hidden,
            )
            for layer, prediction in trial_predictions.items():
                # For each layer add the predictions to corresponding list of all predictions.
                prediction = torch.cat(prediction, dim=1)
                if layer not in dict_predictions:
                    dict_predictions[layer] = [prediction]
                else:
                    dict_predictions[layer].append(prediction)

        return dict_predictions

    def _predict_for_evaluation(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_trials: int,
    ) -> Dict[str, torch.Tensor]:
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
        dict_predictions = self._get_all_trials_predictions(inputs, targets, num_trials)
        torch.cuda.empty_cache()

        # Stack all predictions into one torch array.
        dict_stacked_predictions = {
            key: torch.stack(value_list, dim=0)
            for key, value_list in dict_predictions.items()
        }

        # Reshape the prediction to shape:  `(num_trials, batch_size, time, num_neurons)`
        dict_stacked_predictions = {
            layer: predictions.permute(1, 0, 2, 3)
            for layer, predictions in dict_stacked_predictions.items()
        }

        return dict_stacked_predictions

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
        predictions: Dict[str, torch.Tensor],
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
            pickle.dump({"predictions": predictions, "targets": targets}, f)

    def _save_full_evaluation(
        self,
        batch_index: int,
        predictions: Dict[str, torch.Tensor],
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
                layer: torch.mean(
                    prediction, dim=0
                )  # Trials dimension is the first because we reshape it during prediction step.
                for layer, prediction in predictions.items()
            },
            {
                layer: torch.mean(
                    target, dim=1
                )  # We want to skip the first time step + the trials dimension is the second.
                for layer, target in targets.items()
            },
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
                predictions = self._predict_for_evaluation(
                    inputs, targets, inputs[LayerType.X_ON.value].shape[1]
                )

                if save_predictions:
                    self._save_full_evaluation(i, predictions, targets)

                cc_norm, cc_abs = self.compute_evaluation_score(
                    # Compute evaluation for all time steps except the first step (0-th).
                    {layer: target[:, :, 1:, :] for layer, target in targets.items()},
                    predictions,
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

    def _select_neurons_and_trials_mean(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Takes prediction/target tensors, selects the neurons for further analysis
        (`neuron_selection`), and computes the average over trials.

        :param data: Prediction/target tensor to perform the operations on.
        :return: Returns subset of input data in neurons dimension for selected neurons
        subset for further evaluation. This subset is averaged over the trials dimension.
        """
        return {
            layer: torch.mean(data[layer][:, :, :, self.neuron_selection[layer]], dim=1)
            for layer in RNNCellModel.layers_input_parameters
            if layer in data
        }

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

    def selections_evaluation(self) -> Tuple[Dict, Dict]:
        """
        TODO: maybe get rid of it
        Runs evaluation on the selected subset of experiments (images). After that it
        takes only the selected neurons, computes average over trials and returns
        `np.array` of predictions and targets for these for each layer.

        :return: Returns tuple of predictions and targets for all selected neurons on the
        selected experiments averaged over trials.
        """
        self._load_best_model()  # TODO: Maybe solve better

        self.model.eval()
        self.test_dataset.switch_dataset_selection(selected_experiments=True)

        # Store all batches' data for each layer
        all_prediction_batches: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in RNNCellModel.layers_input_parameters
        }
        all_target_batches: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in RNNCellModel.layers_input_parameters
        }

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader):
                inputs, targets = self._get_data(inputs, targets, test=True)
                predictions = self._predict_for_evaluation(
                    inputs, targets, inputs[LayerType.X_ON.value].shape[1]
                )

                # Select neurons and average their responses over trials.
                selected_predictions = self._select_neurons_and_trials_mean(predictions)
                selected_targets = self._select_neurons_and_trials_mean(targets)

                # Append each layer's data to the corresponding list
                for layer in RNNCellModel.layers_input_parameters:
                    all_prediction_batches[layer].append(selected_predictions[layer])
                    all_target_batches[layer].append(
                        selected_targets[layer][
                            :, 1:, :
                        ]  # Get rid of the first target time step (not used in prediction)
                    )

        self.test_dataset.switch_dataset_selection(selected_experiments=False)

        selected_predictions = self._prepare_selected_data_for_analysis(
            all_prediction_batches
        )
        selected_targets = self._prepare_selected_data_for_analysis(all_target_batches)

        # Convert all selected neurons and image predictions/targets to numpy array.
        return selected_predictions, selected_targets
