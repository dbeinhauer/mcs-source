#!/usr/bin/env python3
"""
This script defines manipulation with the models and is used to execute 
model training and evaluation.
"""

import argparse
import os
from typing import Tuple, Dict, Optional, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use the second GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import globals
from type_variants import LayerType, ModelTypes
from dataset_loader import SparseSpikeDataset, different_times_collate_fn
from models import (
    RNNCellModel,
    ConstrainedRNNCell,
    # ComplexConstrainedRNNCell,
)
from evaluation_metrics import NormalizedCrossCorrelation


class ModelExecuter:
    """
    Class used for execution of training and testing steps of the models.
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
        self.layer_sizes = globals.MODEL_SIZES

        # Dataset Init
        self.train_dataset, self.test_dataset = self._init_datasets(arguments)
        self.train_loader, self.test_loader = self._init_data_loaders()

        # Model Init
        self.model = self._init_model(arguments).to(device=globals.device0)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(arguments.learning_rate)

        # Evaluation metric
        self.evaluation_metrics = NormalizedCrossCorrelation()
        self._print_experiment_info(arguments)

    # def _get_model_type(self, model_identifier: str):
    #     """
    #     Based on the model identifier returns type of the model layer.

    #     :param model_identifier: string identifier of the model.
    #     :return: Returns class of the layer that should be used in the model.
    #     """
    #     if model_identifier == ModelTypes.SIMPLE.value:
    #         # Simple model -> used classical cells without additional NN instead of neuron.
    #         return ConstrainedRNNCell
    #     if model_identifier == ModelTypes.COMPLEX.value:
    #         # Complex model -> use additional shared NN for each layer instead of simple neuron.
    #         return ComplexConstrainedRNNCell

    def _split_input_output_layers(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Splits layers sizes to input and output ones.

        :return: Returns tuple of dictionaries of input and output layer sizes.
        """
        input_layers = {
            key: self.layer_sizes[key] for key in ModelExecuter.input_layers
        }
        output_layers = {
            key: value
            for key, value in self.layer_sizes.items()
            if key not in input_layers
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
        input_layers, output_layers = self._split_input_output_layers()

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
        )

        return train_dataset, test_dataset

    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Initialized train and test `DataLoader` objects.

        :return: Returns initialized train and test `Dataloader` classes.
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=globals.train_batch_size,
            shuffle=True,
            collate_fn=different_times_collate_fn,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=globals.test_batch_size,
            collate_fn=different_times_collate_fn,
        )

        return train_loader, test_loader

    def _get_complexity_kwargs(self, arguments):
        if arguments.model == ModelTypes.SIMPLE.value:
            return {}
        if arguments.model == ModelTypes.COMPLEX.value:
            return {
                ModelTypes.COMPLEX.value: {
                    "num_layers": arguments.neuron_num_layers,
                    "layer_size": arguments.neuron_layer_size,
                    "residual": arguments.neuron_not_residual,
                }
            }

        # Wrongly defined complexity type
        print("Wrong complexity, using simple complexity layer.")
        return {}

    def _init_model(self, arguments) -> RNNCellModel:
        """
        Initializes model based on the provided arguments.

        :param arguments: command line arguments containing model setup info.
        :return: Returns initializes model.
        """
        # if arguments.model == ModelTypes.SIMPLE.value:
        #     # Simple model (without shared complexity).
        #     return RNNCellModel(self.layer_sizes).to(globals.device1)
        # if arguments.model == ModelTypes.COMPLEX.value:
        #     # Complex model (with shared complexity).
        #     return RNNCellModel(
        #         self.layer_sizes,
        #         ComplexConstrainedRNNCell,
        #         complexity_kwargs={
        #             ModelTypes.COMPLEX.value: {
        #                 "complexity_size": arguments.complexity_size
        #             }
        #         },
        #     ).to(globals.device1)

        # complexity_kwargs = self._get_complexity_kwargs

        return RNNCellModel(
            self.layer_sizes,
            arguments.model,
            complexity_kwargs=self._get_complexity_kwargs(arguments),
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
                    # f"Complexity model size: {argument.complexity_size}",
                    f"Neuron number of layers: {argument.neuron_num_layers}",
                    f"Neurons layer sizes: {argument.neuron_layer_size}",
                    f"Neuron use residual connection: {not argument.neuron_not_residual}",
                    f"Batch size: {globals.train_batch_size}",
                    f"Learning rate: {argument.learning_rate}",
                    f"Num epochs: {argument.num_epochs}",
                ]
            )
        )

    def _get_data(self, inputs, targets, test: bool = False) -> Tuple[Dict, Dict]:
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
            layer: input_data[:, slice_, :, :].float().to(globals.device0)
            for layer, input_data in inputs.items()
        }
        targets = {
            layer: output_data[
                :, slice_, :, :
            ].float()  # Do not move it to GPU as it is not always used there (only in training).
            for layer, output_data in targets.items()
        }

        return inputs, targets

    def _compute_loss(self, predictions, targets) -> torch.Tensor:
        """
        Computes model loss of all model layer predictions.

        :param predictions: model predictions.
        :param targets: model targets.
        :return: Returns sum of losses for all model layer predictions.
        """
        loss = torch.zeros((1))
        for layer, target in targets.items():
            loss += self.criterion(
                torch.cat(predictions[layer], dim=1).float().cpu(),
                target[:, 1:, :].float(),
                # Compute loss only for the values from time step 1
                # (we start with the step 0 initial hidden steps).
            )

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
    ):
        """
        Runs continuous evaluation steps in selected training epochs on
        selected subset of test examples.

        :param epoch: current epoch number.
        :param epoch_offset: after how many epochs perform evaluation. If `-1` then never run.
        :param evaluation_subset_size: how many batches use for evaluation.
        If `-1` then all test dataset.
        """
        if epoch_offset != -1 and epoch % epoch_offset == 0:
            # Do continuous evaluation after this step.
            self.evaluation(subset=evaluation_subset_size)
            self.model.train()

    def train(
        self,
        continuous_evaluation_kwargs: Dict = {},
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
        self.model.train()
        for epoch in range(self.num_epochs):
            loss = None
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

                del target_batch, predictions
                torch.cuda.empty_cache()

                # Perform backward step.
                loss.backward()
                self.optimizer.step()

                # Apply weight constrains (excitatory/inhibitory) for all the layers.
                self._apply_model_constraints()

                torch.cuda.empty_cache()

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}")
            self._epoch_evaluation_step(
                epoch,
                **continuous_evaluation_kwargs,
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
        Inputs of shape: (batch_size, num_trials, time, num_neurons)
        :param hidden_states: dictionary of hidden states for each output (hidden) layer
        Of shape: (batch_size, num_trials, time, num_neurons)
        :param num_trials: total number of trials in provided data.
        :return: Returns dictionary of lists of predictions for all trials with key layer name.
        """
        dict_predictions = {}

        for trial in range(num_trials):
            # Get predictions for each trial.
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

    def _prepare_predictions_for_evaluation(self, inputs, targets, num_trials):

        # Get predictions for all trials.
        dict_predictions = self._get_all_trials_predictions(inputs, targets, num_trials)
        torch.cuda.empty_cache()

        # Stack all predictions into one torch array.
        dict_predictions = {
            key: torch.stack(value_list, dim=0)
            for key, value_list in dict_predictions.items()
        }

        # Reshape the prediction to shape:  (num_trials, batch_size, time, num_neurons)
        dict_predictions = {
            key: array.permute(1, 0, 2, 3) for key, array in dict_predictions.items()
        }

        return dict_predictions
    
    # def _select_random_subset():

    def compute_evaluation_score(self, targets, predictions):
        cross_correlation = 0

        # TODO: randomly select neurons in each layer (approx. 10 of them) -> get rid of most neurons
        # TODO: randomly select image in each layer (approx. 10 of them)    -> get rid of batch size
        #   - here is a problem with image IDs (does it make sense to choose random image for each experiment?)
        #       - probably it does make sense
        # TODO: compute mean of the responses through the trials -> get rid of trials dimension
        # TODO: Ideally we want dictionary of outputs:
        #       {layer: tensor(shape=(selected_images, time_duration, avg_selected_neuron_responses))}
        # TODO: We want to save these for the best results in regards to CC

        # predictions shape:    (10, 20, 34, 3750)  -> (batch, trials, time_steps, num_neurons)
        # Concatenate predictions and targets across all layers.
        all_predictions = torch.cat(
            [prediction for prediction in predictions.values()],
            dim=-1,
        ).to(globals.device0)
        all_targets = torch.cat([target for target in targets.values()], dim=-1).to(
            globals.device0
        )

        # Run the calculate function once on the concatenated tensors.
        cross_correlation = self.evaluation_metrics.calculate(
            all_predictions, all_targets
        )

        return cross_correlation

    def evaluation(self, subset: int = -1, print_each_step: int = 10):
        self.model.eval()
        correlation_sum = 0
        num_examples = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                if subset != -1 and i > subset:
                    # Evaluate only subset of test data.
                    break

                inputs, targets = self._get_data(inputs, targets, test=True)
                predictions = self._prepare_predictions_for_evaluation(
                    inputs, targets, inputs["X_ON"].shape[1]
                )
                correlation_sum += self.compute_evaluation_score(
                    {layer: target[:, :, 1:, :] for layer, target in targets.items()},
                    predictions,
                )
                num_examples += 1

                if i % print_each_step == 0:
                    print(
                        f"Cross correlation after step {i+1} is: {correlation_sum / num_examples}"
                    )

        print(f"Total cross correlation {correlation_sum / num_examples}")


# from torchviz import make_dot


def main(args):
    model_executer = ModelExecuter(args)

    model_executer.train(
        continuous_evaluation_kwargs={
            "epoch_offset": 1,
            "evaluation_subset_size": 10,
        },
        debugging_stop_index=3,
    )
    model_executer.evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_dir",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/compressed_spikes/trimmed/size_{globals.TIME_STEP}",
        help="",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_{globals.TIME_STEP}",
        help="",
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        default=f"/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_{int(globals.SIZE_MULTIPLIER*100)}.pkl",
        help="",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=[model_type.value for model_type in ModelTypes],
        help="",
    )
    # parser.add_argument("--complexity_size", type=int, default=64, help="")
    parser.add_argument("--neuron_num_layers", type=int, default=5, help="")
    parser.add_argument("--neuron_layer_size", type=int, default=10, help="")
    parser.set_defaults(neuron_not_residual=False)
    # parser.set_defaults(neuron_not_residual=True)
    parser.add_argument("--neuron_not_residual", action="store_true", help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="")
    parser.add_argument("--num_epochs", type=int, default=10, help="")

    args = parser.parse_args()
    main(args)
