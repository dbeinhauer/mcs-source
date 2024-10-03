#!/usr/bin/env python3

import argparse
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use the second GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from tqdm import tqdm
import torch.nn
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader

from dataset_loader import SparseSpikeDataset, custom_collate_fn
from model import RNNCellModel, ConstrainedRNNCell
import globals

def normalized_cross_correlation_trials(prediction, target):
    """
    prediction should be in silico response (`r` from the paper)
    target should be in vivo response (`y` from the paper)

    Inspired by the model testing in the paper:
    https://www.biorxiv.org/content/10.1101/2023.03.21.533548v1.full.pdf
    """
    # Assuming prediction and target are PyTorch tensors
    batch_size, num_trials, time_duration, num_neurons = prediction.shape

    # Round the prediction to closest integer.
    prediction = prediction.round()

    # Reshape to have time and neurons in one dimension
    prediction = prediction.view(batch_size, num_trials, time_duration * num_neurons)
    target = target.view(batch_size, num_trials, time_duration * num_neurons)

    # Averages across the trials
    prediction_avg = prediction.mean(dim=1)
    target_avg = target.mean(dim=1)

    pred_flat = prediction_avg
    target_flat = target_avg

    # Calculate means
    mean_pred = pred_flat.mean(dim=1, keepdim=True)
    mean_target = target_flat.mean(dim=1, keepdim=True)

    # Calculate covariance between prediction and target
    cov = ((pred_flat - mean_pred) * (target_flat - mean_target)).mean(dim=1)

    # Calculate standard deviations
    std_pred = pred_flat.std(dim=1, unbiased=False)
    std_target = target_flat.std(dim=1, unbiased=False)

    # Calculate correlation coefficients
    cc_abs = cov / (std_pred * std_target + 0.00000001)

    # CC_MAX CALCULATION
    std_target_original = target.std(dim=1, unbiased=False)

    var_target_original = std_target_original * std_target_original

    var_target_original_mean = var_target_original.mean(dim=1)

    numerator = (std_target * std_target) * num_trials - var_target_original_mean
    denominator = (num_trials - 1) * (std_target * std_target) + 0.000000001

    cc_max = torch.sqrt(numerator / denominator)

    cc_norm = cc_abs / cc_max

    cc_norm_batch_mean = cc_norm.mean(dim=0).item()
    # print(cc_norm_batch_mean)

    return cc_norm_batch_mean

class ModelExecuter():
    # Input layer keys (LGN).
    input_layers = ['X_ON', 'X_OFF']

    def __init__(self, args):
        self.layer_sizes = {
            'X_ON': globals.X_ON_SIZE,
            'X_OFF': globals.X_OFF_SIZE,
            'V1_Exc_L4': globals.L4_EXC_SIZE,
            'V1_Inh_L4': globals.L4_INH_SIZE,
            'V1_Exc_L23': globals.L23_EXC_SIZE, 
            'V1_Inh_L23': globals.L23_INH_SIZE, 
        }

        self.train_dataset, self.test_dataset = self._init_datasets(args)
        self.train_loader, self.test_loader = self._init_data_loaders()
        self.model = self._init_model()

        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer(args.learning_rate)

        self.num_epochs = args.num_epochs

        self._print_experiment_info(args)


    def _split_input_output_layers(self):# -> tuple[dict[str, int], dict[str, int]]:
        input_layers = {
            key: self.layer_sizes[key] for key in ModelExecuter.input_layers
        }
        output_layers = {
            key: value for key, value in self.layer_sizes.items() if key not in input_layers
        }

        return input_layers, output_layers
    
    def _init_datasets(self, args):# -> tuple[SparseSpikeDataset, SparseSpikeDataset]:
        input_layers, output_layers = self._split_input_output_layers()

        train_dataset = SparseSpikeDataset(
                args.train_dir, 
                input_layers, 
                output_layers,
                is_test=False,
                model_subset_path=args.subset_dir, 
            )
        test_dataset = SparseSpikeDataset(
                args.test_dir, 
                input_layers, 
                output_layers, 
                is_test=True,
                model_subset_path=args.subset_dir, 
            )
        
        return train_dataset, test_dataset

    def _init_data_loaders(self):# -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=globals.train_batch_size, 
            shuffle=True, 
            collate_fn=custom_collate_fn
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=globals.test_batch_size, 
            collate_fn=custom_collate_fn
        )

        return train_loader, test_loader

    def _init_model(self) -> RNNCellModel:
        return RNNCellModel(self.layer_sizes).to(globals.device1)
    
    def _init_criterion(self):
        return torch.nn.MSELoss()
    
    def _init_optimizer(self, learning_rate):
        return optim.Adam(self.model.parameters(), lr=learning_rate)

    def _print_experiment_info(self, args):
        print("\n".join([
                "NEW EXPERIMENT",
                "---------------------------------",
                "Running with parameters:",
                f"Batch size: {globals.train_batch_size}",
                f"Learning rate: {args.learning_rate}",
                f"Num epochs: {args.num_epochs}",
            ])
        )

    def _get_data(self, inputs, targets, test: bool=False):
        # Define what part of trials dimension we want to take.
        # Take `0` for train or all trials `slice(None) == :` for test.
        slice_ = slice(None) if test else 0

        inputs = {
            layer: input_data[:, slice_, :, :].float().to(globals.device0) 
            for layer, input_data in inputs.items()
        }
        targets = {
            layer: output_data[:, slice_, :, :].float() 
            for layer, output_data in targets.items()
        }

        return inputs, targets

    def _init_model_weights(self, batch_size):
        h4_exc = torch.zeros(batch_size, self.model.l4_exc_size).to(globals.device0)
        h4_inh = torch.zeros(batch_size, self.model.l4_inh_size).to(globals.device0)
        h23_exc = torch.zeros(batch_size, self.model.l23_exc_size).to(globals.device1)
        h23_inh = torch.zeros(batch_size, self.model.l23_inh_size).to(globals.device1)

        return h4_exc, h4_inh, h23_exc, h23_inh

    def _compute_loss(self, predictions, targets):
        loss = 0
        for layer, target in targets.items():
            loss += self.criterion(
                    torch.cat(predictions[layer], dim=1).float().cpu(),
                    target.float(),
                )
        
        return loss
    
    def _apply_model_constraints(self):
        for module in self.model.modules():
            if isinstance(module, ConstrainedRNNCell):
                module.apply_constraints()

    def train(self, evaluation_step: int=-1):
        self.model.train()
        for epoch in range(self.num_epochs):
            loss = 0
            for i, (inputs, targets) in enumerate(tqdm(self.train_loader)):
                # if i > 10:
                #     break
                inputs, targets = self._get_data(inputs, targets)

                self.optimizer.zero_grad()

                h4_exc, h4_inh, h23_exc, h23_inh = self._init_model_weights(
                        globals.train_batch_size,
                    )

                predictions = self.model(
                    inputs['X_ON'], 
                    inputs['X_OFF'], 
                    h4_exc, 
                    h4_inh, 
                    h23_exc, 
                    h23_inh,
                )            
                # print("Predictions done")
                del inputs, h4_exc, h4_inh, h23_inh, h23_exc
                torch.cuda.empty_cache()

                loss = self._compute_loss(predictions, targets)

                del targets, predictions
                torch.cuda.empty_cache()

                loss.float().backward()
                self.optimizer.step()

                # Apply weight constrains for all the layers.
                self._apply_model_constraints()

                torch.cuda.empty_cache()
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
            if evaluation_step != -1 and epoch % evaluation_step == 0:
                # Do control evaluation after this step.
                self.evaluation(subset=10)
                self.model.train()

    def _get_all_trials_predictions(self, inputs, num_trials):
        dict_predictions = {}
        h4_exc, h4_inh, h23_exc, h23_inh = self._init_model_weights(
                globals.test_batch_size,
            )
        for i in range(num_trials):
            trial_predictions = self.model(
                    inputs['X_ON'][:, i, :, :], 
                    inputs['X_OFF'][:, i, :, :], 
                    h4_exc, 
                    h4_inh, 
                    h23_exc, 
                    h23_inh,
                )
            # predictions.append(trial_predictions)
            for key, prediction in trial_predictions.items():
                prediction = torch.cat(prediction, dim=1)
                if key not in dict_predictions.keys():
                    dict_predictions[key] = [prediction]
                else:
                    dict_predictions[key].append(prediction)
                # dict_predictions[key]
            # del trial_predictions
            # torch.cuda.empty_cache()

        return dict_predictions

    def _prepare_predictions_for_evaluation(self, inputs, num_trials):

        # Get predictions for all trials.
        dict_predictions = self._get_all_trials_predictions(inputs, num_trials)
        torch.cuda.empty_cache()

        # Stack all predictions into one torch array.
        dict_predictions = {
            key: torch.stack(value_list, dim=0) 
            for key, value_list in dict_predictions.items()
        }

        # Reshape the prediction to shape:  (num_trials, batch_size, time, num_neurons)
        dict_predictions = {
            key: array.permute(1, 0, 2, 3) 
            for key, array in dict_predictions.items()
        }

        return dict_predictions
    
    def compute_evaluation_score(self, targets, predictions):
        cross_correlation = 0

        for layer, target in targets.items():
            print(target)
            if layer == "V1_Inh_L23":
                print("problematic part")
            cross_correlation += normalized_cross_correlation_trials(
                    predictions[layer].to(globals.device0), 
                    target.to(globals.device0)
                )
            del target, predictions[layer]
            torch.cuda.empty_cache()

        return cross_correlation

    def evaluation(self, subset: int=-1, print_each_step: int=10):
        self.model.eval()
        correlation_sum = 0
        num_examples = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(tqdm(self.test_loader)):
                if i != -1 and i > subset:
                    # Evaluate only subset of test data.
                    break

                inputs, targets = self._get_data(inputs, targets, test=True)
                predictions = self._prepare_predictions_for_evaluation(inputs, inputs['X_ON'].shape[1])
                correlation_sum += self.compute_evaluation_score(targets, predictions)
                num_examples += 1

                if i % print_each_step == 0:
                    print(f"Cross correlation after step {i+1} is: {correlation_sum / num_examples}")

        print(f"Total cross correlation {correlation_sum / num_examples}")

# from torchviz import make_dot

def main(args):

    model_executer = ModelExecuter(args)

    model_executer.train(evaluation_step=1)
    model_executer.evaluation(subset=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("--train_dir", type=str, default=f"/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/compressed_spikes/trimmed/size_{globals.TIME_STEP}", 
        help="")
    parser.add_argument("--test_dir", type=str, default=f"/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_{globals.TIME_STEP}", 
        help="")
    parser.add_argument("--subset_dir", type=str, default=f"/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_{int(globals.SIZE_MULTIPLIER*100)}.pkl", 
        help="")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
        help="")
    parser.add_argument("--num_epochs", type=int, default=10,
        help="")

    args = parser.parse_args()
    main(args)
