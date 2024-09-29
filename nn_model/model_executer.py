#!/usr/bin/env python3

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

def compute_trial_means(prediction, target):
    # Ensure the input sets are tensors of shape (trials, neurons, time)
    # num_trials, time_duration, num_neurons = prediction.shape

    # Means across trials.
    # Those should be the `r^{dash}` and `y^{dash}`.
    return prediction.mean(dim=0), target.mean(dim=0)

def compute_var_across_time(trial_mean_pred, trial_mean_tar):
    return torch.var(trial_mean_pred, dim=0), torch.var(trial_mean_tar, dim=0)

def compute_covariance(trial_mean_pred, trial_mean_tar, time_duration):
    # Means across time (for covariance calculation).
    time_mean_pred = trial_mean_pred.mean(dim=0)
    time_mean_tar = trial_mean_tar.mean(dim=0)

    # Centered matrices
    pred_centered = trial_mean_pred - time_mean_pred
    Y_centered = trial_mean_tar- time_mean_tar

    # Cov(X, Y)
    return (pred_centered * Y_centered).sum(dim=0) / (time_duration - 1)    

def compute_cc_abs(cov, var_trial_mean_pred, var_trial_mean_tar):
    return cov / torch.sqrt(var_trial_mean_pred * var_trial_mean_tar)

def compute_cc_max(target, num_trials, var_trial_mean_tar):
    all_time_var_tar = torch.var(target, dim=1)

    mean_var_tar = all_time_var_tar.mean(dim=0)

    cc_max_numerator = num_trials*var_trial_mean_tar - mean_var_tar 
    cc_max_denominator = (num_trials - 1) * var_trial_mean_tar

    return torch.sqrt(cc_max_numerator / cc_max_denominator)


def normalized_cross_correlation_trials(prediction, target):
    # Assuming prediction and target are PyTorch tensors
    batch_size, num_trials, time_duration, num_neurons = prediction.shape

    # Reshape to have time and neurons in one dimension
    prediction = prediction.view(batch_size, num_trials, time_duration * num_neurons)
    target = target.view(batch_size, num_trials, time_duration * num_neurons)

    # Averages across the trials
    prediction_avg = prediction.mean(dim=1)
    target_avg = target.mean(dim=1)

    # Reshape to combine all vectors together for batch processing
    # pred_flat = prediction_avg.view(batch_size, -1)  # Shape: (batch_size, time_duration * num_neurons)
    # target_flat = target_avg.view(batch_size, -1)    # Same shape

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




# def normalized_cross_correlation_trials(prediction, target):
#     """
#     prediction should be in silico response (`r` from the paper)
#     target should be in vivo response (`y` from the paper)

#     Inspired by the model testing in the paper:
#     https://www.biorxiv.org/content/10.1101/2023.03.21.533548v1.full.pdf
#     """
#     # Ensure the input sets are tensors of shape (trials, neurons, time)
#     # print(prediction)
#     batch_size, num_trials, time_duration, num_neurons = prediction.shape
#     # Reshape to have time and neurons in one dimension 
#     # (it does not very depend which corresponds to which because we 
#     # compute the correlation of the whole vector (it just needs 
#     # to be in reshaped in the same order)).
#     prediction = prediction.reshape(batch_size, num_trials, time_duration*num_neurons)
#     target = target.reshape(batch_size, num_trials, time_duration*num_neurons)

#     # prediction = prediction.float() 
#     # target = target.float()

#     # Averages across the trials
#     prediction_avg = np.mean(prediction, axis=1)
#     target_avg = np.mean(target, axis=1)

#     # Reshape to combine all vectors together for batch processing
#     pred_flat = prediction_avg.reshape(batch_size, -1)  # Shape: (batch_size, time_duration * num_neurons)
#     target_flat = target_avg.reshape(batch_size, -1)  # Same shape

#     # Calculate means
#     mean_pred = np.mean(pred_flat, axis=1, keepdims=True)
#     mean_target = np.mean(target_flat, axis=1, keepdims=True)

#     # Calculate covariance between prediction and target
#     cov = np.mean((pred_flat - mean_pred) * (target_flat - mean_target), axis=1)

#     # Calculate standard deviations
#     std_pred = np.std(pred_flat, axis=1)
#     std_target = np.std(target_flat, axis=1)

#     # Calculate correlation coefficients
#     cc_abs = cov / (std_pred * std_target)


#     # CC_MAX CALCULATION

#     std_target_original = np.std(target_flat, axis=2)

#     var_target_original = var_target_original * var_target_original

#     var_target_original_mean = np.mean(var_target_original, axis=1)
    
#     numerator = (std_target*std_target)*num_trials - var_target_original
#     denominator = (num_trials - 1) * (std_target*std_target)

#     cc_max = np.sqrt(numerator/denominator)

#     cc_norm = cc_abs/cc_max

#     cc_norm_batch_mean = np.mean(cc_norm, axis=0)

#     return cc_norm_batch_mean


    # for i in range(batch_size):
    #     for j in range(num_trials):
    #         corr_abs = np.corrcoef(prediction[i, j, :], target[i, j, :])[0,1]

    # trial_mean_pred, trial_mean_tar = compute_trial_means(prediction, target)

    # var_trial_mean_pred, var_trial_mean_tar = compute_var_across_time(trial_mean_pred, trial_mean_tar)

    # cov = compute_covariance(trial_mean_pred, trial_mean_tar, time_duration)
    # cc_abs = compute_cc_abs(cov, var_trial_mean_pred, var_trial_mean_tar)
    # cc_max = compute_cc_max(target, num_trials, var_trial_mean_tar)

    # cc_norm = cc_abs / cc_max
    
    # return cc_norm.mean(dim=0)    


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    # Initialize GradScaler for mixed precision training
    # scaler = GradScaler()
    for epoch in range(num_epochs):
        loss = 0
        for i, (inputs, targets) in enumerate(tqdm(train_loader)):
            if i > 2:
                break
            # print("I get to training")
            inputs = {layer: input_data[:, 0, :, :].float().to(globals.device0) for layer, input_data in inputs.items()}
            # print("I get to inputs")
            targets = {layer: output_data[:, 0, :, :].float() for layer, output_data in targets.items()}
            # print("I get to outputs")

            optimizer.zero_grad()

            batch_size = globals.train_batch_size#inputs['X_ON'].size(0)

            h4_exc = torch.zeros(batch_size, model.l4_exc_size).to(globals.device0)#.cuda()
            h4_inh = torch.zeros(batch_size, model.l4_inh_size).to(globals.device0)#.cuda()
            h23_exc = torch.zeros(batch_size, model.l23_exc_size).to(globals.device1)#.cuda()
            h23_inh = torch.zeros(batch_size, model.l23_inh_size).to(globals.device1)#.cuda()

            loss = 0

            # with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(inputs['X_ON'], inputs['X_OFF'], h4_exc, h4_inh, h23_exc, h23_inh)            
            # print("Predictions done")
            del inputs, h4_exc, h4_inh, h23_inh, h23_exc
            torch.cuda.empty_cache()

            # loss = 0
            for layer, target in targets.items():
                loss += criterion(torch.cat(predictions[layer], dim=1).float().cpu(), target.float())
                # print("Loss done")

            del targets, predictions
            torch.cuda.empty_cache()


            loss.float().backward()
            # print("Backward done")
            optimizer.step()
            # print("Optimizer step done")

            # Apply constraints to all constrained RNN cells
            for module in model.modules():
                if isinstance(module, ConstrainedRNNCell):
                    module.apply_constraints()

            torch.cuda.empty_cache()
            # print(loss.item())
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluation(model, data_loader, criterion):
    model.eval()
    correlation_sum = 0
    num_examples = 0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader):
            inputs = {layer: input_data.float().to(globals.device0) for layer, input_data in inputs.items()}
            # print("I get to inputs")
            targets = {layer: output_data.float() for layer, output_data in targets.items()}
            # print("I get to outputs")
            
            batch_size = globals.test_batch_size#inputs['X_ON'].size(1)

            h4_exc = torch.zeros(batch_size, model.l4_exc_size).to(globals.device0)
            h4_inh = torch.zeros(batch_size, model.l4_inh_size).to(globals.device0)
            h23_exc = torch.zeros(batch_size, model.l23_exc_size).to(globals.device1)
            h23_inh = torch.zeros(batch_size, model.l23_inh_size).to(globals.device1)

            # loss = 0
            predictions = []

            dict_predictions = {}

            # with autocast(device_type="cuda", dtype=torch.float16):
            for i in range(inputs['X_ON'].shape[1]):
                trial_predictions = model(inputs['X_ON'][:, i, :, :], inputs['X_OFF'][:, i, :, :], h4_exc, h4_inh, h23_exc, h23_inh)       
                # predictions.append(trial_predictions)
                for key, prediction in trial_predictions.items():
                    prediction = torch.cat(prediction, dim=1)
                    if key not in dict_predictions.keys():
                        dict_predictions[key] = [prediction]
                    else:
                        dict_predictions[key].append(prediction)
                    # dict_predictions[key]
                del trial_predictions
                torch.cuda.empty_cache()


            del inputs, h4_exc, h4_inh, h23_inh, h23_exc
            torch.cuda.empty_cache()


            # for key, list in dict_predictions.items():
            #     print(key)
            #     for arr in list:
            #         print(arr.shape)
            
            # # Step 2: Stack the lists into 4D arrays for each key
            # stacked_arrays_by_key = {key: np.stack(value_list, axis=0) for key, value_list in dict_predictions.items()}

            # # Step 3: Move the new axis to the 2nd position
            # reshaped_arrays_by_key = {key: np.moveaxis(array, 0, 1) for key, array in stacked_arrays_by_key.items()}
            stacked_arrays_by_key = {key: torch.stack(value_list, dim=0) for key, value_list in dict_predictions.items()}

            # Step 2: Move the new axis to the 2nd position
            reshaped_arrays_by_key = {key: array.permute(1, 0, 2, 3) for key, array in stacked_arrays_by_key.items()}

                
            # predictions = np.stack(predictions, axis=0)

            # predictions = np.moveaxis(predictions, 0, 1)

            cross_correlation = 0
            # for i in range(len(predictions)):
            #     # print(predictions)
            #     # for prediction in predictions[i]:
            #     for layer, target in targets.items():
            #         # print(target)
            #         cross_correlation += normalized_cross_correlation_trials(torch.cat(predictions[i][layer], dim=1).float().cpu(), target[i, :, :])
            
            # # cross_correlation /= (len(predictions) * 4)

            for layer, target in targets.items():
                # print(target)
                # if layer == "V1_Inh_L23":
                #     print("problematic part")
                cross_correlation += normalized_cross_correlation_trials(reshaped_arrays_by_key[layer].to(globals.device0), target.to(globals.device0))
                del target, reshaped_arrays_by_key[layer]
                torch.cuda.empty_cache()

            
            # # cross_correlation /= (len(predictions) * 4)
            print(f"Test cross correlation {cross_correlation}")
            correlation_sum += cross_correlation
            num_examples += 1
            # print(f'Test Loss: {loss.item():.4f}')
    print(f"Total cross correlation {correlation_sum / num_examples}")

# from torchviz import make_dot

def main():
    # Define directories and layers
    # base_dir = "testing_dataset/size_5"
    train_dir = "/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/compressed_spikes/trimmed/size_5"
    test_dir = "/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_5"

    # base_dir = "/home/beinhaud/diplomka/mcs-source/testing_dataset/test"

    model_subset_path = "/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_76.pkl"
    model_subset_path = "/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_10.pkl"
    model_subset_path = f"/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_{int(globals.SIZE_MULTIPLIER*100)}.pkl"


    train_test_path = "/home/beinhaud/diplomka/mcs-source/dataset/train_test_splits/size_10.pkl"

    layer_sizes = {
        'X_ON': globals.X_ON_SIZE,
        'X_OFF': globals.X_OFF_SIZE,
        'V1_Exc_L4': globals.L4_EXC_SIZE,
        'V1_Inh_L4': globals.L4_INH_SIZE,
        'V1_Exc_L23': globals.L23_EXC_SIZE, 
        'V1_Inh_L23': globals.L23_INH_SIZE, 
    }
    input_layers = {
        'X_ON': globals.X_ON_SIZE, 
        'X_OFF': globals.X_OFF_SIZE,
    }
    output_layers = {
        'V1_Exc_L4': globals.L4_EXC_SIZE,
        'V1_Inh_L4': globals.L4_INH_SIZE,
        'V1_Exc_L23': globals.L23_EXC_SIZE, 
        'V1_Inh_L23': globals.L23_INH_SIZE, 
    }

    # Create dataset and dataloader
    train_dataset = SparseSpikeDataset(
            train_dir, 
            input_layers, 
            output_layers,
            is_test=False,
            model_subset_path=model_subset_path, 
            # train_test_path=train_test_path, 
            # include_experiments=False,
        )
    test_dataset = SparseSpikeDataset(
            test_dir, 
            input_layers, 
            output_layers, 
            is_test=True,
            model_subset_path=model_subset_path, 
            # train_test_path=train_test_path, 
            # include_experiments=True,
        )

    # train_batch_size = 10
    # test_batch_size = 1
    
    train_loader = DataLoader(train_dataset, batch_size=globals.train_batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=globals.test_batch_size, collate_fn=custom_collate_fn)

    model = RNNCellModel(layer_sizes).to(globals.device1)#.half()

    print("Criterion")
    criterion = torch.nn.MSELoss()
    print("optimizer")
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    print("training")
    train(model, train_loader, criterion, optimizer, num_epochs)
    evaluation(model, test_loader, criterion)

if __name__ == "__main__":
    main()
