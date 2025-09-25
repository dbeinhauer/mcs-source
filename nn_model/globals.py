"""
This source code contains definition of all global variables that
are used across multiple source files. Typically information
about the layer and model parameters.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch

from nn_model.type_variants import (
    LayerType,
    PathDefaultFields,
    ModelTypes,
    LayerParent,
)

PROJECT_ROOT = Path(__file__).parent.parent

# GPU Devices:
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Model Parameters:

# Model Diminishing factor
DEFAULT_SIZE_MULTIPLIER = 0.02
# Allow overriding SIZE_MULTIPLIER via an environment variable -> model size using env variables.
SIZE_MULTIPLIER = float(os.getenv("SIZE_MULTIPLIER", DEFAULT_SIZE_MULTIPLIER))

# Model time step size
DEFAULT_TIME_STEP = 20
# Set time steps size from environment variables (in case specified).
TIME_STEP = int(os.getenv("TIME_STEP", DEFAULT_TIME_STEP))

# Batch sizes:
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 10

LAYER_TO_PARENT = {
    LayerType.X_ON.value: LayerParent.LGN.value,
    LayerType.X_OFF.value: LayerParent.LGN.value,
    LayerType.V1_EXC_L4.value: LayerParent.L4.value,
    LayerType.V1_INH_L4.value: LayerParent.L4.value,
    LayerType.V1_EXC_L23.value: LayerParent.L23.value,
    LayerType.V1_INH_L23.value: LayerParent.L23.value,
}

# Layers containing LGN layer.
LGN_NEURONS = {
    LayerType.X_ON.value,
    LayerType.X_OFF.value,
}

L4_NEURONS = {
    LayerType.V1_EXC_L4.value,
    LayerType.V1_INH_L4.value,
}

L23_NEURONS = {
    LayerType.V1_EXC_L23.value,
    LayerType.V1_INH_L23.value,
}

# Will return values as its names
EXCITATORY_LAYERS = {
    LayerType.X_ON.value,
    LayerType.X_OFF.value,
    LayerType.V1_EXC_L4.value,
    LayerType.V1_EXC_L23.value,
}
INHIBITORY_LAYERS = {
    LayerType.V1_INH_L4.value,
    LayerType.V1_INH_L23.value,
}

# Original SNN model size.
ORIGINAL_SIZES = {
    LayerType.X_ON.value: 7200,
    LayerType.X_OFF.value: 7200,
    LayerType.V1_EXC_L4.value: 37500,
    LayerType.V1_INH_L4.value: 9375,
    LayerType.V1_EXC_L23.value: 37500,
    LayerType.V1_INH_L23.value: 9375,
}

# Original number of time steps.
BLANK_DURATION = 151
IMAGE_DURATION = 561

# Number of time steps using selected time binning.
NORMAL_NUM_TIME_STEPS = (BLANK_DURATION + IMAGE_DURATION) // TIME_STEP

# Our Model Parameters
MODEL_SIZES = {
    layer: int(size * SIZE_MULTIPLIER) for layer, size in ORIGINAL_SIZES.items()
}

# All DNN neuron models
DNN_MODELS = [
    ModelTypes.DNN_JOINT.value,
    ModelTypes.DNN_SEPARATE.value,
]

# All RNN neuron models
RNN_MODELS = [
    ModelTypes.RNN_JOINT.value,
    ModelTypes.RNN_SEPARATE.value,
]

# All neuron module models.
COMPLEX_MODELS = DNN_MODELS + RNN_MODELS

# All models that expects RNN output as 1 value.
JOINT_MODELS = [
    ModelTypes.SIMPLE.value,
    ModelTypes.DNN_JOINT.value,
    ModelTypes.RNN_JOINT.value,
]

# All models that expect RNN output to be 2 values (first excitatory and second inhibitory).
SEPARATE_MODELS = [
    ModelTypes.DNN_SEPARATE.value,
    ModelTypes.RNN_SEPARATE.value,
]

DATASET_NAME = "testing_dataset"

# String to represent model size in indices selection (if int then remove zeros).
# SIZE_MULTIPLIER_STRING = (
#     int(SIZE_MULTIPLIER * 100) if SIZE_MULTIPLIER % 1 == 0 else SIZE_MULTIPLIER * 100
# )

# All default input paths that are used in model executer.
DEFAULT_PATHS = {
    PathDefaultFields.TRAIN_DIR.value: f"{PROJECT_ROOT}/dataset/train_dataset/compressed_spikes/trimmed/size_{TIME_STEP}",
    PathDefaultFields.TEST_DIR.value: f"{PROJECT_ROOT}/dataset/test_dataset/compressed_spikes/trimmed/size_{TIME_STEP}",
    PathDefaultFields.SUBSET_DIR.value: f"{PROJECT_ROOT}/dataset/model_subsets/size_{SIZE_MULTIPLIER*100}.pkl",
    PathDefaultFields.VISIBLE_NEURONS_DIR.value: f"{PROJECT_ROOT}/dataset/visible_neurons/",
    PathDefaultFields.MODEL_DIR.value: f"{PROJECT_ROOT}/evaluation_tools/evaluation_results/best_models/",
    PathDefaultFields.EXPERIMENT_SELECTION_PATH.value: f"{PROJECT_ROOT}/evaluation_tools/evaluation_subsets/experiments/experiments_subset_10.pkl",
    PathDefaultFields.NEURON_SELECTION_PATH.value: f"{PROJECT_ROOT}/evaluation_tools/evaluation_subsets/neurons/model_size_{int(SIZE_MULTIPLIER * 100)}_subset_10.pkl",
    PathDefaultFields.SELECTION_RESULTS_DIR.value: f"{PROJECT_ROOT}/evaluation_tools/evaluation_results/neuron_responses/",
    PathDefaultFields.FULL_EVALUATION_DIR.value: f"{PROJECT_ROOT}/evaluation_tools/evaluation_results/full_evaluation_results/",
    PathDefaultFields.NEURON_MODEL_RESPONSES_DIR.value: f"{PROJECT_ROOT}/evaluation_tools/evaluation_results/neuron_model_responses/",
}

# Change default dataset paths in case the time step is 1 (not compressed).
if TIME_STEP == 1:
    DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value] = (
        f"{PROJECT_ROOT}/dataset/train_dataset/trimmed_spikes"
    )
    DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value] = (
        f"{PROJECT_ROOT}/dataset/test_dataset/trimmed_spikes"
    )


def reinitialize_time_step(time_step_size: int):
    """
    Sets the time step to specified values. This function serves for change of
    time step other way than rewriting the source code itself. Mainly for evaluation tools.

    :param time_step_size: Time step to assign.
    """
    global TIME_STEP, NORMAL_NUM_TIME_STEPS, DEFAULT_PATHS
    TIME_STEP = time_step_size
    NORMAL_NUM_TIME_STEPS = (BLANK_DURATION + IMAGE_DURATION) // TIME_STEP
    DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value] = (
        f"{PROJECT_ROOT}/dataset/train_dataset/compressed_spikes/trimmed/size_{TIME_STEP}"
    )
    DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value] = (
        f"{PROJECT_ROOT}/dataset/test_dataset/compressed_spikes/trimmed/size_{TIME_STEP}"
    )

    # Change default dataset paths in case the time step is 1 (not compressed).
    if TIME_STEP == 1:
        DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value] = (
            f"{PROJECT_ROOT}/dataset/train_dataset/trimmed_spikes"
        )
        DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value] = (
            f"{PROJECT_ROOT}/dataset/test_dataset/trimmed_spikes"
        )


def rewrite_test_batch_size(new_batch_size: int):
    """
    Sets test batch size other as global variable. Used mainly for the evaluation tools.

    :param new_batch_size: New batch size to assign.
    """
    global TEST_BATCH_SIZE
    TEST_BATCH_SIZE = new_batch_size


def unflatten_dict(d):
    """Rebuild nested dict from flattened keys."""
    result = {}
    for k, v in d.items():
        keys = k.split(".")
        cur = result
        for sub in keys[:-1]:
            cur = cur.setdefault(sub, {})
        cur[keys[-1]] = v
    return result


# Load position and orientation dictionary. TODO: works also in old numpy
POS_ORI_DICT = unflatten_dict(
    dict(
        np.load(
            f"{PROJECT_ROOT}/testing_dataset/pos_ori_phase_dictionary.npz",
            allow_pickle=True,
        )
    )
)

# TODO: Comment the following functionalities.
# with open(f"{PROJECT_ROOT}/testing_dataset/pos_ori_phase_dictionary.pickle", "rb") as f:
#     POS_ORI_DICT = pickle.load(f)

NEURON_SELECTION = None


def define_neuron_selection(subset_dir: str):
    """
    Defines the neuron selection from the specified path.

    :param subset_dir: Path to the neuron subset pickle file.
    """
    global NEURON_SELECTION, POS_ORI_DICT
    with open(subset_dir, "rb") as f:
        NEURON_SELECTION = pickle.load(f)

    for layer, xyo in POS_ORI_DICT.items():
        subset_filter = np.array(NEURON_SELECTION[layer]).astype(int)
        for attr in xyo.keys():
            POS_ORI_DICT[layer][attr] = np.array(POS_ORI_DICT[layer][attr])[
                subset_filter
            ].astype(float)
            POS_ORI_DICT[layer][attr] = (
                torch.from_numpy(POS_ORI_DICT[layer][attr]).float().to(DEVICE)
            )
