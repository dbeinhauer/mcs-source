"""
This source code contains definition of all global variables that
are used across multiple source files. Typically information
about the layer and model parameters.
"""

from nn_model.type_variants import LayerType, PathDefaultFields, PathPlotDefaults

# GPU Devices:
DEVICE = "cuda"

# Model Parameters:

# Model Diminishing factor
# SIZE_MULTIPLIER = 0.53
SIZE_MULTIPLIER = 0.1
# SIZE_MULTIPLIER = 0.25
# SIZE_MULTIPLIER = 0.5

# Model time step size
# TIME_STEP = 1
# TIME_STEP = 5
# TIME_STEP = 10
# TIME_STEP = 15
TIME_STEP = 20

# Batch sizes:
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 10

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

ORIGINAL_SIZES = {
    LayerType.X_ON.value: 7200,
    LayerType.X_OFF.value: 7200,
    LayerType.V1_EXC_L4.value: 37500,
    LayerType.V1_INH_L4.value: 9375,
    LayerType.V1_EXC_L23.value: 37500,
    LayerType.V1_INH_L23.value: 9375,
}

# Number of time steps
BLANK_DURATION = 151
IMAGE_DURATION = 561

NORMAL_NUM_TIME_STEPS = (BLANK_DURATION + IMAGE_DURATION) // TIME_STEP

# Our Model Parameters
MODEL_SIZES = {
    layer: int(size * SIZE_MULTIPLIER) for layer, size in ORIGINAL_SIZES.items()
}

# All default input paths that are used in model executer.
DEFAULT_PATHS = {
    PathDefaultFields.TRAIN_DIR.value: f"/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/compressed_spikes/trimmed/size_{TIME_STEP}",
    PathDefaultFields.TEST_DIR.value: f"/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/compressed_spikes/trimmed/size_{TIME_STEP}",
    PathDefaultFields.SUBSET_DIR.value: f"/home/beinhaud/diplomka/mcs-source/dataset/model_subsets/size_{int(SIZE_MULTIPLIER*100)}.pkl",
    PathDefaultFields.MODEL_DIR.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/best_models/",
    PathDefaultFields.EXPERIMENT_SELECTION_PATH.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/experiments/experiments_subset_10.pkl",
    PathDefaultFields.NEURON_SELECTION_PATH.value: f"/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_subsets/neurons/model_size_{int(SIZE_MULTIPLIER*100)}_subset_10.pkl",
    PathDefaultFields.SELECTION_RESULTS_DIR.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/neuron_responses/",
    PathDefaultFields.FULL_EVALUATION_DIR.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/full_evaluation_results/",
    PathDefaultFields.NEURON_MODEL_RESPONSES_DIR.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/evaluation_results/neuron_model_responses/",
}

# Change default dataset paths in case the time step is 1 (not compressed).
if TIME_STEP == 1:
    DEFAULT_PATHS[PathDefaultFields.TRAIN_DIR.value] = (
        "/home/beinhaud/diplomka/mcs-source/dataset/train_dataset/trimmed_spikes"
    )
    DEFAULT_PATHS[PathDefaultFields.TEST_DIR.value] = (
        "/home/beinhaud/diplomka/mcs-source/dataset/test_dataset/trimmed_spikes"
    )


# All default paths of the plots.
DEFAULT_PLOT_PATHS = {
    PathPlotDefaults.NEURON_MODULE_SEPARATE.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/plot_images/dnn_module_dependencies_separate.png",
    PathPlotDefaults.NEURON_MODULE_TOGETHER.value: "/home/beinhaud/diplomka/mcs-source/evaluation_tools/plot_images/dnn_module_dependencies_together.png",
    PathPlotDefaults.MEAN_LAYER_RESPONSES.value: "",
}
