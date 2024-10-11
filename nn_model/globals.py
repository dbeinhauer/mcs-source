"""
This source code contains definition of all global variables that
are used accorss multiple source files. Typically information
about the layer and model parameters.
"""

from type_variants import LayerType

# Model Parameters:

# Model Diminishing factor
# SIZE_MULTIPLIER = 0.53
SIZE_MULTIPLIER = 0.1
# SIZE_MULTIPLIER = 0.25
# SIZE_MULTIPLIER = 0.5

# Model time step size
TIME_STEP = 5


# Will return values as its names
EXCITATORY_LAYERS = {
    LayerType.X_ON.value,
    LayerType.X_OFF.value,
    LayerType.V1_Exc_L4.value,
    LayerType.V1_Exc_L23.value,
}
INHIBITORY_LAYERS = {
    LayerType.V1_Inh_L4.value,
    LayerType.V1_Inh_L23.value,
}


ORIGINAL_SIZES = {
    LayerType.X_ON.value: 7200,
    LayerType.X_OFF.value: 7200,
    LayerType.V1_Exc_L4.value: 37500,
    LayerType.V1_Inh_L4.value: 9375,
    LayerType.V1_Exc_L23.value: 37500,
    LayerType.V1_Inh_L23.value: 9375,
}

# Number of time steps
BLANK_DURATION = 151
IMAGE_DURATION = 561

# Our Model Parameters
MODEL_SIZES = {
    layer: int(size * SIZE_MULTIPLIER) for layer, size in ORIGINAL_SIZES.items()
}

# GPU Devices:
device0 = "cuda:1"
device1 = "cuda:0"
device1 = "cuda"
device0 = device1

# Batch sizes:
train_batch_size = 50
test_batch_size = 10
