"""
This source code contains definition of all global variables that
are used accorss multiple source files. Typically information
about the layer and model parameters.
"""
from enum import Enum#, auto

# Model Parameters:

# Model Diminishing factor
# SIZE_MULTIPLIER = 0.53
# SIZE_MULTIPLIER = 0.1
SIZE_MULTIPLIER = 0.25
# SIZE_MULTIPLIER = 0.5

# Model time step size
TIME_STEP = 5

class LayerType(Enum):
    X_ON = "X_ON"
    X_OFF = "X_OFF"
    V1_Exc_L4 = "V1_Exc_L4"
    V1_Inh_L4 = "V1_Inh_L4"
    V1_Exc_L23 = "V1_Exc_L23"
    V1_Inh_L23 = "V1_Inh_L23"

    # Will return values as its names
    EXCITATORY_LAYERS = {X_ON, X_OFF, V1_Exc_L4, V1_Exc_L23}
    INHIBITORY_LAYERS = {V1_Inh_L4, V1_Inh_L23}

ORIGINAL_SIZES = {
    LayerType.X_ON.value: 7200,
    LayerType.X_OFF.value: 7200,
    LayerType.V1_Exc_L4.value: 37500,
    LayerType.V1_Inh_L4.value: 9375,
    LayerType.V1_Exc_L23.value: 37500,
    LayerType.V1_Inh_L23.value: 9375,
}

# # Parameters of the original model
# ORIGINAL_X_ON_SIZE = 7200
# ORIGINAL_X_OFF_SIZE = 7200
# ORIGINAL_L4_EXC_SIZE = 37500
# ORIGINAL_L4_INH_SIZE = 9375
# ORIGINAL_L23_EXC_SIZE = 37500
# ORIGINAL_L23_INH_SIZE = 9375

# Number of time steps
BLANK_DURATION = 151
IMAGE_DURATION = 561

# Our Model Parameters
MODEL_SIZES = {
    layer: int(size * SIZE_MULTIPLIER) 
    for layer, size in ORIGINAL_SIZES.items()
}

# # Our model parameters:
# X_ON_SIZE = int(ORIGINAL_X_ON_SIZE * SIZE_MULTIPLIER)
# X_OFF_SIZE = int(ORIGINAL_X_OFF_SIZE * SIZE_MULTIPLIER)
# L4_EXC_SIZE = int(ORIGINAL_L4_EXC_SIZE * SIZE_MULTIPLIER)
# L4_INH_SIZE = int(ORIGINAL_L4_INH_SIZE * SIZE_MULTIPLIER)
# L23_EXC_SIZE = int(ORIGINAL_L23_EXC_SIZE * SIZE_MULTIPLIER)
# L23_INH_SIZE = int(ORIGINAL_L23_INH_SIZE * SIZE_MULTIPLIER)

# GPU Devices:
device0 = 'cuda:1'
device1 = 'cuda:0'
device1 = 'cuda'
device0 = device1

# Batch sizes:
train_batch_size = 50
test_batch_size = 10
