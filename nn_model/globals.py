
# Model Parameters:

# Model Diminishing factor
# SIZE_MULTIPLIER = 0.76
SIZE_MULTIPLIER = 0.1

# Model time step size
TIME_STEP = 5

# Parameters of the original model
ORIGINAL_X_ON_SIZE = 7200
ORIGINAL_X_OFF_SIZE = 7200
ORIGINAL_L4_EXC_SIZE = 37500
ORIGINAL_L4_INH_SIZE = 9375
ORIGINAL_L23_EXC_SIZE = 37500
ORIGINAL_L23_INH_SIZE = 9375

# Number of time steps
BLANK_DURATION = 151
IMAGE_DURATION = 561

# Our model parameters:
X_ON_SIZE = int(ORIGINAL_X_ON_SIZE * SIZE_MULTIPLIER)
X_OFF_SIZE = int(ORIGINAL_X_OFF_SIZE * SIZE_MULTIPLIER)
L4_EXC_SIZE = int(ORIGINAL_L4_EXC_SIZE * SIZE_MULTIPLIER)
L4_INH_SIZE = int(ORIGINAL_L4_INH_SIZE * SIZE_MULTIPLIER)
L23_EXC_SIZE = int(ORIGINAL_L23_EXC_SIZE * SIZE_MULTIPLIER)
L23_INH_SIZE = int(ORIGINAL_L23_INH_SIZE * SIZE_MULTIPLIER)


# GPU Devices:
device0 = 'cuda:1'
device1 = 'cuda:0'
device1 = 'cuda'
device0 = device1