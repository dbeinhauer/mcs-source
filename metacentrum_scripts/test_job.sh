#!/bin/bash
#PBS -N VISUAL_CORTEX_MODEL
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=40gb:mem=100gb:scratch_local=100gb:cluster=^bee

#PBS -m ae
#PBS -j oe

set -e

# Ensure clean_scratch runs on exit, even on error
cleanup() {
echo "Running clean_scratch at $(date)"
clean_scratch
}
trap cleanup EXIT

nvidia-smi

echo "Task started at $(date)"

PROJECT_NAME="mcs-source"
SERVER_LOCATION="praha1"
DATADIR="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME"
SIF_IMAGE="/storage/$SERVER_LOCATION/home/$USER/visual_cortex.sif"  # or .sandbox

export TMPDIR=$SCRATCHDIR

test -n "$SCRATCHDIR" || {
echo >&2 "SCRATCHDIR is not set!"
exit 1
}

cd "$DATADIR" || {
echo >&2 "Failed to enter DATADIR"
exit 1
}

cd "$SCRATCHDIR"

# Create the runner script for inside container
cat << 'EOF' > run_inside.sh
#!/bin/bash
export PATH=/opt/conda/bin:$PATH
source /opt/conda/etc/profile.d/conda.sh
conda activate metacentrum_env

WANDB_API_KEY=$(cat /mnt/.wandb_api_key)

wandb login "$WANDB_API_KEY" || {
echo >&2 "Failed to log into wandb"
exit 1
}

echo "Logged in wandb at $(date)"

echo "Starting model execution at $(date)"

python /mnt/execute_model.py --num_data_workers=8 \
--learning_rate=0.001 \
--model=dnn_separate \
--num_epochs=10 \
--num_backpropagation_time_steps=1 \
--neuron_num_layers=4 \
--neuron_layer_size=5 \
--neuron_rnn_variant=gru \
--synaptic_adaptation_size=10 \
--synaptic_adaptation_num_layers=1 \
--wandb_project_name=my_project \
--save_all_predictions \
--neuron_residual \
--subset_variant=1 \
--train_subset=1 ||
    {
echo >&2 "Python script failed"
exit 1
}
EOF

chmod +x run_inside.sh

echo "Running inside Singularity at $(date)"

# Bind your project directory as /mnt inside container
singularity exec -nv -B "$DATADIR":/mnt "$SIF_IMAGE" bash run_inside.sh || {
    echo >&2 "Python script failed"
    exit 1
}

echo "Task finished at $(date)"

clean_scratch
