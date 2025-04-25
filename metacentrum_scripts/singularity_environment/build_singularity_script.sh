#!/bin/bash

set -e

# === CONFIGURATION ===
SANDBOX_NAME="debian12_conda.sandbox"
SIF_NAME="metacentrum_env.sif"
CONDA_ENV_NAME="    "
PROJECT_NAME="mcs-source"
SERVER_LOCATION="praha1"
ENV_YAML_HOST_PATH="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME/environment.yaml"
ENV_YAML_CONTAINER_PATH="/mnt/environment.yaml"          # <-- Will appear inside the container
BASE_IMAGE="docker://debian:12"

# === CHECK FOR ROOT ===
if [[ "$EUID" -ne 0 ]]; then
  echo "Please run this script as root (e.g., with sudo)."
  exit 1
fi

# === CREATE SANDBOX ===
echo "Building sandbox from $BASE_IMAGE..."
singularity build --sandbox "$SANDBOX_NAME" "$BASE_IMAGE"

# === INSTALL MINICONDA & ENVIRONMENT INSIDE SANDBOX ===
echo "Installing Miniconda and creating Conda environment..."

singularity exec --writable -B "$(dirname $ENV_YAML_HOST_PATH)":/mnt "$SANDBOX_NAME" bash -c "
    export DEBIAN_FRONTEND=noninteractive
    apt update && apt install -y wget bzip2

    # Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /opt/conda
    rm miniconda.sh

    # Set PATH and enable conda shell
    export PATH=/opt/conda/bin:\$PATH
    echo 'export PATH=/opt/conda/bin:\$PATH' >> /environment
    source /opt/conda/etc/profile.d/conda.sh

    # Create environment from mounted environment.yaml
    conda env create -f $ENV_YAML_CONTAINER_PATH
    conda clean --all -y

    # Export for verification (optional)
    conda activate $CONDA_ENV_NAME
    conda env export > /env_final.yaml
"

echo "Environment installed successfully in sandbox: $SANDBOX_NAME"

# === OPTIONAL: CONVERT TO SIF IMAGE ===
read -p "Do you want to convert the sandbox to a .sif file? (y/n): " BUILD_SIF
if [[ "$BUILD_SIF" == "y" ]]; then
  echo "Converting sandbox to $SIF_NAME..."
  singularity build "$SIF_NAME" "$SANDBOX_NAME"
  echo "Created $SIF_NAME successfully."
else
  echo "Skipped SIF build. You can still use the sandbox directly."
fi
