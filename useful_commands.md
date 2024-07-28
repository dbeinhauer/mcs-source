# Start SSH with port open
```bash
ssh -L 8080:localhost:8080 neuro
```

# Start Jupyter Lab on port
```bash
jupyter lab --no-browser --port=8080
```

# Open directory in browser
```bash
python -m http.server 8081
```

# Create virtualenv for GPUs working
- cuda work in `gpu-neuro` server

## Remove old environment
conda env list
conda deactivate
conda env remove --name old_env_name  # Replace old_env_name with the actual name

## Create new environment
conda create --name new_env_name python=3.8
conda activate new_env_name
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
