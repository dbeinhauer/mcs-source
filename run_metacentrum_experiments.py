import itertools
import os
import sys
import subprocess


def load_settings(settings_file):
    """Load settings from the configuration file."""
    settings = {}
    with open(settings_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Handle lists (e.g., LEARNING_RATE=(0.001 0.0001))
            if value.startswith("(") and value.endswith(")"):
                value = value[1:-1].split()
                # Convert true/false strings in lists to booleans
                value = [
                    v.lower() == "true" if v.lower() in ["true", "false"] else v
                    for v in value
                ]
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            settings[key] = value
    return settings


def generate_combinations(settings):
    """Generate all combinations of parameters that are lists."""
    keys = []
    values = []
    for key, value in settings.items():
        if isinstance(value, list) and len(value) > 1:
            keys.append(key)
            values.append(value)
    # Generate all combinations of list parameters
    combinations = list(itertools.product(*values))
    return keys, combinations


def build_command(base_command, settings, keys, combination):
    """Build the command for a specific combination of parameters."""
    cmd = [base_command]
    for key, value in settings.items():
        if key in keys:
            # Use the value from the current combination
            value = combination[keys.index(key)]
        if isinstance(value, bool):
            if value:  # Add the flag only if it's true
                cmd.append(f"--{key.lower()}")
        else:
            cmd.append(f"--{key.lower()} {value}")
    return " ".join(cmd)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python metacentrum_scripts/thesis_analysis/run_selected_experiments.py <settings_file>"
        )
        sys.exit(1)

    settings_file = sys.argv[1]

    if not os.path.isfile(settings_file):
        print(f"Error: Settings file '{settings_file}' not found.")
        sys.exit(1)

    # Load settings
    settings = load_settings(settings_file)

    # Ensure WANDB_NAME is provided
    if "WANDB_NAME" not in settings or not settings["WANDB_NAME"]:
        print("Error: WANDB_NAME is required in the settings file.")
        sys.exit(1)

    # Generate combinations of list parameters
    keys, combinations = generate_combinations(settings)

    # If no combinations, run with the default settings
    if not combinations:
        combinations = [()]

    # Base command
    base_command = "bash metacentrum_scripts/thesis_analysis/job_template.sh"

    # Handle SIZE_MULTIPLIER separately
    size_multipliers = settings.get("SIZE_MULTIPLIER", [None])

    # Run each combination
    for combination in combinations:
        cmd = build_command(base_command, settings, keys, combination)
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
