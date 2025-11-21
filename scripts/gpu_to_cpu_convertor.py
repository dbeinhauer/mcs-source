import pickle

def convert_gpu_to_cpu(input_file: str, output_file: str):
    """
    Converts a pickle file containing GPU tensors to CPU tensors.

    :param input_file: Path to the input pickle file with GPU tensors.
    :param output_file: Path to the output pickle file with CPU tensors.
    """
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    def move_to_cpu(obj):
        if hasattr(obj, 'cpu'):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(move_to_cpu(v) for v in obj)
        else:
            return obj

    cpu_data = move_to_cpu(data)

    with open(output_file, "wb") as f:
        pickle.dump(cpu_data, f)

    print(f"Converted data saved to {output_file}")
    
def convert_all_directory_to_cpu(input_dir: str, output_dir: str):
    import os
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".pkl"):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                convert_gpu_to_cpu(input_file, output_file)
    
if __name__ == "__main__":
    base_path = "/home/beinhaud/diplomka/mcs-source/thesis_results/evaluation/invisible_0.9_reg_0.01/"
    model_path = "model-2_sub-var-0_visib-0.9_step-20_lr-3e-05_rnn_separate_opt-steps-10_dis-reg-0.0095-sig-0.2_neuron-layers-3-size-10-res-True_hid-time-1_optim-default_p-red-False_loss-poisson_synaptic-True-size-10-layers-3"
    # input_file = base_path + model_path + file_name
    # output_path = base_path + "/full_evaluation_results/" + model_path + file_name
    # convert_gpu_to_cpu(input_file, output_path)
    convert_all_directory_to_cpu(base_path + model_path, base_path + "/full_evaluation_results/" + model_path)