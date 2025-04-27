"""
This scripts serves for either generating the scripts for executing metacentrum jobs that runs either
model training or evaluation based on the provided setup or for directly submitting the metacentrum jobs.
"""

# Model arguments:
import argparse
import subprocess


def load_template(template_path):
    """Reads the job script template from a file."""
    with open(template_path, "r") as file:
        return file.read()


def generate_job_script(template_path, **params):
    """Generates a PBS job script using a template file and writes it to a file."""
    script_template = load_template(template_path)
    return script_template.format(**params)


def submit_job(script_content):
    """Submits the PBS job script using qsub."""
    process = subprocess.Popen(["qsub"], stdin=subprocess.PIPE)
    process.communicate(input=script_content.encode())
    print("Job submitted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or submit a PBS job script.")
    # Input output arguments:
    parser.add_argument(
        "--template", default="job_template.pbs", help="Path to the job script template"
    )
    parser.add_argument(
        "--filename",
        default="job_script.sh",
        help="Filename for the generated job script",
    )
    # Machine specification arguments:
    parser.add_argument("--walltime", default="15:0:0", help="Walltime for the job")
    parser.add_argument("--ncpus", type=int, default=8, help="Number of CPU cores")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu_mem", default="40gb", help="GPU memory")
    parser.add_argument("--mem", default="100gb", help="Total system memory")
    parser.add_argument("--scratch_local", default="100gb", help="Local scratch space")
    parser.set_defaults(use_opt_arguments=False)
    parser.add_argument(
        "--use_opt_arguments",
        action="store_true",
        help="Use optional machine specification arguments",
    )
    parser.add_argument(
        "--opt_machine_args",
        type=str,
        default=":spec=8.0:gpu_cap=compute_86:osfamily=debian",
        help="Optional machine specification arguments",
    )
    parser.add_argument(
        "--size_multiplier",
        type=float,
        default=0.1,
        help="What model size we want to use (if not default).",
    )
    # Model arguments:
    parser.add_argument(
        "--model_params",
        type=str,
        default="",
        help="Additional model parameters to pass to execute_model.py",
    )

    # Submit job arguments:
    parser.set_defaults(submit_job=False)
    parser.add_argument(
        "--submit_job",
        action="store_true",
        help="Directly submits the job.",
    )
    args = parser.parse_args()

    if not args.use_opt_arguments:
        args.opt_machine_args = ""

    # Generate the job script from template
    script_content = generate_job_script(
        args.template,
        walltime=args.walltime,
        ncpus=args.ncpus,
        ngpus=args.ngpus,
        gpu_mem=args.gpu_mem,
        mem=args.mem,
        scratch_local=args.scratch_local,
        opt_machine_args=args.opt_machine_args,
        size_multiplier=args.size_multiplier,
        model_params=args.model_params,
    )

    # Submit the job
    if not args.submit_job:
        with open(args.filename, "w") as file:
            file.write(script_content)
        print(f"Job script saved as {args.filename}")
    else:
        # Submit the job directly
        submit_job(script_content)
