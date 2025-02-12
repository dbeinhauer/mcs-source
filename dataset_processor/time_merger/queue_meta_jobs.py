from pathlib import Path
import subprocess

PROJECT_ROOT = Path("/storage/praha1/home/$USER/mcs-source")

SHEETS = (
    "V1_Exc_L23",
    "V1_Exc_L4",
    "V1_Inh_L23",
    "V1_Inh_L4",
    "X_OFF",
    "X_ON",
)
VARIANTS = ('train_dataset', 'test_dataset')
TIME_MERGER_DIR = PROJECT_ROOT / 'dataset_processor' / 'time_merger'
BASE_DIR = PROJECT_ROOT / 'dataset'
INTERVAL_SIZE = 20
SHELL_SCRIPT = TIME_MERGER_DIR / "meta_job.sh"

for variant in VARIANTS:
    input_dir = BASE_DIR / variant / 'trimmed_spikes'
    out_dir = BASE_DIR / variant / 'compressed_spikes' / 'trimmed' / f'size_{INTERVAL_SIZE}'
    for sheet in SHEETS:
        args = [str(x) for x in [SHELL_SCRIPT, input_dir, out_dir, INTERVAL_SIZE, sheet]]
        args = ['qsub', '--'] + args
        print(' '.join(args))
        subprocess.run(args)