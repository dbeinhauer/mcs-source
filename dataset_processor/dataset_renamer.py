"""
Script to rename all the spiketrains files for given neuronal population to predefined pattern.
"""

import os
import re
import argparse

SPIKES_PREFIX = "spikes_"
TRIALS_PREFIX = "trial_"
FILE_POSTFIX = ".npz"


def rename_files(args):
    """
    Iterates through the directory and renames all files containing the spikes
    based on the given template.

    Note: Default template and also expected variant in further operations
    with the dataset is:
        `spikes_[trial_{trial_id}]_{experiment_id}.npz
    Where `[]` means that this part is optional (only when multitrial experiments).
    :param args: command line arguments specifying the renaming process.
    """
    # Regular expression to match filenames containing numbers.
    pattern = re.compile(r"(\d+)")

    for filename in os.listdir(args.directory):
        # Find all numbers in the filename.
        match = pattern.findall(filename)

        # Check there is a number in the filename.
        if match:
            # Assuming we take the last number found in the filename.
            image_id = match[-1]
            trial_id = ""
            if args.multitrial:
                # Multitrial processing -> trial ID is the penultimate number in the filename.
                trial_id = match[0]

            new_filename = args.template.format(trial_id=trial_id, image_id=image_id)
            old_file = os.path.join(args.directory, filename)
            new_file = os.path.join(args.directory, new_filename)

            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {filename} to {new_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Rename files in a directory based on numbers in the filenames."
    )
    parser.add_argument(
        "directory", type=str, help="The directory containing the files to rename."
    )
    parser.add_argument(
        "--multitrial", action="store_true", help="If the processing is multitrial."
    )
    parser.set_defaults(multitrial=False)
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="The template for the new filenames prefix.",
    )

    args = parser.parse_args()
    if args.template == None:
        args.template = SPIKES_PREFIX
        if args.multitrial:
            args.template += TRIALS_PREFIX + "{trial_id}_"
        args.template += "{image_id}" + FILE_POSTFIX
    rename_files(args)


if __name__ == "__main__":
    main()
