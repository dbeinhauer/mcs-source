"""
Script to rename all the spiketrains files for given neuronal population to predefined pattern.
"""
import os
import re
import argparse

SPIKES_PREFIX = "spikes_"
TRIALS_PREFIX = "trials_"
FILE_POSTFIX = ".npz"

def rename_files(directory: str, template: str, multitrial: bool=False):
    """
    Iterates through the directory and renames all files containing the spikes 
    based on the given template.
    :param directory: directory containing all the spike files for renaming.
    :param template: template for the new filename (typically 'spikes_{number}.npz').
    :param multitrial: `True` if processing multitrials, else `False`.
    """
    # Regular expression to match filenames containing numbers.
    pattern = re.compile(r'(\d+)')
    
    for filename in os.listdir(directory):
        # Find all numbers in the filename.
        match = pattern.findall(filename)
        
        # Check there is a number in the filename.
        if match:
            # Assuming we take the last number found in the filename.
            image_id = match[-1]
            trial_id = ""
            if multitrial:
                # Multitrial processing -> trial ID is the penultimate number in the filename.
                trial_id = match[-2]
            new_filename = template.format(trial_id=trial_id, image_id=image_id)
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {filename} to {new_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Rename files in a directory based on numbers in the filenames.")
    parser.add_argument('directory', type=str, 
        help='The directory containing the files to rename.')
    parser.add_argument('--multitrial', type=bool, default=False,
        help="If the processing is multitrial.")
    parser.add_argument('--template', type=str, default=None, 
        help='The template for the new filenames prefix.')
    
    args = parser.parse_args()
    if args.template == None:
        args.template = SPIKES_PREFIX
        if args.multitrial:
            args.template += TRIALS_PREFIX + "{trial_id}_"
        args.template += "{image_id}." + FILE_POSTFIX
    rename_files(args.directory, args.template)

if __name__ == "__main__":
    main()
