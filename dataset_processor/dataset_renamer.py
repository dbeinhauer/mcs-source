"""
Script to rename all the spiketrains files for given neuronal population to predefined pattern.
"""


import os
import re
import argparse

def rename_files(directory, template):
    # Regular expression to match filenames containing numbers
    pattern = re.compile(r'(\d+)')
    
    for filename in os.listdir(directory):
        # Find all numbers in the filename
        match = pattern.findall(filename)
        
        if match:
            # Assuming we take the last number found in the filename
            number = match[-1]
            new_filename = template.format(number=number)
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed {filename} to {new_filename}")

def main():
    parser = argparse.ArgumentParser(description="Rename files in a directory based on numbers in the filenames.")
    parser.add_argument('directory', type=str, help='The directory containing the files to rename.')
    parser.add_argument('--template', type=str, default='spikes_{number}.npz', help='The template for the new filenames.')
    
    args = parser.parse_args()
    
    rename_files(args.directory, args.template)

if __name__ == "__main__":
    main()
