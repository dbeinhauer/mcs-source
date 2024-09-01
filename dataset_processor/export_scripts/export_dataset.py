#!/usr/bin/env python3

import os
import gc
import pickle
import sys
import logging
import argparse 
from unicodedata import name

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from parameters import ParameterSet
import imagen 
from imagen.image import BoundingBox
import mozaik
from mozaik.controller import Global, setup_logging
from mozaik.storage.queries import param_filter_query
from mozaik.storage.datastore import PickledDataStore
from mozaik.tools.mozaik_parametrized import MozaikParametrized
from mozaik.stimuli.vision.topographica_based import MaximumDynamicRange

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

# Names of the output subdirectories
IMAGES_IDS_SUBDIR = "/image_ids/"
NEURONS_IDS_SUBDIR = "/neuron_ids/"
SPIKES_SUBDIR = "/spikes/"

# Files prefixes:
IMAGE_IDS_PREFIX = "image_ids_"
NEURONS_IDS_PREFIX = "neuron_ids_"
SPIKES_PREFIX = "spikes_"
# Prefix when multiple trials 
TRIALS_PREFIX = "trial_"


# Default output directory:
OUTPUT_DIR = "/home/beinhaud/diplomka/mcs-source/dataset"


def get_datastore(root: str):
    """
    Gets mozaik datastore from the given path.
    :param root: root directory where the mozaik datastore is stored.
    :return: returns the mozaik datastore object from the given root.
    """
    Global.root_directory = root
    datastore = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": root, "store_stimuli": False}),
        replace=True,
    )
    return datastore

def extract_images(path: str) -> str:
    """
    # TODO: not used now.
    :param path: _description_
    :return: _description_
    """
    path = path.split("from_")[1].split("_")[0]
    return path

def pickledump(path: str, file):
    """
    # TODO: now used now
    Creates pickle dump file of the provided file.
    :param path: path where the pickle path should be stored.
    :param file: file to convert to pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(file, f)

def reconstruct_stimuli(s):
    """
    TODO: not used in the current version.
    # Should reconstruct the image from the stimuli.
    :param s: _description_
    :return: _description_
    """
    pattern_sampler = imagen.image.PatternSampler(
            size_normalization="fit_shortest",
            whole_pattern_output_fns=[MaximumDynamicRange()],
        )

    img = imagen.image.FileImage(
        filename=s.image_path,
        x=0,
        y=0,
        orientation=0,
        xdensity=s.density,
        ydensity=s.density,
        size=s.size,
        bounds=BoundingBox(
            points=(
                (-s.size_x / 2, -s.size_y / 2),
                (s.size_x / 2, s.size_y / 2),
            )
        ),
        scale=2 * s.background_luminance,
        pattern_sampler=pattern_sampler,
    )
    return img()

def get_sheetname(sheet: str) -> str:
    """
    TODO: maybe just use the correct sheet name instead of running this function.
    Renames the sheet name to format that can be used in the filename.
    :param sheet: original sheet name provided to program.
    :return: returns modified sheetname to format without '/' symbols.
    """
    if sheet == "V1_Inh_L2/3":
        sheet = "V1_Inh_L23"
    if sheet == "V1_Exc_L2/3":
        sheet= "V1_Exc_L23"
    return sheet

def get_modified_datastore(path: str):
    """
    Loads the provided datastore, filters it to take only 'NaturalImage' 
    and creates the list of trials for this datastore.
    :param path: path of the datastore. 
    :return: returns the tuple of modified datastore and 
    list of trials sorted by ID (used mainly in test dataset).
    """
    dsv = get_datastore(path)
    dsv = param_filter_query(dsv, st_name='NaturalImage')
    # Trials list necessary for selecting specific trials stimuli (mainly for test dataset).
    trials = sorted(list(set( MozaikParametrized.idd(s).trial for s in dsv.get_stimuli())))

    return dsv, trials

def get_all_segments(dsv, trial, sheet: str):
    """
    Retrieves segments for both blanks and images in chronological order as they
    were presented in the experiments.
    :param dsv: mozaik datastore object filtered for 'NaturalImage'.
    :param trial: trial to get data for (from trials object).
    :param sheet: sheet identifier (type of neuronal population). Possible values
    are: ["X_ON", "X_OFF", "V1_Exc_L23", "V1_Inh_L23", "V1_Exc_L4", "V1_Inh_L4"]
    :returns: tuple of blank segments and stimuli (image) 
    segments extracted from provided datstore.
    """
    # Get data for specific sheet and trial.
    dsv_modified = param_filter_query(dsv, sheet_name=sheet)
    dsv_modified = param_filter_query(dsv_modified, st_trial=trial)

    # Retrieve ordered segments (watch the parameters for 
    # selecting the blank and image segments).
    segs_blank = dsv.get_segments(null=True, ordered=True)
    segs_image = dsv.get_segments(ordered=True)

    return segs_blank, segs_image

def get_image_id(segment) -> str:
    """
    Retrieves index of the image of the given segment.
    :param segment: segment to obtain image information from.
    :returns: id of the image corresponding to segment.
    """
    stimulus = MozaikParametrized.idd(segment.annotations['stimulus'])
    return stimulus.image_path.split('/')[-1].split('_')[0]

def sort_spiketrains(spike_trains):
    """
    Sort spiketrains based on the neuron ID.
    :param spike_trains: spike trains object to be sorded.
    :return: returns sorted spike trains object based on its neuron IDs.
    """
    def sorting_key(spike_train):
        """
        Get sorting key for spike trains dataset.
        :param spike_train: spike train to get the sorting key.
        :return: returns ID of the source neuron.
        """
        return spike_train.annotations['source_id']
    
    return sorted(spike_trains, key=sorting_key)

def get_neurons_info(segment) -> int:
    """
    Retrieves information about number of neurons.    
    :param segment: segment to retrieve information from.
    :returns: total number of neurons in the given segment.
    """
    return len(segment.spiketrains)

def get_segment_duration(segment) -> int:
    """
    Retrieves duration of the segment.
    :param segment: segment to get duration from.
    :returns: duration of the segment in ms.
    """
    # We add 1 because the the time steps are stored in interval (0, duration).
    return int(segment.spiketrains[0].duration) + 1

def image_iteration(
        segs_blank, 
        segs_images, 
        spikes: np.array, 
        blank_duration: int, 
        logs: bool=False,
    ):
    """
    Iterates through all images and extracts spiketrains info from them.
    :param segs_blank: segments containing time intervals for blank
    period (blank image presented).
    :param segs_images: segments contatining the time intervals 
    for stimulus period (images presented).
    :param spikes: array of spikes for all images and neurons. 
    (shape: num_images*num_neurons*blank_and_image_duration), blank is always before image.    
    :param blank_duration: duration of the blank segment during the experiment.
    :param logs: `True` if we want to print logs.
    """
    for img_id, (seg_blank, seg_image) in enumerate(tqdm(zip(segs_blank, segs_images))):
        if logs:
            # Get Image Index
            print("NEW TRIAL")
            print("-------------------------------------------")
            print(f"Trial number: {img_id}")
            print()
            
        neuron_iteration(img_id, seg_blank, seg_image, spikes, blank_duration)

    print("Iteration finished!")


def neuron_iteration(
        img_id: int, 
        seg_blank, 
        seg_image, 
        spikes: np.array, 
        blank_offset: int,
    ):
    """
    Iterates through all neurons and extracts all spikes for the given image.
    :param img_id: index of the image in the spikes array.
    :param seg_blank: blank part of the segments.
    :param seg_image: stimulus (image) part of the segments.
    :param spikes: np.array of spikes for all images and neurons.
    (shape: num_images*num_neurons*blank_and_image_duration), blank is always before image.
    :param blank_offset: duration of blank interval in the experiment.
    """
    for neuron_id, (spikes_blank, spikes_image) in enumerate(zip(
            seg_blank.spiketrains, 
            seg_image.spiketrains
        )):
        # Assign each spike of the given neuron in the given image to our spike representation.
        spikes[img_id, neuron_id, spikes_blank.times.magnitude.astype(int)] += 1
        spikes[img_id, neuron_id, spikes_image.times.magnitude.astype(int) + blank_offset] += 1


def save_image_ids(segs, filename: str):
    """
    Saves image IDs into file of `np.array` object.
    :param segs: segments object to get the IDs from.
    :param filename: name of the file where to store the IDs.
    """
    print("Storing Img IDs")
    np.save(filename, np.array([get_image_id(seg) for seg in segs]))

def save_neuron_ids(segs, filename: str):
    """
    Saves neuron IDs into file of `np.array` object.
    :param segs: segments object to get the IDs from.
    :param filename: name of the file where to store the ids.
    """
    print("Storing neuron IDs")
    np.save(filename, np.array([spikes.annotations['source_id'] for spikes in segs[0].spiketrains]))

def save_spiketrains(
        trial_spikes: list,
        num_neurons: int, 
        args, 
        spikes_subdirectory: str=SPIKES_SUBDIR, 
        spikes_prefix: str=SPIKES_PREFIX,
        trials_prefix: str=TRIALS_PREFIX,
    ):
    """
    Reshapes the spikes for each trial into shape (num_neurons * (images*time_slots)), 
    converts it to sparse representation and stores it into the .npz file. 
    The output filename format is: 
        `output_path/spike_subdirectory/sheet/spikes_[trial_{trial_ID}]_{sheet}_{ID}.npz`
    Where `trial_{trial_ID}` is optional and is present only for multiple trials processing.
    :param trial_spikes: list of `np.array` objects of spikes for all trials.
    :param num_neurons: number of neurons in the given sheet.
    :param args: command line arguments.
    :param spikes_subdirectory: where to store the spikes.
    :param spikes_prefix: prefix of the file containing spikes.
    :param trials_prefix: part of the prefix when multiple trials processing.
    """
    for i, spikes in enumerate(trial_spikes):
        # Reshape to 2D matrix (num_neurons * (images*time_slots)) and convert to sparse representation.
        sparse_spikes = csr_matrix(spikes.transpose(1, 0, 2).reshape(num_neurons, -1))
        print(f"Saving spike trains for trial: {i}")
        base_path = args.output_path + spikes_subdirectory + get_sheetname(args.sheet) + "/"
        filename = create_spikes_filename(
                args, 
                spikes_prefix, 
                f"{trials_prefix}_{i}_", # Specify trial ID in the filename. 
                len(trial_spikes) == 1,
            )
        save_npz(
            base_path + filename,
            sparse_spikes,
        )

def prealocate_spikes(segs_blank, segs_images):
    """
    Creates spikes `np.array` (prealocates the array).
    :param segs_blank: all blank segments.
    :param segs_images: all stimuli (image) segments.
    :return: returns `np.array` of zeros in the shape 
    (num_images, num_neurons, blank_and_image_duration).
    """
    num_images = len(segs_blank)
    # Prealocate memory for the dataset at the begining (for faster extraction).
    num_neurons = get_neurons_info(segs_blank[0])
    blank_duration = get_segment_duration(segs_blank[0])
    image_duration = get_segment_duration(segs_images[0])
    return np.zeros(
        (num_images, num_neurons, blank_duration + image_duration), 
        dtype=np.uint8,
    )

def get_dataset_part_id(input_path: str) -> str:
    """
    Filters the dataset part ID from the path.
    :param input_path: path to get ID from.
    :return: ID of the dataset part.
    """
    return input_path.split(':')[-2].split('_')[0]

def create_filename(args, variant: str, np_postfix=False) -> str:
    """
    Creates filename based on the provided parameters.
    :param args: command line arguments.
    :param variant: what to store (variants: 'images_ids', 'neurons_ids', 'spikes_')
    :param np_postfix: `True` if we want to store numpy array (posfix 'npy'), 
    otherwise `False` store 'npz' object.
    :return: filename in format '{variant}_{sheet_name}_{part_id}.npz'.
    """
    part_id = get_dataset_part_id(args.input_path)
    print(f"Part id: {part_id}")
    postfix = ".npz"
    if np_postfix:
        # Numpy object filename.
        postfix = ".npy"
    return variant + get_sheetname(args.sheet) + "_" + part_id + postfix

def create_spikes_filename(args, spikes_prefix: str, trials_prefix: str, single_trial: bool) -> str:
    """
    Creates filename for the spikes file.
    :param args: command line arguments.    
    :param spikes_prefix: prefix of all spikes files.
    :param trials_prefix: trial part of prefix (in case there are multiple files).
    :param single_trial: `True` if single trial processing.
    :return: Filename in format:
           `output_path/spike_subdirectory/sheet/spikes_[trial_{trial_ID}]_{sheet}_{ID}.npz`
    Where `trial_{trial_ID}` is optional and is present only for multiple trials processing.
    """
    prefix = spikes_prefix
    if not single_trial:
        # Multiple trials -> add also info about the trial number.
        prefix += trials_prefix
    return create_filename(args, prefix)

def main(args):
    num_neurons = 0
    num_images = 0
    blank_duration = 0
    image_duration = 0
    spikes = None
    
    print("----------NEW TRIAL---------------")
    print(f"EXPERIMENT_SETTING: sheet-{get_sheetname(args.sheet)}, ID-{get_dataset_part_id(args.input_path)}")
    print("----------------------------------")

    setup_logging()
    logger = mozaik.getMozaikLogger()


    dsv, trials = get_modified_datastore(args.input_path)
    trials_spikes = []
    # Process all the trials.
    for i, trial in enumerate(trials):
        if i != -1 and i > args.num_trials:
            # All wanted trials extracted (do not extract the rest).
            break

        # Get segments for the 
        segs_blank, segs_images = get_all_segments(dsv, trial, args.sheet)

        if args.subset != -1:    
            # Take just subset of the segments (for testing).
            segs_blank = segs_blank[0:args.subset]
            segs_images = segs_images[0:args.subset]

        spikes = prealocate_spikes(segs_blank, segs_images)

        # Save and extract the spike trains.
        image_iteration(segs_blank, segs_images, spikes, blank_duration, logs=False)
        trials_spikes.append(spikes)            
    
    # Save Image IDs and Neuron IDs
    save_image_ids(
        segs_blank, 
        args.output_path + IMAGES_IDS_SUBDIR + create_filename(args, IMAGE_IDS_PREFIX, np_postfix=True),
    )
    save_neuron_ids(
        segs_blank, 
        args.output_path + NEURONS_IDS_SUBDIR + create_filename(args, NEURONS_IDS_PREFIX, np_postfix=True),
    )
    # Save the spike trains.
    save_spiketrains(trials_spikes, num_neurons, args)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the dataset from the provided raw data for the given sheet.")
    parser.add_argument("--input_path", type=str, #/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/NewDataset_Images_from_50000_to_50100_ParameterSearch_____baseline:50000_trial:0", 
        help="Path to input data.")
    parser.add_argument("--output_path", type=str, default=None,# default="/home/beinhaud/diplomka/mcs-source/dataset", 
        help="Path where to store the output.")
    parser.add_argument("--sheet", type=str,
        help="Name of the sheet. Possibilities: ['X_ON', 'X_OFF', 'V1_Exc_L23', 'V1_Inh_L23', 'V1_Exc_L4', 'V1_Inh_L4']")
    parser.add_argument("--subset", type=int, default=-1, 
        help="How big subset of the sheet to take (if `-1` then whole sheet).")
    parser.add_argument("--num_trials", type=int, default=-1,
        help="How many trials extract (if `-1` then all of them).")

    args = parser.parse_args()

    if args.output_path == None:
        args.output_path = OUTPUT_DIR
    main(args)