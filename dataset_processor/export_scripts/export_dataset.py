#!/usr/bin/env python3

import sys
from unicodedata import name
from mozaik.controller import Global, setup_logging
from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet
from mozaik.storage.queries import param_filter_query
import mozaik
import os
import gc
import pickle
import numpy as np
from mozaik.tools.mozaik_parametrized import MozaikParametrized
from tqdm import tqdm
import logging
import imagen 
from imagen.image import BoundingBox
from mozaik.stimuli.vision.topographica_based import MaximumDynamicRange

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

import argparse 

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process directory name and sheet.")

    parser.add_argument("--input_path", 
                        type=str, 
                        default="/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/NewDataset_Images_from_50000_to_50100_ParameterSearch_____baseline:50000_trial:0", 
                        help="Path to input data.")
    parser.add_argument("--output_path", type=str, default="/home/beinhaud/diplomka/mcs-source/dataset", help="Path where to store the output.")
    parser.add_argument("--sheet", type=str, default="V1_Exc_L2/3", help="Name of the sheet.")
    parser.add_argument("--subset", type=int, default=-1, help="How big subset of sheet to take.")


    return parser.parse_args()



def get_datastore(root):
    Global.root_directory = root
    datastore = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": root, "store_stimuli": False}),
        replace=True,
    )
    return datastore


def extract_images(path: str) -> str:
    path = path.split("from_")[1].split("_")[0]
    return path

def pickledump(path: str, file):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def reconstruct_stimuli(s):
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
    """
    if sheet == "V1_Inh_L2/3":
        sheet = "V1_Inh_L23"
    if sheet == "V1_Exc_L2/3":
        sheet= "V1_Exc_L23"
    return sheet

# run export


def get_segments(sheet: str, datastore_path: str):
    """
    Retrieves segments for both blanks and images in chronological order as they 
    were presented in the experiments.
    :param sheet: sheet identifier (type of neuron population).
    :param datastore_path: path to data.
    :returns: Two objects of segments for blanks and images from `dsv`.
    """
    # Get datastore
    dsv = get_datastore(datastore_path)#args.input_path)
    dsv = param_filter_query(dsv, st_name='NaturalImage')

    # TODO: Probably not needed because number of trials should be always `==1`
    trials = sorted(list(set( MozaikParametrized.idd(s).trial for s in dsv.get_stimuli())))
    # img_paths =  sorted(list(set(MozaikParametrized.idd(s).image_path for s in dsv.get_stimuli())))

    # Get data for specific sheet and trial.
    dsv = param_filter_query(dsv, sheet_name=sheet)
    dsv = param_filter_query(dsv, st_trial=trials[0])

    # Retrieve ordered segments.
    segs_blank = dsv.get_segments(null=True,ordered=True)
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
    Sort based on the neuron ID.
    """
    def sorting_key(spike_train):
        # Calculate the average firing rate for each SpikeTrain
        return spike_train.annotations['source_id']
    
    return sorted(spike_trains, key=sorting_key)



def get_neurons_info(sorted_segment):
    """
    Retrieve information about number of neurons and create mapping dictionary of neuron indices.
    :param sorted_segment: segment to retrieve information from.
    :returns: total number of neurons and mapping dictionary of original neuron indices to the new one.
    """
    return len(sorted_segment.spiketrains)

def get_segment_duration(segment) -> int:
    """
    Retrieve duration of the segment.
    :param segment: segment to get duration from.
    :returns: duration of the segment in ms.
    """
    return int(segment.spiketrains[0].duration) + 1


def image_iteration(segs_blank, segs_images, spikes: np.array, blank_duration: int, logs: bool=False):
    """
    Iterate through all images, and extract spiketrains info from them.
    :param segs_blank: segments containing the time intervals for blank period (blank image presented).
    :param segs_images: segments contatining the time intervals for stimulus period (images presented).
    :param spikes: np.array of spikes for all images and neurons 
    (shape: num_images*num_neurons*blank_and_image_duration), blank is always before image.    
    :param blank_duration: duration of the blank segment during the experiment.
    :param logs: `True` if we want to print logs.
        
    :param blank_spikes: array to store spikes from blank periods.
    :param image_spikes: array to store spikes from stimulus periods.
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


def neuron_iteration(img_id: int, seg_blank, seg_image, spikes: np.array, blank_offset: int):
    """
    Iterate through all neurons, and extract their spikes for the given image.
    :param img_id: index of the image in the spikes array.
    :param seg_blank: blank part of the segment.
    :param seg_image: stimulus part of the segment.
    :param spikes: np.array of spikes for all images and neurons 
    (shape: num_images*num_neurons*blank_and_image_duration), blank is always before image.
    :param blank_offset: size of blank_time_interval.

    :param blank_spikes: np.array of spikes for all blank parts and neurons.
    :param image_spikes: np.array of spikes for all stimulus parts and neurons.
    """
    for neuron_id, (spikes_blank, spikes_image) in enumerate(zip(
            seg_blank.spiketrains, 
            seg_image.spiketrains
        )):
        spikes[img_id, neuron_id, spikes_blank.times.magnitude.astype(int)] += 1
        spikes[img_id, neuron_id, spikes_image.times.magnitude.astype(int) + blank_offset] += 1


def save_image_ids(segs, filename: str):
    """
    Save image IDs into file of `np.array` object.
    :param segs: segments object to get the IDs from.
    :param filename: name of the file where to store the ids.
    """
    print("Storing Img IDs")
    np.save(filename, np.array([get_image_id(seg) for seg in segs]))


def save_neuron_ids(segs, filename: str):
    """
    Save neuron IDs into file of `np.array` object.
    :param segs: segments object to get the IDs from.
    :param filename: name of the file where to store the ids.
    """
    print("Storing neuron IDs")
    np.save(filename, np.array([spikes.annotations['source_id'] for spikes in segs[0].spiketrains]))


def save_spiketrains(
        spikes: np.array, 
        num_neurons: int, 
        args, 
        spikes_subdirectory: str="/spikes/", 
        spikes_prefix: str="spikes",
    ):
    """
    Reshapes the spikes into shape (num_neurons * (images*time_slots)), converts it to 
    sparse representation, and stores it into the file.
    :param spikes: array of spikes.
    :param num_neurons: number of neurons in the given sheet.
    :param args: CL arguments.
    :param spikes_subdirectory: where to store the spikes.
    :param spikes_prefix: prefix of the file containing spikes.
    """

    # Reshape to 2D matrix (num_neurons * (images*time_slots)) and convert to sparse representation.
    sparse_spikes = csr_matrix(spikes.transpose(1, 0, 2).reshape(num_neurons, -1))
    print("Saving spike trains")
    save_npz(
        args.output_path + spikes_subdirectory + get_sheetname(args.sheet) + "/" + create_filename(args, spikes_prefix), 
        sparse_spikes
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
    :param args:
    :param variant:
    :return: filename in format '{variant}_{sheet_name}_{part_id}.npz'
    """
    # part_id = args.input_path.split(':')[-2].split('_')[0]
    part_id = get_dataset_part_id(args.input_path)
    print(f"part id: {part_id}")
    postfix = ".npz"
    if np_postfix:
        postfix = "npy"
    return variant + "_" + get_sheetname(args.sheet) + "_" + part_id + postfix



# @profile
def main(args):
    # Names of the output subdirectories
    image_ids_subdirectory = "/image_ids/"
    neuron_ids_subdirectory = "/neuron_ids/"
    spikes_subdirectory = "/spikes/"

    # Files prefixes:
    image_ids_prefix = "image_ids"
    neuron_ids_prefix = "neuron_ids"
    spikes_prefix = "spikes"

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

    segs_blank, segs_images = get_segments(args.sheet, args.input_path)

    num_images = len(segs_blank)

    # Take just subset of the segments (for testing)
    if args.subset != -1:    
        segs_blank = segs_blank[0:args.subset]
        segs_images = segs_images[0:args.subset]

    # Prealocate memory for the dataset.
    if spikes is None:
        num_neurons = get_neurons_info(segs_blank[0])
        blank_duration = get_segment_duration(segs_blank[0])
        image_duration = get_segment_duration(segs_images[0])
        spikes = np.zeros((num_images, num_neurons, blank_duration + image_duration), dtype=np.uint8)        
        
    
    # Save Image IDs and Neuron IDs
    save_image_ids(
        segs_blank, 
        args.output_path + image_ids_subdirectory + create_filename(args, image_ids_prefix, np_postfix=True)
    )
    save_neuron_ids(
        segs_blank, 
        args.output_path + neuron_ids_subdirectory + create_filename(args, neuron_ids_prefix, np_postfix=True)
    )

    # Save and extract the spike trains.
    image_iteration(segs_blank, segs_images, spikes, blank_duration, logs=False)
    save_spiketrains(spikes, num_neurons, args)

    print()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)