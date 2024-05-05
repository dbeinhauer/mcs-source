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


def get_sheetname(sheet: str):
    """
    """
    if sheet == "V1_Inh_L2/3":
        sheet = "V1_Inh_L23"
    if sheet == "V1_Exc_L2/3":
        sheet= "V1_Exc_L23"
    return sheet

# run export




def get_segments(dsv):
    """
    Retrieves segments for both blanks and images in chronological order as they 
    were presented in the experiments.
    :param dsv: datastore containing data information.
    :returns: Two objects of segments for blanks and images from `dsv`.
    """
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



def get_neurons_info(sorted_segment): #-> (int, dict):
    """
    Retrieve information about number of neurons and create mapping dictionary of neuron indices.
    :param sorted_segment: segment to retrieve information from.
    :returns: total number of neurons and mapping dictionary of original neuron indices to the new one.
    """
    # index_mapping = {neuron.annotations['source_id']: i for i, neuron in enumerate(sorted_segment.spiketrains)}
    return len(sorted_segment.spiketrains)#, index_mapping

def get_segment_duration(segment) -> int:
    """
    Retrieve duration of the segment.
    :param segment: segment to get duration from.
    :returns: duration of the segment in ms.
    """
    return int(segment.spiketrains[0].duration) + 1

import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process directory name and sheet.")

    parser.add_argument("--path", 
                        type=str, 
                        default="/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/NewDataset_Images_from_50000_to_50100_ParameterSearch_____baseline:50000_trial:0", 
                        help="Name of the directory.")
    parser.add_argument("--sheet", type=str, default="V1_Exc_L2/3", help="Name of the sheet.")
    parser.add_argument("--subset", type=int, default=-1, help="How big subset of sheet to take.")

    return parser.parse_args()


def neuron_iteration(img_id: int, seg_blank, seg_image, blank_spikes, image_spikes):
    """
    Iterate through all neurons, and extract their spikes for the given image.
    :param img_id: index of the image in the spikes array.
    :param seg_blank: blank part of the segment.
    :param seg_image: stimulus part of the segment.
    :param blank_spikes: np.array of spikes for all blank parts and neurons.
    :param image_spikes: np.array of spikes for all stimulus parts and neurons.
    """
    for neuron_id, (spikes_blank, spikes_image) in enumerate(zip(
            # sorted_blank_spiketrains,
            # sorted_images_spiketrains
            seg_blank.spiketrains, 
            seg_image.spiketrains
        )):

        blank_spikes[img_id, neuron_id, spikes_blank.times.magnitude.astype(int)] += 1
        image_spikes[img_id, neuron_id, spikes_image.times.magnitude.astype(int)] += 1



# from memory_profiler import profile

# @profile
def main(args):
    # args = parse_arguments()

    # path = sys.argv[1]
    # sheet = sys.argv[2]
    # sheet = 'V1_Exc_L2/3'
    # path = '/CSNG/baroni/mozaik-models/LSV1M/20240116-093251[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:[0]}/NewDataset_Images_from_0_to_100_ParameterSearch_____baseline:0_trial:0'
    sheet = args.sheet
    dsv = get_datastore(args.path)


    dsv = param_filter_query(dsv, st_name='NaturalImage')
    # sheets = ['V1_Exc_L2/3', 'V1_Inh_L2/3', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
    # sheet_folders  = ['V1_Exc_L23', 'V1_Inh_L23', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']

    trials = sorted(list(set( MozaikParametrized.idd(s).trial for s in dsv.get_stimuli())))
    img_paths =  sorted(list(set(MozaikParametrized.idd(s).image_path for s in dsv.get_stimuli())))

    setup_logging()
    logger = mozaik.getMozaikLogger()



    images_id_list = []
    images_mapping = {}
    num_neurons = 0
    index_mapping = None
    num_images = 0
    blank_duration = 0
    image_duration = 0

    blank_spikes = None
    image_spikes = None

   
    if len(trials) == 1:
        print(f'There is a single trial')
        dsv = param_filter_query(dsv, sheet_name = sheet)
        for trial in trials:
            dsv = param_filter_query(dsv, st_trial = trial)

            # Should be chronologically sorted segments.
            segs_blank, segs_images = get_segments(dsv)
            dsv = None

            
            num_images = len(segs_blank)
            print(num_images)


            # Take just subset of the segments (for testing)
            if args.subset != -1:    
                segs_blank = segs_blank[0:args.subset]
                segs_images = segs_images[0:args.subset]

            if blank_spikes is None:
                num_neurons = get_neurons_info(segs_blank[0])
                # num_neurons, index_mapping = get_neurons_info(segs_blank[0])
                blank_duration = get_segment_duration(segs_blank[0])
                image_duration = get_segment_duration(segs_images[0])
                blank_spikes = np.zeros((num_images, num_neurons, blank_duration), dtype=np.uint8)
                image_spikes = np.zeros((num_images, num_neurons, image_duration), dtype=np.uint8)
                # blank_spikes = np.zeros((num_neurons, blank_duration), dtype=np.uint8)
                # image_spikes = np.zeros((num_neurons, image_duration), dtype=np.uint8)
                

            
            # Images_id_list -> store it
            # images_id_list = np.array([get_image_id(seg) for seg in segs_blank])
            print("Storing Img IDs")
            np.save("/home/beinhaud/diplomka/dataset_creation/dataset/testing_img_ids.npy",
                    np.array([get_image_id(seg) for seg in segs_blank]))

            # Neuron IDs list -> store it
            # neuron_ids_list = [spikes.annotations['source_id'] for spikes in segs_blank[0].spiketrains]
            print("Storing neuron IDs")
            np.save("/home/beinhaud/diplomka/dataset_creation/dataset/testing_neuron_ids.npy",
                    np.array([spikes.annotations['source_id'] for spikes in segs_blank[0].spiketrains]))

            # parallel_process_trials(segs_blank, segs_images, blank_spikes, image_spikes)
            # Iterate pairs blank->image (in this order)
            # Note: They are order chronologically (in correct order).
            for img_id, (seg_blank, seg_image) in enumerate(tqdm(zip(segs_blank, segs_images))):

                # Get Image Index
                # image_id = get_image_id(seg_blank)
                print("NEW TRIAL")
                print("-------------------------------------------")
                # print(f"Image number: {image_id}")
                print(f"Trial number: {img_id}")
                print()
              
                neuron_iteration(img_id, seg_blank, seg_image, blank_spikes, image_spikes)
                # Get ID of the neuron, its spike times, and duration of the image presentation.
                # for neuron_id, (spikes_blank, spikes_image) in enumerate(zip(
                #         # sorted_blank_spiketrains,
                #         # sorted_images_spiketrains
                #         seg_blank.spiketrains, 
                #         seg_image.spiketrains
                #     )):

                #     blank_spikes[img_id, neuron_id, spikes_blank.times.magnitude.astype(int)] += 1
                #     image_spikes[img_id, neuron_id, spikes_image.times.magnitude.astype(int)] += 1


                # gc.collect()

    print("Iteration finished!")
    print("Saving blanks")
    np.save("/home/beinhaud/diplomka/dataset_creation/dataset/testing_blank.npy",  blank_spikes)
    print("Saving images")
    np.save("/home/beinhaud/diplomka/dataset_creation/dataset/testing_image.npy",  image_spikes)

    # print("Job finished")



if __name__ == "__main__":

    args = parse_arguments()
    main(args)

    # # path = sys.argv[1]
    # # sheet = sys.argv[2]
    # # sheet = 'V1_Exc_L2/3'
    # # path = '/CSNG/baroni/mozaik-models/LSV1M/20240116-093251[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:[0]}/NewDataset_Images_from_0_to_100_ParameterSearch_____baseline:0_trial:0'
    # sheet = args.sheet
    # dsv = get_datastore(args.path)


    # dsv = param_filter_query(dsv, st_name='NaturalImage')
    # # sheets = ['V1_Exc_L2/3', 'V1_Inh_L2/3', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
    # # sheet_folders  = ['V1_Exc_L23', 'V1_Inh_L23', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']

    # trials = sorted(list(set( MozaikParametrized.idd(s).trial for s in dsv.get_stimuli())))
    # img_paths =  sorted(list(set(MozaikParametrized.idd(s).image_path for s in dsv.get_stimuli())))

    # setup_logging()
    # logger = mozaik.getMozaikLogger()



    # images_id_list = []
    # images_mapping = {}
    # num_neurons = 0
    # index_mapping = None
    # num_images = 0
    # blank_duration = 0
    # image_duration = 0

    # blank_spikes = None
    # image_spikes = None

   
    # if len(trials) == 1:
    #     print(f'There is a single trial')
    #     dsv = param_filter_query(dsv, sheet_name = sheet)
    #     for trial in trials:
    #         dsv = param_filter_query(dsv, st_trial = trial)

    #         # Should be chronologically sorted segments.
    #         segs_blank, segs_images = get_segments(dsv)
    #         dsv = None

            
    #         num_images = len(segs_blank)
    #         print(num_images)


    #         # Take just subset of the segments (for testing)
    #         # segs_blank = segs_blank[0:2]
    #         # segs_images = segs_images[0:2]

    #         if blank_spikes is None:
    #             num_neurons = get_neurons_info(segs_blank[0])
    #             # num_neurons, index_mapping = get_neurons_info(segs_blank[0])
    #             blank_duration = get_segment_duration(segs_blank[0])
    #             image_duration = get_segment_duration(segs_images[0])
    #             # blank_spikes = np.zeros((num_images, num_neurons, blank_duration), dtype=np.uint8)
    #             # image_spikes = np.zeros((num_images, num_neurons, image_duration), dtype=np.uint8)
    #             blank_spikes = np.zeros((num_neurons, blank_duration), dtype=np.uint8)
    #             image_spikes = np.zeros((num_neurons, image_duration), dtype=np.uint8)
                

            

    #         images_id_list = [get_image_id(seg) for seg in segs_blank]
    #         # img_ids = np.arange(num_images)
    #         neuron_ids_list = [spikes.annotations['source_id'] for spikes in segs_blank[0].spiketrains]
    #         # neuron_ids = np.arange(num_neurons)

    #         # parallel_process_trials(segs_blank, segs_images, blank_spikes, image_spikes)
    #         # Iterate pairs blank->image (in this order)
    #         # Note: They are order chronologically (in correct order).
    #         for img_id, (seg_blank, seg_image) in enumerate(tqdm(zip(segs_blank, segs_images))):

    #             # Get Image Index
    #             # image_id = get_image_id(seg_blank)
    #             print("NEW TRIAL")
    #             print("-------------------------------------------")
    #             # print(f"Image number: {image_id}")
    #             print(f"Trial number: {img_id}")
    #             print()
              
    #             # Get ID of the neuron, its spike times, and duration of the image presentation.
    #             for neuron_id, (spikes_blank, spikes_image) in enumerate(zip(
    #                     # sorted_blank_spiketrains,
    #                     # sorted_images_spiketrains
    #                     seg_blank.spiketrains, 
    #                     seg_image.spiketrains
    #                 )):

    #                 blank_spikes[img_id, neuron_id, spikes_blank.times.magnitude.astype(int)] += 1
    #                 image_spikes[img_id, neuron_id, spikes_image.times.magnitude.astype(int)] += 1

    #             # gc.collect()

    # # print("Job finished")
