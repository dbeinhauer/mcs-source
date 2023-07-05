# Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes

## Abstract
- notes about the paper

- estimation of neuronal receptive fields is essential for understanding sensory
processing in the early visual system
- full characterization is still incomplete (especially natural stimuli and in 
complete population of cortical neurons)
- previous works:
    * incomporated known structural properties of the early VS
        - lateral connectivity
        - imposing simple-cell-like receptive field structure
    * no study:
        - exploited the fact that nearby V1 neurons share common feed-forward 
        input from thalamus and other upstream cortical neurons
- new method for estimating receptive fields simultaneously for population of V1 
(model based analysis incorporationg knowledge of the feed-forward visual hierarchy)
- we assume:
    * population of V1 neurons shares a common pool of thalamic inputs
    * V1 into 2 layers:
        - simple neurons
        - complex-like neurons
- fitted to recordings of a local population of mouse layer 2/3 V1 neurons
    * accurate description of their response to natural images
    * significant improvement of prediction power
- response of a large local population of V1 with locally diverse receptive fields
    * can be described with surprisingly limited number of thalamic inputs (consistent with experimental finding)
- improved functional characterization of V1, also a framework

## Author Summary
- goal in sensory neuroscience is to understand the relationship between sensory stimuli and patterns of activity they elicit in networks of sensory neurons
- many models in past
    * largely ignored the known architecture of primary visual cortex (experimentaly revealed)
    * limiting ability to accurately describe neural responses to sensory stimuli
- model of V1 - using known architecture of visual cortex
    * only a limited number of thalamic inputs with stereotypical receptive fields are shared withing a local area of visual cortex
    * hierarchical progression from linear receptive fiels (simple cells) 
    -> neurons with non-linear receptive fields (complex cells)
- model outpurforms state-of-the-art methods for receptive field estimation

## Introduction
- different external stimuli -> distinct activity patterns in 
early sensory processing (encodes the content of the stimuli)
- large set of stimuli while recordint the responses of individual neurons -> fit each neuron with model
- accuracy of model - comparing predicted and actual activities in response 
to novel stimulus set (classic ML)
- old studies:
    * filter functions of linear neurons in the retina, LGN and simple cells in V1
        - obtained using artificial sets of stimuli (sparse noise, M-sequences)
    * even less linear neurons - complex cells in V1 and neurons in V2
        - more representative stimuli (movies of natural scenes)
- recent years:
    * spike-triggered covariance (STC)
    * multi-layer NN
    -> estimation of non-linear receptive fields of complex cells
- previous work - data from single cells independent on each other
    * using genearlized linear models (GLM)
        - info about nearby neurons a few milliseconds in the past 
        -> improvement of predictive power
        - it is restricted to linear representation of RF
    * pre-defined baks of linear and non-linear filters to pre-process the 
    visual input -> using linear regression in the transformed input space to fit the model
        - better accuracy
    * no previous method:
        - RFs of a local population of neurons are constructed from limited number of shared LGN inputs
            * stereotypical center-surround RF structure
            * using two-photon calcium imaging -> possibility to record the activity from complete 
            population of neurons (allows estimation of a model containing these constrains)
- new method for estimating RFs in V1 - Hierarchical Structural Model (HSM)
    * assumption that the local neuronal population shares a limited number of afferent 
    inputs from the LGN
    * incorporates hierarchical sub-cortical and cortical processing
    * center-surround thalomo-cortical inputs 
        - are summed in the first layer of neurons - single cells
        - second layer - sums inputs from single cells -> single and complex-cell like RF
        - model takes advantage of the RF redundancies among nearby V1 neurons
            * simultaneously fitting the entire local population of recorded neurons


## Methods



## Results
- measurement:
    * measured neuronal responses with two-photon calcium imaging (TODO: co to je) 
    of local populations of mouse V1 neurons
    * presentation of large set of full-field natural images
    * recorded Ca signals in populations of layer 2/3 in V1 (in response to images)
    * each image in 500 ms interleaved with blank gray for 1474 ms
        - extra images as validation set
    * convertion of Ca signals into spikes - fast non-negative de-convolution method (TODO: co to je)

### Model-based RF estimation in mouse V1
- inspired by anatomical and functional organization of V1 in mammals
- assumptions:
    * LGN units described as difference-of-Gaussian functions
    * local population of V1 shares input from limited number of  LGN units
    * simple cells - constructed by summing several RFs fo LGN neurons (TODO: don't understand)
    * complex cells - summing inputs from the local population of simple cells that are selective to
    the same orientation but different RF phases
- HMS (hierarchical structural model):
    * 3 layers:
        1. linear kernels of LGN units - 2D difference-of-Gaussians
        2. sum of LGN-like responses -> passing result to logistic-loss non-linearity
            - contruct oriented RFs using feed-forward summation of thalomocortical inputs
        3. again sum with logistic-loss non-linearity
            - RFs tuned to orientation but unsensitive to spatial phase (complex cells)
    * generation of RFs which do not conform standard idealized models of either simple and 
    complex cells 