# TODO:

## Data Aggregation from the dataset for analysis
1. histogram neuron spike rate - `HISTOGRAM_NEURON_SPIKE_RATES`
    - `num bins: 712` (highest possible spike rate)
2. histogram time bin spike rate - `HISTOGRAM_TIME_BIN_SPIKE_RATES`
    - `num bins: 20` (highest possible spike rate)
3. sum of spikes in each time bin - `TIME_BIN_SPIKE_COUNTS`
    - shape:    `[num_time_bins]`
        - maximal size: 712

Dimension `[num_experiments, num_trials]` experiments:
1. spike counts - `TOTAL_COUNT`
2. mean across neurons - `MEAN`
3. variance across neurons - `VARIANCE`
4. spike density - `DENSITY`
    - count non-zero time bins -> mean across time
    - how sparse is the dataset
5. average Per-bin synchrony - how much neurons spiked across one time bin - `SYNCHRONY`

4. sum spikes in each trial and experiment: - `EXPERIMENT_SPIKE_COUNTS`
    - shape:    `[num_experiments, num_trials]`
    - max size: 50000
5. mean and variance of spikes per neuron in separate trials - `NEURON_MEAN_VARIANCE_EXPERIMENT`
    - shape: `[num_experiments, num_trials]`
    - max size: 50000
6. for each neuron total spikes: 
    - shape `[num_neurons]`
    - max size: 33000
7. a) Fano factor across trials (makes sense only for the test dataset)
    - shape: `[experiment, num_neurons]`
        - max size ~1650000000
            - might be problem
                - less than 8. (since only for test)
    - need:
        - sum across time -> `[trials, neurons]`
            - mean and variance across trials -> fano_factors = vars / means (over neurons)
7. b) Fano Factor across trials - makes sence only for test
    - shape: `[experiment]`
    - need:
        - sum across spikes and time bins -> fano = vars / mean (over trials)
8. Fano factor across time bins:
    - shape: `[experiment, num_neurons]`
        - max size ~1650000000
            - might be problem
            - it is approx 
    - need:
        - mean across trials -> `[time_bins, neurons]`
            - mean and variance across time bins -> fano = vars / means (over time)
9. Temporal Synchrony:
    - shape: `[experiment, time_bins]`
        - max size: 35600000
            - might be problem
    - needs:
        - mean across trials -> `[time_bins, neurons]`
            - sum across neurons -> synchrony curve
10. Per trial Synchrony:
    - shape: `[experiment, trials, time_bins]`
        - max size: 35600000
            - might be problem
            - may overflow in test with multiple trials (probably not really)
    - same as 9. just do not mean across trials

- We know that shape: `[experiment, 35, neurons]` is not suitable
    - approx max size: ~57750000000