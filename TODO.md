# TODO:

## Data Aggregation from the dataset for analysis
1. histogram neuron spike rate
    - `num bins: 712` (highest possible spike rate)
2. histogram time bin spike rate
    - `num bins: 20` (highest possible spike rate)
3. sum of spikes in each time bin
    - shape:    `[num_time_bins]`
        - maximal size: 712
4. sum spikes in each trial and experiment:
    - shape:    `[num_experiments, num_trials]`
    - max size: 50000
5. mean and variance of spikes per neuron in separate trials
    - shape: `[num_experiments, num_trials]`
    - max size: 50000
6. for each neuron total spikes: 
    - shape `[num_neurons]`
    - max size: 33000
7. Fano factor across trials (makes sense only for the test dataset)
    - shape: `[experiment, num_neurons]`
        - max size ~1650000000
            - might be problem
                - less than 8. (since only for test)
    - need:
        - sum across time -> `[trials, neurons]`
            - mean and variance across trials -> fano_factors = vars / means
8. Fano factor across time bins:
    - shape: `[experiment, num_neurons]`
        - max size ~1650000000
            - might be problem
            - it is approx 
    - need:
        - mean across trials -> `[time_bins, neurons]`
            - mean and variance across time bins -> fano = vars / means
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