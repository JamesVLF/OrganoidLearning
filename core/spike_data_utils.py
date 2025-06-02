# spike_data_utils.py - Utilities for SpikeData analysis
import numpy as np

def calculate_mean_firing_rates(spike_data):
    return np.array([
        len(neuron_spikes) / (spike_data.length / 1000)
        for neuron_spikes in spike_data.train
    ])
