# analysis_utils.py - Matrix-based analyses (correlation, PCA, etc.)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def causal_plot(causal_info):
    figlayout = "AB"
    fig, plot = plt.subplot_mosaic(figlayout, figsize=(12, 5))
    fig.suptitle("Causal Connectivity Matrices")

    # First-order
    pltA = plot["A"].imshow(causal_info["first_order_connectivity"], cmap='Greens')
    plot["A"].set_title("First Order (10–15 ms)")
    plot["A"].set_xlabel("Reactivity Index")
    plot["A"].set_ylabel("Stimulus Index")
    fig.colorbar(pltA, ax=plot["A"], shrink=0.7)

    # Multi-order
    pltB = plot["B"].imshow(causal_info["multi_order_connectivity"], cmap='Greens')
    plot["B"].set_title("Nth Order (200 ms)")
    plot["B"].set_xlabel("Reactivity Index")
    plot["B"].set_ylabel("Stimulus Index")
    fig.colorbar(pltB, ax=plot["B"], shrink=0.7)

    plt.tight_layout()
    plt.show()

def get_correlation_matrix(spike_data, bin_size=1):
    raster = spike_data.raster(bin_size=bin_size).astype(float)
    raster = gaussian_filter1d(raster, sigma=5)
    return np.corrcoef(raster)
