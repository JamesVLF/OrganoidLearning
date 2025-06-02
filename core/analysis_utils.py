# analysis_utils.py - Matrix-based analyses (correlation, PCA, etc.)
import numpy as np
from scipy.ndimage import gaussian_filter1d

def get_correlation_matrix(spike_data, bin_size=1):
    raster = spike_data.raster(bin_size=bin_size).astype(float)
    raster = gaussian_filter1d(raster, sigma=5)
    return np.corrcoef(raster)
