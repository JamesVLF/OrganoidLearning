# OrgLearningEval.py - Central class interface

from core.data_loader import load_datasets
from core.spike_data_utils import calculate_mean_firing_rates
from core.analysis_utils import get_correlation_matrix
from viz.plots_general import plot_raster, plot_firing_rates, plot_correlation_matrix

DATASET_PATHS = {
    "Baseline": "data/baseline_spike_data.pkl",
    "Adaptive": "data/exp1_cartpole_long_6_spike_data.pkl",
    "Random": "data/exp1_cartpole_long_7_spike_data.pkl",
    "None": "data/exp1_cartpole_long_8_spike_data.pkl"
}

class OrgLearningEval:
    def __init__(self, dataset_paths=DATASET_PATHS):
        self.datasets = load_datasets(dataset_paths)
        self.sd_main = self.datasets.get("Baseline")
        print(f"Loaded default dataset: 'Baseline'")

    def set_dataset(self, name):
        if name in self.datasets:
            self.sd_main = self.datasets[name]
            print(f"Switched to dataset: {name}")
        else:
            raise ValueError(f"Dataset '{name}' not found.")

    def show_raster(self):
        plot_raster(self.sd_main)

    def show_mean_firing_rates(self):
        rates = calculate_mean_firing_rates(self.sd_main)
        plot_firing_rates(rates)

    def show_correlation_matrix(self):
        matrix = get_correlation_matrix(self.sd_main)
        plot_correlation_matrix(matrix)
