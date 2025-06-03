# OrgLearningEval.py - Central class interface

from core.data_loader import load_datasets, load_spike_data, load_log_data, load_causal_info, load_metadata
from core.spike_data_utils import calculate_mean_firing_rates
from core.analysis_utils import get_correlation_matrix, causal_plot
from viz.plots_general import plot_raster, plot_firing_rates, plot_correlation_matrix, plot_smoothed_reward, plot_all_rewards_overlay

DEFAULT_SPIKE_PATHS = {
    "Baseline": "data/baseline_spike_data.pkl",
    "Adaptive": "data/exp1_cartpole_long_6_spike_data.pkl",
    "Random":   "data/exp1_cartpole_long_7_spike_data.pkl",
    "Null":     "data/exp1_cartpole_long_8_spike_data.pkl"
}

DEFAULT_LOG_PATHS = {
    "Adaptive": "data/exp1_cartpole_long_6_logs.pkl",
    "Random":   "data/exp1_cartpole_long_7_logs.pkl",
    "Null":     "data/exp1_cartpole_long_8_logs.pkl"
}

class OrgLearningEval:
    def __init__(self, spike_paths=None, log_paths=None):
        self.spike_paths = spike_paths or DEFAULT_SPIKE_PATHS
        self.log_paths = log_paths or DEFAULT_LOG_PATHS

        self.spike_data = load_spike_data(self.spike_paths)
        self.log_data = load_log_data(self.log_paths)

        self.causal_info = load_causal_info()
        self.metadata = load_metadata()

        # Set default spike dataset
        if "Baseline" in self.spike_data:
            self.sd_main = self.spike_data["Baseline"]
            print("Loaded default dataset: 'Baseline'")
        else:
            raise ValueError("Baseline dataset not found in spike_data.")

        # Set default dataset
        self.set_dataset("Baseline")

        print("Loaded default dataset: 'Baseline'")

    def set_dataset(self, name):
        if name in self.spike_data:
            self.sd_main = self.spike_data[name]
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

    def show_causal_plot(self):
        causal_plot(self.causal_info)

    def get_reward_df(self, condition):
        # Return reward DataFrame for a training condition
        if condition in self.log_data:
            return self.log_data[condition]["reward"]
        else:
            raise ValueError(f"No log data found for condition '{condition}'")

    def show_reward_plot(self, condition, color=None):
        # Show smoothed reward plot for a training condition
        if condition not in self.log_data:
            raise ValueError(f"No log data found for condition '{condition}'")

        df = self.log_data[condition]["reward"]
        plot_smoothed_reward(df, label=condition, color=color or "darkblue")

    def show_all_rewards_overlay(self):
        plot_all_rewards_overlay(self.log_data)