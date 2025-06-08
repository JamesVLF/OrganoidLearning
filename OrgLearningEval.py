"""
OrgLearningEval.py - Central class interface for methods aimed at evaluating the adaptive learning
capacity of organoids through dynamic control tasks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.data_loader import load_spike_data, load_log_data, load_causal_info, load_metadata
from core.spike_data_utils import calculate_mean_firing_rates
from core.analysis_utils import get_correlation_matrix
import core.analysis_utils
import core.map_utils
import viz.plots_general
import core.spike_data_utils
import viz.burst_analysis


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

        self.latency_histograms = {}  # condition → {(i, j): histogram}
        self.latency_bins = None
        self.firing_orders = {}       # condition → list of (sequence, count)

        self.bursts = {}  # condition -> list of bursts (from detect_population_bursts)

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
        viz.plots_general.plot_raster(self.sd_main)

    def show_mean_firing_rates(self):
        rates = core.spike_data_utils.calculate_mean_firing_rates(self.sd_main)
        viz.plots_general.plot_firing_rates(rates)

    def show_correlation_matrix(self):
        matrix = core.analysis_utils.get_correlation_matrix(self.sd_main)
        viz.plots_general.plot_correlation_matrix(matrix)

    def show_causal_plot(self):
        core.analysis_utils.causal_plot(self.causal_info)

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
        viz.plots_general.plot_smoothed_reward(df, label=condition, color=color or "darkblue")

    def show_all_rewards_overlay(self):
        viz.plots_general.plot_all_rewards_overlay(self.log_data)

    def show_reward_vs_causal_plot(self, condition="Adaptive"):
        pattern_log = self.log_data[condition]["pattern"]
        return viz.plots_general.plot_reward_vs_causal_metrics(self.causal_info, pattern_log, return_df=True)

    def show_performance_summary(self, bin_size=60, conditions=None):
        """
        Compare time-balanced performance (episode duration) across conditions.

        Parameters:
            bin_size: size of each time bin in seconds
            conditions: list of condition names (e.g. ["Adaptive", "Random", "Null"])
        """
        if conditions is None:
            conditions = list(self.log_data.keys())

        summary = core.analysis_utils.compute_time_balanced_over_time(self.log_data, conditions, bin_size)
        viz.plots_general.plot_performance_summary(summary)

    def show_pole_angle_trajectories(self, condition="Adaptive", angle_thresh_deg=16):
        """
        Plot pole angle with overlayed training times from reward log,
        filtered to occur near the failure threshold (±angle_thresh_deg).
        """
        if condition not in self.log_data:
            raise ValueError(f"Condition '{condition}' not found in log data.")

        game_df = self.log_data[condition]["game"]
        reward_df = self.log_data[condition]["reward"]

        viz.plots_general.plot_pole_angle_trajectories(
            game_df,
            reward_df,
            condition_label=condition,
            angle_thresh_deg=angle_thresh_deg
        )

    def show_time_balanced_vs_training(self, condition="Adaptive", color="green", max_time_min=15.0, window=5):
        """
        Plot episode durations (time balanced) and training signal over time.
        """
        if condition not in self.log_data:
            raise ValueError(f"Condition '{condition}' not found.")

        reward_df = self.log_data[condition]["reward"]
        pattern_df = self.log_data[condition]["pattern"]
        viz.plots_general.plot_time_balanced_with_training(
            reward_df=reward_df,
            pattern_df=pattern_df,
            label=condition,
            color=color,
            window=window,
            max_time_min=max_time_min
        )

    def show_time_balanced_cumulative_plot(self, conditions=None, window=5, max_minutes=None, show_raw=True):
        # Show cumulative time balanced plot for given conditions
        viz.plots_general.plot_time_balanced_cumulative(
            log_data=self.log_data,
            conditions=conditions,
            window=window,
            max_minutes=max_minutes,
            show_raw=show_raw
        )


    def show_training_pair_plot(self, num_neurons=10):
        viz.plots_general.plot_training_pair_metrics(
            self.causal_info["first_order_connectivity"],
            self.causal_info["multi_order_connectivity"],
            self.causal_info["burst_percent"],
            num_neurons
        )

    def show_connectivity_heatmaps(self):
        viz.plots_general.plot_connectivity_heatmaps(
            self.causal_info["first_order_connectivity"],
            self.causal_info["multi_order_connectivity"]
        )

    def show_burst_raster(self, start_time=0, end_time=None):
        viz.plots_general.plot_raster_pretty(self.sd_main, l1=start_time, l2=end_time or (self.sd_main.length / 1000), analyze=True)

    def show_neuron_raster_comparison(self, neuron_id=0, start_s=0, end_s=20):
        """
        Show stacked spike raster plots for a neuron across conditions.
        """

        spike_datasets = []
        for condition in ["Adaptive", "Random", "Null"]:
            if condition in self.spike_data:
                spike_datasets.append((condition, self.spike_data[condition]))

        viz.plots_general.plot_neuron_raster_comparison_stacked(
            spike_datasets,
            neuron_id=neuron_id,
            start_s=start_s,
            end_s=end_s
        )

    def compute_latency_histograms(self, window_ms=30, bin_size=5):
        for cond, sd in self.spike_data.items():
            histograms, bins = core.spike_data_utils.compute_latency_histograms(sd, window_ms, bin_size)
            self.latency_histograms[cond] = histograms
            self.latency_bins = bins
    def show_latency_histogram(self, condition, i, j):
        if condition not in self.latency_histograms:
            raise ValueError(f"Call compute_all_latency_histograms() first.")
        viz.plots_general.plot_latency_histogram_pair(self.latency_histograms[condition], self.latency_bins, i, j)

    def compute_firing_orders(self, window_ms=50, top_k=5):
        for cond, sd in self.spike_data.items():
            self.firing_orders[cond] = core.spike_data_utils.extract_common_firing_orders(sd, window_ms=window_ms, top_k=top_k)

    def show_firing_order_summary(self, condition):
        if condition not in self.firing_orders:
            raise ValueError(f"Call compute_firing_orders() first.")
        viz.plots_general.plot_top_firing_orders(self.firing_orders[condition])

    def show_rank_order_analysis(self, condition="Adaptive", unit_ids=None, threshold_std=2.0, **kwargs):
        """
        Visualize rank-order consistency between bursts using Spearman correlation and z-scores.
        """
        if condition not in self.spike_data:
            raise ValueError(f"Condition '{condition}' not found in spike data.")

        sd = self.spike_data[condition]
        bursts = core.spike_data_utils.detect_population_bursts(sd, threshold_std=threshold_std, **kwargs)

        if not bursts:
            print("No bursts found.")
            return

        all_units = np.unique(sd.idces_times()[0])
        if unit_ids is None:
            unit_ids = all_units

        peak_time_matrix = core.spike_data_utils.extract_peak_times(sd, bursts, unit_ids)
        rho, z = core.spike_data_utils.compute_rank_corr_and_zscores(peak_time_matrix)

        viz.plots_general.plot_rank_order_matrix(rho, title=f"Rank Order Correlation - {condition}")
        viz.plots_general.plot_zscore_matrix(z, title=f"Z-scored Correlation - {condition}")

    def show_rank_order_violin(self):
        """
        Display rank order correlation z-scores across conditions.
        """
        data_dict = {}
        unit_ids = list(range(self.sd_main.N))

        # May be refined to better reflect actual burst structure
        for condition in ["Null", "Random", "Adaptive"]:
            sd = self.spike_data[condition]
            bursts = sd.metadata.get("bursts", None)
            if not bursts:
                print(f"No burst metadata for {condition}")
                continue

            # Possibly split Adaptive into pre/post if time permits
            if condition == "Adaptive" and "log" in self.log_data[condition]:
                stim_time = self.log_data[condition]["log"]["time"].min()
                pre = [b for b in bursts if b.get("t_peak", 0) < stim_time * 1000]
                post = [b for b in bursts if b.get("t_peak", 0) >= stim_time * 1000]

                for label, subset in zip(["Adaptive-Pre", "Adaptive-Post"], [pre, post]):
                    if len(subset) >= 2:
                        peak_mat = core.spike_data_utils.extract_peak_times(sd, subset, unit_ids)
                        _, z = core.spike_data_utils.compute_rank_corr_and_zscores(peak_mat)
                        data_dict[label] = z
            else:
                if len(bursts) >= 2:
                    peak_mat = core.spike_data_utils.extract_peak_times(sd, bursts, unit_ids)
                    _, z = core.spike_data_utils.compute_rank_corr_and_zscores(peak_mat)
                    data_dict[condition] = z

        viz.plots_general.plot_rank_order_violin(data_dict)

    def show_backbone_burst_heatmaps(self, condition="Adaptive", window_ms=1000, threshold_std=2.0, bin_size=10):
        spike_data = self.spike_data[condition]
        bursts = core.spike_data_utils.detect_population_bursts(spike_data, threshold_std=threshold_std)
        aligned = core.spike_data_utils.extract_aligned_spikes(spike_data, bursts, window_ms=window_ms)

        # Filter backbone units only
        min_burst_frac = 0.8
        num_bursts = len(bursts)
        backbone_units = [i for i, spks in enumerate(aligned) if (np.sum(np.array(spks) >= 0) / num_bursts) >= min_burst_frac]
        aligned_backbone = [aligned[i] for i in backbone_units]

        fr_matrix = core.spike_data_utils.compute_normalized_firing_rate_matrix(aligned_backbone, duration_ms=window_ms, bin_size=bin_size)
        core.spike_data_utils.plot_backbone_aligned_heatmaps(aligned_backbone, fr_matrix, window_ms=window_ms, bin_size=bin_size)

    def show_architecture_map(self):
        core.map_utils.plot_architecture_map(self.metadata)
    def show_combined_electrode_neuron_map(self):
        core.map_utils.plot_combined_electrode_neuron_map(self.metadata)

    def compute_bursts(self, threshold_std=2.0, **kwargs):
        for cond, sd in self.spike_data.items():
            self.bursts[cond] = core.spike_data_utils.detect_population_bursts(sd, threshold_std=threshold_std, **kwargs)

    def show_burst_stats(self, condition: str):
        sd = self.spike_data[condition]
        bursts = self.bursts[condition]
        return core.spike_data_utils.analyze_burst_distributions(sd, condition_label=condition, burst_func=lambda *_: bursts)

    def show_within_burst_dynamics(self, condition: str):
        sd = self.spike_data[condition]
        bursts = self.bursts[condition]
        return core.spike_data_utils.analyze_within_burst_firing(sd, bursts)

    def show_burst_latency_consistency(self, condition: str):
        sd = self.spike_data[condition]
        bursts = self.bursts[condition]
        return core.spike_data_utils.analyze_latency_consistency(sd, bursts)

    def show_burst_propagation(self, condition: str):
        sd = self.spike_data[condition]
        bursts = self.bursts[condition]
        return core.spike_data_utils.analyze_burst_propagation(sd, bursts)

    def get_reward_linked_bursts(self, condition: str, window: int = 250):
        bursts = self.bursts[condition]
        log_df = self.log_data[condition]["log"]
        return viz.burst_analysis.match_bursts_to_log(bursts, log_df, window=window)

    def plot_reward_outcome_distribution(self, condition: str, window: int = 250, ax=None):
        rewards, _ = self.get_reward_linked_bursts(condition, window)
        reward_df = viz.burst_analysis.compute_reward_labels(rewards)
        return viz.burst_analysis.plot_burst_reward_histogram(reward_df, ax=ax)

    def plot_reward_linked_condition_comparison(self, cond1: str, cond2: str, window: int = 250, ax=None):
        rewards1, _ = self.get_reward_linked_bursts(cond1, window)
        rewards2, _ = self.get_reward_linked_bursts(cond2, window)
        return viz.burst_analysis.plot_reward_linked_bursts(rewards1, rewards2, ax=ax)

    def compare_burst_stats(self):
        stats = {}
        for cond, sd in self.spike_data.items():
            bursts = self.bursts[cond]
            result = core.spike_data_utils.analyze_burst_distributions(
                sd,
                condition_label=cond,
                burst_func=lambda *_: bursts
            )
            stats[cond] = result
        return stats

    def compare_within_burst_dynamics(self, bin_size=20):
        plt.figure(figsize=(8, 5))
        for cond, sd in self.spike_data.items():
            bursts = self.bursts[cond]
            avg_rate = core.spike_data_utils.analyze_within_burst_firing(sd, bursts, bin_size=bin_size)
            t = np.arange(len(avg_rate)) * bin_size
            plt.plot(t, avg_rate, label=cond)

        plt.xlabel('Time in Burst (ms)')
        plt.ylabel('Avg Pop Firing Rate')
        plt.title('Within-Burst Dynamics by Condition')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_burst_latency_consistency(self):
        plt.figure(figsize=(10, 4))
        for cond, sd in self.spike_data.items():
            bursts = self.bursts[cond]
            std_dev = core.spike_data_utils.analyze_latency_consistency(sd, bursts)
            plt.plot(std_dev, label=cond)

        plt.xlabel("Neuron Index")
        plt.ylabel("Latency Std Dev (ms)")
        plt.title("Burst Latency Consistency Across Conditions")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_burst_propagation(self):
        plt.figure(figsize=(10, 4))
        for cond, sd in self.spike_data.items():
            bursts = self.bursts[cond]
            com_mean = core.spike_data_utils.analyze_burst_propagation(sd, bursts)
            plt.plot(com_mean, label=cond)

        plt.xlabel("Neuron Index")
        plt.ylabel("Mean First Spike Time (ms)")
        plt.title("Burst Propagation Across Conditions")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def compute_causal_matrices(self, condition="Adaptive", max_latency_ms=200, bin_size=5):
        """
        Compute first- and multi-order causal matrices based on latency histograms.
        Stores matrices under `self.causal_matrices`.
        """
        sd = self.spike_data[condition]
        first, multi = core.spike_data_utils.infer_causal_matrices(sd, max_latency_ms=max_latency_ms, bin_size=bin_size)
        self.causal_latency_matrices = {condition: first}
        self.multi_order_matrices = {condition: multi}

    def show_latency_histogram_for_pair(self, condition="Adaptive", i=2, j=5):
        """
        Show latency histogram for neuron pair (i → j).
        """
        if condition not in self.latency_histograms:
            raise ValueError("Run compute_causal_latency_matrices() first.")
        viz.plots_general.plot_latency_histogram_pair(
            self.latency_histograms[condition],
            self.latency_bins,
            i,
            j,
            condition_label=condition
        )

    def show_spatial_connectivity_map(self, condition="Adaptive", order="first"):
        """
        Visualize spatial connectivity between neurons based on inferred causal matrix.

        Parameters:
            condition: one of ["Adaptive", "Random", "Null"]
            order: "first" or "multi"
        """
        matrix = None
        if order == "first":
            matrix = self.causal_latency_matrices.get(condition)
        elif order == "multi":
            matrix = self.multi_order_matrices.get(condition)
        else:
            raise ValueError("order must be 'first' or 'multi'")

        if matrix is None:
            raise ValueError("Run compute_causal_latency_matrices() first.")

        positions = self.metadata.get("mapping", None)
        if positions is None:
            raise ValueError("No neuron spatial mapping found in metadata.")

        viz.plots_general.plot_spatial_connectivity_map(matrix, positions, title=f"{condition} - {order.title()} Order")

    @staticmethod
    def show_metric_scatter_plots(df):
        viz.plots_general.plot_reward_vs_metrics(df)
    @staticmethod
    def show_quartile_comparison_plot(df):
        viz.plots_general.plot_top_vs_bottom_quartiles(df)
    @staticmethod
    def show_metric_correlation_plot(df):
        viz.plots_general.plot_metric_correlations(df)