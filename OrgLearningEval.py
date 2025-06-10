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

        self.causal_latency_matrices = {}  # (condition, start_ms, end_ms) → matrix
        self.multi_order_matrices = {}     # (condition, start_ms, end_ms) → matrix

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

    def show_latency_spatial_map(self, condition="Adaptive", i=2, j=5, start_ms=0, end_ms=300000):
        sd = self.spike_data[condition].subtime(start_ms, end_ms)
        pos = self.metadata.get("mapping", None)
        if pos is None:
            raise ValueError("No neuron mapping found in metadata.")
        viz.plots_general.plot_latency_spatial_map(sd, pos, i, j)

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


    def compute_causal_matrices(self, condition="Adaptive", start_ms=0, end_ms=None,
                                max_latency_ms=200, bin_size=5):
        """
        Compute and store causal matrices for a condition, possibly over a time window.
        """
        sd = self.spike_data[condition]
        if end_ms is None:
            end_ms = sd.length

        # Time-slice the spike data
        sd_window = sd.subtime(start_ms, end_ms)

        first, multi = core.spike_data_utils.infer_causal_matrices(
            sd_window, max_latency_ms=max_latency_ms, bin_size=bin_size
        )
        self.causal_latency_matrices[(condition, start_ms, end_ms)] = first
        self.multi_order_matrices[(condition, start_ms, end_ms)] = multi

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

    def show_spatial_connectivity_map(self, condition="Adaptive", order="first", start_ms=0, end_ms=None):
        """
        Visualize spatial connectivity based on inferred causal matrix.

        Parameters:
            condition: one of ["Adaptive", "Random", "Null", etc.]
            order: "first" or "multi"
            start_ms, end_ms: optional time window (default is full recording)
        """
        key = (condition, start_ms, end_ms)

        if order == "first":
            matrix = self.causal_latency_matrices.get(key)
        elif order == "multi":
            matrix = self.multi_order_matrices.get(key)
        else:
            raise ValueError("order must be 'first' or 'multi'")

        if matrix is None:
            raise ValueError(f"No matrix found for {condition} [{start_ms}–{end_ms} ms]. "
                             "Run compute_causal_matrices() first.")

        positions = self.metadata.get("mapping", None)
        if positions is None:
            raise ValueError("No neuron spatial mapping found in metadata.")

        viz.plots_general.plot_spatial_connectivity_map(
            matrix, positions, title=f"{condition} - {order.title()} Order ({start_ms}–{end_ms} ms)"
        )

    def show_training_spatial_connectivity_all_conditions(self):
        """
        Show spatial connectivity plots (0–5 min) for training neurons across all conditions.
        Uses causal latency matrices.
        """
        spike_locs = np.array(self.metadata["spike_locs"])
        spikes_x = spike_locs[:, 0]
        spikes_y = spike_locs[:, 1]
        electrode_mapping = self.metadata["mapping"]

        viz.plots_general.plot_training_spatial_connectivity_all_conditions(
            self, spikes_x, spikes_y, electrode_mapping
        )

    def compare_causal_matrices(self, cond_a, cond_b, order="first", start_ms=0, end_ms=None, show_plot=True):
        """
        Compare causal matrices between two conditions or time windows.

        Parameters:
            cond_a, cond_b: condition names (e.g., "Baseline", "Adaptive")
            order: "first" or "multi"
            start_ms, end_ms: optional time window
            show_plot: if True, display difference heatmap

        Returns:
            diff_matrix: the matrix of differences (cond_b - cond_a)
            stats: dictionary of summary stats
        """
        key_a = (cond_a, start_ms, end_ms)
        key_b = (cond_b, start_ms, end_ms)

        if order == "first":
            mat_a = self.causal_latency_matrices.get(key_a)
            mat_b = self.causal_latency_matrices.get(key_b)
        elif order == "multi":
            mat_a = self.multi_order_matrices.get(key_a)
            mat_b = self.multi_order_matrices.get(key_b)
        else:
            raise ValueError("order must be 'first' or 'multi'")

        if mat_a is None or mat_b is None:
            raise ValueError(f"Missing matrix for comparison. Ensure compute_causal_matrices() "
                             f"was called for both {cond_a} and {cond_b}.")

        diff = mat_b - mat_a
        stats = {
            "mean_diff": np.mean(diff),
            "sum_abs_diff": np.sum(np.abs(diff)),
            "max_change": np.max(np.abs(diff))
        }

        if show_plot:
            plt.imshow(diff, cmap="bwr", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
            plt.colorbar(label="Δ Causal Value")
            plt.title(f"Δ Causal Matrix ({cond_b} – {cond_a}) [{order.title()}]")
            plt.xlabel("Target Neuron")
            plt.ylabel("Source Neuron")
            plt.show()

        return diff, stats

    def show_causal_plot_from_matrices(self, first_order, multi_order, title="", inds=None):
        """
        Plot causal connectivity heatmaps from matrices, optionally restricted to training neurons.
        """
        viz.plots_general.causal_plot_from_matrices(first_order, multi_order, title=title, training_inds=inds)

    def analyze_firing_orders_all_conditions(self, time_windows=None, save_dir=None):
        """
        Analyze firing orders for all conditions across multiple time windows

        Parameters:
        -----------
        time_windows : list of tuples
            List of (start_ms, end_ms) time windows to analyze
            Default: [(0, 300000), (600000, 900000)]
        save_dir : str
            Directory to save results and plots
        """

        if time_windows is None:
            time_windows = [(0, 300000), (600000, 900000)]

        if save_dir is None:
            save_dir = "firing_order_analysis"

        # Store results
        self.firing_order_results = {}

        print(f"Analyzing firing orders for {len(self.spike_data)} conditions...")
        print(f"Time windows: {time_windows}")

        for condition in self.spike_data.keys():
            print(f"\n=== Analyzing condition: {condition} ===")
            self.firing_order_results[condition] = {}

            for start_ms, end_ms in time_windows:
                window_key = f"{start_ms}_{end_ms}"
                print(f"  Time window: {start_ms}-{end_ms} ms")

                try:
                    # Analyze this condition and time window
                    results = self._analyze_single_condition_window(
                        condition, start_ms, end_ms, save_dir
                    )

                    self.firing_order_results[condition][window_key] = results

                except Exception as e:
                    print(f"    Error analyzing {condition} {window_key}: {e}")
                    continue

        # Generate comparison plots
        self._generate_comparison_plots(time_windows, save_dir)

        print(f"\nAnalysis complete! Results saved to: {save_dir}")
        return self.firing_order_results

    def _analyze_single_condition_window(self, condition, start_ms, end_ms, save_dir):
        """
        Analyze firing order for a single condition and time window
        """

        # Get causal matrices for this condition and time window
        first_order_key = (condition, start_ms, end_ms, 'first_order')
        multi_order_key = (condition, start_ms, end_ms, 'multi_order')

        # Check if matrices exist, if not compute them
        if first_order_key not in self.causal_latency_matrices:
            self._compute_causal_matrices(condition, start_ms, end_ms)

        first_order_matrix = self.causal_latency_matrices.get(first_order_key)
        multi_order_matrix = self.multi_order_matrices.get(multi_order_key)

        if first_order_matrix is None:
            raise ValueError(f"Could not compute causal matrices for {condition} {start_ms}-{end_ms}")

        # Run comprehensive firing order analysis
        analysis_results = self._run_comprehensive_analysis(
            first_order_matrix, multi_order_matrix
        )

        # Create and save visualizations
        self._create_condition_visualizations(
            condition, start_ms, end_ms, analysis_results,
            first_order_matrix, multi_order_matrix, save_dir
        )

        return analysis_results

    def _compute_causal_matrices(self, condition, start_ms, end_ms):
        """
        Compute causal matrices if they don't exist
        This should call your existing causal matrix computation methods
        """
        # This is a placeholder - replace with your actual causal matrix computation
        # You'll need to adapt this to use your existing methods

        print(f"    Computing causal matrices for {condition} {start_ms}-{end_ms}")

        # Example structure - replace with your actual computation:
        try:
            # Set the dataset
            self.set_dataset(condition)

            # Compute first-order matrix (±15ms)
            first_order_matrix = self._compute_first_order_matrix(start_ms, end_ms)
            first_order_key = (condition, start_ms, end_ms, 'first_order')
            self.causal_latency_matrices[first_order_key] = first_order_matrix

            # Compute multi-order matrix (±200ms)
            multi_order_matrix = self._compute_multi_order_matrix(start_ms, end_ms)
            multi_order_key = (condition, start_ms, end_ms, 'multi_order')
            self.multi_order_matrices[multi_order_key] = multi_order_matrix

        except Exception as e:
            print(f"    Error computing matrices: {e}")
            raise

    def _compute_first_order_matrix(self, start_ms, end_ms):
        """
        Placeholder for first-order causal matrix computation
        Replace with your actual method
        """
        # This should call your existing causal matrix computation with ±15ms window
        # Return a numpy array representing the causal connectivity matrix

        # Placeholder - replace with actual computation
        n_neurons = len(self.causal_info['neuron_ids'])
        return np.random.randn(n_neurons, n_neurons)  # Replace with actual computation

    def _compute_multi_order_matrix(self, start_ms, end_ms):
        """
        Placeholder for multi-order causal matrix computation
        Replace with your actual method
        """
        # This should call your existing causal matrix computation with ±200ms window
        # Return a numpy array representing the causal connectivity matrix

        # Placeholder - replace with actual computation
        n_neurons = len(self.causal_info['neuron_ids'])
        return np.random.randn(n_neurons, n_neurons)  # Replace with actual computation

    def _run_comprehensive_analysis(self, first_order_matrix, multi_order_matrix):
        """
        Run all firing order analysis methods
        """
        from core.analysis_utils import (  # Replace with actual import
            infer_firing_order_enhanced,
            infer_firing_order_hierarchical,
            infer_firing_order_topological,
            infer_firing_order_temporal_chains,
            infer_firing_order_consensus
        )

        results = {}

        # Enhanced method
        try:
            order, scores, details = infer_firing_order_enhanced(first_order_matrix)
            results['enhanced'] = {'order': order, 'scores': scores, 'details': details}
        except Exception as e:
            print(f"    Enhanced method failed: {e}")
            results['enhanced'] = None

        # Hierarchical method
        try:
            order, scores = infer_firing_order_hierarchical(first_order_matrix)
            results['hierarchical'] = {'order': order, 'scores': scores}
        except Exception as e:
            print(f"    Hierarchical method failed: {e}")
            results['hierarchical'] = None

        # Topological method
        try:
            order, status = infer_firing_order_topological(first_order_matrix)
            results['topological'] = {'order': order, 'status': status}
        except Exception as e:
            print(f"    Topological method failed: {e}")
            results['topological'] = None

        # Temporal chains method (if both matrices available)
        if multi_order_matrix is not None:
            try:
                order, scores, details = infer_firing_order_temporal_chains(
                    first_order_matrix, multi_order_matrix
                )
                results['temporal_chains'] = {'order': order, 'scores': scores, 'details': details}
            except Exception as e:
                print(f"    Temporal chains method failed: {e}")
                results['temporal_chains'] = None

        # Consensus method
        try:
            consensus_order, consensus_scores, all_orders = infer_firing_order_consensus(
                first_order_matrix
            )
            results['consensus'] = {
                'order': consensus_order,
                'scores': consensus_scores,
                'all_methods': all_orders
            }
        except Exception as e:
            print(f"    Consensus method failed: {e}")
            results['consensus'] = None

        return results

    def _create_condition_visualizations(self, condition, start_ms, end_ms,
                                         analysis_results, first_order_matrix,
                                         multi_order_matrix, save_dir):
        """
        Create and save all visualizations for a single condition/time window
        """
        from viz.plots_general import (
            create_directed_graph_from_consensus,
            visualize_firing_graph,
            create_multiple_graph_views,
            analyze_graph_properties
        )

        if analysis_results.get('consensus') is None:
            print(f"    Skipping visualization - no consensus results")
            return

        consensus_order = analysis_results['consensus']['order']
        consensus_scores = analysis_results['consensus']['scores']


        try:
            # Create directed graph
            G = create_directed_graph_from_consensus(
                consensus_order, consensus_scores, first_order_matrix
            )

            # Single view visualizations
            layouts = ['spring', 'hierarchical', 'circular']
            for layout in layouts:
                plt.figure(figsize=(12, 8))
                visualize_firing_graph(G, layout=layout,
                                       save_path=condition_dir / f"graph_{layout}.png")
                plt.close()

            # Multiple views if both matrices available
            if multi_order_matrix is not None:
                plt.figure(figsize=(16, 12))
                create_multiple_graph_views(
                    consensus_order, consensus_scores, first_order_matrix,
                    first_order_matrix, multi_order_matrix
                )
                plt.savefig(condition_dir / "multiple_views.png", dpi=300, bbox_inches='tight')
                plt.close()

            # Analyze and save graph properties
            properties = analyze_graph_properties(G)

            # Save properties to text file
            with open(condition_dir / "graph_properties.txt", 'w') as f:
                f.write(f"=== Graph Properties: {condition} {start_ms}-{end_ms}ms ===\n")
                f.write(f"Firing order: {properties['firing_order']}\n")
                f.write(f"Most influential neurons: {properties['influences'][:3]}\n")
                f.write(f"Is acyclic: {properties['is_acyclic']}\n")
                f.write(f"Network density: {properties['density']:.3f}\n")

                # Add method comparison
                f.write(f"\n=== Method Comparison ===\n")
                for method, results in analysis_results.items():
                    if results is not None and 'order' in results:
                        f.write(f"{method}: {results['order']}\n")

            print(f"    Saved visualizations to: {condition_dir}")

        except Exception as e:
            print(f"    Error creating visualizations: {e}")

    def _generate_comparison_plots(self, time_windows, save_dir):
        """
        Generate comparison plots across conditions and time windows
        """

        print("\nGenerating comparison plots...")

        # Create comparison directory
        comp_dir = Path(save_dir) / "comparisons"
        comp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Plot 1: Firing order consistency across time windows
            self._plot_firing_order_consistency(time_windows, comp_dir)

            # Plot 2: Method agreement across conditions
            self._plot_method_agreement(time_windows, comp_dir)

            # Plot 3: Network properties comparison
            self._plot_network_properties_comparison(time_windows, comp_dir)

        except Exception as e:
            print(f"Error generating comparison plots: {e}")

    def _plot_firing_order_consistency(self, time_windows, comp_dir):
        """
        Plot how consistent firing orders are across time windows for each condition
        """

        fig, axes = plt.subplots(1, len(time_windows), figsize=(6*len(time_windows), 8))
        if len(time_windows) == 1:
            axes = [axes]

        for i, (start_ms, end_ms) in enumerate(time_windows):
            window_key = f"{start_ms}_{end_ms}"

            # Collect consensus orders for this time window
            orders_data = []
            conditions = []

            for condition in self.firing_order_results.keys():
                if window_key in self.firing_order_results[condition]:
                    consensus_result = self.firing_order_results[condition][window_key].get('consensus')
                    if consensus_result is not None:
                        orders_data.append(consensus_result['order'])
                        conditions.append(condition)

            if orders_data:
                # Create heatmap of firing orders
                orders_matrix = np.array(orders_data)
                im = axes[i].imshow(orders_matrix, aspect='auto', cmap='viridis')

                axes[i].set_title(f'{start_ms}-{end_ms} ms')
                axes[i].set_xlabel('Neuron Position in Firing Order')
                axes[i].set_ylabel('Condition')
                axes[i].set_yticks(range(len(conditions)))
                axes[i].set_yticklabels(conditions)

                # Add colorbar
                plt.colorbar(im, ax=axes[i], label='Neuron ID')

        plt.tight_layout()
        plt.savefig(comp_dir / "firing_order_consistency.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_method_agreement(self, time_windows, comp_dir):
        """
        Plot how well different methods agree with each other
        """

        methods = ['enhanced', 'hierarchical', 'topological', 'temporal_chains', 'consensus']

        fig, axes = plt.subplots(len(time_windows), len(self.firing_order_results),
                                 figsize=(4*len(self.firing_order_results), 4*len(time_windows)))

        if len(time_windows) == 1:
            axes = axes.reshape(1, -1)
        if len(self.firing_order_results) == 1:
            axes = axes.reshape(-1, 1)

        for i, (start_ms, end_ms) in enumerate(time_windows):
            window_key = f"{start_ms}_{end_ms}"

            for j, condition in enumerate(self.firing_order_results.keys()):
                if window_key not in self.firing_order_results[condition]:
                    continue

                results = self.firing_order_results[condition][window_key]

                # Create agreement matrix between methods
                agreement_matrix = np.zeros((len(methods), len(methods)))

                for m1_idx, method1 in enumerate(methods):
                    for m2_idx, method2 in enumerate(methods):
                        if (results.get(method1) is not None and
                                results.get(method2) is not None and
                                'order' in results[method1] and
                                'order' in results[method2]):

                            order1 = results[method1]['order']
                            order2 = results[method2]['order']

                            # Calculate rank correlation as agreement measure
                            try:
                                from scipy.stats import spearmanr
                                correlation, _ = spearmanr(order1, order2)
                                agreement_matrix[m1_idx, m2_idx] = correlation
                            except:
                                # Fallback: simple overlap measure
                                agreement = np.mean(order1 == order2)
                                agreement_matrix[m1_idx, m2_idx] = agreement

                # Plot agreement matrix
                im = axes[i, j].imshow(agreement_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
                axes[i, j].set_title(f'{condition}\n{start_ms}-{end_ms} ms')
                axes[i, j].set_xticks(range(len(methods)))
                axes[i, j].set_yticks(range(len(methods)))
                axes[i, j].set_xticklabels(methods, rotation=45)
                axes[i, j].set_yticklabels(methods)

                # Add text annotations
                for mi in range(len(methods)):
                    for mj in range(len(methods)):
                        text = axes[i, j].text(mj, mi, f'{agreement_matrix[mi, mj]:.2f}',
                                               ha="center", va="center", color="black", fontsize=8)

        plt.tight_layout()
        plt.savefig(comp_dir / "method_agreement.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_network_properties_comparison(self, time_windows, comp_dir):
        """
        Compare network properties across conditions and time windows
        """

        # This would create plots comparing density, acyclicity, etc.
        # Implementation depends on what properties you want to compare

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot network densities, influence distributions, etc.
        # This is a placeholder - implement based on your specific needs

        axes[0, 0].text(0.5, 0.5, 'Network Density\nComparison',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'Influence Distribution\nComparison',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[1, 0].text(0.5, 0.5, 'Acyclicity\nComparison',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'Firing Order\nStability',
                        ha='center', va='center', transform=axes[1, 1].transAxes)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(comp_dir / "network_properties_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def get_firing_order_summary(self):
        """
        Get a summary of firing order results across all conditions and time windows
        """

        if not hasattr(self, 'firing_order_results'):
            print("No firing order analysis results found. Run analyze_firing_orders_all_conditions() first.")
            return None

        summary = {}

        for condition, windows in self.firing_order_results.items():
            summary[condition] = {}

            for window_key, results in windows.items():
                if results.get('consensus') is not None:
                    consensus_order = results['consensus']['order']
                    summary[condition][window_key] = {
                        'firing_order': consensus_order.tolist(),
                        'methods_used': list(results.keys()),
                        'consensus_available': True
                    }
                else:
                    summary[condition][window_key] = {
                        'firing_order': None,
                        'methods_used': list(results.keys()),
                        'consensus_available': False
                    }

        return summary



    @staticmethod
    def show_metric_scatter_plots(df):
        viz.plots_general.plot_reward_vs_metrics(df)
    @staticmethod
    def show_quartile_comparison_plot(df):
        viz.plots_general.plot_top_vs_bottom_quartiles(df)
    @staticmethod
    def show_metric_correlation_plot(df):
        viz.plots_general.plot_metric_correlations(df)

