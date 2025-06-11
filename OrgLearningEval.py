"""
OrgLearningEval.py - Central class interface for methods aimed at evaluating the adaptive learning
capacity of organoids through dynamic control tasks.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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

        self.log_data = load_log_data(self.log_paths)
        self.causal_info = load_causal_info()
        self.metadata = load_metadata()
        self.spike_data = load_spike_data(self.spike_paths)

        # Load mapping
        mapping_df = self.metadata["mapping"]
        mapper = core.map_utils.Mapping.from_df(mapping_df)

        # Get electrode groups
        training_electrodes = self.metadata["training_electrodes"]
        encode_electrodes = self.metadata["encode_electrodes"]
        decode_electrodes = self.metadata["decode_electrodes"]

        all_task_electrodes = (
                [("training", e) for e in training_electrodes] +
                [("encode", e) for e in encode_electrodes] +
                [("decode", e) for e in decode_electrodes]
        )

        # Convert task electrodes to corresponding channels
        valid_electrodes = list({e for _, e in all_task_electrodes})
        try:
            valid_channels = mapper.get_channels(valid_electrodes)
        except Exception as e:
            print(f"[ERROR] Failed to map electrodes to channels: {e}")
            valid_channels = []

        # Filter SpikeData to include only units with channel IDs matching task channels
        for label, sd in self.spike_data.items():
            try:
                unit_ids, spike_times = sd.idces_times()
                filtered_unit_ids = []
                filtered_spike_times = []

                for uid, times in zip(unit_ids, spike_times):
                    if uid in valid_channels:
                        filtered_unit_ids.append(uid)
                        filtered_spike_times.append(times)

                if filtered_unit_ids:
                    self.spike_data[label] = sd.__class__.from_idces_times(filtered_unit_ids, filtered_spike_times)
                    print(f"[INFO] Filtered {label} SpikeData to {len(filtered_unit_ids)} matching channels.")
                else:
                    print(f"[WARNING] No matching channels found in {label} data.")

            except Exception as e:
                print(f"[ERROR] Filtering {label} SpikeData failed: {e}")

        # Build task_neuron_info (keys = channel IDs)
        self.task_neuron_info = {}
        spike_channels = [ch for sublist in self.metadata["spike_channels"] for ch in sublist]
        self.spike_channels = spike_channels

        for role, electrode in all_task_electrodes:
            match = mapping_df[mapping_df["electrode"] == electrode]
            if not match.empty:
                row = match.iloc[0]
                neuron_channel = int(row["channel"])
                if neuron_channel in spike_channels:
                    x, y = row["x"], row["y"]
                    self.task_neuron_info[neuron_channel] = {
                        "electrode": electrode,
                        "role": role,
                        "x": x,
                        "y": y
                    }
                else:
                    print(f"[WARNING] Channel {neuron_channel} from electrode {electrode} not in spike data.")
            else:
                print(f"[WARNING] Electrode {electrode} not found in mapping.")

        # Force add channel 909 (if not present)
        try:
            row = mapping_df[mapping_df["electrode"] == 909].iloc[0]
            neuron_channel = 909
            if neuron_channel not in self.task_neuron_info:
                self.task_neuron_info[neuron_channel] = {
                    "electrode": 909,
                    "role": "training",
                    "x": row["x"],
                    "y": row["y"]
                }
                print(f"[INFO] Forced inclusion of channel 909 (electrode 909)")
        except Exception:
            print("[WARNING] Could not force-add channel 909")

        # Set baseline spike data
        if "Baseline" in self.spike_data:
            self.sd_main = self.spike_data["Baseline"]
            print("Loaded default dataset: 'Baseline'")
        else:
            raise ValueError("Baseline dataset not found in spike_data.")

        # Confirm which neurons exist in spike data
        unit_ids, _ = self.sd_main.idces_times()
        self.task_neuron_inds = sorted([
            nid for nid in self.task_neuron_info if nid in unit_ids
        ])
        print("Task Neuron Indices in Data:", self.task_neuron_inds)

        self.task_neuron_coords = np.array([
            [self.task_neuron_info[nid]["x"], self.task_neuron_info[nid]["y"]]
            for nid in self.task_neuron_inds
        ])

        self.set_dataset("Baseline")

        self.latency_histograms = {}
        self.latency_bins = None
        self.firing_orders = {}
        self.bursts = {}
        self.firing_order_results = {}
        self.causal_latency_matrices = {}
        self.multi_order_matrices = {}
        self.sttc_matrices = {}


    def set_dataset(self, name):
        if name in self.spike_data:
            self.sd_main = self.spike_data[name]
            print(f"Switched to dataset: {name}")
        else:
            raise ValueError(f"Dataset '{name}' not found.")



    def inspect_spike_datasets(self, conditions=None):
        """
        Inspect one, multiple, or all spike datasets loaded in the class instance.

        Parameters:
        - conditions (str, list of str, or None): Dataset condition(s) to inspect. If None, inspects all.
        """
        if not self.spike_data:
            print("[ERROR] No spike datasets loaded.")
            return

        if conditions is None:
            conditions = list(self.spike_data.keys())
        elif isinstance(conditions, str):
            conditions = [conditions]

        for condition in conditions:
            sd = self.spike_data.get(condition)
            if sd is None:
                print(f"[WARNING] Condition '{condition}' not found.")
                continue

            try:
                unit_ids, spike_times = sd.idces_times()
            except Exception as e:
                print(f"[WARNING] Could not extract spike info for '{condition}': {e}")
                unit_ids, spike_times = [], []

            print("\n--- Dataset Inspection ---")
            print(f"Condition           : {condition}")
            print(f"Number of Neurons   : {sd.N}")
            print(f"Recording Length    : {sd.length / 1000:.2f} seconds")
            print(f"Sample Unit IDs     : {unit_ids[:10]}")
            print(f"Sample Spike Times  : {spike_times[:10]}")
            print(f"Available Attributes: {dir(sd)}")
            print("-----------------------------")


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

    def show_neuron_raster_comparison(self, neuron_id=None, task_neuron_rank=None, start_s=0, end_s=20):
        """
        Show stacked spike raster plots for a given neuron across conditions.

        You can specify:
        - `task_neuron_rank`: index into self.task_neuron_inds (e.g., 0 for the first task neuron)
        - `neuron_id`: an explicit neuron ID (unit ID in the spike data)

        One of these must be provided.
        """
        if neuron_id is None:
            if task_neuron_rank is None:
                raise ValueError("Must provide either `neuron_id` or `task_neuron_rank`.")
            if task_neuron_rank >= len(self.task_neuron_inds):
                raise IndexError(f"Invalid rank: {task_neuron_rank}. Only {len(self.task_neuron_inds)} task neurons.")
            neuron_id = self.task_neuron_inds[task_neuron_rank]

        # Confirm neuron ID exists in the spike data
        all_unit_ids, _ = self.sd_main.idces_times()
        if neuron_id not in all_unit_ids:
            raise ValueError(f"Neuron ID {neuron_id} not found in spike data unit IDs.")

        # Prepare datasets
        spike_datasets = []
        for condition in ["Adaptive", "Random", "Null"]:
            if condition in self.spike_data:
                spike_datasets.append((condition, self.spike_data[condition]))

        # Plot
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
                                max_latency_ms=200, bin_size=5, neuron_ids=None):
        """
        Compute and store causal matrices for a condition, possibly over a time window.

        Parameters:
            condition: str
            start_ms: int
            end_ms: int or None
            max_latency_ms: int
            bin_size: int
            neuron_ids: list of neuron indices to restrict analysis restricted to task neurons below
        """
        sd = self.spike_data[condition]
        if end_ms is None:
            end_ms = sd.length

        if neuron_ids is None:
            neuron_ids = self.task_neuron_inds

        # Time-slice the spike data
        sd_window = sd.subtime(start_ms, end_ms)

        first, multi = core.spike_data_utils.infer_causal_matrices(
            sd_window, max_latency_ms=max_latency_ms, bin_size=bin_size, neuron_ids=neuron_ids
        )

        key = (condition, start_ms, end_ms)
        self.causal_latency_matrices[key] = first
        self.multi_order_matrices[key] = multi

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
        if inds is None:
            inds = self.task_neuron_inds  # Use all 10 task-related neurons by default
        viz.plots_general.causal_plot_from_matrices(first_order, multi_order, title=title, training_inds=inds)

    def compute_sttc_for_condition(self, condition, start_ms=0, end_ms=None):
        """
        Compute STTC matrix for training neurons in a given condition and time window.
        """
        from core.spike_data_utils import get_sttc

        if end_ms is None:
            end_ms = self.spike_data[condition].length

        sd = self.spike_data[condition].subtime(start_ms, end_ms)
        neuron_ids = self.task_neuron_inds

        sttc_matrix = get_sttc(sd, neuron_ids)
        key = (condition, start_ms, end_ms)
        self.sttc_matrices[key] = sttc_matrix

    def show_firing_order_overlay(self, condition, start_ms, end_ms):
        """
        Show firing order overlay for the task neurons (stim + encode + training)
        in a given condition and time window.
        """
        from viz.plots_general import plot_firing_order_overlay

        key = (condition, start_ms, end_ms)
        matrix = self.causal_latency_matrices.get(key)
        if matrix is None:
            raise ValueError(f"Causal matrix not found for {key}")

        task_ids = self.task_neuron_inds  # use unified list of task neurons
        coords = np.array(self.metadata["spike_locs"])[task_ids]
        matrix_task = matrix[np.ix_(task_ids, task_ids)]

        plot_firing_order_overlay(
            matrix_task,
            coords,
            title=f"Firing Order - {condition} ({start_ms}-{end_ms} ms)",
            neuron_labels=task_ids
        )


    def plot_sttc_matrix_overlay(self, condition, start_ms=0, end_ms=None):
        """
        Plot STTC matrix for a condition and time window (task neurons only).
        """
        from core.spike_data_utils import plot_sttc_matrix
        if end_ms is None:
            end_ms = self.spike_data[condition].length
        sd = self.spike_data[condition].subtime(start_ms, end_ms)
        plot_sttc_matrix(sd, self.task_neuron_inds, title=f"{condition} STTC: {start_ms}-{end_ms} ms")

    def plot_sttc_spatial_overlay(self, condition, start_ms=0, end_ms=None, threshold=0.1):
        """
        Plot spatial STTC connections for task neurons.
        """
        from core.spike_data_utils import plot_sttc_spatial
        if end_ms is None:
            end_ms = self.spike_data[condition].length
        coords = np.array(self.metadata["spike_locs"])[self.task_neuron_inds]
        sd = self.spike_data[condition].subtime(start_ms, end_ms)
        plot_sttc_spatial(sd, self.task_neuron_inds, coords, threshold=threshold,
                          title=f"{condition} Spatial STTC: {start_ms}-{end_ms} ms")

    def show_combined_firing_sttc_overlay(self, condition, start_ms=0, end_ms=None,
                                          order="first", top_n=None, sttc_threshold=0.1):
        """
        Show combined plot of firing order and STTC connection strength for task neurons.

        Parameters:
            condition : str
            start_ms : int
            end_ms : int or None
            order : "first" or "multi"
            top_n : int or None
            sttc_threshold : float, min STTC for connection lines
        """
        from viz.plots_general import plot_firing_sttc_combined

        if end_ms is None:
            end_ms = self.spike_data[condition].length

        key = (condition, start_ms, end_ms)

        # Retrieve causal matrix
        if order == "first":
            matrix = self.causal_latency_matrices.get(key)
        elif order == "multi":
            matrix = self.multi_order_matrices.get(key)
        else:
            raise ValueError("order must be 'first' or 'multi'")

        if matrix is None:
            raise ValueError(f"Missing causal matrix for {key}. Did you run compute_causal_matrices()?")

        # Ensure STTC matrix is computed
        if key not in self.sttc_matrices:
            self.compute_sttc_for_condition(condition, start_ms, end_ms)

        sttc = self.sttc_matrices[key]
        task_neuron_inds = self.task_neuron_inds
        coords = np.array(self.metadata["spike_locs"])[task_neuron_inds]

        plot_firing_sttc_combined(
            matrix=matrix,
            sttc=sttc,
            coords=coords,
            title=f"{condition} ({order.title()} Order) {start_ms}-{end_ms} ms",
            top_n=top_n,
            sttc_threshold=sttc_threshold,
        )

    def plot_sttc_over_time(self, conditions, window_ms=30000, step_ms=60000):
        """
        Visualize how STTC scores for the 10 task neurons change over time across conditions.

        Parameters:
            ole: OrgLearningEval instance
            conditions: list of str (condition names)
            window_ms: size of time window for each STTC calculation
            step_ms: step size between windows
        """
        task_ids = self.task_neuron_inds
        n = len(task_ids)
        total_time = min([self.spike_data[cond].length for cond in conditions])
        time_bins = list(range(0, total_time - window_ms + 1, step_ms))

        sttc_results = []

        for cond in conditions:
            for start in time_bins:
                end = start + window_ms
                sd = self.spike_data[cond].subtime(start, end)
                sttc = sd.subset(task_ids).spike_time_tilings()
                sttc = np.nan_to_num(sttc)
                upper_tri_values = sttc[np.triu_indices(n, k=1)]
                mean_sttc = np.mean(upper_tri_values)

                sttc_results.append({
                    "condition": cond,
                    "start_ms": start,
                    "end_ms": end,
                    "mean_sttc": mean_sttc
                })

        df = pd.DataFrame(sttc_results)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        for cond in conditions:
            subset = df[df["condition"] == cond]
            times = subset["start_ms"] / 60000  # Convert ms to minutes
            ax.plot(times, subset["mean_sttc"], label=cond, marker='o')

        ax.set_title("STTC Changes Over Time (Task Neurons)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Mean STTC (Upper Triangle)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return df


    @staticmethod
    def show_metric_scatter_plots(df):
        viz.plots_general.plot_reward_vs_metrics(df)
    @staticmethod
    def show_quartile_comparison_plot(df):
        viz.plots_general.plot_top_vs_bottom_quartiles(df)
    @staticmethod
    def show_metric_correlation_plot(df):
        viz.plots_general.plot_metric_correlations(df)

    @property
    def task_neuron_ids(self):
        """List of original neuron IDs for task neurons (in order)."""
        return list(self.task_neuron_info.keys())

    @property
    def task_coords(self):
        """(n, 2) array of (x, y) coordinates for task neurons."""
        return np.array([[info["x"], info["y"]] for info in self.task_neuron_info.values()])

    @property
    def task_roles(self):
        """List of roles (training, encode, decode) in same order as task_neuron_ids."""
        return [info["role"] for info in self.task_neuron_info.values()]

    @property
    def task_channels(self):
        """List of spike data channel IDs in same order as task_neuron_ids."""
        return [info.get("channel") for info in self.task_neuron_info.values()]