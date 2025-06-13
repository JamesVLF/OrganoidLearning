"""
OrgLearningEval.py - Central class interface for methods aimed at evaluating the adaptive learning
capacity of organoids through dynamic control tasks.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from IPython.display import display
from scipy.stats import spearmanr, pearsonr
from core.data_loader import load_spike_data, load_log_data, load_causal_info, load_metadata, label_task_units
from core.spike_data_utils import calculate_mean_firing_rates
from core.analysis_utils import get_correlation_matrix
import core.data_loader
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
        self.unit_role_map, self.task_unit_ids = label_task_units(self.metadata)
        self.spike_data = load_spike_data(self.spike_paths)
        self.set_dataset("Baseline")  # sets self.sd_main

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
            self.current_dataset = name
            print(f"Switched to dataset: {name}")
        else:
            raise ValueError(f"Dataset '{name}' not found.")

    def show_raster(self, task_units_only=None, start=None, end=None, unit_ids=None):
        if unit_ids is not None:
            units = unit_ids
        elif task_units_only:
            units = list(self.task_unit_ids)
        else:
            units = None  # Show all units

        # Call plot function with customization
        viz.plots_general.plot_raster(
            self.sd_main,
            title=f"Spike Raster ({self.current_dataset})",
            start=start if start is not None else 0,
            end=end,
            unit_ids=units
        )

    def show_mean_firing_rates(self, unit_ids=None):
        """
        Show a bar plot of mean firing rates.

        Parameters:
            unit_ids: list or array of unit IDs to include (default: all)
        """
        rates = core.spike_data_utils.calculate_mean_firing_rates(self.sd_main, unit_ids=unit_ids)
        viz.plots_general.plot_firing_rates(rates)

    def show_correlation_matrix(self, unit_ids=None):
        """
        Show a correlation matrix of firing activity.

        Parameters:
            unit_ids: list or array of unit IDs to include (default: all)
        """
        matrix = core.analysis_utils.get_correlation_matrix(self.sd_main, unit_ids=unit_ids)
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

    def show_burst_raster(self, start_time=0, end_time=None, task_units_only=False, unit_ids=None):
        units = unit_ids or (list(self.task_unit_ids) if task_units_only else None)
        end_time = end_time or (self.sd_main.length / 1000)

        viz.plots_general.plot_raster_pretty(
            self.sd_main,
            l1=start_time,
            l2=end_time,
            analyze=True,
            unit_ids=units,
            title=f"Burst Raster – {self.current_dataset}"
        )


    def show_neuron_raster_comparison(self, unit_id=None, start_s=0, end_s=20):
        """
        Show stacked spike raster plots for a given neuron across conditions.

        Parameters:
        - unit_id: int, neuron index (0-based)
        - start_s, end_s: float, time window in seconds
        """
        if unit_id is None:
            raise ValueError("You must provide a valid `unit_id`.")

        if unit_id < 0 or unit_id >= self.sd_main.N:
            raise ValueError(f"Neuron ID {unit_id} is out of range (0 to {self.sd_main.N - 1}).")

        spike_datasets = [
            (condition, self.spike_data[condition])
            for condition in ["Adaptive", "Random", "Null"]
            if condition in self.spike_data
        ]

        viz.plots_general.plot_neuron_raster_comparison_stacked(
            spike_datasets,
            unit_id=unit_id,
            start_s=start_s,
            end_s=end_s
        )

    def compute_latency_histograms(self, window_ms=30, bin_size=5, unit_ids=None):
        self.latency_bins = {}
        for cond, sd in self.spike_data.items():
            histograms, bins = core.spike_data_utils.compute_latency_histograms(
                sd, window_ms=window_ms, bin_size=bin_size, unit_ids=unit_ids
            )
            self.latency_histograms[cond] = histograms
            self.latency_bins[cond] = bins

    def show_latency_histogram(self, condition, i, j):
        if condition not in self.latency_histograms:
            raise ValueError("Call compute_latency_histograms() first.")
        if condition not in self.latency_bins:
            raise ValueError("Latency bins missing for this condition.")

        viz.plots_general.plot_latency_histogram_pair(
            self.latency_histograms[condition],
            self.latency_bins[condition],
            i,j
        )

    def compute_causal_matrices(self, condition="Adaptive", start_ms=0, end_ms=None,
                                max_latency_ms=200, bin_size=5, unit_ids=None):
        """
        Compute and store causal matrices for a given condition and time window.

        Parameters:
            condition: str – dataset label
            start_ms: int – start of window (in ms)
            end_ms: int or None – end of window (in ms)
            max_latency_ms: int – latency window (± ms)
            bin_size: int – histogram bin size in ms
            unit_ids: list of unit IDs to include; defaults to all units
        """
        sd = self.spike_data[condition]

        if end_ms is None:
            end_ms = sd.length

        if unit_ids is None:
            unit_ids = list(range(sd.N))  # Use all internal unit indices

        sd_window = sd.subtime(start_ms, end_ms)

        first, multi = core.spike_data_utils.infer_causal_matrices(
            sd_window,
            max_latency_ms=max_latency_ms,
            bin_size=bin_size,
            unit_ids=unit_ids
        )

        key = (condition, start_ms, end_ms)
        self.causal_latency_matrices[key] = first
        self.multi_order_matrices[key] = multi

    def show_causal_plot_from_matrices(self, first_order, multi_order, title="", unit_ids=None):
        """
        Plot causal matrices using a general-purpose utility.

        Parameters:
            first_order: np.ndarray – result from compute_causal_matrices
            multi_order: np.ndarray – result from compute_causal_matrices
            title: str – title for the plot
            unit_ids: list[int] or None – subset of unit indices to visualize; defaults to all
        """
        if unit_ids is None:
            unit_ids = list(range(self.sd_main.N))

        viz.plots_general.causal_plot_from_matrices(
            first_order,
            multi_order,
            title=title,
            unit_ids=unit_ids
        )


    def compute_causal_matrices_counts(self, condition="Adaptive", start_ms=0, end_ms=None,
                                       latency_window_first=(-15, 15), latency_window_multi=(-200, 200),
                                       unit_ids=None):
        """
        Compute causal connection count matrices for a condition.

        Parameters:
            condition: str – dataset label
            start_ms: int – start time (ms)
            end_ms: int – end time (ms); defaults to full length
            latency_window_first: tuple – window for first-order causal links
            latency_window_multi: tuple – window for multi-order causal links
            unit_ids: list – optional subset of unit indices (defaults to all)
        """
        sd = self.spike_data[condition]
        if end_ms is None:
            end_ms = sd.length

        if unit_ids is None:
            unit_ids = list(range(sd.N))

        sd_window = sd.subtime(start_ms, end_ms)

        # Restrict the spike data if needed
        sd_window = sd_window.select_units(unit_ids)

        first, multi = core.spike_data_utils.infer_causal_matrices_counts(
            sd_window,
            latency_window_first=latency_window_first,
            latency_window_multi=latency_window_multi
        )

        key = (condition, start_ms, end_ms)
        self.causal_latency_matrices[key] = first
        self.multi_order_matrices[key] = multi


    def show_causal_plot_counts(self, first_order, multi_order, title="", unit_ids=None, vmin=None, vmax=None):
        """
        Plot count-based causal matrices using baseline-style heatmap.

        Parameters:
            first_order: np.ndarray – from compute_causal_matrices_counts
            multi_order: np.ndarray – from compute_causal_matrices_counts
            title: str – plot title
            unit_ids: list[int] or None – optional subset of unit indices; defaults to all
            vmin/vmax: float – optional color scaling limits
        """
        if unit_ids is None:
            unit_ids = list(range(self.sd_main.N))  # Default to all units in current dataset

        viz.plots_general.causal_plot_from_matrices_counts(
            first_order,
            multi_order,
            title=title,
            unit_ids=unit_ids,
            vmin=vmin,
            vmax=vmax
        )

    def compute_firing_order_from_causal_matrix(self, matrix, condition=None, label="first_order"):
        """
        Compute firing order using net influence scores from an external utility.

        Parameters:
            matrix: np.ndarray – causal matrix (e.g., first_order or multi_order)
            condition: str or None – optional condition label for caching
            label: str – string to describe matrix type (e.g., 'first_order')

        Returns:
            firing_order: np.ndarray – unit indices from most to least influential
            net_score: np.ndarray – net influence scores for all units
        """
        firing_order, net_score = core.analysis_utils.infer_firing_order(matrix)

        key = (condition, label) if condition else label
        self.firing_order_results[key] = {
            "firing_order": firing_order,
            "net_score": net_score
        }
        return firing_order, net_score

    def show_spike_time_tilings(self, condition, start_ms=0, end_ms=None, unit_ids=None, title="STTC Matrix"):
        """
        Plot the spike time tiling coefficient matrix for a given condition and subset of units.
        """
        sd = self.spike_data[condition]
        if end_ms is None:
            end_ms = sd.length

        sd = sd.subtime(start_ms, end_ms)

        if unit_ids is None:
            unit_ids = list(sd.unit_ids)  # Default to all units in dataset

        subset = sd.subset(unit_ids)
        sttc = subset.spike_time_tilings()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(np.nan_to_num(sttc), vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Neuron Index")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def plot_firing_order_spatial_with_sttc(self, firing_order, sttc_matrix, title="Firing Order with STTC", arrow_width=0.005):
        """
        Plot neuron firing order using spatial coordinates and color-coded arrows by STTC.

        Parameters:
            firing_order (list or np.ndarray): Ordered list of unit_ids from most to least influential.
            sttc_matrix (np.ndarray): NxN STTC matrix between unit_ids.
            title (str): Plot title.
            arrow_width (float): Width of the arrows.
        """
        mapping_df = self.metadata["mapping"].set_index("channel")
        coords = []
        used_unit_ids = []
        missing_units = []

        for unit_id in firing_order:
            if unit_id in mapping_df.index:
                x, y = mapping_df.loc[unit_id, ["x", "y"]]
                coords.append((x, y))
                used_unit_ids.append(unit_id)
            else:
                missing_units.append(unit_id)

        if missing_units:
            print(f" Warning: {len(missing_units)} unit IDs not found in mapping and were skipped.")

        coords = np.array(coords)
        cmap = get_cmap("plasma")
        norm = Normalize(vmin=0, vmax=1)  # STTC values range from 0 to 1

        summary_rows = []

        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap="viridis", s=50)
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")

        for i in range(len(coords) - 1):
            source_id = used_unit_ids[i]
            target_id = used_unit_ids[i + 1]
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            dx, dy = x1 - x0, y1 - y0

            # Find index positions for STTC matrix lookup
            try:
                idx_source = self.sd_main.unit_ids.index(source_id)
                idx_target = self.sd_main.unit_ids.index(target_id)
                sttc_val = sttc_matrix[idx_source, idx_target]
            except ValueError:
                sttc_val = np.nan

            # Plot arrow with STTC-colored magnitude
            plt.arrow(
                x0, y0, dx, dy,
                head_width=arrow_width * 50,
                head_length=arrow_width * 50,
                fc=cmap(norm(sttc_val)),
                ec=cmap(norm(sttc_val)),
                linewidth=1.0,
                alpha=0.9,
                length_includes_head=True
            )

            summary_rows.append({
                "source_unit": source_id,
                "target_unit": target_id,
                "source_x": x0,
                "source_y": y0,
                "target_x": x1,
                "target_y": y1,
                "sttc_score": round(sttc_val, 4)
            })

        plt.text(float(coords[0, 0]), float(coords[0, 1]), "Start", fontsize=8, color="green")
        plt.text(float(coords[-1, 0]), float(coords[-1, 1]), "End", fontsize=8, color="red")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label="STTC Score")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

        summary_df = pd.DataFrame(summary_rows)

        display(summary_df)
        return summary_df

    def compare_causal_matrices(
            self,
            cond_a,
            cond_b,
            time_a=(0, None),
            time_b=(0, None),
            order="first",
            show_plot=True,
            perform_stats=False,
            alpha=0.05,
            threshold=5
    ):
        """
        Compare causal matrices between two conditions or time windows.

        Parameters:
            cond_a (str): First condition label (e.g., "Baseline" or "Adaptive").
            cond_b (str): Second condition label.
            time_a (tuple): (start_ms, end_ms) window for cond_a.
            time_b (tuple): (start_ms, end_ms) window for cond_b.
            order (str): 'first' or 'multi' – which type of causal matrix to compare.
            show_plot (bool): Whether to show the difference heatmap.
            perform_stats (bool): Whether to compute a significance mask.
            alpha (float): Significance level (not yet used).
            threshold (float): Latency change threshold (in ms) to flag significant pairs.

        Returns:
            diff (np.ndarray): Difference matrix (cond_b - cond_a).
            stats (dict): Summary statistics.
            sig_mask (np.ndarray, optional): Boolean matrix for significance, if perform_stats=True.
        """
        key_a = (cond_a, time_a[0], time_a[1] or self.spike_data[cond_a].length)
        key_b = (cond_b, time_b[0], time_b[1] or self.spike_data[cond_b].length)

        # Select matrices
        if order == "first":
            mat_a = self.causal_latency_matrices.get(key_a)
            mat_b = self.causal_latency_matrices.get(key_b)
        elif order == "multi":
            mat_a = self.multi_order_matrices.get(key_a)
            mat_b = self.multi_order_matrices.get(key_b)
        else:
            raise ValueError("order must be 'first' or 'multi'")

        if mat_a is None or mat_b is None:
            raise ValueError(f"Missing matrices: ensure compute_causal_matrices() was run for both keys.")

        # Compute difference
        diff = mat_b - mat_a
        stats = {
            "mean_diff": np.mean(diff),
            "sum_abs_diff": np.sum(np.abs(diff)),
            "max_change": np.max(np.abs(diff)),
        }

        sig_mask = None
        if perform_stats:
            sig_mask = np.abs(diff) >= threshold
            stats["num_significant_pairs"] = int(np.sum(sig_mask))

        # Plotting
        if show_plot:
            vmax = np.max(np.abs(diff))
            plt.imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax, alpha=alpha)
            plt.colorbar(label="Δ Causal Value")
            plt.title(f"Δ Causal Matrix ({cond_b} – {cond_a}) [{order.title()}]")
            plt.xlabel("Source Neuron")
            plt.ylabel("Target Neuron")
            plt.tight_layout()
            plt.show()

            if perform_stats:
                print(f"Significant changes (|Δ| ≥ {threshold} ms): {stats['num_significant_pairs']} pairs")

        return (diff, stats, sig_mask) if perform_stats else (diff, stats)

    def segment_and_compute_causal(self, condition, bin_size_s=60, max_latency_ms=200, bin_size=5, order="first", unit_ids=None):
        """
        Segment a dataset into time bins and compute causal matrices for each.

        Returns:
            List of matrices with format: [(start, end, matrix), ...]
        """
        sd = self.spike_data[condition]
        total_ms = sd.length
        bin_ms = bin_size_s * 1000
        segments = []

        for start in range(0, total_ms, bin_ms):
            end = min(start + bin_ms, total_ms)
            self.compute_causal_matrices(
                condition=condition,
                start_ms=start,
                end_ms=end,
                max_latency_ms=max_latency_ms,
                bin_size=bin_size,
                unit_ids=unit_ids
            )

            key = (condition, start, end)
            matrix = (
                self.causal_latency_matrices[key]
                if order == "first"
                else self.multi_order_matrices[key]
            )
            segments.append((start, end, matrix))

        return segments

    def extract_pairwise_timeseries(self, matrices):
        """
        Extract per-pair latency time series from a list of matrices.

        Returns:
            Dictionary of {(i, j): [v_t0, v_t1, ..., v_tN]} across bins
        """
        timeseries = {}
        for _, _, matrix in matrices:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if (i, j) not in timeseries:
                        timeseries[(i, j)] = []
                    timeseries[(i, j)].append(matrix[i, j])
        return timeseries

    def compare_adaptive_to_controls(self, pair, adaptive_ts, control_ts_dict, min_corr_diff=0.3):
        """
        Compare one pair's latency time series across conditions using correlation.

        Returns:
            dict with correlation values and a flag if Adaptive is significantly different
        """
        control_corrs = {}
        for cond, ctrl_ts in control_ts_dict.items():
            if len(ctrl_ts) != len(adaptive_ts):
                raise ValueError(f"Length mismatch: Adaptive vs {cond}")
            r, _ = pearsonr(adaptive_ts, ctrl_ts)
            control_corrs[cond] = r

        mean_ctrl_corr = np.mean(list(control_corrs.values()))
        adaptive_r = 1.0  # Adaptive compared to itself
        delta = adaptive_r - mean_ctrl_corr
        is_significant = delta > min_corr_diff

        return {
            "pair": pair,
            "adaptive_vs_controls_corr_delta": delta,
            "control_corrs": control_corrs,
            "significant": is_significant
        }

    def analyze_all_latency_changes(
            self,
            bin_size_s=60,
            order="first",
            unit_ids=None,
            min_corr_diff=0.3,
            conditions=("Adaptive", "Baseline", "Null")
    ):
        """
        Analyze causal latency change over time for all neuron pairs, comparing Adaptive to controls.

        Parameters:
            bin_size_s: int – size of each time window in seconds
            order: str – 'first' or 'multi'
            unit_ids: list – unit IDs to restrict analysis (default: all)
            min_corr_diff: float – threshold to flag adaptive changes as significant
            conditions: tuple – conditions to compare, where the first is "Adaptive"

        Returns:
            results: list of dicts with change metrics for each neuron pair
        """
        if len(conditions) < 2:
            raise ValueError("Must provide at least one control condition along with Adaptive")

        cond_adaptive = conditions[0]
        control_conditions = conditions[1:]

        # Step 1: Segment and compute matrices
        print("Segmenting and computing causal matrices...")
        segmented_data = {
            cond: self.segment_and_compute_causal(
                cond,
                bin_size_s=bin_size_s,
                order=order,
                unit_ids=unit_ids
            ) for cond in conditions
        }

        # Step 2: Extract time series
        print("Extracting pairwise latency time series...")
        pairwise_ts = {
            cond: self.extract_pairwise_timeseries(matrices)
            for cond, matrices in segmented_data.items()
        }

        # Step 3: Analyze all pairs
        print("Analyzing changes in latency trajectories...")
        all_pairs = list(pairwise_ts[cond_adaptive].keys())
        results = []

        for pair in all_pairs:
            adaptive_ts = pairwise_ts[cond_adaptive][pair]
            control_ts_dict = {}

            try:
                for cond in control_conditions:
                    ctrl_ts = pairwise_ts[cond][pair]
                    control_ts_dict[cond] = ctrl_ts

                # Perform correlation-based analysis
                result = self.compare_adaptive_to_controls(
                    pair=pair,
                    adaptive_ts=adaptive_ts,
                    control_ts_dict=control_ts_dict,
                    min_corr_diff=min_corr_diff
                )
                results.append(result)

            except Exception as e:
                print(f"Skipping pair {pair}: {e}")

        return results

    def compute_firing_orders(self, window_ms=50, top_k=5, unit_ids=None):
        self.firing_orders = {}  # Reset
        for cond, sd in self.spike_data.items():
            self.firing_orders[cond] = core.spike_data_utils.extract_common_firing_orders(
                sd, window_ms=window_ms, top_k=top_k, unit_ids=unit_ids
            )

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

    from scipy.stats import ttest_ind






    def show_firing_order_overlay(self, condition, start_ms, end_ms):
        """
        Show firing order overlay for training neurons in the given condition and time window.
        """
        from viz.plots_general import plot_firing_order_overlay

        key = (condition, start_ms, end_ms)
        matrix = self.causal_latency_matrices.get(key)
        if matrix is None:
            print(f"[WARNING] Causal matrix not found for {key}. Skipping.")
            return

        unit_ids, _ = self.sd_main.idces_times()
        training_inds = [
            uid for uid, info in self.task_neuron_info.items()
            if info["role"] == "training" and uid in unit_ids
        ]
        coords = np.array([
            [self.task_neuron_info[uid]["x"], self.task_neuron_info[uid]["y"]]
            for uid in training_inds
        ])
        matrix_task = matrix[np.ix_(training_inds, training_inds)]

        plot_firing_order_overlay(
            matrix_task,
            coords,
            title=f"Firing Order - {condition} ({start_ms}-{end_ms} ms)",
            neuron_labels=training_inds
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