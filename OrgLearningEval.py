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
from scipy.stats import spearmanr, pearsonr, wilcoxon
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
        self.latency_change_cache = {}


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

    def plot_firing_order_spatial_with_sttc(self, firing_order, sttc_matrix,
                                            title="Firing Order with STTC",
                                            arrow_width=0.004):
        """
        Plot neuron firing order using spatial coordinates and STTC-colored arrows.

        Parameters:
            firing_order: list of unit indices (ordered firing)
            sttc_matrix: np.ndarray – square STTC matrix (same order as firing_order)
            title: str
            arrow_width: float – base width of arrows
        """
        # Ensure firing_order is a list
        if isinstance(firing_order, np.ndarray):
            firing_order = firing_order.tolist()

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
            print(f"  {len(missing_units)} unit(s) missing from mapping: {missing_units}")

        coords = np.array(coords)
        cmap = get_cmap("plasma")
        norm = Normalize(vmin=0, vmax=1)

        summary_rows = []
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.arange(len(coords)), cmap="viridis", s=40)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        for i in range(len(coords) - 1):
            source_id = used_unit_ids[i]
            target_id = used_unit_ids[i + 1]
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]

            try:
                idx_source = firing_order.index(source_id)
                idx_target = firing_order.index(target_id)
                sttc_val = sttc_matrix[idx_source, idx_target]
                if np.isnan(sttc_val):
                    continue
            except ValueError:
                continue

            scaled_width = arrow_width * (0.6 + sttc_val)

            # Use annotate for proper arrows
            ax.annotate("",
                        xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=cmap(norm(sttc_val)),
                            lw=2,
                            alpha=0.9
                        )
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

        # Annotate start and end
        ax.text(float(coords[0, 0]), float(coords[0, 1]), "Start", fontsize=8, color="green")
        ax.text(float(coords[-1, 0]), float(coords[-1, 1]), "End", fontsize=8, color="red")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="STTC Score")

        ax.grid(True, linestyle="--", alpha=0.3)
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

    def segment_and_compute_causal(
            self,
            condition,
            bin_size_s=60,
            max_latency_ms=200,
            bin_size=5,
            order="first",
            unit_ids=None
    ):
        """
        Segment a dataset into time bins and compute causal and STTC matrices.

        Returns:
            List of tuples: [(start, end, causal_matrix), ...]
        """
        sd = self.spike_data[condition]
        total_ms = int(sd.length)
        bin_ms = bin_size_s * 1000
        segments = []

        for start in range(0, total_ms, bin_ms):
            end = min(start + bin_ms, total_ms)

            # Compute causal matrices
            self.compute_causal_matrices(
                condition=condition,
                start_ms=start,
                end_ms=end,
                max_latency_ms=max_latency_ms,
                bin_size=bin_size,
                unit_ids=unit_ids
            )

            # Retrieve causal matrix
            key = (condition, start, end)
            causal_matrix = (
                self.causal_latency_matrices[key]
                if order == "first"
                else self.multi_order_matrices[key]
            )

            # Cache STTC matrix
            sd_window = self.spike_data[condition].subtime(start, end)
            if unit_ids is not None:
                sd_window = sd_window.select_units(unit_ids)

            sttc_matrix = sd_window.spike_time_tilings()
            self.sttc_matrices[key] = sttc_matrix

            # Store segment tuple
            segments.append((start, end, causal_matrix))

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

    def segment_and_compute_sttc(self, condition, bin_size_s=60, unit_ids=None):
        """
        Segment the dataset and compute STTC matrices per bin.
        Returns: list of (start, end, matrix)
        """
        sd = self.spike_data[condition]
        total_ms = int(sd.length)
        bin_ms = bin_size_s * 1000
        segments = []

        for start in range(0, total_ms, bin_ms):
            end = min(start + bin_ms, total_ms)
            sd_window = sd.subtime(start, end)
            if unit_ids:
                sd_window = sd_window.select_units(unit_ids)
            sttc = sd_window.spike_time_tilings()
            segments.append((start, end, sttc))

        return segments

    def analyze_all_connection_changes(
            self,
            bin_size_s=60,
            order="first",
            unit_ids=None,
            min_corr_diff=0.3,
            zscore_thresh=1.5,
            baseline_sttc_thresh=0.2,
            smooth=True,
            smooth_window=3,
            conditions=("Adaptive", "Null", "Random"),
            cache_key="connection_strengthening"
    ):
        """
        Analyze both causal latency and STTC changes over time, comparing Adaptive vs controls.

        Parameters:
            bin_size_s (int): Segment size in seconds.
            order (str): "first" or "multi" latency matrix.
            unit_ids (list): Subset of unit IDs.
            min_corr_diff (float): Correlation delta threshold for divergence.
            zscore_thresh (float): Z-score threshold for trend stability.
            baseline_sttc_thresh (float): Minimum baseline STTC to include pair.
            smooth (bool): Apply temporal smoothing.
            smooth_window (int): Window size for smoothing.
            conditions (tuple): Conditions to compare.
            cache_key (str): Save results under this key.

        Returns:
            results (list of dicts): Each dict summarizes a pair's metrics.
        """
        from scipy.stats import wilcoxon
        from scipy.ndimage import uniform_filter1d

        if len(conditions) < 2:
            raise ValueError("Must provide Adaptive and at least one control condition")

        cond_adaptive = conditions[0]
        control_conditions = conditions[1:]

        # Step 1: Segment and compute both latency and STTC matrices
        print("Segmenting and computing matrices...")
        segmented_latency = {
            cond: self.segment_and_compute_causal(
                cond, bin_size_s=bin_size_s, order=order, unit_ids=unit_ids
            ) for cond in conditions
        }

        segmented_sttc = {
            cond: self.segment_and_compute_sttc(
                cond, bin_size_s=bin_size_s, unit_ids=unit_ids
            ) for cond in conditions
        }

        # Step 2: Extract pairwise time series
        print("Extracting time series...")
        ts_latency = {cond: self.extract_pairwise_timeseries(seg) for cond, seg in segmented_latency.items()}
        ts_sttc = {cond: self.extract_pairwise_timeseries(seg) for cond, seg in segmented_sttc.items()}
        self.ts_latency = ts_latency
        self.ts_sttc = ts_sttc

        # Step 3: Analyze each pair
        print("Analyzing pairs...")
        results = []
        all_pairs = list(ts_latency[cond_adaptive].keys())

        for pair in all_pairs:
            try:
                # Retrieve adaptive time series
                lat_ad = np.array(ts_latency[cond_adaptive].get(pair, []))
                sttc_ad = np.array(ts_sttc[cond_adaptive].get(pair, []))

                if len(lat_ad) < 2 or len(sttc_ad) < 2:
                    continue

                # Smooth
                if smooth:
                    lat_ad = uniform_filter1d(lat_ad, size=smooth_window)
                    sttc_ad = uniform_filter1d(sttc_ad, size=smooth_window)

                # Check for constant arrays before z-score
                std_lat = np.std(lat_ad)
                std_sttc = np.std(sttc_ad)
                if std_lat == 0 or std_sttc == 0:
                    continue

                # Z-score
                z_lat = (lat_ad - np.mean(lat_ad)) / std_lat
                z_sttc = (sttc_ad - np.mean(sttc_ad)) / std_sttc

                slope_lat = np.polyfit(np.arange(len(lat_ad)), lat_ad, 1)[0]
                slope_sttc = np.polyfit(np.arange(len(sttc_ad)), sttc_ad, 1)[0]

                # Compare with controls
                control_corrs = {}
                p_values = []

                for cond in control_conditions:
                    lat_ctrl = np.array(ts_latency[cond].get(pair, []))
                    sttc_ctrl = np.array(ts_sttc[cond].get(pair, []))

                    if len(lat_ctrl) != len(lat_ad) or len(lat_ctrl) < 2:
                        continue

                    if smooth:
                        lat_ctrl = uniform_filter1d(lat_ctrl, size=smooth_window)
                        sttc_ctrl = uniform_filter1d(sttc_ctrl, size=smooth_window)

                    # Correlation safeguards
                    r_lat = np.nan if np.std(lat_ctrl) == 0 or std_lat == 0 else pearsonr(lat_ad, lat_ctrl)[0]
                    r_sttc = np.nan if np.std(sttc_ctrl) == 0 or std_sttc == 0 else pearsonr(sttc_ad, sttc_ctrl)[0]

                    control_corrs[cond] = {
                        "latency_corr": r_lat,
                        "sttc_corr": r_sttc
                    }

                    # Wilcoxon test
                    try:
                        if (
                                len(lat_ad) == len(lat_ctrl) and
                                len(lat_ad) >= 2 and
                                not np.allclose(lat_ad, lat_ctrl)
                        ):
                            result = wilcoxon(lat_ad, lat_ctrl)
                            # Robust handling: tuple OR WilcoxonResult object
                            p = result.pvalue if hasattr(result, "pvalue") else result[1]
                            p_values.append(p)
                    except Exception as e:
                        print(f"Wilcoxon failed for pair {pair} in {cond}: {e}")
                        continue

                # Final aggregation
                mean_ctrl_corr_latency = np.nanmean([v["latency_corr"] for v in control_corrs.values()])
                mean_ctrl_corr_sttc = np.nanmean([v["sttc_corr"] for v in control_corrs.values()])
                corr_delta_latency = 1.0 - mean_ctrl_corr_latency
                corr_delta_sttc = 1.0 - mean_ctrl_corr_sttc
                min_p = np.min(p_values)

                # Check STTC baseline
                if np.mean(sttc_ad[:3]) < baseline_sttc_thresh:
                    continue

                result = {
                    "pair": pair,
                    "slope_latency": slope_lat,
                    "slope_sttc": slope_sttc,
                    "zscore_peak_latency": np.max(np.abs(z_lat)),
                    "zscore_peak_sttc": np.max(np.abs(z_sttc)),
                    "corr_delta_latency": corr_delta_latency,
                    "corr_delta_sttc": corr_delta_sttc,
                    "wilcoxon_min_p": min_p,
                    "significant_strengthening": (
                            slope_lat > 0 and
                            slope_sttc > 0 and
                            corr_delta_latency > min_corr_diff and
                            min_p < 0.05 and
                            np.max(np.abs(z_sttc)) > zscore_thresh
                    ),
                    "control_corrs": control_corrs
                }
                results.append(result)

            except Exception as e:
                print(f"Skipping {pair}: {e}")
                continue

        # Cache the result
        self.latency_change_cache[cache_key] = results
        return results

    def plot_top_connection_trajectories(self, results, ts_latency, ts_sttc, top_n=5):
        """
        Plot latency and STTC trajectories for top N most significant connection changes.

        Parameters:
            results (list): Output from analyze_all_latency_changes (must be sorted).
            ts_latency (dict): Time series of latency values per condition.
            ts_sttc (dict): Time series of STTC values per condition.
            top_n (int): Number of top connections to plot.
        """
        top_results = [res for res in results if res.get("significant_strengthening", False)][:top_n]
        n_bins = len(next(iter(ts_latency["Adaptive"].values())))
        time = np.arange(n_bins)

        for res in top_results:
            pair = res["pair"]
            fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            fig.suptitle(f"Connection {pair} | Δr={res['corr_delta_latency']:.2f}, slope={res['slope_latency']:.3f}")

            for cond in ts_latency:
                lat_series = ts_latency[cond].get(pair)
                sttc_series = ts_sttc[cond].get(pair)

                if lat_series and sttc_series:
                    axs[0].plot(time, lat_series, label=cond)
                    axs[1].plot(time, sttc_series, label=cond)

            axs[0].set_ylabel("Latency (ms)")
            axs[1].set_ylabel("STTC")
            axs[1].set_xlabel("Time Bin")
            axs[0].legend()
            axs[1].legend()
            axs[0].grid(True, linestyle="--", alpha=0.3)
            axs[1].grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.show()


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


    def show_adaptive_spatial_network(self, dt=0.02, min_spikes=5, sttc_threshold=0.05):
        """
        Create spatial network visualization for the adaptive dataset, similar to fig5.py output.
        Shows neurons as nodes positioned spatially with connections based on STTC.
        Includes ALL neurons in the dataset to capture maximum network connectivity.
        """
        # Set to adaptive dataset
        self.set_dataset("Adaptive")

        # Get spike times and basic info
        spike_times = self.get_spike_times()
        n_units = len(spike_times)
        time_range = [0, self.sd_main.length]

        # Get spatial positions for neurons using spike_locs from metadata
        spike_locs = np.array(self.metadata["spike_locs"])

        # Create position mapping for ALL neurons using spike_locs indices
        neuron_positions = {}
        for unit_id in range(min(n_units, len(spike_locs))):
            try:
                x, y = spike_locs[unit_id]
                neuron_positions[unit_id] = (x, y)
            except (IndexError, ValueError):
                continue

        # Include ALL neurons with any spikes and spatial positions (much more inclusive)
        valid_units = [i for i in range(n_units)
                       if len(spike_times[i]) >= min_spikes and i in neuron_positions]

        print(f"Total neurons in dataset: {n_units}")
        print(f"Neurons with spatial positions: {len(neuron_positions)}")
        print(f"Analyzing {len(valid_units)} neurons with >= {min_spikes} spikes and spatial positions")

        # Compute STTC matrix for valid units
        sttc_matrix = np.zeros((len(valid_units), len(valid_units)))

        for i, neuron_i in enumerate(valid_units):
            for j, neuron_j in enumerate(valid_units):
                if i != j:
                    sttc_coef = self.sttc(
                        len(spike_times[neuron_i]),
                        len(spike_times[neuron_j]),
                        dt, time_range,
                        spike_times[neuron_i],
                        spike_times[neuron_j]
                    )
                    sttc_matrix[i, j] = sttc_coef if not np.isnan(sttc_coef) else 0

        # Create positions array for active neurons
        positions = np.array([spike_locs[i] for i in valid_units])

        # Create spatial network plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Spatial network with connections
        # Plot all electrode positions as background
        ax1.scatter(spike_locs[:, 0], spike_locs[:, 1], c='lightgray', alpha=0.2, s=5,
                    label=f'All Electrodes ({len(spike_locs)})')

        # Color neurons by their role and activity level
        colors = []
        sizes = []
        alphas = []

        for i, neuron_id in enumerate(valid_units):
            spike_count = len(self.sd_main.train[neuron_id])

            # Size based on activity level
            base_size = 30
            activity_size = min(80, base_size + spike_count / 50)

            # Color and alpha based on role
            if neuron_id in self.metadata["training_inds"]:
                colors.append('red')
                alphas.append(0.9)
                sizes.append(activity_size * 1.5)  # Larger for training neurons
            else:
                colors.append('blue')
                alphas.append(0.7)
                sizes.append(activity_size)

        # Plot active neurons
        scatter = ax1.scatter(positions[:, 0], positions[:, 1],
                              c=colors, s=sizes, alpha=0.8,
                              edgecolors='black', linewidth=0.5, zorder=5)

        # Draw connections above threshold
        connection_count = 0
        connection_strengths = []

        for i in range(len(valid_units)):
            for j in range(len(valid_units)):
                if i != j and sttc_matrix[i, j] > sttc_threshold:
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]

                    strength = sttc_matrix[i, j]
                    connection_strengths.append(strength)

                    # Line properties based on connection strength
                    max_sttc = np.max(sttc_matrix) if np.max(sttc_matrix) > 0 else 1.0
                    linewidth = 0.3 + 2 * (strength / max_sttc)
                    alpha = 0.2 + 0.6 * (strength / max_sttc)

                    # Different colors for training neuron connections
                    neuron_i_id = valid_units[i]
                    neuron_j_id = valid_units[j]

                    if (neuron_i_id in self.metadata["training_inds"] and
                            neuron_j_id in self.metadata["training_inds"]):
                        color = 'orange'  # Training-training connections
                    elif (neuron_i_id in self.metadata["training_inds"] or
                          neuron_j_id in self.metadata["training_inds"]):
                        color = 'purple'  # Training-other connections
                    else:
                        color = 'gray'    # Other-other connections

                    ax1.plot([x1, x2], [y1, y2], color=color,
                             linewidth=linewidth, alpha=alpha, zorder=2)
                    connection_count += 1

        # Network plot styling
        training_count = sum(1 for nid in valid_units if nid in self.metadata["training_inds"])
        ax1.set_title(f'Comprehensive Spatial Network - Adaptive Dataset\n'
                      f'{connection_count} connections (STTC > {sttc_threshold})\n'
                      f'{len(valid_units)} active neurons ({training_count} training)')
        ax1.set_xlabel('X position (μm)')
        ax1.set_ylabel('Y position (μm)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='Training Neurons'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=8, label='Other Active Neurons'),
            Line2D([0], [0], color='orange', linewidth=2,
                   label='Training-Training'),
            Line2D([0], [0], color='purple', linewidth=2,
                   label='Training-Other'),
            Line2D([0], [0], color='gray', linewidth=2,
                   label='Other-Other')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Right plot: STTC matrix heatmap
        im = ax2.imshow(sttc_matrix, cmap='viridis', aspect='auto',
                        vmin=0, vmax=np.percentile(sttc_matrix, 95))
        ax2.set_title('STTC Matrix (Active Neurons)')
        ax2.set_xlabel('Neuron Index')
        ax2.set_ylabel('Neuron Index')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, label='STTC')

        plt.tight_layout()
        plt.show()

        # Print comprehensive statistics
        mean_sttc = np.mean(sttc_matrix[sttc_matrix > 0]) if np.any(sttc_matrix > 0) else 0
        max_sttc = np.max(sttc_matrix)
        total_possible_connections = len(valid_units) * (len(valid_units) - 1)
        connectivity_density = connection_count / total_possible_connections if total_possible_connections > 0 else 0

        # Connection type statistics
        training_indices = [i for i, nid in enumerate(valid_units) if nid in self.metadata["training_inds"]]

        training_training_conns = 0
        training_other_conns = 0
        other_other_conns = 0

        for i in range(len(valid_units)):
            for j in range(len(valid_units)):
                if i != j and sttc_matrix[i, j] > sttc_threshold:
                    if i in training_indices and j in training_indices:
                        training_training_conns += 1
                    elif i in training_indices or j in training_indices:
                        training_other_conns += 1
                    else:
                        other_other_conns += 1

        print(f"\nComprehensive Network Statistics:")
        print(f"  Active neurons analyzed: {len(valid_units)}")
        print(f"  Training neurons: {training_count}")
        print(f"  Mean STTC (non-zero): {mean_sttc:.4f}")
        print(f"  Max STTC: {max_sttc:.4f}")
        print(f"  Total connections: {connection_count}")
        print(f"  Network density: {connectivity_density:.2%}")
        print(f"  Connection breakdown:")
        print(f"    Training-Training: {training_training_conns}")
        print(f"    Training-Other: {training_other_conns}")
        print(f"    Other-Other: {other_other_conns}")

        if connection_strengths:
            print(f"  Connection strength stats:")
            print(f"    Mean: {np.mean(connection_strengths):.4f}")
            print(f"    Std: {np.std(connection_strengths):.4f}")
            print(f"    95th percentile: {np.percentile(connection_strengths, 95):.4f}")

        return sttc_matrix, valid_units, positions




    def get_spike_times(self):
        """
        Get spike times from the currently active dataset.
        Returns:
        --------
        list of np.ndarray :
            List of spike time arrays, one for each unit in the dataset.
            Each array contains spike times in milliseconds.
        """
        if not hasattr(self, 'sd_main') or self.sd_main is None:
            raise ValueError("No dataset currently active. Call set_dataset first.")
        return self.sd_main.train  # train contains list of spike times for each unit

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