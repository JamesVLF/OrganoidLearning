import numpy as np
import zipfile
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import os
import networkx as nx
from diptest import diptest
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as path_effects
import csv
import matplotlib.patches as mpatches
import pandas as pd
from scipy.signal import find_peaks
from IPython.display import display

class STTCAnalyzer:
    def __init__(self, dataset_name, spike_trains, recording_length=None, neuron_data=None,
                 delt=0.02, num_shuffles=10, threshold=2.5, fs=20000, layout_positions=None):
        self.dataset_name = dataset_name
        self.delt          = delt
        self.num_shuffles  = num_shuffles
        self.threshold     = threshold
        self.fs            = fs

        # compute recording_length if missing
        if recording_length is None:
            recording_length = max((max(s) for s in spike_trains.values() if len(s)>0), default=0)
        self.recording_length = recording_length

        # sort & align keys
        sorted_spike_keys = sorted(spike_trains.keys())
        if neuron_data:
            sorted_meta_keys = sorted(neuron_data.keys())
            if len(sorted_spike_keys) != len(sorted_meta_keys):
                raise ValueError(f"Mismatch: {len(sorted_spike_keys)} spikes vs {len(sorted_meta_keys)} metadata")
        else:
            sorted_meta_keys = []

        self.neuron_keys   = sorted_spike_keys
        self.spike_trains  = [spike_trains[k] for k in sorted_spike_keys]
        self.neuron_data   = [neuron_data[k]  for k in sorted_meta_keys] if neuron_data else None

        # placeholders for STTC
        self.original_sttc   = None
        self.randomized_sttc = None
        self.filtered_sttc   = None

        # ── Build electrode layout map: channel → (x_um, y_um) ─────────
        if layout_positions is not None:
            self.layout_positions = layout_positions
        else:
            self.layout_positions = {}
            if neuron_data:
                # use the raw dict passed in to __init__
                for ch, unit in neuron_data.items():
                    pos = unit.get('position')
                    if pos is not None:
                        self.layout_positions[str(ch)] = tuple(pos)
                    # also pick up any neighbor channels
                    neigh_chs  = unit.get('neighbor_channels', [])
                    neigh_poss = unit.get('neighbor_positions', [])
                    for nc, np_pos in zip(neigh_chs, neigh_poss):
                        nc = str(nc)
                        if nc not in self.layout_positions:
                            self.layout_positions[nc] = tuple(np_pos)

        # ── Attach neighbor waveforms and positions if neuron_data is provided ─────────
        if self.neuron_data:
            for i, unit in enumerate(self.neuron_data):
                pos_i = unit.get('position')
                if pos_i is None:
                    continue

                neighbors = []
                positions = []

                for j, other in enumerate(self.neuron_data):
                    if i == j:
                        continue
                    pos_j = other.get('position')
                    if pos_j is None:
                        continue
                    dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                    if dist <= 100:  # radius in µm; adjust as needed
                        tmpl = other.get('template')
                        if tmpl is not None:
                            neighbors.append(tmpl)
                            positions.append(pos_j)

                if neighbors:
                    unit['neighbor_templates'] = neighbors
                    unit['neighbor_positions'] = positions

    def get_neuron_info(self, index):
        """
        Retrieve spike train and metadata for a given neuron index.

        Parameters:
            index (int): Index of the neuron (based on sorted order)

        Returns:
            dict: Contains 'neuron_key', 'spike_train', and 'metadata' (if available)
        """
        if index < 0 or index >= len(self.spike_trains):
            raise IndexError("Neuron index out of range.")

        return {
            "neuron_key": self.neuron_keys[index],
            "spike_train": self.spike_trains[index],
            "metadata": self.neuron_data[index] if self.neuron_data else None
        }

    ###########################################################
    # STTC Computation
    ###########################################################

    def compute_sttc_matrix(self, spike_trains=None):
        if spike_trains is None:
            spike_trains = self.spike_trains

        N = len(spike_trains)
        sttc_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                val, _, _, _, _ = self.sttc_pairs(spike_trains[i], spike_trains[j])
                sttc_matrix[i, j] = sttc_matrix[j, i] = val

        return sttc_matrix

    def sttc_pairs(self, tA, tB):
        if len(tA) == 0 or len(tB) == 0:
            return 0.0, 0, 0, 0, 0

        TA = self._sttc_ta_tb(tA) / self.recording_length
        TB = self._sttc_ta_tb(tB) / self.recording_length
        NA, NB = self._sttc_na_nb(tA, tB)

        PA = min(NA / len(tA), 1.0)
        PB = min(NB / len(tB), 1.0)

        epsilon = 1e-10
        sttc = ((PA - TB) / (1 - PA * TB + epsilon) + (PB - TA) / (1 - PB * TA + epsilon)) / 2
        return sttc, PA, PB, TA, TB

    def _sttc_ta_tb(self, t):
        if len(t) == 0 or self.recording_length == 0:
            return 0
        base = min(self.delt, t[0]) + min(self.delt, self.recording_length - t[-1])
        total_time = base + np.minimum(np.diff(t), 2 * self.delt).sum()
        return total_time

    def _sttc_na_nb(self, tA, tB):
        tA, tB = np.asarray(tA), np.asarray(tB)
        iB = np.searchsorted(tB, tA)
        iB = np.clip(iB, 1, len(tB) - 1)
        dt_left = np.abs(tB[iB] - tA)
        dt_right = np.abs(tB[iB - 1] - tA)
        count = np.sum((dt_left <= self.delt) | (dt_right <= self.delt))
        return count, count

    ###########################################################
    # Randomization
    ###########################################################
    def random_okun(self, bin_size=0.001, swap_per_spike=5, duration=None, seed_val=None):
        """
        Applies spike swap randomization similar to Okun 2015 supplemental method.
        - Preserves spike count per neuron and population rate
        - Destroys fine temporal correlations
        - Works on binarized spike-time matrix

        Returns:
            List of shuffled spike trains (same format as self.spike_trains)
        """
        if seed_val is not None:
            np.random.seed(seed_val)

        if duration is None:
            duration = self.recording_length

        num_neurons = len(self.spike_trains)
        num_bins = int(np.ceil(duration / bin_size))
        spike_matrix = np.zeros((num_neurons, num_bins), dtype=np.float32)

        # Bin the spike trains
        for i, spikes in enumerate(self.spike_trains):
            binned = (np.floor(spikes / bin_size)).astype(int)
            binned = binned[binned < num_bins]
            spike_matrix[i, binned] = 1.0

        # Swap-based shuffling
        def swap(ar, idxs):
            idx0 = np.random.randint(len(idxs[0]))
            idx1 = np.random.randint(len(idxs[0]))
            i0, j0 = idxs[0][idx0], idxs[1][idx0]
            i1, j1 = idxs[0][idx1], idxs[1][idx1]
            if i0 == i1 or j0 == j1 or ar[i0, j1] == 1.0 or ar[i1, j0] == 1.0:
                return False
            ar[i0, j0] = ar[i1, j1] = 0.0
            ar[i0, j1] = ar[i1, j0] = 1.0
            idxs[0][idx0], idxs[1][idx0] = i0, j1
            idxs[0][idx1], idxs[1][idx1] = i1, j0
            return True

        def randomize(matrix, swap_per_spike):
            mat = matrix.copy()
            idxs = np.where(mat == 1.0)
            total_swaps = swap_per_spike * len(idxs[0])
            count = 0
            attempts = int(1.5 * total_swaps)

            for _ in range(attempts):
                if swap(mat, idxs):
                    count += 1
                if count >= total_swaps:
                    break
            return mat

        shuffled_matrix = randomize(spike_matrix, swap_per_spike=swap_per_spike)

        # Convert back to spike times
        shuffled_trains = []
        for i in range(num_neurons):
            spike_bins = np.where(shuffled_matrix[i] == 1.0)[0]
            times = (spike_bins + np.random.rand(len(spike_bins))) * bin_size
            shuffled_trains.append(np.sort(times[times < duration]))

        return shuffled_trains

    def plot_sttc_zscore_distribution(self, threshold=2.0, save_path=None, show=True):
        """
        Plots a histogram of z-scored STTC values using the precomputed shuffled matrices.

        Args:
            threshold (float): z-score threshold line to plot.
            save_path (str): Optional path to save the figure.
            show (bool): Whether to display the plot.
        """
        if self.original_sttc is None:
            self.original_sttc = self.compute_sttc_matrix()

        if not hasattr(self, "randomized_sttc") or self.randomized_sttc is None:
            raise ValueError("No shuffled STTC matrices found. Please generate or load them first.")

        # Stack the shuffled matrices and compute mean and std
        rand_array = np.stack(self.randomized_sttc)
        mean_rand = np.mean(rand_array, axis=0)
        std_rand = np.std(rand_array, axis=0)

        # Compute z-scores with error handling for division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = (self.original_sttc - mean_rand) / std_rand
            z_scores[std_rand == 0] = np.nan

        z_vals = z_scores[np.triu_indices_from(z_scores, k=1)]

        # Calculate stats
        threshold = self.threshold  # use class threshold if set
        significant_count = np.sum(z_scores > threshold)
        total_pairs = np.sum(~np.isnan(z_scores))  # count of valid comparisons
        percent = 100 * significant_count / total_pairs if total_pairs else 0

        # === Plotting ===
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(z_vals[~np.isnan(z_vals)], bins=50, color='skyblue', edgecolor='black')
        ax.axvline(threshold, color='red', linestyle='--', label=f'z = {threshold:.1f}')
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Pair count")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()

        # Annotate summary
        annotation_text = (
            f"{significant_count:,} pairs > z={threshold:.1f}\n"
            f"({percent:.1f}% of total)"
        )
        ax.text(
            0.95, 0.85, annotation_text,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        # Set final title
        ax.set_title(
            f"{self.dataset_name} — STTC Z-score Distribution\n"
            f"{significant_count:,} pairs > z={threshold:.1f}",
            fontsize=13
        )

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        return z_scores

    def plot_raster_original_vs_shuffled(self, shuffled_trains=None, bin_size=0.03, duration=None,
                                         save_path=None, show=True, rate_bin_size=0.03, rate_smooth_sigma=2):
        """
        Compares original vs shuffled spike rasters with population firing rate (dual y-axis).
        If shuffled_trains is not provided, defaults to using Okun shuffling.
        """
        if duration is None:
            duration = self.recording_length

        if shuffled_trains is None:
            shuffled_trains = self.random_okun(bin_size=bin_size)

        num_neurons = len(self.spike_trains)
        time_bins = np.arange(0, duration + rate_bin_size, rate_bin_size)

        # === Compute smoothed population firing rates ===
        def pop_rate(trains):
            pop_counts = np.zeros(len(time_bins) - 1)
            for train in trains:
                pop_counts += np.histogram(train, bins=time_bins)[0]
            rate = pop_counts / (rate_bin_size * num_neurons)
            return gaussian_filter1d(rate, sigma=rate_smooth_sigma)

        rate_orig = pop_rate(self.spike_trains)
        rate_shuf = pop_rate(shuffled_trains)

        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

        # === ORIGINAL RASTER + FIRING RATE ===
        ax0 = fig.add_subplot(gs[0])
        ax0.eventplot(self.spike_trains, colors='black', linewidths=0.5)
        ax0.set_title(f"{self.dataset_name} – Original", fontsize=12)
        ax0.set_ylabel("Neuron index")
        ax0.set_xlim(0, duration)

        ax0b = ax0.twinx()
        ax0b.plot(time_bins[:-1], rate_orig, color='red', lw=1, alpha=0.8, label='Population rate')
        ax0b.set_ylabel("Pop. rate (Hz)", color='red')
        ax0b.tick_params(axis='y', colors='red')

        # === SHUFFLED RASTER + FIRING RATE ===
        ax1 = fig.add_subplot(gs[1])
        ax1.eventplot(shuffled_trains, colors='gray', linewidths=0.5)
        ax1.set_title("Shuffled Spike Trains", fontsize=12)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Neuron index")
        ax1.set_xlim(0, duration)

        ax1b = ax1.twinx()
        ax1b.plot(time_bins[:-1], rate_shuf, color='red', lw=1, alpha=0.8, label='Population rate')
        ax1b.set_ylabel("Pop. rate (Hz)", color='red')
        ax1b.tick_params(axis='y', colors='red')

        if save_path:
            plt.savefig(save_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_raster_population_overlay(self, bin_size=0.03, duration=None,
                                  save_path=None, show=True,
                                  rate_bin_size=0.03, rate_smooth_sigma=2):
        """
        Plot only the original spike raster and its smoothed population firing rate.

        Parameters:
        - bin_size: not used (included for interface compatibility)
        - duration: length of time to display
        - save_path: if provided, saves figure to this path
        - show: whether to display the plot
        - rate_bin_size: time resolution for population rate
        - rate_smooth_sigma: smoothing factor (Gaussian std dev)
        """
        if duration is None:
            duration = self.recording_length

        num_neurons = len(self.spike_trains)
        time_bins = np.arange(0, duration + rate_bin_size, rate_bin_size)

        # === Compute population rate ===
        pop_counts = np.zeros(len(time_bins) - 1)
        for train in self.spike_trains:
            pop_counts += np.histogram(train, bins=time_bins)[0]
        rate = pop_counts / (rate_bin_size * num_neurons)
        rate_smooth = gaussian_filter1d(rate, sigma=rate_smooth_sigma)

        # === Plot ===
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.eventplot(self.spike_trains, colors='black', linewidths=0.5)
        ax.set_title(f"{self.dataset_name} – Spike Raster", fontsize=12)
        ax.set_ylabel("Neuron index")
        ax.set_xlim(0, duration)

        ax2 = ax.twinx()
        ax2.plot(time_bins[:-1], rate_smooth, color='red', lw=1.2, alpha=0.8, label='Population rate')
        ax2.set_ylabel("Pop. rate (Hz)", color='red')
        ax2.tick_params(axis='y', colors='red')

        if save_path:
            plt.savefig(save_path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()


    def compute_randomized_sttc(self):
        print(f"Shuffling spike trains using Okun method for {self.dataset_name}...")

        self.randomized_sttc = []
        self.cached_shuffled_trains = None  # Will store the first shuffle

        for i in range(self.num_shuffles):
            shuffled = self.random_okun()
            if i == 0:
                self.cached_shuffled_trains = shuffled  # Save the first one
            mat = self.compute_sttc_matrix(spike_trains=shuffled)
            self.randomized_sttc.append(mat)

        self.randomized_sttc = np.array(self.randomized_sttc)
        print(f"Completed {self.num_shuffles} shuffled STTC calculations for {self.dataset_name}.")

    def load_randomized_sttc(self, save_dir="outputs/shuffled_sttc"):
        path = os.path.join(save_dir, f"{self.dataset_name}_randomized_sttc.npy")
        if os.path.exists(path):
            self.randomized_sttc = np.load(path, allow_pickle=False)

            # === Sanity Check ===
            if self.randomized_sttc.ndim != 3:
                raise ValueError(f"Loaded randomized STTC has invalid shape: {self.randomized_sttc.shape} — expected 3D array (shuffles x N x N)")

            print(f" Randomized STTC loaded: {path} | Shape = {self.randomized_sttc.shape}")
        else:
            print(f" No saved randomized STTC found for {self.dataset_name}")
            raise FileNotFoundError(f"Missing shuffled STTC: {path}")

    def save_randomized_sttc(self, save_dir="outputs/shuffled_sttc"):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{self.dataset_name}_randomized_sttc.npy")
        np.save(path, self.randomized_sttc)
        print(f" Saved randomized STTC to: {path} | Shape = {self.randomized_sttc.shape}")

    ###########################################################
    # Filtering
    ###########################################################

    def filter_sttc(self):
        mean_shuffled = np.mean(self.randomized_sttc, axis=0)
        std_shuffled = np.std(self.randomized_sttc, axis=0)
        std_shuffled[std_shuffled == 0] = np.mean(std_shuffled[std_shuffled > 0]) if np.any(std_shuffled > 0) else 1e-6

        z_scores = (self.original_sttc - mean_shuffled) / std_shuffled
        self.filtered_sttc = np.where(z_scores >= self.threshold, self.original_sttc, 0)
        np.fill_diagonal(self.filtered_sttc, 1)

        self.num_significant_neurons = np.count_nonzero(np.any(self.filtered_sttc != 0, axis=1))

        # Metrics
        N = self.original_sttc.shape[0]
        total_possible = (self.num_significant_neurons * (self.num_significant_neurons - 1)) // 2

        # Count actual connections (non-zero, excluding diagonal)
        def count_nonzero_offdiag(mat):
            upper = np.triu(mat, k=1)
            return np.count_nonzero(upper)

        connections_original = count_nonzero_offdiag(self.original_sttc)
        connections_shuffled_mean = count_nonzero_offdiag(mean_shuffled)
        connections_filtered = count_nonzero_offdiag(self.filtered_sttc)

        # Difference = connections that were filtered out
        filtered_out = connections_original - connections_filtered
        percent_filtered = (filtered_out / connections_original * 100) if connections_original > 0 else 0

        print(f" Significant neurons: {self.num_significant_neurons}")
        print(f" Theoretical max connections: {total_possible}")
        print(f" Original matrix connections: {connections_original}")
        print(f" Mean shuffled matrix connections: {connections_shuffled_mean}")
        print(f" Filtered matrix connections: {connections_filtered}")
        print(f" Filtered (random/noise) connections removed: {filtered_out}")
        print(f" % of original connections filtered out: {percent_filtered:.1f}%")

    def plot_sttc_matrix(self, matrix, title="STTC Matrix"):
        """
        Plots a heatmap of the given STTC matrix (original or filtered).
        """
        if matrix is None:
            print("No STTC matrix provided.")
            return

        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap="YlGnBu", square=True, cbar=True)
        plt.title(title)
        plt.xlabel("Neuron Index")
        plt.ylabel("Neuron Index")
        plt.tight_layout()
        plt.show()

    def plot_sttc_distribution_overlay(self, save_path=None):
        """
        Overlays original and randomized STTC distributions.
        Displays the z-score threshold used for filtering.
        """
        if self.original_sttc is None or self.randomized_sttc is None:
            print("STTC data missing. Run analysis first.")
            return

        print(f"Plotting STTC distribution overlay for {self.dataset_name}...")

        # Upper triangle only (excluding diagonal)
        original_flat = self.original_sttc[np.triu_indices_from(self.original_sttc, k=1)]
        randomized_flat = self.randomized_sttc[:, np.triu_indices_from(self.randomized_sttc[0], k=1)[0],
                          np.triu_indices_from(self.randomized_sttc[0], k=1)[1]].flatten()

        mean_rand = np.mean(randomized_flat)
        std_rand = np.std(randomized_flat)
        threshold_val = mean_rand + self.threshold * std_rand

        # Plot histograms
        plt.figure(figsize=(10, 5))
        plt.hist(original_flat, bins=50, alpha=0.6, label="Original STTC", color="blue", density=True)
        plt.hist(randomized_flat, bins=50, alpha=0.6, label="Randomized STTC", color="red", density=True)

        '''
        # Add annotation showing z-score threshold
        text = f"Z-score threshold = {self.threshold:.1f}σ\n(Mean + {self.threshold:.1f} × Std)"
        plt.text(threshold_val + 0.02, plt.ylim()[1] * 0.8, text,
                 fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
        '''
        # Labels and layout
        plt.title(f"STTC Distribution Overlay - {self.dataset_name}", fontsize=14)
        plt.xlabel("STTC Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot to {save_path}")
        plt.show()

    def plot_filtered_population_raster(self, rate_bin_size=0.1, smooth_sigma=2, duration=None, save_path=None, show=True):
        """
        Plots a spike raster and population firing rate trace for neurons that passed the Z-score threshold filter.

        Parameters:
            rate_bin_size (float): Bin size in seconds for firing rate histogram.
            smooth_sigma (float): Sigma for Gaussian smoothing of firing rate.
            duration (float): Length of time to display (defaults to full recording).
            save_path (str): If provided, saves the figure.
            show (bool): If True, shows the plot.
        """
        if self.filtered_sttc is None:
            print("No filtered STTC matrix available. Run filter_sttc() first.")
            return

        # Identify significant neurons (non-zero rows or columns)
        mask = np.any(self.filtered_sttc != 0, axis=1)
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            print("No significant neurons found after filtering.")
            return

        # Clip to duration
        if duration is None:
            duration = self.recording_length

        # Prepare spike trains
        spike_trains = [self.spike_trains[i] for i in filtered_indices]
        spike_trains_clipped = [[t for t in train if t <= duration] for train in spike_trains]

        # Compute population rate
        time_bins = np.arange(0, duration + rate_bin_size, rate_bin_size)
        counts = np.zeros(len(time_bins) - 1)
        for train in spike_trains_clipped:
            counts += np.histogram(train, bins=time_bins)[0]

        pop_rate = counts / rate_bin_size / len(spike_trains)  # Hz
        smoothed_rate = gaussian_filter1d(pop_rate, sigma=smooth_sigma)

        # === Plot ===
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Raster
        ax1.eventplot(spike_trains_clipped, colors='black', linewidths=0.5)
        ax1.set_ylabel("Neuron index")
        ax1.set_xlim(0, duration)
        ax1.set_title(f"{self.dataset_name} – Z-filtered Raster + Population Rate ({len(filtered_indices)} neurons)")

        # Population rate (twin axis)
        ax2 = ax1.twinx()
        ax2.plot(time_bins[:-1], smoothed_rate, color='red', lw=1.2, label='Population firing rate')
        ax2.set_ylabel("Firing rate (Hz)", color='red')
        ax2.tick_params(axis='y', colors='red')

        ax1.set_xlabel("Time (s)")
        ax1.spines['right'].set_color('red')

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()



    ###########################################################
    # Visualization (Spike Trains, Latency, Network)
    ###########################################################

    def plot_unit_spike_trains(self, unit_a, unit_b, start_time=0, save_path=None):
        """
        Plots spike trains for a 90-second interval starting at start_time.
        Includes a 30-second scale bar at 1/3 of the interval.
        """
        duration = 90  # fixed window length
        end_time = start_time + duration

        # Filter spike times within the selected window
        spikes_a = [t for t in self.spike_trains[unit_a] if start_time <= t <= end_time]
        spikes_b = [t for t in self.spike_trains[unit_b] if start_time <= t <= end_time]

        fig, ax = plt.subplots(figsize=(6, 2))

        ax.eventplot([spikes_a, spikes_b],
                     colors='black',
                     lineoffsets=[1.86, 1.14],
                     linelengths=[0.7, 0.7],
                     linewidths=0.75)

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(0.5, 2.5)

        ax.set_yticks([1.86, 1.14])
        ax.set_yticklabels([f"Unit {unit_a}", f"Unit {unit_b}"])
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Spike trains: {start_time}–{end_time} s")

        # 30-second scale bar at 1/3 of window
        bar_start = start_time + duration / 3
        bar_y = 0.3
        ax.plot([bar_start, bar_start + 30], [bar_y, bar_y], color='black', lw=3)
        ax.text(bar_start + 15, bar_y - 0.2, "30 s", ha="center", va="top", fontsize=9)

        ax.tick_params(left=False)
        ax.spines[['top', 'right', 'left']].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()



    def plot_top_sttc_pairs_window(self, n=5, start_time=0, min_spikes=5, save_dir=None):
        """
        Plots spike trains for top N STTC pairs in a specific 90s window starting at start_time.
        """
        if self.original_sttc is None:
            self.original_sttc = self.compute_sttc_matrix()

        N = len(self.spike_trains)
        matrix = self.original_sttc.copy()
        np.fill_diagonal(matrix, -np.inf)

        # Filter and sort by STTC values
        pairs = [(i, j, matrix[i, j]) for i in range(N) for j in range(i + 1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]
        top = sorted(pairs, key=lambda x: x[2], reverse=True)[:n]

        for idx, (i, j, score) in enumerate(top):
            print(f"\nTop {idx + 1}: Unit {i} ↔ Unit {j} | STTC = {score:.3f}")
            base = f"{save_dir}/{self.dataset_name}_win{start_time}_pair{idx+1}_U{i}_U{j}" if save_dir else None
            self.plot_unit_spike_trains(i, j, start_time=start_time, save_path=f"{base}_spike_trains.png" if base else None)


    def plot_latency_histogram(self,
                               unit_a,
                               unit_b,
                               window=0.03,
                               bin_size=0.001,
                               latency_range=(None, None),
                               pval_range=(None, None),
                               ax=None,
                               show_plot=True,
                               save_path=None):
        """
        Plot latency histogram between spikes of unit_a and unit_b using nearest-spike logic.

        Parameters:
        - unit_a, unit_b: unit indices
        - window: half-width of time window in seconds (default: 30ms)
        - bin_size: histogram bin width in seconds (default: 1ms)
        - ax: matplotlib axis to plot into (optional)
        - show_plot: whether to show the plot (default: True)
        - save_path: if provided, saves the figure to this path

        Returns:
        - mean_latency (ms)
        - diptest p-value
        - peak_latency (ms)
        """
        # --- Compute lags ---
        spikes_a = np.array(self.spike_trains[unit_a])
        spikes_b = np.array(self.spike_trains[unit_b])
        lags = []
        for t_a in spikes_a:
            if spikes_b.size == 0:
                continue
            diffs = spikes_b - t_a
            idx = np.argmin(np.abs(diffs))
            latency = diffs[idx]
            if abs(latency) <= window:
                lags.append(latency)
        lags = np.array(lags)

        # --- Histogram ---
        bins = np.arange(-window, window + bin_size, bin_size)
        counts, _ = np.histogram(lags, bins=bins)

        # --- Summary Stats ---
        mean_latency = lags.mean() * 1000 if lags.size else 0

        try:
            p_val = diptest(lags)[1] if lags.size > 10 else np.nan
        except:
            p_val = np.nan

        # --- Peak Detection ---
        peak_indices, _ = find_peaks(counts)
        if peak_indices.size > 0:
            peak_idx = peak_indices[np.argmax(counts[peak_indices])]
            peak_latency = (bins[peak_idx] + bin_size / 2) * 1000  # ms
        else:
            peak_latency = np.nan

        # --- Filtering logic ---
        min_lat, max_lat = latency_range
        min_p, max_p = pval_range
        passed_filter = True
        if ((min_lat is not None and mean_latency < min_lat) or
                (max_lat is not None and mean_latency > max_lat) or
                (min_p is not None and (np.isnan(p_val) or p_val < min_p)) or
                (max_p is not None and (np.isnan(p_val) or p_val > max_p))):
            passed_filter = False

        # --- Plotting ---
        own_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
            own_ax = True

        ax.bar(bins[:-1]*1000, counts, width=bin_size*1000,
               color='skyblue', edgecolor='black')
        ax.axvline(0, linestyle='--', color='red', linewidth=1)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_xlim(-window*1000, window*1000)
        ax.set_title(f"Latency (mean={mean_latency:.1f} ms, "
                     f"peak={peak_latency:.1f} ms, p={p_val:.3f})")
        ax.grid(linestyle='--', linewidth=0.5, color='lightgray')

        if save_path:
            plt.savefig(save_path, dpi=300)
        if own_ax and show_plot:
            plt.show()

        return mean_latency, p_val, peak_latency

    def plot_top_sttc_pairs_with_latency(self, n=5, start_time=0, min_spikes=5, save_dir=None,
                                         window=0.03, bin_size=0.001, peak_prominence=3):
        """
        Plots spike trains and corrected latency histograms (using nearest spike logic)
        for the top N STTC pairs in the dataset, with enhanced multimodality visualization.
        """

        if self.filtered_sttc is None:
            self.filtered_sttc = self.compute_sttc_matrix()

        N = len(self.spike_trains)
        matrix = self.original_sttc.copy()
        np.fill_diagonal(matrix, -np.inf)

        pairs = [(i, j, matrix[i, j]) for i in range(N) for j in range(i + 1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]
        top = sorted(pairs, key=lambda x: x[2], reverse=True)[:n]

        for idx, (i, j, score) in enumerate(top):
            print(f"\nTop {idx + 1}: Unit {i} → Unit {j} | STTC = {score:.3f}")
            base = f"{save_dir}/{self.dataset_name}_win{start_time}_pair{idx+1}_U{i}_U{j}" if save_dir else None

            # === Panel 1: Spike trains (windowed for clarity only)
            self.plot_unit_spike_trains(i, j, start_time=start_time,
                                        save_path=f"{base}_spike_trains.png" if base else None)

            # === Latency histogram from full recording
            spikes_a = np.array(self.spike_trains[i])
            spikes_b = np.array(self.spike_trains[j])
            lags = []

            if len(spikes_a) == 0 or len(spikes_b) == 0:
                print(f" Not enough spikes for units {i} or {j}")
                continue

            for t_a in spikes_a:
                diffs = spikes_b - t_a
                if len(diffs) == 0:
                    continue
                nearest_idx = np.argmin(np.abs(diffs))
                latency = diffs[nearest_idx]
                if np.abs(latency) <= window:
                    lags.append(latency)

            lags = np.array(lags)
            bins = np.arange(-window, window + bin_size, bin_size)
            counts, _ = np.histogram(lags, bins=bins)

            # === Stats
            mean_latency = np.mean(lags) * 1000 if len(lags) > 0 else 0
            try:
                p_val = diptest(lags)[1] if len(lags) > 10 else np.nan
            except Exception as e:
                p_val = np.nan
                print(f" Dip test failed for ({i},{j}): {e}")

            # === Plot
            fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
            ax.bar(bins[:-1] * 1000, counts, width=bin_size * 1000,
                   color='skyblue', edgecolor='black', label="Histogram")
            ax.axvline(0, linestyle="--", color="red", label="t=0")
            ax.set_xlim(-window * 1000, window * 1000)
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Spike count")
            ax.set_title(f"Latency Histogram | Unit {i} → Unit {j}")

            # === KDE overlay for multimodal structure
            if len(lags) > 3:
                sns.kdeplot(lags * 1000, ax=ax, color="black", lw=1.5, label="KDE")

            # === Peak markers (visual only)
            peak_indices, _ = find_peaks(counts, prominence=peak_prominence)
            for peak_idx in peak_indices:
                peak_x = bins[peak_idx] * 1000
                ax.axvline(peak_x, linestyle=":", color="gray", alpha=0.6)

            # === Annotation
            annotation_text = f"Mean latency = {mean_latency:.1f} ms\nP = {p_val:.3f}" if not np.isnan(p_val) else f"Mean latency = {mean_latency:.1f} ms"
            annotation = ax.text(
                0.98, 0.95, annotation_text,
                ha='right', va='top',
                transform=ax.transAxes,
                fontsize=9, fontweight='bold'
            )
            annotation.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='white'),
                path_effects.Normal()
            ])

            ax.legend(loc='upper left', fontsize=8, frameon=True)

            if base:
                fig.savefig(f"{base}_latency.png", dpi=300)
                print(f"  ↪ latency plot saved to: {base}_latency.png")

            plt.show()

    def plot_top_sttc_pairs_grid(self, n=5, start_time=0, duration=60, min_spikes=5,
                                 latency_range=(None, None), pval_range=(None, None),
                                 save_dir=None, return_csv=False, show_plots=False):
        """
        Plots spike trains, latency histograms, and prints coordinates for the top N STTC pairs.
        Metadata is now accessed by index, not by dict lookup.
        """
        if self.original_sttc is None:
            self.original_sttc = self.compute_sttc_matrix()

        N = len(self.spike_trains)
        matrix = self.original_sttc.copy()
        np.fill_diagonal(matrix, -np.inf)

        if save_dir is None:
            save_dir = f"{self.dataset_name}_latency_histograms"
        os.makedirs(save_dir, exist_ok=True)

        # Find top-N pairs by STTC
        pairs = [(i, j, matrix[i, j])
                 for i in range(N) for j in range(i+1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]
        top = sorted(pairs, key=lambda x: x[2], reverse=True)[:n]

        results = []

        for idx, (i, j, score) in enumerate(top):
            # Windowed spike trains
            spikes_a_window = [t for t in self.spike_trains[i] if start_time <= t <= start_time + duration]
            spikes_b_window = [t for t in self.spike_trains[j] if start_time <= t <= start_time + duration]

            # Full trains for latency
            spikes_a_full = np.array(self.spike_trains[i])
            spikes_b_full = np.array(self.spike_trains[j])
            lags = []
            window = 0.03
            for t in spikes_a_full:
                diffs = spikes_b_full - t
                lags.extend(diffs[(diffs >= -window) & (diffs <= window)])
            lags = np.array(lags)

            # Histogram and stats
            bin_size = 0.001
            bins = np.arange(-window, window + bin_size, bin_size)
            counts, _ = np.histogram(lags, bins=bins)
            dip, pval = diptest(lags) if len(lags) > 3 else (np.nan, np.nan)
            mean_latency = np.mean(lags) * 1000 if len(lags) > 0 else 0

            # Apply filtering criteria
            if latency_range[0] is not None and mean_latency < latency_range[0]:
                continue
            if latency_range[1] is not None and mean_latency > latency_range[1]:
                continue
            if pval_range[0] is not None and pval < pval_range[0]:
                continue
            if pval_range[1] is not None and pval > pval_range[1]:
                continue

            # --- NEW: metadata lookup by index ---
            if self.neuron_data:
                unit_a = self.neuron_data[i]
                unit_b = self.neuron_data[j]
                x_a, y_a = unit_a.get("position", ("NA", "NA"))
                x_b, y_b = unit_b.get("position", ("NA", "NA"))
            else:
                x_a = y_a = x_b = y_b = "NA"

            coord_str_a = f"({x_a}, {y_a})"
            coord_str_b = f"({x_b}, {y_b})"

            print(f"Pair ({i}, {j}) | STTC={score:.3f} | latency={mean_latency:.1f} ms | "
                  f"p={pval:.3f} | Coords: {coord_str_a} vs {coord_str_b}")

            results.append((i, j, score, mean_latency, pval, coord_str_a, coord_str_b))

            # --- Plot panels ---
            fig = plt.figure(figsize=(12, 4))
            gs = GridSpec(1, 2, width_ratios=[1, 1.1], figure=fig)

            # Raster
            ax1 = fig.add_subplot(gs[0])
            ax1.eventplot([spikes_a_window, spikes_b_window], colors='blue',
                          lineoffsets=[2, 1], linelengths=0.8, linewidths=0.75)
            ax1.set_xlim(start_time, start_time + duration)
            ax1.set_ylim(0.5, 2.5)
            ax1.set_yticks([2, 1])
            ax1.set_yticklabels([f"Unit {i}", f"Unit {j}"])
            ax1.set_xlabel("Time (s)")
            ax1.set_title("Spike trains")
            ax1.spines[['top', 'right', 'left']].set_visible(False)
            bar_x = start_time + duration / 3
            ax1.plot([bar_x, bar_x + 30], [0.3, 0.3], color='black', lw=3)
            ax1.text(bar_x + 15, 0.1, "30 s", ha="center", va="top", fontsize=9)

            # Latency histogram
            ax2 = fig.add_subplot(gs[1])
            ax2.bar(bins[:-1] * 1000, counts, width=bin_size * 1000, color='skyblue', edgecolor='black')
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("Spike count")
            ax2.set_xlim(-window * 1000, window * 1000)
            annotation = f"Mean = {mean_latency:.1f} ms\nP = {pval:.3f}"
            ax2.text(0.98, 0.95, annotation, ha='right', va='top', transform=ax2.transAxes,
                     fontsize=9, fontweight='bold')

            # Save/close or show
            fname = f"{self.dataset_name}_win{start_time}_pair{idx+1}_U{i}_U{j}.png"
            plt.savefig(os.path.join(save_dir, fname), dpi=300)
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        # --- CSV export at end ---
        if return_csv and results:
            csv_path = os.path.join(save_dir, f"{self.dataset_name}_summary.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Unit A', 'Unit B', 'STTC', 'Mean Latency (ms)',
                                 'P-Value', 'Unit A Coord (x,y)', 'Unit B Coord (x,y)'])
                writer.writerows(results)
            print(f"→ Saved summary to {csv_path}")


    def resolve_coordinates(self, i, j):
        """
        Returns: coords_i (str), coords_j (str), dist (float or 'N/A'),
                 neuron_key_i, neuron_key_j
        """
        coords_i = coords_j = "N/A"
        dist     = "N/A"
        key_i = key_j = None

        if self.neuron_data:
            key_i, key_j = self.neuron_keys[i], self.neuron_keys[j]
            unit_i, unit_j = self.neuron_data[i], self.neuron_data[j]

            pos_i = unit_i.get("position") if isinstance(unit_i, dict) else None
            pos_j = unit_j.get("position") if isinstance(unit_j, dict) else None

            if (isinstance(pos_i, (list,tuple,np.ndarray)) and len(pos_i)==2 and
                    isinstance(pos_j, (list,tuple,np.ndarray)) and len(pos_j)==2):
                x_i,y_i = float(pos_i[0]), float(pos_i[1])
                x_j,y_j = float(pos_j[0]), float(pos_j[1])
                coords_i = f"({x_i:.1f}, {y_i:.1f})"
                coords_j = f"({x_j:.1f}, {y_j:.1f})"
                dist     = np.hypot(x_i-x_j, y_i-y_j)

        return coords_i, coords_j, dist, key_i, key_j

    def plot_waveform_com_footprint(self, unit_idx_i, unit_idx_j,
                                    ax=None, show_plot=True, save_path=None):
        """
        Draw the two COMs and full waveforms offset spatially.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))  # slightly taller
        else:
            fig = None

        ui = self.neuron_data[unit_idx_i]
        uj = self.neuron_data[unit_idx_j]

        # Compute COMs
        com_i = self._compute_center_of_mass(ui)
        com_j = self._compute_center_of_mass(uj)

        xi, yi = com_i
        xj, yj = com_j

        # Draw distance line
        ax.plot([xi, xj], [yi, yj], '--', color='gray', zorder=1)
        dist = np.hypot(xi - xj, yi - yj)
        ax.text((xi + xj) / 2 + 5, (yi + yj) / 2 + 5,
                f"{dist:.1f} µm", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', pad=0.2))

        # Plot waveforms at COMs
        for (com, unit, color, label) in (
                (com_i, ui, 'red', 'Sender'),
                (com_j, uj, 'blue', 'Receiver')
        ):
            cx, cy = com
            wf = np.asarray(unit['template'], float)
            wf = (wf - wf.mean()) / (wf.ptp() + 1e-9) * 15  # scale height
            t_space = np.linspace(-15, 15, wf.size)         # waveform width
            ax.plot(cx + t_space, cy + wf, color=color, lw=1.5, label=f"{label} waveform")
            ax.scatter([cx], [cy], color=color, s=60,
                       edgecolor='black', zorder=5, label=f"{label} COM")

        # Dynamic aspect: use ratio based on spread
        x_margin = 50
        y_margin = 30
        all_x = [xi, xj]
        all_y = [yi, yj]

        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        ax.set_xlabel("X position (µm)")
        ax.set_ylabel("Y position (µm)")
        ax.set_title("Spatial + Waveform Footprint (COM‐aligned)")

        # Cleaner legend handling
        ax.legend(loc='upper right', fontsize=7, frameon=True)

        if save_path:
            plt.savefig(save_path, dpi=300)
            if fig:
                plt.close(fig)
        elif show_plot and fig:
            plt.show()
    '''
    def plot_waveform_footprint(self, unit_idx_i, unit_idx_j, ax=None, show_plot=True, save_path=None):
        """
        Plots spatially aligned waveforms for a neuron pair (sender → receiver),
        overlayed on their spatial coordinates. Metadata is accessed by index.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = None

        neuron_key_i = self.neuron_keys[unit_idx_i]
        neuron_key_j = self.neuron_keys[unit_idx_j]
        unit_i = self.neuron_data[unit_idx_i]
        unit_j = self.neuron_data[unit_idx_j]

        def draw_overlay(unit, color, label, spatial_radius=20, neighbor_cutoff=100):
            pos = unit.get("position")
            tmpl = unit.get("template")
            if pos is None or tmpl is None:
                return
            x0, y0 = pos
            wf = np.asarray(tmpl, dtype=float)
            wf = (wf - wf.mean()) / (wf.ptp() + 1e-9) * spatial_radius
            t_space = np.linspace(-spatial_radius, spatial_radius, wf.size)
            ax.plot(x0 + t_space, y0 + wf, color=color, lw=1.5, label=label, zorder=4)
            ax.scatter([x0], [y0], color=color, s=60, edgecolor="black", zorder=5)
            neigh_tm = unit.get("neighbor_templates", [])
            neigh_pos = unit.get("neighbor_positions", [])
            for wf_n, pos_n in zip(neigh_tm, neigh_pos):
                xn, yn = pos_n
                if np.hypot(xn - x0, yn - y0) > neighbor_cutoff:
                    continue
                wf2 = np.asarray(wf_n, dtype=float)
                wf2 = (wf2 - wf2.mean()) / (wf2.ptp() + 1e-9) * spatial_radius
                ax.plot(xn + t_space, yn + wf2, color=color, alpha=0.3, lw=0.8)

        if isinstance(unit_i.get("position"), (list, tuple, np.ndarray)):
            x_i, y_i = unit_i["position"]
            ax.scatter([x_i], [y_i], color="red", s=60, edgecolor="black",
                       label=f"Unit {unit_idx_i} (ID {neuron_key_i})", zorder=5)
        if isinstance(unit_j.get("position"), (list, tuple, np.ndarray)):
            x_j, y_j = unit_j["position"]
            ax.scatter([x_j], [y_j], color="blue", s=60, edgecolor="black",
                       label=f"Unit {unit_idx_j} (ID {neuron_key_j})", zorder=5)
        if (isinstance(unit_i.get("position"), (list, tuple, np.ndarray)) and
                isinstance(unit_j.get("position"), (list, tuple, np.ndarray))):
            ax.plot([x_i, x_j], [y_i, y_j], linestyle="--", color="gray", zorder=1)
            dist = np.hypot(x_i - x_j, y_i - y_j)
            ax.text((x_i + x_j) / 2 + 15, (y_i + y_j) / 2 + 15,
                    f"{dist:.1f} µm", fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))

        draw_overlay(unit_i, color="red", label="Sender waveform")
        draw_overlay(unit_j, color="blue", label="Receiver waveform")

        ax.set_aspect("equal")
        ax.set_xlabel("X position (µm)")
        ax.set_ylabel("Y position (µm)")
        ax.set_title("Waveform + Spatial Footprint")
        ax.grid(True, linestyle="--", alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, frameon=True, loc="upper right")

        if save_path:
            plt.savefig(save_path, dpi=300)
            if fig:
                plt.close(fig)
        elif show_plot and fig:
            plt.show()
    '''

    def plot_waveform_footprint(self, unit_idx_i, unit_idx_j, ax=None, show_plot=True, save_path=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))  # match aspect ratio
        else:
            fig = None

        neuron_key_i = self.neuron_keys[unit_idx_i]
        neuron_key_j = self.neuron_keys[unit_idx_j]
        unit_i = self.neuron_data[unit_idx_i]
        unit_j = self.neuron_data[unit_idx_j]

        positions = []

        def draw_overlay(unit, color, label, spatial_radius=20, neighbor_cutoff=100):
            pos = unit.get("position")
            tmpl = unit.get("template")
            if pos is None or tmpl is None:
                return
            x0, y0 = pos
            positions.append((x0, y0))  # Track for axis limits
            wf = np.asarray(tmpl, dtype=float)
            wf = (wf - wf.mean()) / (wf.ptp() + 1e-9) * spatial_radius
            t_space = np.linspace(-spatial_radius, spatial_radius, wf.size)
            ax.plot(x0 + t_space, y0 + wf, color=color, lw=1.5, label=label, zorder=4)
            ax.scatter([x0], [y0], color=color, s=60, edgecolor="black", zorder=5)

            neigh_tm = unit.get("neighbor_templates", [])
            neigh_pos = unit.get("neighbor_positions", [])
            for wf_n, pos_n in zip(neigh_tm, neigh_pos):
                xn, yn = pos_n
                if np.hypot(xn - x0, yn - y0) > neighbor_cutoff:
                    continue
                positions.append((xn, yn))
                wf2 = np.asarray(wf_n, dtype=float)
                wf2 = (wf2 - wf2.mean()) / (wf2.ptp() + 1e-9) * spatial_radius
                ax.plot(xn + t_space, yn + wf2, color=color, alpha=0.3, lw=0.8)

        if isinstance(unit_i.get("position"), (list, tuple, np.ndarray)):
            x_i, y_i = unit_i["position"]
            ax.scatter([x_i], [y_i], color="red", s=60, edgecolor="black",
                       label=f"Unit {unit_idx_i} (ID {neuron_key_i})", zorder=5)
            positions.append((x_i, y_i))
        if isinstance(unit_j.get("position"), (list, tuple, np.ndarray)):
            x_j, y_j = unit_j["position"]
            ax.scatter([x_j], [y_j], color="blue", s=60, edgecolor="black",
                       label=f"Unit {unit_idx_j} (ID {neuron_key_j})", zorder=5)
            positions.append((x_j, y_j))
        if (isinstance(unit_i.get("position"), (list, tuple, np.ndarray)) and
                isinstance(unit_j.get("position"), (list, tuple, np.ndarray))):
            ax.plot([x_i, x_j], [y_i, y_j], linestyle="--", color="gray", zorder=1)
            dist = np.hypot(x_i - x_j, y_i - y_j)
            ax.text((x_i + x_j) / 2 + 15, (y_i + y_j) / 2 + 15,
                    f"{dist:.1f} µm", fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))

        draw_overlay(unit_i, color="red", label="Sender waveform")
        draw_overlay(unit_j, color="blue", label="Receiver waveform")

        # Dynamic axis limits to match COM-style framing
        if positions:
            xs, ys = zip(*positions)
            x_margin, y_margin = 50, 30
            ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            ax.set_ylim(min(ys) - y_margin, max(ys) + y_margin)

        ax.set_xlabel("X position (µm)")
        ax.set_ylabel("Y position (µm)")
        ax.set_title("Waveform + Spatial Footprint")
        ax.grid(True, linestyle="--", alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7, frameon=True, loc="upper right")

        if save_path:
            plt.savefig(save_path, dpi=300)
            if fig:
                plt.close(fig)
        elif show_plot and fig:
            plt.show()

    def plot_combined_waveform_footprint(self, unit_idx_i, unit_idx_j,
                                         ax=None, show_plot=True, save_path=None):
        """
        Draws the two COMs (centers of mass) and full waveforms offset spatially,
        with optional overlay of neighboring unit templates from both neurons.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = None

        ui = self.neuron_data[unit_idx_i]
        uj = self.neuron_data[unit_idx_j]

        # Compute COMs
        com_i = self._compute_center_of_mass(ui)
        com_j = self._compute_center_of_mass(uj)

        xi, yi = com_i
        xj, yj = com_j

        # Draw connection line and label with distance
        ax.plot([xi, xj], [yi, yj], '--', color='gray', zorder=1)
        dist = np.hypot(xi - xj, yi - yj)
        ax.text((xi + xj) / 2 + 5, (yi + yj) / 2 + 5,
                f"{dist:.1f} µm", fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', pad=0.2))

        # Store all coordinate positions for dynamic axis limits
        all_x = [xi, xj]
        all_y = [yi, yj]

        # Plot waveforms and neighbors
        for (com, unit, color, label) in (
                (com_i, ui, 'red', 'Sender'),
                (com_j, uj, 'blue', 'Receiver')
        ):
            cx, cy = com
            wf = np.asarray(unit['template'], float)
            wf = (wf - wf.mean()) / (wf.ptp() + 1e-9) * 15
            t_space = np.linspace(-15, 15, wf.size)

            ax.plot(cx + t_space, cy + wf, color=color, lw=1.5, label=f"{label} waveform")
            ax.scatter([cx], [cy], color=color, s=60, edgecolor='black', zorder=5, label=f"{label} COM")

            # Include main unit COMs in bounds
            all_x.append(cx)
            all_y.append(cy)

            # Neighbor overlays
            neigh_tm = unit.get("neighbor_templates", [])
            neigh_pos = unit.get("neighbor_positions", [])

            for wf_n, pos_n in zip(neigh_tm, neigh_pos):
                xn, yn = pos_n
                wf2 = np.asarray(wf_n, float)
                wf2 = (wf2 - wf2.mean()) / (wf2.ptp() + 1e-9) * 15
                ax.plot(xn + t_space, yn + wf2, color=color, alpha=0.3, lw=0.8)

                all_x.append(xn)
                all_y.append(yn)

        # Apply consistent margins
        x_margin = 50
        y_margin = 50
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X position (µm)")
        ax.set_ylabel("Y position (µm)")
        ax.set_title("Spatial + Waveform Footprint (COM‐aligned)")
        ax.legend(loc='upper right', fontsize=7, frameon=True)

        if save_path:
            plt.savefig(save_path, dpi=300)
            if fig:
                plt.close(fig)
        elif show_plot and fig:
            plt.show()


    def compute_top_pairs(self,
                          n: int = 5,
                          min_spikes: int = 5,
                          sttc_range: Tuple[float, float] = (0.5, 1.0),
                          distance_range: Tuple[float, float] = (0, 1000),
                          latency_range: Tuple[float, float] = (1, 10),
                          pval_range: Tuple[float, float] = (0.01, 1.0)) -> List[Tuple[int, int, float]]:
        """
        Compute top STTC neuron pairs based on filtering logic.
        Returns the list of top (i, j, sttc) tuples.

        No plots are generated. This is intended for programmatic reuse.
        """

        if self.filtered_sttc is None:
            self.filter_sttc()

        mat = self.filtered_sttc.copy()
        np.fill_diagonal(mat, -np.inf)
        N = len(self.spike_trains)

        # Step 1: Initial candidate pairs
        pairs = [(i, j, mat[i, j]) for i in range(N) for j in range(i + 1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]

        # Step 2: STTC range filtering
        min_sttc, max_sttc = sttc_range
        pairs = [p for p in pairs if min_sttc <= p[2] <= max_sttc]

        # Step 3: Distance range filtering
        min_dist, max_dist = distance_range
        pairs_with_distance = []
        for i, j, sttc in pairs:
            coords_i, coords_j, dist, *_ = self.resolve_coordinates(i, j)
            if min_dist <= dist <= max_dist:
                pairs_with_distance.append((i, j, sttc, dist))

        # Step 4: Latency and p-value filtering
        min_lat, max_lat = latency_range
        min_p, max_p = pval_range
        win = 0.03  # latency window
        retained = []
        for i, j, sttc, dist in pairs_with_distance:
            A = np.array(self.spike_trains[i])
            B = np.array(self.spike_trains[j])
            lags = []
            for t in A:
                if B.size == 0:
                    continue
                diffs = B - t
                idx = np.argmin(np.abs(diffs))
                lat = diffs[idx]
                if abs(lat) <= win:
                    lags.append(lat)
            lags = np.array(lags)
            mean_latency = lags.mean() * 1000 if lags.size else 0
            try:
                from diptest import diptest
                dip_p = diptest(lags)[1] if lags.size > 3 else np.nan
            except:
                dip_p = np.nan

            if ((min_lat is not None and mean_latency < min_lat) or
                    (max_lat is not None and mean_latency > max_lat) or
                    (min_p is not None and (np.isnan(dip_p) or dip_p < min_p)) or
                    (max_p is not None and (np.isnan(dip_p) or dip_p > max_p))):
                continue

            retained.append((i, j, sttc))

            if len(retained) == n:
                break

        return retained


    def compare_spike_trains_stacked(self, unit_a, unit_b, bin_size=1.0, duration=90, start_time=0,
                                     show_latency=False, save_path=None, show_plot=True):
        """
        Compares original and shuffled spike trains for a pair of units, and shows STTC info.
        Includes latency histogram if show_latency=True.
        """
        if not hasattr(self, 'cached_shuffled_trains') or not self.cached_shuffled_trains:
            cache_path = f"outputs/shuffled_sttc/{self.dataset_name}_shuffled_spike_trains.npy"
            if os.path.exists(cache_path):
                print(f" Loading cached shuffled spike trains from {cache_path}")
                self.cached_shuffled_trains = np.load(cache_path, allow_pickle=True)
            else:
                print("️ Cached shuffled spike trains not found. Falling back to random_okun().")
                self.cached_shuffled_trains = self.random_okun(bin_size=bin_size)

        # Use same okun shuffled trains used to generate filtered sttc matrices
        if hasattr(self, 'cached_shuffled_trains') and self.cached_shuffled_trains:
            shuffled_trains = self.cached_shuffled_trains
        else:
            print("Warning: cached shuffled spike trains not found. Using a new shuffled set.")
            shuffled_trains = self.random_okun(bin_size=bin_size)

        # Time window
        end_time = start_time + duration

        def clip(spikes):
            return [t for t in spikes if start_time <= t <= end_time]

        # Clip original and shuffled spikes
        orig_a, orig_b = clip(self.spike_trains[unit_a]), clip(self.spike_trains[unit_b])
        shuf_a, shuf_b = clip(shuffled_trains[unit_a]), clip(shuffled_trains[unit_b])

        # === Set up figure ===
        fig, axes = plt.subplots(2 if show_latency else 1, 1,
                                 figsize=(10, 5 if show_latency else 3),
                                 height_ratios=[2, 1] if show_latency else [1],
                                 constrained_layout=True)

        ax0 = axes[0] if show_latency else axes  # raster
        ax1 = axes[1] if show_latency else None  # histogram

        # === Spike train raster ===
        lines = [orig_a, orig_b, shuf_a, shuf_b]
        offsets = [4, 3, 2, 1]
        colors = ['blue', 'blue', 'gray', 'gray']

        ax0.eventplot(lines, lineoffsets=offsets, linelengths=0.8, linewidths=0.75, colors=colors)

        ax0.set_xlim(start_time, end_time)
        ax0.set_ylim(0.5, 4.5)
        ax0.set_yticks(offsets)
        ax0.set_yticklabels([
            f"Unit {unit_a} (original)",
            f"Unit {unit_b} (original)",
            f"Unit {unit_a} (shuffled)",
            f"Unit {unit_b} (shuffled)"
        ])
        ax0.set_xlabel("Time (s)")
        ax0.spines[['top', 'right', 'left']].set_visible(False)
        ax0.tick_params(left=False)

        # === Title with STTC info ===
        sttc_val = self.original_sttc[unit_a, unit_b] if self.original_sttc is not None else None
        threshold = self.threshold
        significant = False

        if self.filtered_sttc is not None and sttc_val is not None:
            significant = self.filtered_sttc[unit_a, unit_b] > 0

        title_text = f"STTC between Unit {unit_a} & {unit_b}: {sttc_val:.3f} "
        if significant:
            title_text += f"(↑ passed z ≥ {threshold})"
        else:
            title_text += f"(not sig, z < {threshold})"

        ax0.set_title(title_text, fontsize=11, fontweight='bold', pad=10,
                      path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
                                    path_effects.Normal()])

        if self.randomized_sttc is not None:
            mu_ij = np.mean([mat[unit_a, unit_b] for mat in self.randomized_sttc])
            sigma_ij = np.std([mat[unit_a, unit_b] for mat in self.randomized_sttc])
            X_ij = sttc_val
            thresh = mu_ij + self.threshold * sigma_ij

            decision = "retained" if X_ij >= thresh else "filtered"

            explanation = (
                f"X_ij = {X_ij:.3f},   μ_ij = {mu_ij:.3f},   σ_ij = {sigma_ij:.3f}\n"
                f"Threshold: {mu_ij:.3f} + {self.threshold:.1f} * {sigma_ij:.3f} = {thresh:.3f} → {decision}"
            )

        ax0.text(0.5, -0.25, explanation,
                 transform=ax0.transAxes,
                 ha="center", va="top", fontsize=8,
                 fontfamily='monospace')

        # === Latency histogram ===
        if show_latency and ax1:
            spikes_a = np.array(self.spike_trains[unit_a])
            spikes_b = np.array(self.spike_trains[unit_b])
            lags = []
            for t in spikes_a:
                diffs = spikes_b - t
                lags.extend(diffs[(diffs >= -0.03) & (diffs <= 0.03)])
            lags = np.array(lags)

            bins = np.arange(-0.03, 0.03 + 0.001, 0.001)
            counts, _ = np.histogram(lags, bins=bins)
            ax1.bar(bins[:-1] * 1000, counts, width=1.0, color='skyblue', edgecolor='black')
            ax1.set_xlim(-30, 30)
            ax1.set_ylabel("Count")
            ax1.set_xlabel("Latency (ms)")
            ax1.spines[['top', 'right']].set_visible(False)

        if save_path:
            plt.savefig(save_path, dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close()

    def compute_fwhm(self, bins, counts):
        """
        Computes Full Width at Half Maximum (FWHM) of a latency histogram.

        Parameters:
            bins (np.ndarray): Bin edges (in seconds or ms)
            counts (np.ndarray): Histogram counts (same length as bins[:-1])

        Returns:
            fwhm (float): Full width at half max in the same time units as bins
        """
        if len(counts) == 0 or np.max(counts) == 0:
            return 0.0

        half_max = np.max(counts) / 2
        above_half = np.where(counts >= half_max)[0]

        if len(above_half) < 2:
            return 0.0  # Not enough points to define width

        left_idx = above_half[0]
        right_idx = above_half[-1]
        fwhm = bins[right_idx + 1] - bins[left_idx]  # bins are edges

        return fwhm


    # Define a batch function for running wide latency analysis across datasets
    @staticmethod
    def run_batch_wide_latency_analysis(analyzers, save_root="outputs/wide_latency_summary",
                                        n=1000, latency_std_threshold=8.0,
                                        mean_latency_range=(None, None),
                                        bin_size=0.001, save_csv=True):

        all_results = []

        for name, analyzer in analyzers.items():
            print(f"\n--- Running wide latency analysis for: {name} ---")
            save_dir = os.path.join(save_root, name)
            os.makedirs(save_dir, exist_ok=True)

            analyzer.find_wide_latency_pairs(
                n=n,
                latency_std_threshold=latency_std_threshold,
                mean_latency_range=mean_latency_range,
                save_dir=save_dir,
                save_csv=save_csv
            )

            csv_path = os.path.join(save_dir, f"{name}_wide_latency_pairs.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["dataset"] = name
                all_results.append(df)

        if all_results:
            full_df = pd.concat(all_results, ignore_index=True)
            summary_csv = os.path.join(save_root, "all_datasets_wide_latency_summary.csv")
            full_df.to_csv(summary_csv, index=False)
            print(f"\n All results combined and saved to:\n→ {summary_csv}")
        else:
            print("️ No wide latency pairs found in any dataset.")

    def plot_autocorrelogram(self, unit, bin_size=0.001, window=0.05, save_path=None):
        """
        Plots the autocorrelogram (spike time differences within the same unit) with annotations.

        Parameters:
            unit (int): Index of the neuron
            bin_size (float): Bin size in seconds (default 1 ms)
            window (float): +/- time window in seconds (default ±50 ms)
            save_path (str): If provided, saves the plot
        """
        spikes = np.array(self.spike_trains[unit])
        if len(spikes) < 2:
            print(f"Unit {unit} has too few spikes.")
            return

        # Compute all pairwise differences (excluding 0s)
        diffs = []
        for i in range(len(spikes)):
            dt = spikes - spikes[i]
            dt = dt[(dt >= -window) & (dt <= window) & (dt != 0)]
            diffs.extend(dt)

        diffs = np.array(diffs)
        bins = np.arange(-window, window + bin_size, bin_size)
        counts, _ = np.histogram(diffs, bins=bins)

        # Resolve spatial coordinates
        coords_i, _, _, neuron_id_i, _ = self.resolve_coordinates(unit, unit)

        # === Plot
        plt.figure(figsize=(5.5, 3.5))
        plt.bar(bins[:-1] * 1000, counts, width=bin_size * 1000, color='gray', edgecolor='black')
        plt.xlabel("Time lag (ms)")
        plt.ylabel("Count")
        title = f"Autocorrelogram – Unit {unit}"
        subtitle = f"{self.dataset_name} | Neuron ID: {neuron_id_i} | Pos: {coords_i}"
        plt.title(f"{title}\n{subtitle}", fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def directional_sttc(self, tA, tB, direction="A_to_B", delt=None):
        """
        Calculates directional STTC where only A → B or B → A contributions are considered.

        Parameters:
            tA, tB (array-like): Spike trains
            direction (str): "A_to_B" or "B_to_A"
            delt (float): Temporal window (defaults to self.delt)

        Returns:
            STTC value (float)
        """
        if delt is None:
            delt = self.delt

        if len(tA) == 0 or len(tB) == 0:
            return 0.0

        tA, tB = np.array(tA), np.array(tB)

        if direction == "A_to_B":
            # Check how many B spikes fall within delt AFTER A spikes
            count = 0
            for ta in tA:
                count += np.sum((tB > ta) & (tB <= ta + delt))
            P = count / len(tA)
            TB = self._sttc_ta_tb(tB) / self.recording_length
            sttc = (P - TB) / (1 - P * TB + 1e-10)
            return sttc

        elif direction == "B_to_A":
            # Same logic reversed
            count = 0
            for tb in tB:
                count += np.sum((tA > tb) & (tA <= tb + delt))
            P = count / len(tB)
            TA = self._sttc_ta_tb(tA) / self.recording_length
            sttc = (P - TA) / (1 - P * TA + 1e-10)
            return sttc

        else:
            raise ValueError("direction must be 'A_to_B' or 'B_to_A'")

    def plot_directional_sttc(self, i, j, bin_size=0.001, window=0.05, save_path=None):
        """
        Plots directional STTC values from i→j and j→i with annotated metadata.

        Parameters:
            i, j (int): Unit indices
            bin_size (float): Histogram bin width in seconds
            window (float): Latency window in seconds
            save_path (str): If provided, saves the plot
        """
        tA = np.array(self.spike_trains[i])
        tB = np.array(self.spike_trains[j])

        sttc_a_to_b = self.directional_sttc(tA, tB, direction="A_to_B")
        sttc_b_to_a = self.directional_sttc(tA, tB, direction="B_to_A")

        # Resolve coordinates
        coords_i, coords_j, dist, neuron_id_i, neuron_id_j = self.resolve_coordinates(i, j)

        # === Plot
        plt.figure(figsize=(5, 3))
        plt.bar(["A → B", "B → A"], [sttc_a_to_b, sttc_b_to_a], color=["skyblue", "salmon"], edgecolor="black")
        plt.ylim(-1, 1)
        plt.axhline(0, linestyle="--", color="gray")

        title = f"Directional STTC – Pair ({i}, {j})"
        subtitle = f"{self.dataset_name} | Neuron IDs: {neuron_id_i}, {neuron_id_j} | Pos: {coords_i}, {coords_j} | Dist: {dist}"
        plt.title(f"{title}\n{subtitle}", fontsize=10)
        plt.ylabel("STTC")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def characterize_connection(self, i, j, save_dir=None):
        """
        Run autocorrelograms and directional STTC for a given neuron pair.
        """
        print(f"\nAnalyzing connection between Unit {i} and Unit {j}")

        tA = self.spike_trains[i]
        tB = self.spike_trains[j]

        sttc_ab = self.directional_sttc(tA, tB, direction="A_to_B")
        sttc_ba = self.directional_sttc(tB, tA, direction="B_to_A")

        print(f"Directional STTC A→B: {sttc_ab:.3f}")
        print(f"Directional STTC B→A: {sttc_ba:.3f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.plot_autocorrelogram(i, save_path=os.path.join(save_dir, f"unit_{i}_autocorr.png"))
            self.plot_autocorrelogram(j, save_path=os.path.join(save_dir, f"unit_{j}_autocorr.png"))
            self.plot_directional_sttc(i, j, save_path=os.path.join(save_dir, f"pair_{i}_{j}_directional_sttc.png"))
        else:
            self.plot_autocorrelogram(i)
            self.plot_autocorrelogram(j)
            self.plot_directional_sttc(i, j)

    def characterize_connections_from_csv(self, csv_path, output_dir="outputs/connection_characterization"):
        """
        For each neuron pair in a CSV, compute directional STTC and plot autocorrelograms.

        Parameters:
            csv_path (str): Path to CSV containing 'unit_i' and 'unit_j' columns.
            output_dir (str): Base directory to save outputs.
        """

        df = pd.read_csv(csv_path)
        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []

        for idx, row in df.iterrows():
            i, j = int(row["unit_i"]), int(row["unit_j"])
            print(f"\n[{idx+1}/{len(df)}] Characterizing pair ({i}, {j})")

            tA = self.spike_trains[i]
            tB = self.spike_trains[j]

            # Compute directional STTC
            sttc_ab = self.directional_sttc(tA, tB, direction="A_to_B")
            sttc_ba = self.directional_sttc(tB, tA, direction="B_to_A")

            # Output directory per pair
            pair_dir = os.path.join(output_dir, f"pair_{i}_{j}")
            os.makedirs(pair_dir, exist_ok=True)

            # Plot autocorrelograms
            self.plot_autocorrelogram(i, save_path=os.path.join(pair_dir, f"unit_{i}_autocorr.png"))
            self.plot_autocorrelogram(j, save_path=os.path.join(pair_dir, f"unit_{j}_autocorr.png"))

            # Spatial info
            coords_i, coords_j, dist, _, _ = self.resolve_coordinates(i, j)

            # Record summary
            summary_rows.append({
                "unit_i": i,
                "unit_j": j,
                "sttc_A_to_B": sttc_ab,
                "sttc_B_to_A": sttc_ba,
                "coords_i": coords_i,
                "coords_j": coords_j,
                "distance_px": dist
            })

        # Save summary CSV
        summary_df = pd.DataFrame(summary_rows)
        csv_out = os.path.join(output_dir, f"{self.dataset_name}_characterization_summary.csv")
        summary_df.to_csv(csv_out, index=False)
        print(f"\n Summary saved to: {csv_out}")



    def plot_top_sttc_pairs(self, n=5, min_spikes=5, save_dir=None):
        if self.filtered_sttc is None:
            self.filtered_sttc = self.compute_sttc_matrix()

        N = len(self.spike_trains)
        matrix = self.filtered_sttc.copy()
        np.fill_diagonal(matrix, -np.inf)

        pairs = [(i, j, matrix[i, j]) for i in range(N) for j in range(i + 1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]
        top = sorted(pairs, key=lambda x: x[2], reverse=True)[:n]

        for idx, (i, j, score) in enumerate(top):
            print(f"\nTop {idx+1}: Unit {i} ↔ Unit {j} | STTC = {score:.3f}")
            base = f"{save_dir}/{self.dataset_name}_pair{idx+1}_U{i}_U{j}" if save_dir else None
            self.plot_unit_spike_trains(i, j, save_path=f"{base}_spike_trains.png" if base else None)
            self.plot_latency_histogram(i, j, save_path=f"{base}_latency.png" if base else None)


    def get_retained_pairs(self, threshold=2.5):
        """
        Returns a set of retained neuron pairs with z-score above the threshold.

        Args:
            threshold (float): Z-score cutoff for retention.

        Returns:
            set of tuple: Set of (i, j) pairs (i < j) retained by z-score.
        """
        if self.original_sttc is None:
            self.original_sttc = self.compute_sttc_matrix()

        if self.randomized_sttc is None:
            raise ValueError("Randomized STTC matrices not available.")

        rand_array = np.stack(self.randomized_sttc)
        mean_rand = np.mean(rand_array, axis=0)
        std_rand = np.std(rand_array, axis=0)
        std_rand[std_rand == 0] = 1e-6  # avoid divide-by-zero

        z_scores = (self.original_sttc - mean_rand) / std_rand

        retained = {
            tuple(sorted((i, j)))
            for i in range(z_scores.shape[0])
            for j in range(i + 1, z_scores.shape[1])
            if z_scores[i, j] > threshold
        }

        return retained

    def run_special_zscore_inspection(self,
                                      zscore_range=None,
                                      zscore_targets=None,
                                      top_n=20,
                                      latency_range=(None, None),
                                      pval_range=(None, None),
                                      sttc_range=(None, None),
                                      start_time=0,
                                      duration=60,
                                      save_dir=None):
        """
        Pick pairs around specific z-scores (range or targets) and top-|z|, then plot 3-panel
        (raster, latency hist, spatial+waveform) exactly as in grid_with_spatial.
        """

        if self.original_sttc is None or self.randomized_sttc is None:
            raise RuntimeError("Run full analysis (including randomization) first.")

        window = 0.03
        bin_size = 0.001

        R = np.stack(self.randomized_sttc)
        mu = R.mean(axis=0)
        sd = R.std(axis=0)
        sd[sd == 0] = 1e-6
        Z = (self.original_sttc - mu) / sd

        iu, ju = np.triu_indices_from(Z, k=1)
        z_flat = Z[iu, ju]

        pairs = []

        # --- Select pairs based on z-score range ---
        if zscore_range is not None and isinstance(zscore_range, (tuple, list)) and len(zscore_range) == 2:
            z_min, z_max = zscore_range
            for k in range(len(z_flat)):
                zval = z_flat[k]
                if z_min <= zval <= z_max:
                    i, j = iu[k], ju[k]
                    pairs.append((i, j, zval))
            print(f"Found {len(pairs)} pairs in z-score range ({z_min}, {z_max})")

            count_in_range = sum(0 <= self.original_sttc[i, j] <= 4 for i, j, _ in pairs)
            print(f"{count_in_range} pairs had STTC in range [0, 4] before STTC filtering.")

            if len(pairs) > top_n:
                pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:top_n]
                print(f"Truncated to top {top_n} by |z|.")
        else:
            for t in (zscore_targets or [-40, -30, -20, -10, 0, 10, 20, 30, 40]):
                idx = np.nanargmin(np.abs(z_flat - t))
                pairs.append((iu[idx], ju[idx], z_flat[idx]))
            top_idx = np.argsort(np.abs(z_flat))[-top_n:]
            for idx in top_idx:
                pairs.append((iu[idx], ju[idx], z_flat[idx]))

        if not pairs:
            print("No pairs passed the z-score selection.")
            return

        # --- STTC filtering ---
        min_sttc, max_sttc = sttc_range

        # Snapshot pairs before filtering for histogram
        pairs_before_sttc_filter = list(pairs)
        sttc_vals_full = [self.original_sttc[i, j] for i, j, _ in pairs_before_sttc_filter]
        sttc_vals_plot = [v for v in sttc_vals_full if 0 <= v <= 4]

        # Optional: save histogram of STTC values (before filtering)
        if save_dir:
            sttc_vals_plot = [self.original_sttc[i, j] for i, j, _ in pairs]


            plt.figure(figsize=(8, 5))
            plt.hist(sttc_vals_plot, bins=np.linspace(0, 4, 81),
                     color='skyblue', edgecolor='black')
            plt.xlabel("STTC Value")
            plt.ylabel("Number of Pairs")
            plt.title("STTC Distribution Within Z-Score Range", fontsize=14)
            plt.xlim(0, 4)
            plt.tight_layout()

            os.makedirs(save_dir, exist_ok=True)
            hist_path = os.path.join(save_dir, "sttc_histogram.png")
            plt.savefig(hist_path, dpi=300)
            plt.close()

        # Apply actual STTC filtering
        min_sttc, max_sttc = sttc_range
        if min_sttc is not None or max_sttc is not None:
            pairs = [
                (i, j, zval) for (i, j, zval) in pairs
                if (min_sttc is None or self.original_sttc[i, j] >= min_sttc) and
                   (max_sttc is None or self.original_sttc[i, j] <= max_sttc)
            ]
            print(f"{len(pairs)} pairs remained after STTC filtering (range: {min_sttc}, {max_sttc})")

        if not pairs:
            print("No pairs left after STTC filtering.")
            return

        # --- Latency and p-value filtering ---
        min_lat, max_lat = latency_range
        min_p, max_p = pval_range
        retained_pairs = []
        for i, j, zval in pairs:
            sttc = self.original_sttc[i, j]
            pair_key = (i, j) if i < j else (j, i)
            status = "Retained" if pair_key in self.get_retained_pairs(self.threshold) else "Filtered"
            coords_i, coords_j, dist, *_ = self.resolve_coordinates(i, j)

            A = np.array(self.spike_trains[i])
            B = np.array(self.spike_trains[j])
            lags = []
            for t_a in A:
                if B.size == 0:
                    continue
                diffs = B - t_a
                idx_min = np.argmin(np.abs(diffs))
                lat = diffs[idx_min]
                if abs(lat) <= window:
                    lags.append(lat)
            lags = np.array(lags)

            bins = np.arange(-window, window + bin_size, bin_size)
            counts, _ = np.histogram(lags, bins=bins)
            mean_latency = lags.mean() * 1000 if lags.size else 0

            try:
                dip_p = diptest(lags)[1] if lags.size > 3 else np.nan
            except:
                dip_p = np.nan

            peak_latency = np.nan
            if lags.size > 0:
                centers = (bins[:-1] + bins[1:]) / 2
                peak_indices, _ = find_peaks(counts)
                if peak_indices.size > 0:
                    peak_idx = peak_indices[np.argmax(counts[peak_indices])]
                    peak_latency = centers[peak_idx] * 1000

            if ((min_lat is not None and mean_latency < min_lat) or
                    (max_lat is not None and mean_latency > max_lat) or
                    (min_p is not None and (np.isnan(dip_p) or dip_p < min_p)) or
                    (max_p is not None and (np.isnan(dip_p) or dip_p > max_p))):
                continue

            retained_pairs.append((i, j, zval, sttc, status, dist, lags, bins, counts,
                                   mean_latency, dip_p, peak_latency))

        print(f"Retained {len(retained_pairs)} pairs after all filters.")

        if not retained_pairs:
            return

        # --- Plotting ---
        for count, (i, j, zval, sttc, status, dist, lags, bins, counts,
                    mean_latency, dip_p, peak_latency) in enumerate(retained_pairs, 1):

            spikes_i = [t for t in self.spike_trains[i] if start_time <= t < start_time + duration]
            spikes_j = [t for t in self.spike_trains[j] if start_time <= t < start_time + duration]

            fig = plt.figure(figsize=(18, 4), constrained_layout=True)
            gs = GridSpec(1, 3, width_ratios=[1, 1.1, 1.2], figure=fig)

            ax1 = fig.add_subplot(gs[0])
            ax1.eventplot([spikes_i, spikes_j], colors='black',
                          lineoffsets=[1.87, 1.13], linelengths=0.7, linewidths=0.75)
            ax1.set_xlim(start_time, start_time + duration)
            ax1.set_yticks([1.87, 1.13])
            ax1.set_yticklabels([f"U{i}", f"U{j}"])
            ax1.set_xlabel("Time (s)")
            ax1.set_title(f"Pair {i}-{j} | STTC={sttc:.3f}\nDist={dist:.1f}µm")
            ax1.plot([0, 30], [0.6, 0.6], 'k-', lw=2)
            ax1.text(15, 0.5, "30 s", ha='center', va='top')
            ax1.spines[['top', 'right', 'left']].set_visible(False)

            ax2 = fig.add_subplot(gs[1])
            ax2.bar((bins[:-1] * 1000), counts, width=bin_size * 1000,
                    color='skyblue', edgecolor='black')
            ax2.axvline(0, ls='--', c='r')
            ax2.set_xlim(-window * 1000, window * 1000)
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("Count")
            txt = f"{status}\nZ={zval:.2f}, STTC={sttc:.3f}\nMean={mean_latency:.1f} ms"
            if not np.isnan(dip_p):
                txt += f"\nP={dip_p:.3f}"
            if not np.isnan(peak_latency):
                txt += f"\nPeak={peak_latency:.1f} ms"
            ax2.text(0.98, 0.95, txt, ha='right', va='top',
                     transform=ax2.transAxes, fontsize=9, fontweight='bold')

            ax3 = fig.add_subplot(gs[2])
            self.plot_waveform_com_footprint(i, j, ax=ax3, show_plot=False)
            wf_sim = self.compute_waveform_similarity_sliding_dot(i, j)
            ax3.text(0.02, 0.98, f"Max-corr = {wf_sim:.3f}",
                     transform=ax3.transAxes, ha='left', va='top',
                     fontsize=9, bbox=dict(facecolor='white', edgecolor='gray', pad=0.2))

            if save_dir:
                fn = f"{self.dataset_name}_zinspect_pair{count}_U{i}_{j}.png"
                fig.savefig(os.path.join(save_dir, fn), dpi=300)
                plt.close(fig)
            else:
                plt.show()


    def plot_custom_pairs_grid_with_spatial(self, pairs, start_time=0, duration=60):
        """
        Plots 3-panel figures (spike trains, latency histogram, spatial footprint)
        for a list of manually specified (i, j, z) pairs.
        """
        for count, (i, j, z) in enumerate(pairs, 1):
            print(f"[{count}/{len(pairs)}] Units {i}-{j} | Z = {z:.2f}")

            spikes_a_window = [t for t in self.spike_trains[i] if start_time <= t <= start_time + duration]
            spikes_b_window = [t for t in self.spike_trains[j] if start_time <= t <= start_time + duration]

            spikes_a_full = np.array(self.spike_trains[i])
            spikes_b_full = np.array(self.spike_trains[j])
            lags = []
            window = 0.03
            for t in spikes_a_full:
                diffs = spikes_b_full - t
                diffs_in_window = diffs[(diffs >= -window) & (diffs <= window)]
                if len(diffs_in_window) > 0:
                    nearest = diffs_in_window[np.argmin(np.abs(diffs_in_window))]
                    lags.append(nearest)
            lags = np.array(lags)

            bin_size = 0.001
            bins = np.arange(-window, window + bin_size, bin_size)
            counts, _ = np.histogram(lags, bins=bins)

            dip, pval = diptest(lags) if len(lags) > 3 else (np.nan, np.nan)
            mean_latency = np.mean(lags) * 1000 if len(lags) > 0 else 0

            fig = plt.figure(figsize=(14, 3.5), constrained_layout=True)
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.2])

            ax1 = fig.add_subplot(gs[0])
            ax1.eventplot([spikes_a_window, spikes_b_window], colors='blue',
                          lineoffsets=[2, 1], linelengths=0.8, linewidths=0.75)
            ax1.set_xlim(start_time, start_time + duration)
            ax1.set_ylim(0.5, 2.5)
            ax1.set_yticks([2, 1])
            ax1.set_yticklabels([f"Unit {i}", f"Unit {j}"])
            ax1.set_xlabel("Time (s)")
            ax1.set_title("Spike trains")
            ax1.tick_params(left=False)
            ax1.spines[['top', 'right', 'left']].set_visible(False)
            bar_x = start_time + duration / 3
            ax1.plot([bar_x, bar_x + 30], [0.3, 0.3], color='black', lw=3)
            ax1.text(bar_x + 15, 0.1, "30 s", ha="center", va="top", fontsize=9)

            ax2 = fig.add_subplot(gs[1])
            ax2.bar(bins[:-1] * 1000, counts, width=bin_size * 1000,
                    color='skyblue', edgecolor='black')
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("Spike count")
            ax2.set_xlim(-window * 1000, window * 1000)
            ax2.text(0.98, 0.95,
                     f"Mean latency = {mean_latency:.1f} ms\nP = {pval:.3f}",
                     ha='right', va='top', transform=ax2.transAxes,
                     fontsize=9, fontweight='bold')

            ax3 = fig.add_subplot(gs[2])
            try:
                self.plot_waveform_footprint(i, j, ax=ax3)
            except Exception as e:
                print(f"Waveform overlay failed for {i}-{j}: {e}")
                ax3.set_title("Error rendering footprint")

            plt.show()

    def compute_directional_cosine_similarity(self, i, j, reference_vector=(1.0, 0.0)):
        """
        Computes the cosine similarity between the vector from neuron i to neuron j
        and a reference direction (default: horizontal).

        Metadata is accessed by index: self.neuron_data[i]['position'], etc.
        Returns:
            float: Cosine similarity between -1 and 1, or None if invalid.
        """
        # Ensure metadata exists
        if not self.neuron_data:
            return None

        # Pull unit dicts by index
        unit_i = self.neuron_data[i]
        unit_j = self.neuron_data[j]

        # Validate dicts and positions
        if not isinstance(unit_i, dict) or not isinstance(unit_j, dict):
            return None
        pos_i = unit_i.get("position")
        pos_j = unit_j.get("position")
        if pos_i is None or pos_j is None:
            return None

        # Convert to numpy arrays
        pos_i = np.asarray(pos_i, dtype=float)
        pos_j = np.asarray(pos_j, dtype=float)

        # Compute vector
        vec_ij = pos_j - pos_i
        norm_vec = np.linalg.norm(vec_ij)
        norm_ref = np.linalg.norm(reference_vector)

        if norm_vec == 0 or norm_ref == 0:
            return None

        # Cosine similarity
        return float(np.dot(vec_ij, reference_vector) / (norm_vec * norm_ref))


    def _compute_center_of_mass(self, unit):
        """
        Given one unit dict with:
          - 'position': [x,y] of main contact
          - 'neighbor_positions': list of [x,y]
          - 'template': main-contact waveform
          - 'neighbor_templates': list of neighbor waveforms
        returns (com_x, com_y) in µm.
        """
        main_pos  = unit['position']
        main_tmpl = unit['template']

        # pair up only the neighbors you have BOTH position and template for
        nbr_pos = unit.get('neighbor_positions', [])
        nbr_tmpl = unit.get('neighbor_templates', [])
        paired = list(zip(nbr_pos, nbr_tmpl))

        # if for some reason they still mismatch, truncate to the shorter
        if len(paired) < max(len(nbr_pos), len(nbr_tmpl)):
            paired = paired[:min(len(nbr_pos), len(nbr_tmpl))]

        # now build your arrays
        all_pos  = np.vstack([main_pos] + [p for p,_ in paired])    # shape (N,2)
        all_tmps =        [main_tmpl] + [t for _,t in paired]       # length N

        # find trough amplitude of each waveform
        amps = np.array([abs(np.min(wf)) for wf in all_tmps], float)  # length N

        # weighted COM
        wsum = amps.sum() + 1e-12
        com_x = (all_pos[:,0] * amps).sum() / wsum
        com_y = (all_pos[:,1] * amps).sum() / wsum

        return float(com_x), float(com_y)

    '''
    def _compute_center_of_mass(self, unit):
        """
        Given one unit dict with:
          - 'position': [x,y] of main contact
          - 'neighbor_positions': list of [x,y]
          - 'template': main-contact waveform
          - 'neighbor_templates': list of neighbor waveforms
        returns (com_x, com_y) in µm.
        """
        # collect all positions & templates
        poss = [unit['position']] + unit.get('neighbor_positions', [])
        tmpls = [unit['template']] + unit.get('neighbor_templates', [])
        # for each waveform, find its trough amplitude
        amps = []
        for wf in tmpls:
            wf = np.asarray(wf, float)
            trough_idx = np.argmin(wf)
            amps.append(abs(wf[trough_idx]))
        amps = np.array(amps)
        poss = np.array(poss, float)
        # weighted average
        wsum = amps.sum() + 1e-12
        com = (poss.T @ amps) / wsum
        return float(com[0]), float(com[1])
    '''

    def compute_waveform_similarity_sliding_dot(self, i, j):
        """
        Sliding-dot-product (normalized) between the two templates,
        returns max cross‐corr coefficient in [–1,1].
        """
        wf_i = np.asarray(self.neuron_data[i]['template'], float)
        wf_j = np.asarray(self.neuron_data[j]['template'], float)
        # zero‐mean
        wf_i = wf_i - wf_i.mean()
        wf_j = wf_j - wf_j.mean()
        # full cross‐correlation
        cc = np.correlate(wf_i, wf_j, mode='full')
        norm = np.linalg.norm(wf_i) * np.linalg.norm(wf_j) + 1e-12
        return float(cc.max() / norm)


    def compute_waveform_cosine_similarity(self, i, j):
        """
        Computes the cosine similarity between the raw templates of two units.

        Parameters:
            i, j (int): Indices in self.spike_trains / self.neuron_data lists.

        Returns:
            float or None: Cosine similarity between the two templates, or None if unavailable.
        """
        # Must have metadata aligned by index
        if not self.neuron_data:
            return None

        # Pull units by index
        unit_i = self.neuron_data[i]
        unit_j = self.neuron_data[j]

        # Validate templates exist
        tmpl_i = unit_i.get("template")
        tmpl_j = unit_j.get("template")
        if tmpl_i is None or tmpl_j is None:
            return None

        # Convert to 1D float arrays
        wf_i = np.asarray(tmpl_i, dtype=float).flatten()
        wf_j = np.asarray(tmpl_j, dtype=float).flatten()

        # Compute norms
        norm_i = np.linalg.norm(wf_i)
        norm_j = np.linalg.norm(wf_j)
        if norm_i == 0 or norm_j == 0:
            return None

        # Cosine similarity
        return float(np.dot(wf_i, wf_j) / (norm_i * norm_j))

    '''
    def plot_top_sttc_pairs_grid_with_com_spatial(self, n=5,
                                                  start_time=0,
                                                  duration=60,
                                                  min_spikes=5,
                                                  sttc_range=(None, None),
                                                  distance_range=(None, None),
                                                  latency_range=(None, None),
                                                  pval_range=(None, None),
                                                  save_dir=None,
                                                  show_plot=False):
        if self.filtered_sttc is None:
            self.filter_sttc()

        mat = self.filtered_sttc.copy()
        np.fill_diagonal(mat, -np.inf)
        N = len(self.spike_trains)

        # STTC range filtering
        pairs = [(i, j, mat[i, j]) for i in range(N) for j in range(i+1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        min_sttc, max_sttc = sttc_range
        pairs = [p for p in pairs if min_sttc <= p[2] <= max_sttc]
        print(f"{len(pairs)} pairs remained after STTC filtering")

        # Distance filtering
        dist_filtered = []
        for i, j, sttc in pairs:
            coords_i, coords_j, dist, *_ = self.resolve_coordinates(i, j)
            if distance_range[0] <= dist <= distance_range[1]:
                dist_filtered.append((i, j, sttc, coords_i, coords_j, dist))
        print(f"{len(dist_filtered)} pairs remained after distance filtering")

        # Latency + pval filtering
        retained = []
        win = 0.03
        bin_size = 0.001
        min_lat, max_lat = latency_range
        min_p, max_p = pval_range

        for i, j, sttc, coords_i, coords_j, dist in dist_filtered:
            A = np.array(self.spike_trains[i])
            B = np.array(self.spike_trains[j])
            lags = []
            for t in A:
                if B.size == 0:
                    continue
                diffs = B - t
                idx = np.argmin(np.abs(diffs))
                lat = diffs[idx]
                if abs(lat) <= win:
                    lags.append(lat)
            lags = np.array(lags)
            mean_latency = lags.mean() * 1000 if lags.size else 0
            try:
                dip_p = diptest(lags)[1] if lags.size > 3 else np.nan
            except:
                dip_p = np.nan

            if ((min_lat is not None and mean_latency < min_lat) or
                    (max_lat is not None and mean_latency > max_lat) or
                    (min_p is not None and (np.isnan(dip_p) or dip_p < min_p)) or
                    (max_p is not None and (np.isnan(dip_p) or dip_p > max_p))):
                continue

            retained.append((i, j, sttc, lags, mean_latency, dip_p, coords_i, coords_j, dist))
            if len(retained) == n:
                break

        print(f"{len(retained)} pairs retained after all filters.")


        # Plotting
        for idx, (i, j, sttc, lags, mean_latency, dip_p, coords_i, coords_j, dist) in enumerate(retained, 1):
            spikes_i = [t for t in self.spike_trains[i] if start_time <= t < start_time + duration]
            spikes_j = [t for t in self.spike_trains[j] if start_time <= t < start_time + duration]

            fig = plt.figure(figsize=(15, 4), constrained_layout=True)
            gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], figure=fig)

            # Panel 1: Raster
            ax1 = fig.add_subplot(gs[0])
            ax1.eventplot([spikes_i, spikes_j], colors='black', lineoffsets=[1.87, 1.13],
                          linelengths=[0.7, 0.7], linewidths=0.75)
            ax1.set_xlim(start_time, start_time + duration)
            ax1.set_ylim(0.5, 2.5)
            ax1.set_yticks([1.87, 1.13])
            ax1.set_yticklabels([f"U{i}", f"U{j}"])
            ax1.set_xlabel("Time (s)")
            ax1.set_title(f"Pair {i}-{j} | STTC={sttc:.3f}\nDist={dist:.1f} µm")
            ax1.plot([0, 30], [0.6, 0.6], 'k-', lw=2)
            ax1.text(15, 0.5, "30 s", ha='center', va='top')
            ax1.spines[['top', 'right', 'left']].set_visible(False)

            # Panel 2: Latency histogram
            ax2 = fig.add_subplot(gs[1])
            bins = np.arange(-win, win + bin_size, bin_size)
            counts, edges = np.histogram(lags, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            peak_latency = centers[np.argmax(counts)] * 1000 if counts.any() else np.nan

            ax2.bar(centers * 1000, counts, width=bin_size * 1000,
                    color='skyblue', edgecolor='black')
            ax2.axvline(0, linestyle='--', color='red', linewidth=1)
            ax2.axvline(peak_latency, linestyle='--', color='green', linewidth=1)
            ax2.set_xlim(-win * 1000, win * 1000)
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("Count")
            ax2.set_title(f"Latency\nMean={mean_latency:.1f} ms | Peak={peak_latency:.1f} ms")
            txt = f"STTC={sttc:.3f}"
            if not np.isnan(dip_p):
                txt += f"\nP={dip_p:.3f}"
            ax2.text(0.98, 0.95, txt, ha='right', va='top', transform=ax2.transAxes,
                     fontsize=9, fontweight='bold')

            # Panel 3: Spatial + waveform
            ax3 = fig.add_subplot(gs[2])
            self.plot_combined_waveform_footprint(unit_idx_i=i, unit_idx_j=j, ax=ax3, show_plot=False)
            wf_sim = self.compute_waveform_similarity_sliding_dot(i, j)
            ax3.text(0.02, 0.98,
                     f"Max-corr={wf_sim:.3f}",
                     transform=ax3.transAxes,
                     ha='left', va='top',
                     fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='gray', pad=0.2))

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fn = f"{self.dataset_name}_pair{idx}_U{i}_{j}.png"
                fig.savefig(os.path.join(save_dir, fn), dpi=300)
                plt.close(fig)
            elif show_plot:
                plt.show()
            else:
                plt.close(fig)
        '''

    def plot_top_sttc_pairs_grid_with_com_spatial(self, n=5,
                                                  start_time=0,
                                                  duration=60,
                                                  min_spikes=5,
                                                  sttc_range=(None, None),
                                                  distance_range=(None, None),
                                                  latency_disjoint_ranges=((-15, -1), (1, 15)),
                                                  pval_range=(None, None),
                                                  fwhm_thresh=10.0,
                                                  save_dir=None,
                                                  show_plot=False):
        def in_disjoint_range(value, ranges):
            return any(low < value < high for (low, high) in ranges)

        if self.filtered_sttc is None:
            self.filter_sttc()

        mat = self.filtered_sttc.copy()
        np.fill_diagonal(mat, -np.inf)
        N = len(self.spike_trains)

        # STTC filtering
        pairs = [(i, j, mat[i, j]) for i in range(N) for j in range(i + 1, N)
                 if len(self.spike_trains[i]) >= min_spikes and len(self.spike_trains[j]) >= min_spikes]
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
        min_sttc, max_sttc = sttc_range
        pairs = [p for p in pairs if min_sttc <= p[2] <= max_sttc]
        print(f"{len(pairs)} pairs remained after STTC filtering")

        # Distance filtering
        dist_filtered = []
        for i, j, sttc in pairs:
            coords_i, coords_j, dist, *_ = self.resolve_coordinates(i, j)
            if distance_range[0] <= dist <= distance_range[1]:
                dist_filtered.append((i, j, sttc, coords_i, coords_j, dist))
        print(f"{len(dist_filtered)} pairs remained after distance filtering")

        # Filtering loop
        retained = []
        win = 0.03
        bin_size = 0.001
        min_p, max_p = pval_range
        drop_fwhm = 0
        drop_latency = 0

        for i, j, sttc, coords_i, coords_j, dist in dist_filtered:
            A = np.array(self.spike_trains[i])
            B = np.array(self.spike_trains[j])
            lags = []
            for t in A:
                if B.size == 0:
                    continue
                diffs = B - t
                idx = np.argmin(np.abs(diffs))
                lat = diffs[idx]
                if abs(lat) <= win:
                    lags.append(lat)
            lags = np.array(lags)
            if not lags.size:
                continue

            mean_latency = lags.mean() * 1000

            try:
                dip_p = diptest(lags)[1] if lags.size > 3 else np.nan
            except:
                dip_p = np.nan

            # FWHM calculation
            bins = np.arange(-win, win + bin_size, bin_size)
            counts, edges = np.histogram(lags, bins=bins)
            max_count = counts.max()
            half_max = max_count / 2.0
            above_half = np.where(counts >= half_max)[0]
            if above_half.size > 1:
                left = edges[above_half[0]]
                right = edges[above_half[-1] + 1]
                fwhm = (right - left) * 1000  # ms
            else:
                fwhm = np.inf
            if fwhm > fwhm_thresh:
                drop_fwhm += 1
                continue

            # Latency range filtering
            if not in_disjoint_range(mean_latency, latency_disjoint_ranges):
                drop_latency += 1
                continue

            # Dip test filtering
            if ((min_p is not None and (np.isnan(dip_p) or dip_p < min_p)) or
                    (max_p is not None and (np.isnan(dip_p) or dip_p > max_p))):
                continue

            retained.append((i, j, sttc, lags, mean_latency, dip_p, fwhm, coords_i, coords_j, dist))
            if len(retained) == n:
                break

        print(f"{drop_fwhm} pairs excluded by FWHM > {fwhm_thresh} ms")
        print(f"{drop_latency} pairs excluded outside disjoint latency ranges: {latency_disjoint_ranges}")
        print(f"{len(retained)} pairs retained after all filters.")

        # Plotting
        for idx, (i, j, sttc, lags, mean_latency, dip_p, fwhm, coords_i, coords_j, dist) in enumerate(retained, 1):
            spikes_i = [t for t in self.spike_trains[i] if start_time <= t < start_time + duration]
            spikes_j = [t for t in self.spike_trains[j] if start_time <= t < start_time + duration]

            fig = plt.figure(figsize=(15, 4), constrained_layout=True)
            gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], figure=fig)

            # Raster plot
            ax1 = fig.add_subplot(gs[0])
            ax1.eventplot([spikes_i, spikes_j], colors='black', lineoffsets=[1.87, 1.13],
                          linelengths=[0.7, 0.7], linewidths=0.75)
            ax1.set_xlim(start_time, start_time + duration)
            ax1.set_ylim(0.5, 2.5)
            ax1.set_yticks([1.87, 1.13])
            ax1.set_yticklabels([f"U{i}", f"U{j}"])
            ax1.set_xlabel("Time (s)")
            ax1.set_title(f"Pair {i}-{j} | STTC={sttc:.3f}\nDist={dist:.1f} µm")
            ax1.plot([0, 30], [0.6, 0.6], 'k-', lw=2)
            ax1.text(15, 0.5, "30 s", ha='center', va='top')
            ax1.spines[['top', 'right', 'left']].set_visible(False)

            # Latency histogram
            ax2 = fig.add_subplot(gs[1])
            counts, edges = np.histogram(lags, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            peak_latency = centers[np.argmax(counts)] * 1000 if counts.any() else np.nan

            ax2.bar(centers * 1000, counts, width=bin_size * 1000, color='skyblue', edgecolor='black')
            ax2.axvline(0, linestyle='--', color='red', linewidth=1)
            ax2.axvline(peak_latency, linestyle='--', color='green', linewidth=1)
            ax2.set_xlim(-win * 1000, win * 1000)
            ax2.set_xlabel("Latency (ms)")
            ax2.set_ylabel("Count")
            ax2.set_title(f"Latency\nMean={mean_latency:.1f} ms | Peak={peak_latency:.1f} ms")
            txt = f"STTC={sttc:.3f}\nFWHM={fwhm:.2f} ms"
            if not np.isnan(dip_p):
                txt += f"\nP={dip_p:.3f}"
            ax2.text(0.98, 0.95, txt, ha='right', va='top', transform=ax2.transAxes,
                     fontsize=9, fontweight='bold')

            # Spatial/waveform plot
            ax3 = fig.add_subplot(gs[2])
            self.plot_combined_waveform_footprint(unit_idx_i=i, unit_idx_j=j, ax=ax3, show_plot=False)
            wf_sim = self.compute_waveform_similarity_sliding_dot(i, j)
            ax3.text(0.02, 0.98, f"Max-corr={wf_sim:.3f}",
                     transform=ax3.transAxes, ha='left', va='top',
                     fontsize=9, bbox=dict(facecolor='white', edgecolor='gray', pad=0.2))

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fn = f"{self.dataset_name}_pair{idx}_U{i}_{j}.png"
                fig.savefig(os.path.join(save_dir, fn), dpi=300)
                plt.close(fig)
            elif show_plot:
                plt.show()
            else:
                plt.close(fig)

    def plot_top_sttc_pairs_grid_with_spatial(self,
                                              n=5, start_time=0, duration=60,
                                              min_spikes=5, latency_range=(None,None),
                                              pval_range=(None,None), save_dir=None,
                                              show_plot=True):
        """
        Top‐N *filtered* STTC pairs → 3‐panel grid (raster, latency, spatial+wf).
        """
        # ensure filtered_sttc exists
        if self.filtered_sttc is None:
            self.filter_sttc()

        # pick top‐N
        N = len(self.spike_trains)
        M = self.filtered_sttc.copy()
        np.fill_diagonal(M, -np.inf)
        pairs = [(i,j,M[i,j]) for i in range(N) for j in range(i+1,N)
                 if len(self.spike_trains[i])>=min_spikes and
                 len(self.spike_trains[j])>=min_spikes]
        top = sorted(pairs,key=lambda x: x[2],reverse=True)[:n]

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for idx,(i,j,sttc) in enumerate(top,1):
            # raster panel
            spikes_i = [t for t in self.spike_trains[i] if start_time<=t<start_time+duration]
            spikes_j = [t for t in self.spike_trains[j] if start_time<=t<start_time+duration]

            # panel2 (latency)
            A = np.array(self.spike_trains[i])
            B = np.array(self.spike_trains[j])
            win = 0.03
            lags = []
            for t in A:
                d = B - t
                if d.size:
                    nn = d[np.argmin(np.abs(d))]
                    if abs(nn) <= win:
                        lags.append(nn)
            lags = np.array(lags)

            # Histogram and peak latency
            bins = np.arange(-win, win + 0.001, 0.001)
            counts, edges = np.histogram(lags, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            peak_latency = centers[np.argmax(counts)] * 1000 if counts.any() else np.nan

            # Summary statistics
            dip_p = diptest(lags)[1] if lags.size > 3 else np.nan
            mean_lat = lags.mean() * 1000 if lags.size else 0

            # Filtering
            min_lat, max_lat = latency_range
            min_p, max_p = pval_range
            if ((min_lat is not None and mean_lat < min_lat) or
                    (max_lat is not None and mean_lat > max_lat) or
                    (min_p   is not None and (np.isnan(dip_p) or dip_p < min_p)) or
                    (max_p   is not None and (np.isnan(dip_p) or dip_p > max_p))):
                continue

            # coords & dist
            coords_i, coords_j, dist, key_i, key_j = self.resolve_coordinates(i,j)

            # 3‐panel
            fig=plt.figure(figsize=(18,4), constrained_layout=True)
            gs=GridSpec(1,3, width_ratios=[1,1.1,1.2], figure=fig)

            # raster
            ax1=fig.add_subplot(gs[0])
            ax1.eventplot([spikes_i,spikes_j],colors='blue',
                          lineoffsets=[2,1],linelengths=0.8,linewidths=0.75)
            ax1.set_xlim(start_time,start_time+duration)
            ax1.set_title(f"STTC={sttc:.3f}")
            ax1.spines[['top','right','left']].set_visible(False)

            # latency
            ax2=fig.add_subplot(gs[1])
            ax2.bar(bins[:-1]*1000,counts,width=1.0)
            ax2.axvline(0,ls='--',c='r')
            txt=f"Mean={mean_lat:.1f}ms"
            if not np.isnan(dip_p): txt+=f"\nP={dip_p:.3f}"
            ax2.text(0.98,0.95,txt,ha='right',va='top',transform=ax2.transAxes)

            # spatial+wf
            ax3=fig.add_subplot(gs[2])
            try:
                self.plot_waveform_footprint(i,j,ax=ax3,show_plot=False)
                ann=f"dist={dist:.1f}µm"
                dir_sim=self.compute_directional_cosine_similarity(i,j)
                wf_sim=self.compute_waveform_cosine_similarity(i,j)
                if dir_sim is not None: ann+=f"\nDirSim={dir_sim:.3f}"
                if wf_sim  is not None: ann+=f"\nWfSim={wf_sim:.3f}"
                x_i,y_i = self.neuron_data[i]['position']
                x_j,y_j = self.neuron_data[j]['position']
                ax3.text((x_i+x_j)/2+15,(y_i+y_j)/2+15,ann,
                         fontsize=8,ha='left',va='bottom',
                         bbox=dict(facecolor='white',edgecolor='gray',pad=0.3))
                ax3.set_title("Spatial + Waveform")
            except:
                ax3.set_title("Plot failed")

            # save/show
            if save_dir:
                fname = f"{self.dataset_name}_pair{idx}_U{i}_{j}.png"
                fig.savefig(os.path.join(save_dir, fname), dpi=300)
                plt.close(fig)
            elif show_plot:
                plt.show()
            else:
                plt.close(fig)


    def find_central_peak_candidates(self, window=0.03, bin_size=0.001, center_width=0.004, flank_range=(0.01, 0.03),
                                     min_ratio=2.0, min_spikes=5, top_n=None):
        """
        Scans all neuron pairs for a strong central peak in the latency histogram.

        Parameters:
            window (float): Total latency window (± in seconds)
            bin_size (float): Histogram bin width (s)
            center_width (float): Width of central region (± this value)
            flank_range (tuple): Range for flank comparison (in seconds)
            min_ratio (float): Threshold ratio of center bin to flank average
            min_spikes (int): Minimum spikes per neuron to be included
            top_n (int): Optional limit on number of candidate pairs to return

        Returns:
            List of candidate pairs: (i, j, central_ratio)
        """
        print(f"Scanning for strong central peaks (center ±{center_width*1000:.1f} ms)...")

        N = len(self.spike_trains)
        bins = np.arange(-window, window + bin_size, bin_size)

        candidates = []
        for i in range(N):
            for j in range(i + 1, N):
                if len(self.spike_trains[i]) < min_spikes or len(self.spike_trains[j]) < min_spikes:
                    continue

                spikes_a = np.array(self.spike_trains[i])
                spikes_b = np.array(self.spike_trains[j])

                lags = []
                for t in spikes_a:
                    diffs = spikes_b - t
                    lags.extend(diffs[(diffs >= -window) & (diffs <= window)])
                if len(lags) < 5:
                    continue

                counts, _ = np.histogram(lags, bins=bins)

                # Define central bin index range
                center_idx = np.where((bins[:-1] >= -center_width) & (bins[:-1] <= center_width))[0]
                flank_idx = np.where(((bins[:-1] >= flank_range[0]) & (bins[:-1] <= window)) |
                                     ((bins[:-1] <= -flank_range[0]) & (bins[:-1] >= -window)))[0]

                if len(center_idx) == 0 or len(flank_idx) == 0:
                    continue

                center_peak = np.max(counts[center_idx])
                flank_mean = np.mean(counts[flank_idx]) + 1e-5  # Avoid division by zero

                ratio = center_peak / flank_mean

                if ratio >= min_ratio:
                    candidates.append((i, j, ratio))

        # Sort and optionally truncate
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        if top_n:
            candidates = candidates[:top_n]

        print(f"Found {len(candidates)} candidate pairs.")
        return candidates

    def save_central_peak_candidates_with_plots(self, save_dir="central_peak_candidates", **kwargs):
        """
        Finds central peak candidates, saves their info to CSV, and generates latency histogram plots.

        Parameters:
            save_dir (str): Directory where plots and CSV will be saved.
            kwargs: Parameters forwarded to find_central_peak_candidates().
        """
        os.makedirs(save_dir, exist_ok=True)

        candidates = self.find_central_peak_candidates(**kwargs)
        rows = []

        for idx, (i, j, ratio) in enumerate(candidates):
            # === Latency histogram
            window = kwargs.get("window", 0.03)
            bin_size = kwargs.get("bin_size", 0.001)

            spikes_a = np.array(self.spike_trains[i])
            spikes_b = np.array(self.spike_trains[j])
            lags = []
            for t in spikes_a:
                diffs = spikes_b - t
                lags.extend(diffs[(diffs >= -window) & (diffs <= window)])
            lags = np.array(lags)

            bins = np.arange(-window, window + bin_size, bin_size)
            counts, _ = np.histogram(lags, bins=bins)

            # === Spatial info
            coords_i, coords_j, dist, _, _ = self.resolve_coordinates(i, j)

            # === Save plot
            plt.figure(figsize=(6, 3))
            plt.bar(bins[:-1] * 1000, counts, width=bin_size * 1000, color="gray", edgecolor="black")
            plt.title(f"Pair ({i}, {j}) | Ratio: {ratio:.2f}")
            plt.xlabel("Latency (ms)")
            plt.ylabel("Spike count")
            plt.axvline(0, color="red", linestyle="--")
            plot_path = os.path.join(save_dir, f"{self.dataset_name}_pair_{i}_{j}_ratio_{ratio:.2f}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

            # === Record metadata
            rows.append({
                "unit_i": i,
                "unit_j": j,
                "ratio": ratio,
                "coords_i": coords_i,
                "coords_j": coords_j,
                "distance_px": dist,
                "plot_path": os.path.relpath(plot_path, start=save_dir)
            })

        # === Save CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, f"{self.dataset_name}_central_peak_candidates.csv")
        df.to_csv(csv_path, index=False)
        print(f" Saved {len(rows)} candidates to:\n→ CSV: {csv_path}\n→ Plots in: {save_dir}")

    def run_jitter_test_from_csv(self, candidate_dir="central_peak_candidates", save_base_dir="jitter_results", n_shuffles=100,
                                 jitter_window=0.05, bin_size=0.001, latency_window=0.03):
        """
        Loads central peak candidates from CSV and runs jitter tests with spatial annotations.

        Parameters:
            candidate_dir (str): Directory where central peak candidate CSVs live
            save_base_dir (str): Base directory to save jitter plots
            n_shuffles (int): Number of jitter shuffles
            jitter_window (float): Time window for jittering
            bin_size (float): Histogram bin size
            latency_window (float): Window around each spike for latency histogram
        """
        # Path to CSV file for this dataset
        csv_path = os.path.join(candidate_dir, f"{self.dataset_name}_central_peak_candidates.csv")
        if not os.path.exists(csv_path):
            print(f" No CSV found for dataset {self.dataset_name} at: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        out_dir = os.path.join(save_base_dir, self.dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n Running jitter test on {len(df)} pairs from: {csv_path}")
        for idx, row in df.iterrows():
            i, j = int(row["unit_i"]), int(row["unit_j"])
            print(f"  → Pair ({i}, {j})")

            fig = self.run_jitter_test_for_pair(
                unit_a=i,
                unit_b=j,
                n_shuffles=n_shuffles,
                jitter_window=jitter_window,
                bin_size=bin_size,
                latency_window=latency_window,
                show_plot=False
            )

            # Annotate spatial coords if available
            coords_i, coords_j, dist, _, _ = self.resolve_coordinates(i, j)
            if coords_i != "N/A" and coords_j != "N/A":
                fig.axes[0].text(
                    0.02, 0.98,
                    f"Coords:\n{i}: {coords_i}\n{j}: {coords_j}\nDist: {dist:.1f}px",
                    transform=fig.axes[0].transAxes,
                    ha="left", va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3")
                )

            fname = f"{self.dataset_name}_pair_{i}_{j}_jitter.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=300)
            plt.close(fig)

        print(f" All jitter test plots saved to: {out_dir}")

    def run_jitter_test_for_pair(self, unit_a, unit_b, n_shuffles=100, jitter_window=0.05,
                                 bin_size=0.001, latency_window=0.03, show_plot=True):
        """
        Performs jitter test on latency histogram between two units.
        Optionally shows or returns the figure.
        """
        spikes_a = np.array(self.spike_trains[unit_a])
        spikes_b = np.array(self.spike_trains[unit_b])
        lags = []

        for t in spikes_a:
            diffs = spikes_b - t
            lags.extend(diffs[(diffs >= -latency_window) & (diffs <= latency_window)])
        lags = np.array(lags)

        bins = np.arange(-latency_window, latency_window + bin_size, bin_size)
        actual_counts, _ = np.histogram(lags, bins=bins)

        # Jitter test: shift A by random jitter in range
        jittered_max_peaks = []
        for _ in range(n_shuffles):
            jitter = np.random.uniform(-jitter_window, jitter_window, size=spikes_a.shape)
            jittered = spikes_a + jitter
            jittered_lags = []
            for t in jittered:
                diffs = spikes_b - t
                jittered_lags.extend(diffs[(diffs >= -latency_window) & (diffs <= latency_window)])
            counts, _ = np.histogram(jittered_lags, bins=bins)
            jittered_max_peaks.append(np.max(counts))

        # Compute p-value: how often is max jittered peak >= actual center peak?
        actual_peak = np.max(actual_counts)
        p_value = np.sum(np.array(jittered_max_peaks) >= actual_peak) / n_shuffles

        # Plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(bins[:-1] * 1000, actual_counts, width=bin_size * 1000,
               color='skyblue', edgecolor='black')
        ax.set_title(f"Latency Histogram: Units {unit_a} vs {unit_b}\nJitter p = {p_value:.3f}")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Spike count")
        ax.axvline(0, linestyle='--', color='gray', lw=1)

        if show_plot:
            plt.show()
        else:
            return fig

    def export_all_figures(self, n=5, threshold=0.3, save_dir="outputs/figures"):
        os.makedirs(save_dir, exist_ok=True)
        self.plot_top_sttc_pairs(n=n, save_dir=save_dir)
        self.plot_network_graph(threshold=threshold, save_path=f"{save_dir}/{self.dataset_name}_network.png")

    @staticmethod
    def load_all_curated_pairs(root_dir="outputs/jitter_results"):

        all_data = []

        for dataset in os.listdir(root_dir):
            csv_path = os.path.join(root_dir, dataset, f"{dataset}_jitter_results.csv")
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                df["dataset"] = dataset
                all_data.append(df)

        if not all_data:
            print(" No curated jitter CSVs found in:", root_dir)
            return None

        full_df = pd.concat(all_data, ignore_index=True)
        print(f" Loaded {len(full_df)} connections from {len(all_data)} datasets.")
        return full_df

    ###########################################################
    # Auditing & Diagnostics
    ###########################################################

    def plot_sttc_heatmap_comparison(self):
        """Compare original, mean shuffled, and filtered STTC matrices."""
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        vmin, vmax = 0, 1

        axs[0].imshow(self.original_sttc, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0].set_title("Original STTC")

        mean_shuffled = np.mean(self.randomized_sttc, axis=0)
        axs[1].imshow(mean_shuffled, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[1].set_title("Mean Shuffled STTC")

        axs[2].imshow(self.filtered_sttc, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[2].set_title("Filtered STTC")

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f"STTC Heatmap Audit — {self.dataset_name}")
        plt.tight_layout()
        plt.show()

    def plot_zscore_distribution(self):
        """Histogram of Z-scores used in filtering."""
        mean = np.mean(self.randomized_sttc, axis=0)
        std = np.std(self.randomized_sttc, axis=0)
        std[std == 0] = 1e-6
        z_scores = (self.original_sttc - mean) / std
        z_flat = z_scores[np.triu_indices_from(z_scores, k=1)]

        plt.figure(figsize=(6, 4))
        plt.hist(z_flat, bins=50, alpha=0.7, color="purple")
        plt.axvline(self.threshold, color="black", linestyle="--", label=f"Threshold: {self.threshold}")
        plt.title(f"Z-Score Distribution — {self.dataset_name}")
        plt.xlabel("Z-Score")
        plt.ylabel("Pair Count")
        plt.legend()
        plt.show()

    def check_matrix_symmetry(self):
        """Prints a check on matrix symmetry."""
        if np.allclose(self.original_sttc, self.original_sttc.T):
            print(f"{self.dataset_name}: Original STTC matrix is symmetric.")
        else:
            print(f"{self.dataset_name}: Original STTC matrix is NOT symmetric!")

    def count_filtered_pairs(self):
        """Prints the number of filtered pairs that passed threshold."""
        if self.filtered_sttc is None:
            print("Filtered STTC matrix not available.")
            return
        upper = np.triu(self.filtered_sttc, k=1)
        passed = np.sum(upper > 0)
        print(f"{self.dataset_name}: {passed} neuron pairs passed threshold (z >= {self.threshold})")

    def plot_zscore_matrix(self):
        """Heatmap of Z-scores across the matrix."""
        mean = np.mean(self.randomized_sttc, axis=0)
        std = np.std(self.randomized_sttc, axis=0)
        std[std == 0] = 1e-6
        z_scores = (self.original_sttc - mean) / std

        plt.figure(figsize=(6, 5))
        sns.heatmap(z_scores, cmap="coolwarm", center=0)
        plt.title(f"Z-Score Matrix — {self.dataset_name}")
        plt.show()

    def audit_all(self):
        """Runs all available STTC audit checks."""
        print(f"\ Running STTC audit for: {self.dataset_name}")
        self.check_matrix_symmetry()
        self.count_filtered_pairs()
        self.plot_sttc_heatmap_comparison()
        self.plot_zscore_distribution()
        self.plot_zscore_matrix()

    def pairwise_integrity_summary(self, show_plot=True, save_path="outputs/pairwise_integrity_summary.csv"):
        """
        Scans all STTC pairs in original and filtered matrices for:
          - missing waveform templates,
          - missing coordinates,
          - duplicate coordinates.
        Metadata is accessed by index (self.neuron_data[i]).
        Returns:
            df (pd.DataFrame): Detailed pairwise integrity records.
            summary (pd.DataFrame): Summary counts by problem type and matrix source.
        """
        records = []

        def assess(i, j, source):
            # Pull metadata by index
            unit_i = self.neuron_data[i] if self.neuron_data else {}
            unit_j = self.neuron_data[j] if self.neuron_data else {}

            # Waveform presence
            tmpl_i = unit_i.get("template")
            tmpl_j = unit_j.get("template")
            has_wf_i = isinstance(tmpl_i, (list, np.ndarray)) and len(tmpl_i) > 0
            has_wf_j = isinstance(tmpl_j, (list, np.ndarray)) and len(tmpl_j) > 0
            missing_waveform = not (has_wf_i and has_wf_j)

            # Coordinate presence
            pos_i = unit_i.get("position")
            pos_j = unit_j.get("position")
            valid_coords_i = isinstance(pos_i, (list, tuple, np.ndarray)) and len(pos_i) == 2
            valid_coords_j = isinstance(pos_j, (list, tuple, np.ndarray)) and len(pos_j) == 2
            missing_coordinates = not (valid_coords_i and valid_coords_j)

            # Duplicate coordinates
            duplicate_coords = valid_coords_i and valid_coords_j and tuple(pos_i) == tuple(pos_j)

            return {
                "i": i,
                "j": j,
                "source": source,
                "missing_waveform": missing_waveform,
                "missing_coordinates": missing_coordinates,
                "duplicate_coords": duplicate_coords,
                "any_problem": missing_waveform or missing_coordinates or duplicate_coords
            }

        def scan(matrix, label):
            N = matrix.shape[0]
            for i in range(N):
                for j in range(i+1, N):
                    if matrix[i, j] > 0:
                        records.append(assess(i, j, source=label))

        # Scan original and filtered matrices
        if self.original_sttc is not None:
            scan(self.original_sttc, "Original")
        if self.filtered_sttc is not None:
            scan(self.filtered_sttc, "Filtered")

        # Build DataFrame
        df = pd.DataFrame(records)

        # Summary aggregation
        summary = df.groupby("source").agg(
            total=("i", "count"),
            missing_waveform=("missing_waveform", "sum"),
            missing_coordinates=("missing_coordinates", "sum"),
            duplicate_coords=("duplicate_coords", "sum"),
            any_problem=("any_problem", "sum")
        ).reset_index()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        if show_plot:
            # Prepare for plotting
            melted = summary.melt(
                id_vars="source",
                value_vars=["missing_waveform", "missing_coordinates", "duplicate_coords", "any_problem"],
                var_name="Problem",
                value_name="Count"
            )
            melted["total"] = melted["source"].map(dict(zip(summary["source"], summary["total"])))
            melted["Percent"] = 100 * melted["Count"] / melted["total"]

            # Plot
            category_order = ["missing_waveform", "missing_coordinates", "duplicate_coords", "any_problem"]
            color_map = {
                "missing_waveform": "#2196f3",
                "missing_coordinates": "#ff9800",
                "duplicate_coords": "#673ab7",
                "any_problem": "#f44336",
            }

            fig, ax = plt.subplots(figsize=(12, 6))
            labels = []
            bar_values = []

            for cat in category_order:
                for src in summary["source"]:
                    row = melted[(melted["Problem"] == cat) & (melted["source"] == src)]
                    if not row.empty:
                        count = int(row["Count"].values[0])
                        pct = row["Percent"].values[0]
                        total = int(row["total"].values[0])
                        idx = len(labels)
                        ax.bar(idx, count, color=color_map[cat], edgecolor="black")
                        ax.text(idx, count + total*0.01, f"{count}\n({pct:.1f}%)",
                                ha="center", va="bottom", fontsize=8)
                        labels.append(f"{cat.replace('_',' ').title()} ({src})")
                        bar_values.append(count)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Number of Connections")
            ax.set_title(f"{self.dataset_name} — Connection Integrity Summary")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_ylim(top=max(bar_values)*1.15)

            # Annotate totals
            total_text = (
                f"Original: {summary.loc[summary.source=='Original','total'].values[0]:,} connections\n"
                f"Filtered: {summary.loc[summary.source=='Filtered','total'].values[0]:,} connections"
            )
            ax.annotate(total_text,
                        xy=(0.88, 0.92), xycoords="axes fraction",
                        ha="left", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

            plt.tight_layout()
            plt.show()

        return df, summary


    ###########################################################
    # Run and Save
    ###########################################################

    def run_analysis(self, use_cached_random=True, save_dir=None):
        """
        Full STTC analysis pipeline: original STTC, randomization, filtering.

        Parameters:
            use_cached_random (bool): If True, tries to load previously saved shuffled STTC matrix.
            save_dir (str): Directory to load/save the shuffled STTC matrix.
        """
        print(f"\n→ Starting analysis for: {self.dataset_name}")
        print(f"  Using cached shuffled STTC: {use_cached_random}")

        self.original_sttc = self.compute_sttc_matrix()

        if use_cached_random:
            try:
                self.load_randomized_sttc(save_dir=save_dir)
            except Exception as e:
                print(f"️ Could not load cached STTC: {e}")
                self.compute_randomized_sttc()
                self.save_randomized_sttc(save_dir=save_dir)
        else:
            self.compute_randomized_sttc()
            self.save_randomized_sttc(save_dir=save_dir)

        self.filter_sttc()


############################################################
# Utility Functions
############################################################

import pickle
import os
from datetime import datetime
import glob

def save_analyzers(analyzers, save_dir="outputs/analyzers"):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"analyzers_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "wb") as f:
        pickle.dump(analyzers, f)
    print(f"Analyzers saved to: {filepath}")


def load_latest_analyzers(save_dir="outputs/analyzers"):
    files = sorted(glob.glob(os.path.join(save_dir, "analyzers_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No analyzer files found in {save_dir}")

    latest_file = files[-1]
    with open(latest_file, "rb") as f:
        analyzers = pickle.load(f)
    print(f"Loaded analyzers from: {latest_file}")
    return analyzers
