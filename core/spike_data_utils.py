# spike_data_utils.py - Utilities for SpikeData analysis
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from typing import List, Tuple
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spikedata.spikedata import SpikeData

def calculate_mean_firing_rates(spike_data, unit_ids=None):
    """
    Compute mean firing rate (Hz) for specified or all units.

    Parameters:
        spike_data: SpikeData object
        unit_ids: list of unit IDs to include (default: all)

    Returns:
        np.ndarray of firing rates (Hz), in order of unit_ids
    """
    if unit_ids is None:
        selected_units = spike_data.train
    else:
        # Map unit IDs to their index in spike_data.unit_ids
        indices = [spike_data.unit_ids.index(uid) for uid in unit_ids]
        selected_units = [spike_data.train[i] for i in indices]

    duration_s = spike_data.length / 1000.0  # ms → s

    return np.array([
        len(spikes) / duration_s for spikes in selected_units
    ])


def compute_instantaneous_firing_rate(spike_data, duration_ms=None, sigma=50):
    """Compute smoothed instantaneous firing rates from SpikeData."""
    idces, times = spike_data.idces_times()
    times = times.astype(int)

    if duration_ms is None:
        duration_ms = int(np.max(times)) + 1

    n_units = int(np.max(idces)) + 1
    rate_mat = np.zeros((duration_ms, n_units))

    for unit in range(n_units):
        spk_times = times[idces == unit]
        if len(spk_times) < 2:
            continue
        isi = np.diff(spk_times)
        isi = np.insert(isi, 0, np.nan)
        isi_rate = 1.0 / isi

        series = np.zeros(duration_ms)
        for i in range(1, len(spk_times)):
            start = spk_times[i - 1]
            end = spk_times[i]
            if end >= duration_ms:
                break
            series[start:end] = isi_rate[i]

        smoothed = 1000 * gaussian_filter1d(series, sigma=sigma)
        rate_mat[:, unit] = smoothed

    return rate_mat

def detect_population_bursts(spike_data, bin_size_ms=10, smooth_sigma=2, threshold_std=2.0, min_duration_ms=30):
    """
    Detect bursts based on smoothed population firing rate.

    Parameters:
        spike_data: SpikeData object
        bin_size_ms: bin size for population rate
        smooth_sigma: Gaussian filter sigma in bins
        threshold_std: threshold for burst detection (in std units above baseline)
        min_duration_ms: minimum duration for a valid burst

    Returns:
        List of dicts: [{"start": t_start, "end": t_end, "t_peak": peak_time}, ...]
    """
    times, rates = spike_data.population_firing_rate(bin_size=bin_size_ms, w=1, average=True)
    smoothed = ndimage.gaussian_filter1d(rates, sigma=smooth_sigma)

    baseline = np.median(smoothed)
    std_dev = np.std(smoothed)
    thresh = baseline + threshold_std * std_dev

    above_thresh = smoothed > thresh
    labeled, num = ndimage.label(above_thresh)

    bursts = []
    for i in range(1, num + 1):
        mask = labeled == i
        if mask.sum() == 0:
            continue
        start_idx, end_idx = np.where(mask)[0][[0, -1]]
        start_time = times[start_idx]
        end_time = times[end_idx]
        duration = end_time - start_time
        if duration < min_duration_ms:
            continue
        peak_idx = start_idx + np.argmax(smoothed[start_idx:end_idx + 1])
        bursts.append({
            "start": start_time,
            "end": end_time,
            "t_peak": times[peak_idx]
        })

    return bursts

def extract_peak_times(spike_data, bursts: List[dict], unit_ids: List[int]) -> np.ndarray:
    """
    Extract peak times (1/ISI) for each unit across a list of burst windows.

    Parameters:
        spike_data: SpikeData instance
        bursts: list of burst dicts with 'start' and 'end' or 't_peak'
        unit_ids: neuron indices to include

    Returns:
        peak_time_matrix: shape [num_units, num_bursts]
    """
    peak_time_matrix = []

    for unit in unit_ids:
        peak_times = []
        for burst in bursts:
            spikes = spike_data.train[unit]
            start, end = burst.get("start", 0), burst.get("end", spike_data.length)
            spks = spikes[(spikes >= start) & (spikes <= end)]
            if len(spks) < 2:
                peak_times.append(np.nan)
                continue
            isi = np.diff(spks)
            rate = 1 / isi
            peak_idx = np.argmax(rate)
            peak_time = spks[peak_idx]
            peak_times.append(peak_time)
        peak_time_matrix.append(peak_times)

    return np.array(peak_time_matrix)


def compute_rank_corr_and_zscores(peak_time_matrix: np.ndarray, num_shuffles: int = 100):
    """
    Compute Spearman rank-order correlation and z-scores across bursts.

    Parameters:
        peak_time_matrix: ndarray of shape [units, bursts], can contain NaNs
        num_shuffles: number of shuffles to build null distribution

    Returns:
        rho_matrix: Spearman correlation matrix [bursts x bursts]
        zscore_matrix: Z-scored matrix based on null distribution
    """
    # Original Spearman correlation across bursts (axis=0 = across units)
    rho_matrix, _ = spearmanr(peak_time_matrix, axis=0, nan_policy='omit')

    shuffled_rhos = []
    for _ in range(num_shuffles):
        shuffled = np.empty_like(peak_time_matrix)
        for i in range(peak_time_matrix.shape[0]):
            row = peak_time_matrix[i]
            valid = ~np.isnan(row)
            shuffled_row = row.copy()
            shuffled_row[valid] = np.random.permutation(row[valid])
            shuffled[i] = shuffled_row

        shuff_rho, _ = spearmanr(shuffled, axis=0, nan_policy='omit')
        shuffled_rhos.append(shuff_rho)

    shuffled_rhos = np.array(shuffled_rhos)
    mean_shuff = np.nanmean(shuffled_rhos, axis=0)
    std_shuff = np.nanstd(shuffled_rhos, axis=0)
    zscore_matrix = (rho_matrix - mean_shuff) / std_shuff

    return rho_matrix, zscore_matrix

def extract_aligned_spikes(spike_data, burst_windows, window_ms=1000):
    """
    Return a list of spike times for each neuron, aligned to burst onset (t=0).
    Only spikes within ±window_ms around each burst onset are included.
    """
    aligned = [[] for _ in range(spike_data.N)]

    for b in burst_windows:
        start = b["start"] if isinstance(b, dict) else b[0]
        t0 = start  # burst onset in ms
        win_start, win_end = t0 - window_ms // 2, t0 + window_ms // 2
        sd_slice = spike_data.subtime(win_start, win_end)
        for i, spikes in enumerate(sd_slice.train):
            aligned[i].extend(spikes - (t0 - win_start))

    return aligned  # List of N neurons, each with list of relative times

def compute_normalized_firing_rate_matrix(aligned_spikes, duration_ms=1000, bin_size=10):
    """
    Returns: matrix (neurons x time_bins), each value normalized to [0, 1] per unit
    """
    num_neurons = len(aligned_spikes)
    num_bins = duration_ms // bin_size
    fr_matrix = np.zeros((num_neurons, num_bins))

    for i, spikes in enumerate(aligned_spikes):
        counts, _ = np.histogram(spikes, bins=num_bins, range=(0, duration_ms))
        smoothed = ndimage.gaussian_filter1d(counts, sigma=1)
        norm = smoothed / smoothed.max() if smoothed.max() > 0 else smoothed
        fr_matrix[i] = norm

    return fr_matrix

def plot_backbone_aligned_heatmaps(aligned_spikes, fr_matrix, window_ms=1000, bin_size=10, burst_onset_ms=500):
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [1, 1.1]})

    # LEFT: Raster Plot
    axs[0].set_title("Aligned Spikes")
    for i, spikes in enumerate(aligned_spikes):
        axs[0].vlines(spikes, i + 0.5, i + 1.5, color='white', linewidth=0.3)
    axs[0].axvline(burst_onset_ms, color='lime', linestyle='--', linewidth=1.2)
    axs[0].set_xlim(0, window_ms)
    axs[0].set_ylim(0.5, len(aligned_spikes) + 0.5)
    axs[0].set_ylabel("Neuron ID")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_facecolor('black')

    # RIGHT: Firing rate heatmap
    axs[1].imshow(fr_matrix, aspect='auto', cmap='hot', interpolation='nearest', extent=[0, window_ms, 0, fr_matrix.shape[0]])
    axs[1].axvline(burst_onset_ms, color='lime', linestyle='--', linewidth=1.2)
    axs[1].set_title("Norm. Firing Rate")
    axs[1].set_xlabel("Time (ms)")
    axs[1].tick_params(left=False)

    # SCALE BAR
    scale_len = 500
    scale_y = -5
    axs[1].add_patch(patches.Rectangle((window_ms - scale_len - 20, scale_y), scale_len, 2, color='black'))
    axs[1].text(window_ms - scale_len // 2 - 20, scale_y - 5, "500 ms", ha='center')

    plt.tight_layout()
    plt.show()

def analyze_burst_distributions(spike_data, condition_label, burst_func, **kwargs):
    bursts = burst_func(spike_data, **kwargs)  # use extract_population_bursts or burst_detection
    durations = [b["end"] - b["start"] for b in bursts]
    freq = len(bursts) / (spike_data.length / 1000)  # bursts per second

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(durations, bins=30, color='gray')
    plt.xlabel('Burst Duration (ms)')
    plt.ylabel('Count')
    plt.title(f'{condition_label}: Burst Durations')

    plt.subplot(1, 2, 2)
    plt.bar([condition_label], [freq])
    plt.ylabel('Burst Frequency (Hz)')
    plt.title(f'{condition_label}: Burst Frequency')
    plt.tight_layout()
    plt.show()

    return {
        "durations_ms": durations,
        "burst_rate_Hz": freq,
        "burst_count": len(bursts)
    }

def analyze_within_burst_firing(spike_data, bursts, bin_size=20):
    rates = []
    for b in bursts:
        start, end = b["start"], b["end"]
        sub_sd = spike_data.subtime(start, end)
        raster = sub_sd.raster(bin_size)
        rates.append(raster.sum(axis=0))  # total population rate

    rates = np.array(rates)
    avg_rate = rates.mean(axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(avg_rate.size) * bin_size, avg_rate)
    plt.xlabel('Time in Burst (ms)')
    plt.ylabel('Mean Population Firing Rate')
    plt.title('Average Within-Burst Dynamics')
    plt.tight_layout()
    plt.show()

    return avg_rate

def analyze_latency_consistency(spike_data, bursts):
    all_latencies = []
    for b in bursts:
        start = b["start"]
        latencies = [
            np.min(train[train >= start] - start) if np.any(train >= start) else np.nan
            for train in spike_data.train
        ]
        all_latencies.append(latencies)

    latencies = np.array(all_latencies)
    std_dev = np.nanstd(latencies, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(std_dev)
    plt.xlabel('Neuron Index')
    plt.ylabel('Latency Std Dev (ms)')
    plt.title('Burst Latency Consistency')
    plt.tight_layout()
    plt.show()

    return std_dev

def analyze_burst_propagation(spike_data, bursts):
    coms = []
    for b in bursts:
        start, end = b["start"], b["end"]
        window = spike_data.subtime(start, end)
        latencies = [np.min(t[t >= 0]) if len(t) > 0 else np.nan for t in window.train]
        coms.append(latencies)

    coms = np.array(coms)
    com_mean = np.nanmean(coms, axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(com_mean)
    plt.xlabel('Neuron Index')
    plt.ylabel('Mean First Spike Time (ms)')
    plt.title('Burst Propagation Profile')
    plt.tight_layout()
    plt.show()

    return com_mean

def compute_latency_histograms(spike_data, window_ms=200, bin_size=5, unit_ids=None):
    """
    Compute latency histograms for specified unit (neuron) pairs.

    Parameters:
        spike_data: SpikeData object
        window_ms: int – time window in milliseconds
        bin_size: int – width of histogram bins in ms
        unit_ids: list of unit indices to compare (default: all)

    Returns:
        histograms: dict[(i, j)] → histogram array
        bin_edges: np.ndarray
    """
    N = spike_data.N
    all_units = list(range(N))

    if unit_ids is None:
        unit_ids = all_units
    else:
        unit_ids = sorted(set(unit_ids))  # Remove duplicates, sort

    histograms = {}
    bin_edges = np.arange(-window_ms, window_ms + bin_size, bin_size)

    for i in unit_ids:
        latencies_list = spike_data.latencies_to_index(i, window_ms=window_ms)

        # Only compare i to j if both are in unit_ids
        for j in unit_ids:
            if i == j or j >= len(latencies_list):
                continue

            latencies = latencies_list[j]
            hist = (
                np.histogram(latencies, bins=bin_edges)[0]
                if len(latencies) > 0
                else np.zeros(len(bin_edges) - 1)
            )

            histograms[(i, j)] = hist

    return histograms, bin_edges

def infer_causal_matrices(spike_data, max_latency_ms=200, bin_size=5, unit_ids=None):
    """
    Construct directional connection matrices from pairwise latency histograms.
    Categorized by weighted mean latency.

    Parameters:
        spike_data: SpikeData object
        max_latency_ms: int, latency window size (± max_latency_ms)
        bin_size: int, bin size in ms
        unit_ids: optional list of unit indices (default: all)

    Returns:
        first_order: NxN matrix of direct causal links (within ±15 ms)
        multi_order: NxN matrix of mean latencies (within ±max_latency_ms)
    """
    latency_histograms, bin_edges = compute_latency_histograms(
        spike_data,
        window_ms=max_latency_ms,
        bin_size=bin_size,
        unit_ids=unit_ids
    )

    lag_ms = (bin_edges[:-1] + bin_edges[1:]) / 2
    N = spike_data.N
    first_order = np.zeros((N, N))
    multi_order = np.zeros((N, N))

    for (i, j), hist in latency_histograms.items():
        total = hist.sum()
        if total == 0:
            continue

        probs = hist / total
        mean_latency = np.sum(lag_ms * probs)

        # Multi-order: always record
        multi_order[i, j] = mean_latency

        # First-order: only if latency is within tight causal band
        if -15 <= mean_latency <= 15:
            first_order[i, j] = mean_latency

    return first_order, multi_order

def infer_causal_matrices_counts(spike_data, latency_window_first=(-15, 15), latency_window_multi=(-200, 200)):
    """
    Construct directional connection matrices based on raw latency counts (baseline-consistent).

    Parameters:
        spike_data: SpikeData object
        latency_window_first: Tuple[int, int], window for first-order (e.g. ±15 ms)
        latency_window_multi: Tuple[int, int], window for multi-order (e.g. ±200 ms)

    Returns:
        first_order: NxN matrix (latency counts in ±15 ms)
        multi_order: NxN matrix (latency counts in ±200 ms)
    """
    N = spike_data.N
    first_order = np.zeros((N, N), dtype=int)
    multi_order = np.zeros((N, N), dtype=int)

    for i in range(N):
        latencies_to_i = spike_data.latencies_to_index(i)
        for j in range(N):
            lats = latencies_to_i[j]
            if not lats:
                continue

            multi_hits = [lat for lat in lats if latency_window_multi[0] < lat < latency_window_multi[1]]
            first_hits = [lat for lat in lats if latency_window_first[0] < lat < latency_window_first[1]]

            multi_order[i, j] = len(multi_hits)
            first_order[i, j] = len(first_hits)

    return first_order, multi_order

def compute_connection_strength_matrix(latency_histograms, bin_edges, latency_range=(5, 200)):
    N = max(max(i, j) for i, j in latency_histograms.keys()) + 1
    matrix = np.zeros((N, N))
    for (i, j), hist in latency_histograms.items():
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = (centers >= latency_range[0]) & (centers <= latency_range[1])
        matrix[i, j] = np.sum(hist[mask])
    return matrix

def get_sttc(sd, neuron_ids):
    """
    Returns STTC matrix for selected neuron IDs.

    Parameters:
        sd : SpikeData
        neuron_ids : list of int
    Returns:
        NxN numpy array of STTC values for selected neurons
    """
    if not neuron_ids:
        raise ValueError("No neuron IDs provided.")

    subset_sd = sd.subset(neuron_ids)
    sttc = subset_sd.spike_time_tilings()
    return np.nan_to_num(sttc)


def plot_sttc_matrix(sd, neuron_ids, title="STTC Matrix"):
    sttc = get_sttc(sd, neuron_ids)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sttc, vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Neuron Index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_sttc_spatial(sd, neuron_ids, coords, threshold=0.1, title="STTC Spatial Connectivity"):
    sttc = get_sttc(sd, neuron_ids)
    coords = np.array(coords)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    print("Max STTC:", np.max(sttc))  # See overall range of values

    # Electrode layout with neuron connections
    axs[0].scatter(coords[:, 0], coords[:, 1], c='red', label='Task Neurons')
    for i in range(len(neuron_ids)):
        for j in range(i + 1, len(neuron_ids)):
            strength = sttc[i, j]
            print(f"STTC[{i}, {j}] = {strength:.2f}")

            if strength >= threshold - 1e-6:
                xi, yi = coords[i]
                xj, yj = coords[j]
                linewidth = 1.0 + 6.0 * strength
                axs[0].plot([xi, xj], [yi, yj], linewidth=linewidth, color='black', alpha=0.7)

    axs[0].set_title("Spatial Connectivity (STTC ≥ {:.2f})".format(threshold))
    axs[0].set_xlabel("X (µm)")
    axs[0].set_ylabel("Y (µm)")
    axs[0].set_aspect("equal")
    axs[0].legend()

    # STTC heatmap
    im = axs[1].imshow(sttc, vmin=0, vmax=1, cmap="viridis")
    axs[1].set_title("STTC Matrix")
    axs[1].set_xlabel("Neuron Index")
    axs[1].set_ylabel("Neuron Index")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()