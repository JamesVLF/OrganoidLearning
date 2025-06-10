# plots_general.py - Generic visualizations for spike data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict
import networkx as nx
from spikedata.spikedata import SpikeData

def plot_raster(spike_data, title="Spike Raster", start=0, end=None):
    idces, times = spike_data.idces_times()
    if end is None:
        end = spike_data.length / 1000
    mask = (times >= start * 1000) & (times <= end * 1000)
    times, idces = times[mask], idces[mask]

    plt.figure(figsize=(15, 6))
    plt.title(title)
    plt.scatter(times / 1000, idces, marker='|', s=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Unit #")
    plt.show()

def plot_firing_rates(rates, title="Mean Firing Rates"):
    plt.figure(figsize=(8, 4))
    plt.hist(rates, bins=30)
    plt.title(title)
    plt.xlabel("Firing Rate (Hz)")
    plt.ylabel("Number of Neurons")
    plt.show()

def plot_correlation_matrix(matrix, title="Correlation Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.title(title)
    plt.colorbar(label="Correlation")
    plt.show()

def plot_smoothed_reward(reward_df, label="Condition", color="darkgreen", window=50):
    """
    Plot smoothed reward over time for a given condition.

    Parameters:
        reward_df (pd.DataFrame): Reward DataFrame with 'time' and 'reward' columns.
        label (str): Label for the legend (e.g. "Adaptive").
        color (str): Line color.
        window (int): Rolling window size for smoothing.
    """
    df = reward_df.copy()
    df["reward_smooth"] = df["reward"].rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["reward_smooth"], label=f"{label} Smoothed Reward", color=color)
    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed Reward")
    plt.title(f"Smoothed Reward Over Time ({label})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_rewards_overlay(log_data, window=50):
    """
    Overlay smoothed reward plots for all conditions in one figure.

    Parameters:
        log_data (dict): Dictionary of log datasets keyed by condition name.
        window (int): Rolling window size for smoothing.
    """
    plt.figure(figsize=(10, 5))

    for condition, logs in log_data.items():
        if "reward" in logs:
            df = logs["reward"]
            smoothed = df["reward"].rolling(window=window, min_periods=1).mean()
            plt.plot(df["time"], smoothed, label=condition)

    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed Reward")
    plt.title("Smoothed Reward Over Time (All Conditions)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_reward_vs_causal_metrics(causal_info, pattern_log, return_df=False):
    """Plot average reward vs. causal metrics and burst %."""
    first_order = causal_info["first_order_connectivity"]
    multi_order = causal_info["multi_order_connectivity"]
    burst_percent = causal_info["burst_percent"]

    pattern_log = pattern_log.copy()
    pattern_log["stim_key"] = pattern_log["stim_indices"].apply(tuple)
    pattern_rewards = pattern_log.groupby("stim_key")["reward"].mean()

    rows = []
    for stim_key, avg_reward in pattern_rewards.items():
        if len(stim_key) != 2:
            continue
        stim_i, stim_j = stim_key
        rows.append({
            "Pattern": f"{stim_i}-{stim_j}",
            "AvgReward": avg_reward,
            "FirstOrder": first_order[stim_i, stim_j],
            "MultiOrder": multi_order[stim_i, stim_j],
            "Burst": burst_percent[stim_j]
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    plt.scatter(df["FirstOrder"], df["AvgReward"], label="First-Order Causal")
    plt.scatter(df["MultiOrder"], df["AvgReward"], label="Multi-Order Causal")
    plt.scatter(df["Burst"], df["AvgReward"], label="Output Burst %")
    plt.xlabel("Metric Value")
    plt.ylabel("Average Reward")
    plt.title("Reward vs. Causal Metrics (Per Pattern)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if return_df:
        return df

def plot_training_pair_metrics(first_order, multi_order, burst_percent, num_neurons=10):
    """Plot causal metrics for all pairs of training neurons."""
    training_pairs = [(i, j) for i in range(num_neurons) for j in range(num_neurons) if i != j]

    first_vals = [first_order[i, j] for i, j in training_pairs]
    multi_vals = [multi_order[i, j] for i, j in training_pairs]
    burst_vals = [burst_percent[j] for _, j in training_pairs]

    plt.figure(figsize=(10, 5))
    bar_width = 0.25
    x = range(len(training_pairs))

    plt.bar([xi - bar_width for xi in x], first_vals, width=bar_width, label='First Order')
    plt.bar(x, multi_vals, width=bar_width, label='Multi Order')
    plt.bar([xi + bar_width for xi in x], burst_vals, width=bar_width, label='Burst %')

    plt.xlabel('Training Neuron Pairs')
    plt.ylabel('Value')
    plt.title('Causal Connectivity and Burst Percent Across Training Neuron Pairs')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_connectivity_heatmaps(first_order, multi_order):
    """Plot heatmaps of first-order and multi-order connectivity."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(first_order, annot=True, fmt=".2f", cmap='viridis')
    plt.title("First Order Connectivity")
    plt.xlabel("Target Neuron (j)")
    plt.ylabel("Source Neuron (i)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(multi_order, annot=True, fmt=".2f", cmap='plasma')
    plt.title("Multi Order Connectivity")
    plt.xlabel("Target Neuron (j)")
    plt.ylabel("Source Neuron (i)")
    plt.tight_layout()
    plt.show()

def plot_reward_vs_metrics(df):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(df["FirstOrder"], df["AvgReward"])
    axs[0].set_xlabel("First-Order Causal")
    axs[0].set_ylabel("Avg Reward")
    axs[0].set_title("First-Order vs. Reward")

    axs[1].scatter(df["MultiOrder"], df["AvgReward"])
    axs[1].set_xlabel("Multi-Order Causal")
    axs[1].set_ylabel("Avg Reward")
    axs[1].set_title("Multi-Order vs. Reward")

    axs[2].scatter(df["Burst"], df["AvgReward"])
    axs[2].set_xlabel("Output Burst %")
    axs[2].set_ylabel("Avg Reward")
    axs[2].set_title("Burst vs. Reward")

    plt.tight_layout()
    plt.show()

def plot_top_vs_bottom_quartiles(df):
    df_sorted = df.sort_values("AvgReward")
    n = len(df_sorted) // 4

    bottom_q = df_sorted.iloc[:n]
    top_q = df_sorted.iloc[-n:]

    metrics = ["FirstOrder", "MultiOrder", "Burst"]
    top_means = [top_q[m].mean() for m in metrics]
    bottom_means = [bottom_q[m].mean() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, bottom_means, width, label="Bottom 25%")
    plt.bar(x + width/2, top_means, width, label="Top 25%")

    plt.xticks(x, ["First-Order", "Multi-Order", "Burst"])
    plt.ylabel("Mean Metric Value")
    plt.title("Top vs. Bottom Quartile Connectivity Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metric_correlations(df):
    metrics = ["FirstOrder", "MultiOrder", "Burst"]
    correlations = [pearsonr(df[m], df["AvgReward"])[0] for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.bar(metrics, correlations)
    plt.ylim(-1, 1)
    plt.ylabel("Pearson r")
    plt.title("Correlation of Metrics with Avg Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_performance_summary(summary_dict):
    """
    Plot mean ± IQR of time balanced performance over time for each condition.

    Parameters:
        summary_dict: dict of condition -> DataFrame with ['time_bin', 'mean', 'iqr']
    """
    plt.figure(figsize=(10, 5))

    for cond, df in summary_dict.items():
        if df.empty:
            continue
        plt.plot(df["time_bin"], df["mean"], label=cond)
        plt.fill_between(df["time_bin"], df["mean"] - df["iqr"]/2, df["mean"] + df["iqr"]/2, alpha=0.3)

    plt.xlabel("Time (minutes)")
    plt.ylabel("Time Balanced (s)")
    plt.title("Mean ± IQR of Time Balanced per Condition")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pole_angle_trajectories(game_df, reward_df, condition_label="Condition", angle_thresh_deg=16):
    """
    Plot pole angle (in degrees) over time with training signal times overlaid
    when the pole is near the angle threshold. Reward log is used as the signal indicator.
    """

    # Copy and convert pole angle from radians to degrees
    df = game_df.copy()
    df["pole_angle_deg"] = np.rad2deg(df["theta"])
    df["time_min"] = df["time"] / 60

    # Convert reward log time to minutes
    reward_df = reward_df.copy()
    reward_df["time_min"] = reward_df["time"] / 60

    # Interpolate angle at reward (training signal) times
    interp_angles = np.interp(reward_df["time_min"], df["time_min"], df["pole_angle_deg"])

    # Only keep training signals near threshold (e.g. within 1.5° buffer)
    near_thresh_mask = np.abs(interp_angles) >= (angle_thresh_deg - 1.5)
    signal_times = reward_df["time_min"][near_thresh_mask]
    signal_angles = interp_angles[near_thresh_mask]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["time_min"], df["pole_angle_deg"], color='black', lw=1, label="Pole Angle")

    # Horizontal threshold lines
    ax.axhline(y=angle_thresh_deg, linestyle="--", color="gray")
    ax.axhline(y=-angle_thresh_deg, linestyle="--", color="gray")

    # Overlay filtered training signals
    ax.scatter(signal_times, signal_angles, color="orange", marker="x", alpha=0.8, label="Training")

    ax.set_title(f"Pole Angle Over Time - {condition_label}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Pole Angle (°)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()



def plot_time_balanced_with_training(
        reward_df: pd.DataFrame,
        pattern_df: pd.DataFrame,
        label: str,
        color: str = "green",
        window: int = 5,
        max_time_min: float = 15.0
):
    """
    Plot time-balanced performance with overlaid training signal events.

    Parameters:
        reward_df: DataFrame with 'time' and 'reward' columns (reward = episode duration)
        pattern_df: DataFrame with 'time' column (training signal times)
        label: Label for this condition
        color: Color for plotting (green/red/blue etc.)
        window: rolling average smoothing window (episodes)
        max_time_min: clip x-axis to this duration (in minutes)
    """
    time = reward_df["time"] / 60  # in minutes
    duration = reward_df["reward"]
    signal_times = pattern_df["time"] / 60

    # Truncate
    mask = time <= max_time_min
    time = time[mask]
    duration = duration[mask]

    signal_times = signal_times[signal_times <= max_time_min]

    # Rolling smoothing
    smooth = duration.rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time, duration, "o", markersize=3, color=color, alpha=0.5, label="Raw")
    ax.plot(time, smooth, "-", color=color, linewidth=2, label="Smoothed")

    ax.scatter(signal_times, [0]*len(signal_times), marker="x", c="orange", label="Training Signal", alpha=0.7)

    ax.set_ylim(bottom=0)
    ax.set_xlim(0, max_time_min)
    ax.set_xlabel("Training time (m)")
    ax.set_ylabel("Time balanced (s)")
    ax.set_title(label)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_raster_pretty(sd, title="Spike Raster", l1=-10, l2=False, xsize=20, ysize=6, analyze=False):
    """
    Plots a configuable raster plot of the spike data.
        sd : spike data object from braingeneers
        title : Title of the plot
        l1 : start time in seconds
        l2 : end time in seconds
        xsize : width of the plot
        ysize : height of the plot
        analyze : If True, will plot the population rate as well
    """

    if l2==False:
        l2 = sd.length / 1000 + 10

    idces, times = sd.idces_times()

    if analyze == True:
        # Get population rate for everything
        pop_rate = sd.binned(bin_size=1)  # in ms
        # Lets smooth this to make it neater
        sigma = 5
        pop_rate_smooth = gaussian_filter1d(pop_rate.astype(float), sigma=sigma)
        t = np.linspace(0, sd.length, pop_rate.shape[0]) / 1000

        # Determine the stop_time if it's not provided
        if l2 is None:
            l2 = t[-1]

        # Filter times and idces within the specified start and stop times
        mask = (times >= l1 * 1000) & (times <= l2 * 1000)
        times = times[mask]
        idces = idces[mask]

    fig, ax = plt.subplots(figsize=(xsize, ysize))
    fig.suptitle(title)
    ax.scatter(times/1000,idces,marker='|',s=1)

    if analyze == True:
        ax2 = ax.twinx()
        ax2.plot(t, pop_rate_smooth, c='r', alpha=0.6)
        ax2.set_ylabel('Firing Rate')

    ax.set_xlabel("Time(s)")
    ax.set_ylabel('Unit #')
    plt.xlim(l1, l2)
    plt.show()

def plot_neuron_raster_comparison_stacked(spike_datasets, neuron_id, start_s=0, end_s=20):
    """
    Plot spike rasters for a specific neuron across multiple conditions,
    with stacked rows for clarity (rather than overlaying them).

    Parameters:
        spike_datasets: list of (label, SpikeData) tuples
        neuron_id: int, neuron index
        start_s: float, start time in seconds
        end_s: float, end time in seconds
    """
    y_positions = {
        label: i + 1 for i, (label, _) in enumerate(spike_datasets)
    }

    plt.figure(figsize=(10, 5))

    for label, sd in spike_datasets:
        idces, times = sd.idces_times()
        times = times / 1000  # ms → s

        mask = (idces == neuron_id) & (times >= start_s) & (times <= end_s)
        spike_times = times[mask]
        y_center = y_positions[label]

        plt.vlines(spike_times, y_center - 0.3, y_center + 0.3, label=label)

    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.xlabel("Time (s)")
    plt.ylabel("Condition")
    plt.title(f"Neuron {neuron_id} Raster (Across Conditions)")
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rank_order_matrix(rho_matrix, title="Rank Order Correlation"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(rho_matrix, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel("Burst Index")
    plt.ylabel("Burst Index")
    plt.tight_layout()
    plt.show()

def plot_zscore_matrix(zscore_matrix, title="Z-scored Rank Correlation"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(zscore_matrix, cmap="vlag", center=0)
    plt.title(title)
    plt.xlabel("Burst Index")
    plt.ylabel("Burst Index")
    plt.tight_layout()
    plt.show()

def plot_rank_order_violin(data_dict: Dict[str, np.ndarray], title: str = "Rank Order Correlation Z-Scores"):
    """
    Plot violin plots from z-scored Spearman correlation matrices (upper triangle only).
    """
    rows = []
    for label, matrix in data_dict.items():
        triu = np.triu_indices_from(matrix, k=1)
        z_values = matrix[triu]
        for val in z_values:
            rows.append({"Condition": label, "ZScore": val})

    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="Condition", y="ZScore", inner="box")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(title)
    plt.ylabel("Z-scored Spearman ρ")
    plt.tight_layout()
    plt.show()

def plot_time_balanced_cumulative(
        log_data: dict,
        conditions: list[str] = None,
        window: int = 5,
        max_minutes: float = None,
        show_raw: bool = True,
):
    """
    Plot episode durations vs. cumulative training time across conditions.

    Parameters:
        log_data: dict of log data per condition
        conditions: list of condition names to include
        window: smoothing window in episodes
        max_minutes: optional cap for x-axis
        show_raw: whether to show raw points
    """

    if conditions is None:
        conditions = ["Adaptive", "Random", "Null"]

    color_map = {
        "Adaptive": "green",
        "Random": "red",
        "Null": "blue"
    }

    plt.figure(figsize=(12, 4))

    for cond in conditions:
        if cond not in log_data:
            print(f"Skipping missing condition: {cond}")
            continue
        if "reward" not in log_data[cond]:
            print(f"Missing reward data for {cond}")
            continue

        df = log_data[cond]["reward"].copy()
        df = df.sort_values("time").reset_index(drop=True)

        # Use time as x (in minutes)
        df["time_min"] = df["time"] / 60
        df["smooth"] = df["reward"].rolling(window=window, min_periods=1).mean()

        color = color_map.get(cond, None)

        if show_raw:
            plt.plot(df["time_min"], df["reward"], ".", markersize=3, alpha=0.3, color=color, label=f"{cond} Raw")

        plt.plot(df["time_min"], df["smooth"], linewidth=2, color=color, label=f"{cond} Smoothed")

    plt.xlabel("Cumulative training time (min)")
    plt.ylabel("Time balanced (s)")
    plt.title("Time Balanced vs. Cumulative Training Time")
    plt.legend()
    if max_minutes:
        plt.xlim(0, max_minutes)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

def plot_latency_histogram_pair(latency_histograms, bin_edges, i, j, condition_label=None):
    """
    Plot latency histogram for a neuron pair (i → j).

    Parameters:
        latency_histograms: dict of {(i, j): histogram}
        bin_edges: numpy array of bin edges
        i: source neuron index
        j: target neuron index
        condition_label: optional string for labeling the plot
    """
    key = (i, j)
    if key not in latency_histograms:
        raise ValueError(f"No latency data for neuron pair ({i} → {j})")

    hist = latency_histograms[key]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist, width=bin_edges[1] - bin_edges[0], color='gray', edgecolor='black')
    title = f"Latency Histogram: Neuron {i} → {j}"
    if condition_label:
        title = f"{title} ({condition_label})"
    plt.title(title)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_top_firing_orders(order_counts):
    """
    Bar plot of the top firing sequences by frequency.
    """

    labels = [' → '.join(map(str, seq)) for seq, _ in order_counts]
    counts = [count for _, count in order_counts]

    plt.figure(figsize=(10, 4))
    plt.barh(labels, counts, color='darkgreen')
    plt.xlabel("Frequency")
    plt.title("Top Firing Sequences")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def causal_plot_from_matrices(first_order, multi_order, title="", training_inds=None):
    """
    Plot causal connectivity heatmaps from precomputed matrices (restricted to training neurons).

    Parameters:
        first_order: np.ndarray, shape (N, N)
        multi_order: np.ndarray, shape (N, N)
        title: str
        training_inds: list of neuron indices (e.g., ole.metadata["training_inds"])
    """
    if training_inds is not None:
        first_order = first_order[np.ix_(training_inds, training_inds)]
        multi_order = multi_order[np.ix_(training_inds, training_inds)]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    im1 = axs[0].imshow(first_order, cmap="Greens")
    axs[0].set_title("First Order (±15 ms)")
    axs[0].set_xlabel("Target Neuron")
    axs[0].set_ylabel("Source Neuron")
    fig.colorbar(im1, ax=axs[0], shrink=0.8)

    im2 = axs[1].imshow(multi_order, cmap="Greens")
    axs[1].set_title("Multi-Order (±200 ms)")
    axs[1].set_xlabel("Target Neuron")
    axs[1].set_ylabel("Source Neuron")
    fig.colorbar(im2, ax=axs[1], shrink=0.8)

    plt.tight_layout()
    plt.show()

def plot_training_spatial_connectivity_all_conditions(ole, spikes_x, spikes_y, electrode_mapping):
    """
    Plot spatial connectivity of training neurons using causal matrices (0–5 min window) for all conditions.

    Parameters:
        ole: OrgLearningEval instance
        spikes_x, spikes_y: arrays of x/y spike positions (indexed by neuron)
        electrode_mapping: DataFrame with full electrode layout (x, y columns)
    """
    training_inds = ole.metadata["training_inds"]

    for cond in ["Baseline", "Adaptive", "Random", "Null"]:
        matrix = ole.causal_latency_matrices.get((cond, 0, 300000))
        if matrix is None:
            print(f"[!] Skipping {cond}: No matrix found.")
            continue

        plt.figure(figsize=(10, 6))
        plt.title(f"Spatial Connectivity of Training Neurons ({cond} 0–5 min)")

        # Background layout
        plt.scatter(electrode_mapping.x.values, electrode_mapping.y.values, s=5, alpha=0.2)

        # Training neurons
        for i in training_inds:
            plt.scatter(spikes_x[i], spikes_y[i], c='r', s=70)
            plt.text(spikes_x[i] + 3, spikes_y[i], str(i), fontsize=9)

        # Arrows for causal links
        for i in training_inds:
            for j in training_inds:
                if i == j:
                    continue
                latency = matrix[i, j]
                if latency != 0:
                    x1, y1 = spikes_x[i], spikes_y[i]
                    x2, y2 = spikes_x[j], spikes_y[j]
                    plt.annotate("",
                                 xy=(x2, y2), xytext=(x1, y1),
                                 arrowprops=dict(arrowstyle="->", color="lime", lw=2))

        plt.xlabel("μm")
        plt.ylabel("μm")
        plt.tight_layout()
        plt.show()

def plot_training_neuron_connectivity_map(matrix, neuron_ids, positions, title="", threshold=0):
    """
    Plot directional connectivity among training neurons as arrows on spatial map.

    Parameters:
        matrix: NxN causal matrix (already scoped to training neurons)
        neuron_ids: list of neuron IDs (training neurons)
        positions: dict {neuron_id: (x, y)}
        title: plot title
        threshold: minimum value in matrix to draw a connection
    """
    roles = {}
    N = len(neuron_ids)
    for idx, neuron in enumerate(neuron_ids):
        out_deg = np.sum(matrix[idx, :] > threshold)
        in_deg = np.sum(matrix[:, idx] > threshold)
        if in_deg > 0 and out_deg > 0:
            roles[neuron] = 'broker'
        elif out_deg > 0:
            roles[neuron] = 'sender'
        elif in_deg > 0:
            roles[neuron] = 'receiver'
        else:
            roles[neuron] = 'isolated'

    # Plot neurons
    plt.figure(figsize=(8, 6))
    for idx, neuron in enumerate(neuron_ids):
        x, y = positions[neuron]
        color = {
            "sender": "red",
            "receiver": "blue",
            "broker": "gray",
            "isolated": "lightgray"
        }[roles[neuron]]
        plt.scatter(x, y, c=color, s=80, edgecolor='black', zorder=3)

    # Plot arrows
    for i in range(N):
        for j in range(N):
            if matrix[i, j] > threshold:
                i_id, j_id = neuron_ids[i], neuron_ids[j]
                x0, y0 = positions[i_id]
                x1, y1 = positions[j_id]
                dx, dy = x1 - x0, y1 - y0
                plt.arrow(x0, y0, dx * 0.85, dy * 0.85,
                          head_width=10, length_includes_head=True,
                          fc='k', ec='k', alpha=0.7, zorder=2)

    plt.title(title or "Training Neuron Connectivity")
    plt.axis('equal')
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_spatial_connectivity_map(matrix, positions, title="Spatial Connectivity"):
    N = matrix.shape[0]
    G = nx.DiGraph()

    for i in range(N):
        G.add_node(i, pos=positions.get(i, (0, 0)))

    for i in range(N):
        for j in range(N):
            weight = matrix[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)

    pos = nx.get_node_attributes(G, 'pos')
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos, ax=ax, with_labels=True, arrows=True, node_size=500,
            edge_color=weights, edge_cmap=plt.cm.viridis, width=2)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(weights)
    fig.colorbar(sm, ax=ax, label="Latency (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def create_directed_graph_from_consensus(consensus_order, consensus_scores,
                                         causal_matrix, threshold=None,
                                         influence_metric='net_influence'):
    """
    Create directed graph from consensus firing order results

    Parameters:
    -----------
    consensus_order : array
        Firing order from consensus method
    consensus_scores : array
        Consensus scores for each neuron
    causal_matrix : array
        Original causal connectivity matrix
    threshold : float
        Minimum connection strength to show (None = auto-detect)
    influence_metric : str
        How to calculate node influence ('net_influence', 'total_output', 'total_input')
    """

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with firing order as attribute
    n_neurons = len(consensus_order)
    firing_rank = np.argsort(consensus_order)  # Convert order to ranks

    for i in range(n_neurons):
        G.add_node(i,
                   firing_rank=firing_rank[i],
                   consensus_score=consensus_scores[i])

    # Determine threshold for showing edges
    if threshold is None:
        # Show top 30% of connections
        threshold = np.percentile(np.abs(causal_matrix), 70)

    # Add edges based on causal matrix and firing order
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j and abs(causal_matrix[i, j]) > threshold:
                # Only add edge if it respects temporal order
                if firing_rank[i] < firing_rank[j]:  # i fires before j
                    G.add_edge(i, j,
                               weight=abs(causal_matrix[i, j]),
                               causal_strength=causal_matrix[i, j])

    # Calculate influence metrics for node sizing/coloring
    if influence_metric == 'net_influence':
        influence = np.sum(causal_matrix, axis=1) - np.sum(causal_matrix, axis=0)
    elif influence_metric == 'total_output':
        influence = np.sum(np.abs(causal_matrix), axis=1)
    elif influence_metric == 'total_input':
        influence = np.sum(np.abs(causal_matrix), axis=0)
    else:
        influence = consensus_scores

    # Add influence as node attribute
    for i in range(n_neurons):
        G.nodes[i]['influence'] = influence[i]

    return G

def visualize_firing_graph(G, layout='spring', figsize=(12, 8),
                           node_size_range=(300, 2000),
                           edge_width_range=(0.5, 5),
                           show_labels=True, save_path=None):
    """
    Visualize the directed firing graph

    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph from create_directed_graph_from_consensus
    layout : str
        Layout algorithm ('spring', 'circular', 'hierarchical')
    """

    plt.figure(figsize=figsize)

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'hierarchical':
        # Arrange by firing order
        firing_ranks = [G.nodes[node]['firing_rank'] for node in G.nodes()]
        pos = {}
        for i, node in enumerate(G.nodes()):
            rank = G.nodes[node]['firing_rank']
            pos[node] = (rank, np.random.uniform(-0.5, 0.5))  # Add some vertical jitter
    else:
        pos = nx.spring_layout(G)

    # Extract node attributes
    influences = [G.nodes[node]['influence'] for node in G.nodes()]
    firing_ranks = [G.nodes[node]['firing_rank'] for node in G.nodes()]

    # Normalize influences for sizing/coloring
    influence_norm = (np.array(influences) - np.min(influences)) / (np.max(influences) - np.min(influences) + 1e-8)

    # Node sizes based on influence
    node_sizes = node_size_range[0] + influence_norm * (node_size_range[1] - node_size_range[0])

    # Node colors based on firing order (early = cool colors, late = warm colors)
    rank_norm = np.array(firing_ranks) / (len(firing_ranks) - 1)
    node_colors = plt.cm.viridis(rank_norm)

    # Edge attributes
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges()]
    if edge_weights:
        edge_weight_norm = (np.array(edge_weights) - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights) + 1e-8)
        edge_widths = edge_width_range[0] + edge_weight_norm * (edge_width_range[1] - edge_width_range[0])
    else:
        edge_widths = [1.0]

    # Draw the graph
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.8,
                           edgecolors='black',
                           linewidths=1)

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           alpha=0.6,
                           edge_color='gray',
                           arrows=True,
                           arrowsize=20,
                           arrowstyle='->')

    if show_labels:
        # Labels showing neuron ID and firing rank
        labels = {node: f'{node}\n(#{G.nodes[node]["firing_rank"]+1})' for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    plt.title('Neural Firing Order Network\n(Node size = influence, Color = firing order, Edges = causal connections)',
              fontsize=14, pad=20)

    # Add colorbar for firing order
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=1, vmax=len(G.nodes())))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Firing Order (1=earliest)', rotation=270, labelpad=20)

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return pos

def create_multiple_graph_views(consensus_order, consensus_scores, causal_matrix,
                                first_order_matrix=None, multi_order_matrix=None):
    """
    Create multiple views of the same network
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # View 1: Standard spring layout
    G1 = create_directed_graph_from_consensus(consensus_order, consensus_scores, causal_matrix)
    plt.sca(axes[0,0])
    pos1 = nx.spring_layout(G1, k=2)
    draw_graph_subplot(G1, pos1, "Spring Layout")

    # View 2: Hierarchical layout (by firing order)
    plt.sca(axes[0,1])
    pos2 = create_hierarchical_layout(G1)
    draw_graph_subplot(G1, pos2, "Hierarchical Layout (by firing order)")

    # View 3: First-order connections only
    if first_order_matrix is not None:
        G3 = create_directed_graph_from_consensus(consensus_order, consensus_scores,
                                                  first_order_matrix, threshold=np.percentile(np.abs(first_order_matrix), 75))
        plt.sca(axes[1,0])
        draw_graph_subplot(G3, pos1, "First-Order Connections (±15ms)")

    # View 4: Multi-order connections only
    if multi_order_matrix is not None:
        G4 = create_directed_graph_from_consensus(consensus_order, consensus_scores,
                                                  multi_order_matrix, threshold=np.percentile(np.abs(multi_order_matrix), 75))
        plt.sca(axes[1,1])
        draw_graph_subplot(G4, pos1, "Multi-Order Connections (±200ms)")

    plt.tight_layout()
    plt.show()

def draw_graph_subplot(G, pos, title):
    """Helper function to draw graph in subplot"""

    # Node attributes
    influences = [G.nodes[node]['influence'] for node in G.nodes()]
    firing_ranks = [G.nodes[node]['firing_rank'] for node in G.nodes()]

    # Normalize for visualization
    influence_norm = (np.array(influences) - np.min(influences)) / (np.max(influences) - np.min(influences) + 1e-8)
    rank_norm = np.array(firing_ranks) / (len(firing_ranks) - 1)

    node_sizes = 300 + influence_norm * 1000
    node_colors = plt.cm.viridis(rank_norm)

    # Edge weights
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges()] if G.edges() else [1]
    if edge_weights:
        edge_weight_norm = (np.array(edge_weights) - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights) + 1e-8)
        edge_widths = 0.5 + edge_weight_norm * 3
    else:
        edge_widths = [1.0]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                           arrows=True, arrowsize=15, edge_color='gray')

    # Labels
    labels = {node: f'{node}' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

    plt.title(title, fontsize=12)
    plt.axis('off')

def create_hierarchical_layout(G):
    """Create layout arranged by firing order"""
    pos = {}
    firing_ranks = {node: G.nodes[node]['firing_rank'] for node in G.nodes()}

    # Group nodes by firing rank
    rank_groups = {}
    for node, rank in firing_ranks.items():
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(node)

    # Position nodes
    for rank, nodes in rank_groups.items():
        for i, node in enumerate(nodes):
            y_offset = (i - len(nodes)/2) * 0.5 if len(nodes) > 1 else 0
            pos[node] = (rank * 2, y_offset)

    return pos

def analyze_graph_properties(G):
    """Analyze properties of the created graph"""

    print("=== Graph Analysis ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.3f}")

    # Firing order
    firing_order = sorted(G.nodes(), key=lambda x: G.nodes[x]['firing_rank'])
    print(f"Firing order: {firing_order}")

    # Most influential nodes
    influences = [(node, G.nodes[node]['influence']) for node in G.nodes()]
    influences.sort(key=lambda x: x[1], reverse=True)
    print(f"Most influential: {influences[:3]}")

    # Network metrics
    if nx.is_directed_acyclic_graph(G):
        print("Network is acyclic (no feedback loops)")
    else:
        print("Network contains cycles (feedback loops)")

    return {
        'firing_order': firing_order,
        'influences': influences,
        'is_acyclic': nx.is_directed_acyclic_graph(G),
        'density': nx.density(G)
    }

# Example usage
def example_usage():
    """
    Example of how to use these functions with your consensus results
    """

    # Assuming you have these from your consensus analysis:
    # consensus_order, consensus_scores, all_methods = infer_firing_order_consensus(matrix)

    # Example data (replace with your actual results)
    consensus_order = np.array([0, 1, 2, 3, 4, 5])
    consensus_scores = np.array([0.8, 0.6, 0.2, -0.1, -0.3, -0.5])
    causal_matrix = np.random.rand(6, 6) - 0.5  # Replace with your actual matrix

    # Create and visualize the graph
    G = create_directed_graph_from_consensus(consensus_order, consensus_scores, causal_matrix)

    # Single view
    visualize_firing_graph(G, layout='hierarchical')

    # Multiple views
    # create_multiple_graph_views(consensus_order, consensus_scores, causal_matrix)

    # Analyze properties
    properties = analyze_graph_properties(G)

    return G, properties