# plots_general.py - Generic visualizations for spike data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict
import networkx as nx

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

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=500,
            edge_color=weights, edge_cmap=plt.cm.viridis, width=2)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(weights)
    plt.colorbar(sm, label="Latency (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
