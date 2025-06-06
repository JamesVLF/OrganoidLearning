# plots_general.py - Generic visualizations for spike data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
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