# analysis_utils.py - Matrix-based analyses (correlation, PCA, etc.)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

def causal_plot(causal_info):
    figlayout = "AB"
    fig, plot = plt.subplot_mosaic(figlayout, figsize=(12, 5))
    fig.suptitle("Causal Connectivity Matrices")

    # First-order
    pltA = plot["A"].imshow(causal_info["first_order_connectivity"], cmap='Greens')
    plot["A"].set_title("First Order (10–15 ms)")
    plot["A"].set_xlabel("Reactivity Index")
    plot["A"].set_ylabel("Stimulus Index")
    fig.colorbar(pltA, ax=plot["A"], shrink=0.7)

    # Multi-order
    pltB = plot["B"].imshow(causal_info["multi_order_connectivity"], cmap='Greens')
    plot["B"].set_title("Nth Order (200 ms)")
    plot["B"].set_xlabel("Reactivity Index")
    plot["B"].set_ylabel("Stimulus Index")
    fig.colorbar(pltB, ax=plot["B"], shrink=0.7)

    plt.tight_layout()
    plt.show()

def get_correlation_matrix(spike_data, bin_size=1):
    raster = spike_data.raster(bin_size=bin_size).astype(float)
    raster = gaussian_filter1d(raster, sigma=5)
    return np.corrcoef(raster)

def compute_episode_durations(reward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute episode durations from reward log.

    Parameters:
        reward_df: DataFrame with columns ['time', 'episode', 'reward']

    Returns:
        DataFrame with columns ['episode', 'time', 'duration']
    """
    df = reward_df.sort_values("time")
    episode_groups = df.groupby("episode")["time"]
    starts = episode_groups.min()
    ends = episode_groups.max()
    durations = ends - starts

    return pd.DataFrame({
        "episode": starts.index,
        "time": starts.values,  # start time of episode
        "duration": durations.values  # time balanced
    })

def get_pole_angle_trajectories(game_df: pd.DataFrame, n_cycles=3, cycle_duration_min=15) -> list:
    """
    Extract pole angle traces for selected training cycles.

    Returns:
        List of DataFrames: each with ['time', 'pole_angle'] for one cycle
    """
    cycles = []
    total_time = game_df["time"].max()
    cycle_sec = cycle_duration_min * 60

    for i in range(n_cycles):
        start = i * cycle_sec
        end = start + cycle_sec
        cycle_df = game_df[(game_df["time"] >= start) & (game_df["time"] < end)].copy()
        cycle_df["time"] -= cycle_df["time"].min()  # normalize within cycle
        cycles.append(cycle_df[["time", "pole_angle"]])

    return cycles

def get_episode_times(reward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the start time of each episode.
    """
    episode_starts = reward_df.groupby("episode")["time"].min().reset_index()
    return episode_starts.rename(columns={"time": "start_time"})

def get_training_times(log_df: pd.DataFrame) -> pd.Series:
    """
    Extract training stimulation times from log.
    """
    return log_df["time"]

def compute_time_balanced_over_time(log_data, conditions, bin_size=60):
    """
    Computes mean ± IQR of time-balanced performance (episode duration) per time bin.

    Parameters:
        log_data: dict of log DataFrames, keyed by condition name.
        conditions: list of conditions (e.g. ["Adaptive", "Random", "Null"])
        bin_size: bin width in seconds.

    Returns:
        Dictionary: condition -> DataFrame with columns ["time_bin", "mean", "iqr"]
    """
    result = {}

    for cond in conditions:
        if cond not in log_data:
            continue

        df = log_data[cond]["reward"].copy()
        if df.empty or "reward" not in df or "time" not in df:
            continue

        # Add time bin
        df["time_bin"] = (df["time"] // bin_size).astype(int)

        # Compute mean and IQR of durations per bin
        grouped = df.groupby("time_bin")["reward"]
        summary = pd.DataFrame({
            "time_bin": grouped.mean().index * (bin_size / 60),  # convert to minutes
            "mean": grouped.mean().values,
            "iqr": grouped.quantile(0.75).values - grouped.quantile(0.25).values
        })
        result[cond] = summary

    return result
