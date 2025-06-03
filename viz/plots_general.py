# plots_general.py - Generic visualizations for spike data
import matplotlib.pyplot as plt

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