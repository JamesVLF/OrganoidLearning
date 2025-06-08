
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_performance_over_cycles(df_cycles):
    """
    Plot time balanced per episode over cumulative training time,
    color-coded by paradigm.

    Parameters:
        df_cycles: DataFrame with columns ['time', 'time_balanced', 'paradigm']
    """

    color_map = {'Null': 'blue', 'Random': 'red', 'Adaptive': 'green'}

    plt.figure(figsize=(12, 4))
    for paradigm in ['Null', 'Random', 'Adaptive']:
        subset = df_cycles[df_cycles['paradigm'] == paradigm]
        plt.plot(subset['time'], subset['time_balanced'],
                 color=color_map[paradigm], label=paradigm, alpha=0.6)

    # Optional: vertical lines for 45-min rest periods
    for i in range(45, int(df_cycles['time'].max()), 60):
        plt.axvline(i, color='k', linestyle='--', alpha=0.3)

    plt.xlabel("Cumulative time training (m)")
    plt.ylabel("Time balanced (s)")
    plt.title("Time-Balanced Performance Over Cycles")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_iqr_per_paradigm(df_cycles):
    """
    Plot average + IQR of time balanced for each paradigm over time.
    """

    plt.figure(figsize=(6, 4))

    for paradigm, color in zip(['Adaptive', 'Random', 'Null'], ['green', 'red', 'blue']):
        subset = df_cycles[df_cycles['paradigm'] == paradigm]
        grouped = subset.groupby('cycle')['time_balanced']
        mean = grouped.mean()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        x = mean.index
        plt.plot(x, mean, label=paradigm, color=color)
        plt.fill_between(x, q1, q3, alpha=0.3, color=color)

    plt.xlabel("Cycle")
    plt.ylabel("Time balanced (s)")
    plt.title("Panel 4d: Mean + IQR per Paradigm")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pole_angle_trajectories(angle_data):
    """
    angle_data: dict with keys 'Adaptive', 'Random', 'Null', each containing
                a list of arrays of pole angles per episode (y = time, x = angle)
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    colors = plt.cm.cool(np.linspace(0, 1, 50))  # color gradient

    for ax, key, color in zip(axs, ['Adaptive', 'Null', 'Random'], ['green', 'blue', 'red']):
        episodes = angle_data[key]
        for i, ep in enumerate(episodes[:50]):
            t = np.linspace(0, len(ep), len(ep))
            ax.plot(ep, t, color=colors[i])
        ax.set_title(f"{key}")
        ax.set_xlabel("Pole Angle")
        ax.set_xlim(-0.3, 0.3)
    axs[0].set_ylabel("Time (s)")
    plt.suptitle("Panel 4e: Pole Angle Trajectories")
    plt.tight_layout()
    plt.show()

def plot_signal_and_performance(df_signals, paradigm):
    """
    Plot time-balanced trace (raw + smoothed) and overlay training signal times.

    df_signals: DataFrame with columns ['time', 'time_balanced', 'smoothed', 'signal']
    paradigm: str, used for labeling
    """
    plt.figure(figsize=(6, 3))
    plt.plot(df_signals['time'], df_signals['time_balanced'], alpha=0.3, label='Raw')
    plt.plot(df_signals['time'], df_signals['smoothed'], lw=2, label='Smoothed')
    plt.scatter(df_signals['time'], df_signals['signal'], color='orange', label='Training Signal', s=30)
    plt.title(f"Panel 4f: {paradigm}")
    plt.xlabel("Training time (m)")
    plt.ylabel("Time balanced (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def build_df_cycles(log_data, cycle_duration=15):
    """
    Build a DataFrame for performance across all paradigms.

    Returns:
        DataFrame with columns: ['time', 'time_balanced', 'paradigm', 'cycle']
    """
    rows = []
    current_time = 0
    cycle = 0

    # Repeat pattern for all paradigms
    for paradigm in ['Null', 'Random', 'Adaptive'] * 7:  # 21 cycles
        log = log_data[paradigm]['log']
        for entry in log:
            if 'time_balanced' in entry:
                rows.append({
                    'time': current_time,
                    'time_balanced': entry['time_balanced'],
                    'paradigm': paradigm,
                    'cycle': cycle
                })
                current_time += entry.get('duration', 0.25)  # assume ~15s episodes unless defined
        cycle += 1
        current_time += 45  # insert rest gap

    return pd.DataFrame(rows)

def build_df_signals(log_dict, window=5):
    """
    Returns a DataFrame with 'time_balanced', 'smoothed', 'signal', 'time'.
    """
    tb = [ep['time_balanced'] for ep in log_dict['log']]
    signals = [1 if ep.get('train_signal') else 0 for ep in log_dict['log']]
    times = np.arange(len(tb)) * 0.25  # 15 min = ~60 episodes

    df = pd.DataFrame({
        'time_balanced': tb,
        'smoothed': pd.Series(tb).rolling(window=window, min_periods=1).mean(),
        'signal': signals,
        'time': times
    })
    return df

def build_angle_data(log_data):
    """
    Extract pole angle trajectories from logs.

    Returns:
        Dictionary of lists of pole angle arrays, keyed by paradigm.
    """
    angle_data = {}
    for condition in ['Adaptive', 'Random', 'Null']:
        angle_data[condition] = [
            np.array(ep['pole_angle']) for ep in log_data[condition]['log']
            if 'pole_angle' in ep
        ]
    return angle_data
