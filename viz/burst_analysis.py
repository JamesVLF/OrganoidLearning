import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def match_bursts_to_log(bursts, log_df, window=250):
    """
    Link each burst (start time) to the closest reward/stim event in time within ±window ms.
    """
    burst_rewards, stim_patterns = [], []
    for start, _ in bursts:
        match = log_df[(log_df['time'] >= start - window) & (log_df['time'] <= start + window)]
        if not match.empty:
            row = match.iloc[0]
            burst_rewards.append(row['reward'])
            stim_patterns.append(tuple(row['stim_indices']))
        else:
            burst_rewards.append(np.nan)
            stim_patterns.append(None)
    return burst_rewards, stim_patterns


def compute_reward_labels(rewards, labels=("Negative", "Neutral", "Positive")):
    """
    Categorize rewards by percentile threshold.
    """
    thresholds = np.nanpercentile(rewards, [33, 66])
    categories = []
    for r in rewards:
        if np.isnan(r):
            categories.append(labels[1])
        elif r < thresholds[0]:
            categories.append(labels[0])
        elif r > thresholds[1]:
            categories.append(labels[2])
        else:
            categories.append(labels[1])
    return pd.DataFrame({'Reward': rewards, 'Category': categories})


def plot_burst_reward_histogram(reward_df, ax=None):
    """
    Plot histogram of bursts grouped by reward category.
    """
    ax = ax or plt.gca()
    counts = reward_df['Category'].value_counts()
    ax.bar(counts.index, counts.values)
    ax.set_title("Burst Counts by Reward Outcome")
    ax.set_ylabel("Count")
    return ax


def plot_reward_linked_bursts(adaptive_rewards, null_rewards, ax=None):
    """
    Compare mean burst-linked rewards between two conditions.
    """
    ax = ax or plt.gca()
    ax.bar(['Adaptive', 'Null'],
           [np.nanmean(adaptive_rewards), np.nanmean(null_rewards)])
    ax.set_title("Reward-Linked Burst Comparison")
    ax.set_ylabel("Mean Reward (Linked to Bursts)")
    return ax