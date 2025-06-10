# analysis_utils.py - Matrix-based analyses (correlation, PCA, etc.)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import networkx as nx


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


def infer_firing_order_enhanced(matrix, threshold=None, use_weights=True):
    """
    Enhanced firing order inference with multiple improvements

    Parameters:
    -----------
    matrix : np.array
        Causal connectivity matrix
    threshold : float, optional
        Threshold for considering connections (removes weak connections)
    use_weights : bool
        Whether to use connection strength weighting
    """

    # Step 1: Apply threshold to focus on strong connections
    if threshold is not None:
        matrix_thresh = np.where(np.abs(matrix) > threshold, matrix, 0)
    else:
        # Auto-threshold: keep top 50% of connections
        threshold = np.percentile(np.abs(matrix), 50)
        matrix_thresh = np.where(np.abs(matrix) > threshold, matrix, 0)

    # Step 2: Weight by connection strength (stronger connections matter more)
    if use_weights:
        # Square the matrix to emphasize strong connections
        weighted_matrix = np.sign(matrix_thresh) * (matrix_thresh ** 2)
    else:
        weighted_matrix = matrix_thresh

    # Step 3: Calculate directional scores
    outgoing = np.sum(weighted_matrix, axis=1)
    incoming = np.sum(weighted_matrix, axis=0)
    net_score = outgoing - incoming

    # Step 4: Normalize by total activity to handle different scales
    total_activity = outgoing + np.abs(incoming)
    normalized_score = np.divide(net_score, total_activity,
                                 out=np.zeros_like(net_score),
                                 where=total_activity!=0)

    firing_order = np.argsort(-normalized_score)

    return firing_order, normalized_score, {'outgoing': outgoing, 'incoming': incoming}

def infer_firing_order_topological(matrix, threshold=None):
    """
    Use topological sorting to find firing order based on directed graph structure
    """
    if threshold is None:
        threshold = np.percentile(np.abs(matrix), 60)

    # Create binary adjacency matrix
    adj_matrix = (np.abs(matrix) > threshold).astype(int)

    # Create directed graph
    G = nx.DiGraph(adj_matrix)

    try:
        # Topological sort gives a valid ordering
        topo_order = list(nx.topological_sort(G))
        return topo_order, None
    except nx.NetworkXError:
        # Graph has cycles, fall back to approximate method
        return infer_firing_order_enhanced(matrix, threshold)[0], "cycles_detected"

def infer_firing_order_hierarchical(matrix, alpha=0.7):
    """
    Hierarchical approach: early neurons influence many, late neurons influence few

    Parameters:
    -----------
    alpha : float
        Weight for outgoing vs incoming influences
    """

    # Calculate influence metrics
    outgoing = np.sum(np.abs(matrix), axis=1)  # Total outgoing influence
    incoming = np.sum(np.abs(matrix), axis=0)  # Total incoming influence

    # Number of connections (breadth of influence)
    out_connections = np.sum(matrix != 0, axis=1)
    in_connections = np.sum(matrix != 0, axis=0)

    # Combined score: strength + breadth, weighted by position in hierarchy
    hierarchy_score = (
            alpha * (outgoing + out_connections) -
            (1 - alpha) * (incoming + in_connections)
    )

    firing_order = np.argsort(-hierarchy_score)

    return firing_order, hierarchy_score

def infer_firing_order_temporal_chains(first_order_matrix, multi_order_matrix):
    """
    Use both first-order and multi-order matrices to detect temporal chains
    """

    # Find strongest first-order connections (immediate causality)
    first_threshold = np.percentile(np.abs(first_order_matrix), 75)
    strong_immediate = np.abs(first_order_matrix) > first_threshold

    # Find multi-order patterns
    multi_threshold = np.percentile(np.abs(multi_order_matrix), 75)
    strong_delayed = np.abs(multi_order_matrix) > multi_threshold

    # Weight immediate connections more heavily
    combined_matrix = (2 * first_order_matrix * strong_immediate +
                       multi_order_matrix * strong_delayed) / 3

    # Apply enhanced method to combined matrix
    firing_order, scores, details = infer_firing_order_enhanced(combined_matrix)

    return firing_order, scores, {
        'immediate_strength': np.sum(np.abs(first_order_matrix), axis=1),
        'delayed_strength': np.sum(np.abs(multi_order_matrix), axis=1),
        'combined_scores': scores
    }



# Example usage function
def analyze_firing_patterns(first_order_matrix, multi_order_matrix=None):
    """
    Comprehensive analysis using multiple approaches
    """

    print("=== Firing Order Analysis ===")

    # Method 1: Enhanced single matrix
    order1, scores1, details1 = infer_firing_order_enhanced(first_order_matrix)
    print(f"Enhanced method: {order1}")
    print(f"Scores: {scores1.round(3)}")

    # Method 2: Hierarchical
    order2, scores2 = infer_firing_order_hierarchical(first_order_matrix)
    print(f"Hierarchical method: {order2}")

    # Method 3: Topological
    order3, status3 = infer_firing_order_topological(first_order_matrix)
    print(f"Topological method: {order3}")
    if status3: print(f"Status: {status3}")

    # Method 4: Temporal chains (if both matrices available)
    if multi_order_matrix is not None:
        order4, scores4, details4 = infer_firing_order_temporal_chains(
            first_order_matrix, multi_order_matrix)
        print(f"Temporal chains method: {order4}")

    # Method 5: Consensus
    consensus_order, consensus_scores, all_orders = infer_firing_order_consensus(first_order_matrix)
    print(f"Consensus method: {consensus_order}")

    return {
        'enhanced': (order1, scores1),
        'hierarchical': (order2, scores2),
        'topological': (order3, status3),
        'consensus': (consensus_order, consensus_scores),
        'all_methods': all_orders
    }

from scipy import stats

def infer_firing_order_consensus(matrix, methods=None):
    """
    Infer a consensus neuron firing order using multiple ranking methods.

    Parameters:
        matrix : np.ndarray
            Causal connectivity matrix (NxN)
        methods : list of str or None
            Which methods to use. Default: ['enhanced', 'hierarchical', 'topological']

    Returns:
        consensus_order : np.ndarray
            Neuron indices in consensus firing order
        consensus_scores : np.ndarray
            Averaged rank scores per neuron
        all_orders : dict
            Method -> firing order list
    """
    if methods is None:
        methods = ['enhanced', 'hierarchical', 'topological']

    orders = {}
    rank_scores = {}

    if 'enhanced' in methods:
        order, score, _ = infer_firing_order_enhanced(matrix)
        orders['enhanced'] = order
        rank_scores['enhanced'] = stats.rankdata(-score)

    if 'hierarchical' in methods:
        order, score = infer_firing_order_hierarchical(matrix)
        orders['hierarchical'] = order
        rank_scores['hierarchical'] = stats.rankdata(-score)

    if 'topological' in methods:
        order, status = infer_firing_order_topological(matrix)
        orders['topological'] = order
        if status != "cycles_detected":
            rank_scores['topological'] = stats.rankdata(order)
        else:
            print("Topological method skipped: graph contains cycles.")

    # Combine available rank scores
    n_neurons = matrix.shape[0]
    consensus_scores = np.zeros(n_neurons)

    valid_methods = list(rank_scores.keys())
    for method in valid_methods:
        consensus_scores += rank_scores[method]

    consensus_scores /= len(valid_methods)
    consensus_order = np.argsort(consensus_scores)

    return consensus_order, consensus_scores, orders