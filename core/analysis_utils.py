# analysis_utils.py - Matrix-based analyses (correlation, PCA, etc.)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import networkx as nx
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from scipy.stats import spearmanr


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

def infer_firing_order(matrix):
    """Compute net influence and firing order from a causal connectivity matrix."""
    outgoing = np.sum(matrix, axis=1)
    incoming = np.sum(matrix, axis=0)
    net_score = outgoing - incoming
    firing_order = np.argsort(-net_score)
    return firing_order, net_score

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

def run_comprehensive_firing_analysis(
        first_order_matrix,
        multi_order_matrix=None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Run all available firing order inference methods on a causal matrix.
    """

    results = {}

    try:
        order, scores, details = infer_firing_order_enhanced(first_order_matrix)
        results['enhanced'] = {'order': order, 'scores': scores, 'details': details}
    except Exception as e:
        print(f"[Enhanced] Failed: {e}")
        results['enhanced'] = None

    try:
        order, scores = infer_firing_order_hierarchical(first_order_matrix)
        results['hierarchical'] = {'order': order, 'scores': scores}
    except Exception as e:
        print(f"[Hierarchical] Failed: {e}")
        results['hierarchical'] = None

    try:
        order, status = infer_firing_order_topological(first_order_matrix)
        results['topological'] = {'order': order, 'status': status}
    except Exception as e:
        print(f"[Topological] Failed: {e}")
        results['topological'] = None

    if multi_order_matrix is not None:
        try:
            order, scores, details = infer_firing_order_temporal_chains(
                first_order_matrix, multi_order_matrix)
            results['temporal_chains'] = {'order': order, 'scores': scores, 'details': details}
        except Exception as e:
            print(f"[Temporal Chains] Failed: {e}")
            results['temporal_chains'] = None

    try:
        consensus_order, consensus_scores, all_orders = infer_firing_order_consensus(first_order_matrix)
        results['consensus'] = {
            'order': consensus_order,
            'scores': consensus_scores,
            'all_methods': all_orders
        }
    except Exception as e:
        print(f"[Consensus] Failed: {e}")
        results['consensus'] = None

    return results

def create_condition_visualizations(
        condition: str,
        start_ms: int,
        end_ms: int,
        analysis_results: Dict[str, Optional[Dict[str, Any]]],
        first_order_matrix: Any,
        multi_order_matrix: Optional[Any],
        save_dir: str
) -> None:
    """
    Create and save visualizations from causal matrices and inferred firing orders.

    Args:
        condition: Name of the experimental condition.
        start_ms: Start time of window (in ms).
        end_ms: End time of window (in ms).
        analysis_results: Dict of results from firing order inference.
        first_order_matrix: Direct causal matrix.
        multi_order_matrix: Optional multi-order causal matrix.
        save_dir: Directory to save figures and reports.
    """
    from org_eval.viz.plots_general import (
        create_directed_graph_from_consensus,
        visualize_firing_graph,
        create_multiple_graph_views,
        analyze_graph_properties
    )

    if analysis_results.get('consensus') is None:
        print(f"[{condition}] Skipped visualization - no consensus results.")
        return

    consensus_order = analysis_results['consensus']['order']
    consensus_scores = analysis_results['consensus']['scores']

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Create directed graph
        G = create_directed_graph_from_consensus(
            consensus_order, consensus_scores, first_order_matrix
        )

        # Graph visualizations
        for layout in ['spring', 'hierarchical', 'circular']:
            fig = plt.figure(figsize=(12, 8))
            visualize_firing_graph(G, layout=layout, save_path=save_path / f"graph_{layout}.png")
            plt.close(fig)

        if multi_order_matrix is not None:
            fig = plt.figure(figsize=(16, 12))
            create_multiple_graph_views(
                consensus_order, consensus_scores,
                first_order_matrix, first_order_matrix, multi_order_matrix
            )
            plt.savefig(save_path / "multiple_views.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Graph properties
        properties = analyze_graph_properties(G)
        with open(save_path / "graph_properties.txt", 'w') as f:
            f.write(f"=== Graph Properties: {condition} {start_ms}-{end_ms} ms ===\n")
            f.write(f"Firing order: {properties['firing_order']}\n")
            f.write(f"Most influential neurons: {properties['influences'][:3]}\n")
            f.write(f"Acyclic: {properties['is_acyclic']}\n")
            f.write(f"Network density: {properties['density']:.3f}\n\n")

            f.write("=== Method Comparison ===\n")
            for method, result in analysis_results.items():
                if result and 'order' in result:
                    f.write(f"{method}: {result['order']}\n")

        print(f"[{condition}] Visualizations saved to: {save_path}")

    except Exception as e:
        print(f"[{condition}] Visualization error: {e}")


def generate_comparison_plots(
        firing_order_results: Dict[str, Dict[str, Dict[str, Any]]],
        time_windows: List[Tuple[int, int]],
        save_dir: str
) -> None:
    """
    Generate comparison plots across conditions and time windows.

    Args:
        firing_order_results: Nested dict of condition -> window_key -> method results.
        time_windows: List of (start_ms, end_ms) tuples.
        save_dir: Directory to save output figures.
    """

    print("\nGenerating comparison plots...")
    comp_dir = Path(save_dir) / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_firing_order_consistency(firing_order_results, time_windows, comp_dir)
        plot_method_agreement(firing_order_results, time_windows, comp_dir)
        plot_network_properties_comparison(firing_order_results, time_windows, comp_dir)
    except Exception as e:
        print(f"[Comparison Error] {e}")

def plot_firing_order_consistency(
        firing_order_results: Dict[str, Dict[str, Dict[str, Any]]],
        time_windows: List[Tuple[int, int]],
        save_dir: Path
) -> None:
    """
    Plot consistency of consensus firing orders across time windows and conditions.

    Args:
        firing_order_results: Nested dict with consensus order data.
        time_windows: List of (start_ms, end_ms) tuples.
        save_dir: Path object to output directory.
    """
    fig, axes = plt.subplots(1, len(time_windows), figsize=(6 * len(time_windows), 8))
    if len(time_windows) == 1:
        axes = [axes]  # Ensure iterable

    for i, (start_ms, end_ms) in enumerate(time_windows):
        window_key = f"{start_ms}_{end_ms}"
        orders_data, conditions = [], []

        for condition, result_by_window in firing_order_results.items():
            consensus = result_by_window.get(window_key, {}).get('consensus')
            if consensus and 'order' in consensus:
                orders_data.append(consensus['order'])
                conditions.append(condition)

        if orders_data:
            orders_matrix = np.array(orders_data)
            ax = axes[i]
            im = ax.imshow(orders_matrix, aspect='auto', cmap='viridis')
            ax.set_title(f'{start_ms}-{end_ms} ms')
            ax.set_xlabel('Neuron Position')
            ax.set_ylabel('Condition')
            ax.set_yticks(range(len(conditions)))
            ax.set_yticklabels(conditions)
            plt.colorbar(im, ax=ax, label='Neuron Index')

    plt.tight_layout()
    out_path = Path(save_dir) / "firing_order_consistency.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved firing order consistency plot to {out_path}")


def plot_method_agreement(
        firing_order_results: Dict[str, Dict[str, Dict[str, Any]]],
        time_windows: List[Tuple[int, int]],
        save_dir: Path,
        methods: List[str] = None
) -> None:
    """
    Plot agreement between different firing order inference methods.

    Args:
        firing_order_results: Nested dictionary of method outputs.
        time_windows: List of (start_ms, end_ms) tuples.
        save_dir: Path to save the plot.
        methods: Optional list of methods to compare.
    """
    if methods is None:
        methods = ['enhanced', 'hierarchical', 'topological', 'temporal_chains', 'consensus']
    conditions = list(firing_order_results.keys())

    n_rows, n_cols = len(time_windows), len(conditions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for i, (start_ms, end_ms) in enumerate(time_windows):
        window_key = f"{start_ms}_{end_ms}"

        for j, condition in enumerate(conditions):
            ax = axes[i][j]
            results = firing_order_results.get(condition, {}).get(window_key, {})

            agreement_matrix = np.zeros((len(methods), len(methods)))

            for m1_idx, m1 in enumerate(methods):
                for m2_idx, m2 in enumerate(methods):
                    data1 = results.get(m1)
                    data2 = results.get(m2)

                    if data1 and data2:
                        o1, o2 = data1.get('order'), data2.get('order')
                        if o1 is not None and o2 is not None:
                            try:
                                corr, _ = spearmanr(o1, o2)
                                agreement_matrix[m1_idx, m2_idx] = corr
                            except Exception:
                                match = np.mean(np.array(o1) == np.array(o2))
                                agreement_matrix[m1_idx, m2_idx] = match

            im = ax.imshow(agreement_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{condition}\n{start_ms}-{end_ms} ms', fontsize=10)
            ax.set_xticks(range(len(methods)))
            ax.set_yticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(methods, fontsize=8)

            for mi in range(len(methods)):
                for mj in range(len(methods)):
                    ax.text(mj, mi, f'{agreement_matrix[mi, mj]:.2f}',
                            ha='center', va='center', fontsize=7)

    plt.tight_layout()
    output_path = Path(save_dir) / "method_agreement.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved method agreement plot to: {output_path}")

def plot_network_properties_comparison(
        graph_properties: Dict[str, Dict[str, Dict[str, float]]],
        time_windows: List[Tuple[int, int]],
        save_dir: Path
) -> None:
    """
    Visual comparison of network properties (density, acyclicity, etc.)
    across conditions and time windows.

    Args:
        graph_properties: Nested dict {condition -> {window_key -> metrics_dict}}
        time_windows: List of (start_ms, end_ms) windows
        save_dir: Path where plot will be saved
    """

    conditions = list(graph_properties.keys())
    metrics = ['density', 'is_acyclic']  # Add more if needed

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx] if len(metrics) > 1 else axes
        for condition in conditions:
            values = []
            labels = []
            for (start_ms, end_ms) in time_windows:
                window_key = f"{start_ms}_{end_ms}"
                props = graph_properties.get(condition, {}).get(window_key, {})
                if metric in props:
                    values.append(props[metric])
                    labels.append(f"{start_ms}-{end_ms}")
            ax.plot(labels, values, label=condition, marker='o')

        ax.set_title(f"Comparison of {metric.replace('_', ' ').title()} Across Conditions")
        ax.set_xlabel("Time Window (ms)")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    output_file = Path(save_dir) / "network_properties_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved network properties comparison plot to {output_file}")


def run_all_firing_order_analyses(
        first_order_matrix: np.ndarray,
        multi_order_matrix: Optional[np.ndarray] = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Runs multiple firing order inference methods on causal connectivity matrices.

    Args:
        first_order_matrix: NxN matrix of immediate causal links
        multi_order_matrix: NxN matrix of extended causal relationships (optional)

    Returns:
        Dictionary mapping method names to their results or None if failed.
    """

    results: Dict[str, Optional[Dict[str, Any]]] = {}

    try:
        order, scores, details = infer_firing_order_enhanced(first_order_matrix)
        results['enhanced'] = {'order': order, 'scores': scores, 'details': details}
    except Exception as e:
        print(f"[ERROR] Enhanced method failed: {e}")
        results['enhanced'] = None

    try:
        order, scores = infer_firing_order_hierarchical(first_order_matrix)
        results['hierarchical'] = {'order': order, 'scores': scores}
    except Exception as e:
        print(f"[ERROR] Hierarchical method failed: {e}")
        results['hierarchical'] = None

    try:
        order, status = infer_firing_order_topological(first_order_matrix)
        results['topological'] = {'order': order, 'status': status}
    except Exception as e:
        print(f"[ERROR] Topological method failed: {e}")
        results['topological'] = None

    if multi_order_matrix is not None:
        try:
            order, scores, details = infer_firing_order_temporal_chains(
                first_order_matrix, multi_order_matrix
            )
            results['temporal_chains'] = {
                'order': order, 'scores': scores, 'details': details
            }
        except Exception as e:
            print(f"[ERROR] Temporal chains method failed: {e}")
            results['temporal_chains'] = None

    try:
        consensus_order, consensus_scores, all_orders = infer_firing_order_consensus(
            first_order_matrix
        )
        results['consensus'] = {
            'order': consensus_order,
            'scores': consensus_scores,
            'all_methods': all_orders
        }
    except Exception as e:
        print(f"[ERROR] Consensus method failed: {e}")
        results['consensus'] = None

    return results

def _analyze_single_condition_window(self, condition, start_ms, end_ms, save_dir):
    """
    Analyze firing order for a single condition and time window
    """

    # Get causal matrices for this condition and time window
    first_order_key = (condition, start_ms, end_ms, 'first_order')
    multi_order_key = (condition, start_ms, end_ms, 'multi_order')

    # Check if matrices exist, if not compute them
    if first_order_key not in self.causal_latency_matrices:
        self._compute_causal_matrices(condition, start_ms, end_ms)

    first_order_matrix = self.causal_latency_matrices.get(first_order_key)
    multi_order_matrix = self.multi_order_matrices.get(multi_order_key)

    if first_order_matrix is None:
        raise ValueError(f"Could not compute causal matrices for {condition} {start_ms}-{end_ms}")

    # Run comprehensive firing order analysis
    analysis_results = self._run_comprehensive_analysis(
        first_order_matrix, multi_order_matrix
    )

    # Create and save visualizations
    self._create_condition_visualizations(
        condition, start_ms, end_ms, analysis_results,
        first_order_matrix, multi_order_matrix, save_dir
    )

    return analysis_results