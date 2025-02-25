import networkx as nx
import random
import numpy as np
from tqdm import tqdm
import pickle
import time
import sys
import os

# Get code folder path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(code_dir)

from config import get_config


def set_random_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def crp_preprocessing(G, L=6, seed=42):
    """
    Perform CRP (Customizable Route Planning) preprocessing on a graph
    Args:
        G: NetworkX graph
        L: Number of partition levels
        seed: Random seed for consistent partitioning
    Returns:
        partitions: List of partitions at each level
        boundary_nodes: List of boundary nodes at each level
    """
    # Set random seed for reproducibility
    set_random_seed(seed)

    partitions = []
    boundary_nodes = []
    nodes = sorted(list(G.nodes()))  # Sort nodes for consistency

    # Initial partition
    partitions.append([nodes])
    boundary_nodes.append([])

    start_time = time.time()

    # Process each level
    for level in tqdm(range(L), desc="Processing levels"):
        new_partitions = []
        new_boundary_nodes = []

        # Process each partition at current level
        for partition in tqdm(partitions[-1], desc=f"Level {level + 1} partitions", leave=False):
            if len(partition) <= 1:
                new_partitions.append(partition)
                new_boundary_nodes.append([])
                continue

            # Sort partition for consistency before splitting
            partition = sorted(partition)
            split_point = len(partition) // 2

            # Deterministic partition split
            p1, p2 = partition[:split_point], partition[split_point:]
            new_partitions.extend([p1, p2])

            # Determine boundary nodes using graph connectivity
            sub_g = G.subgraph(partition)
            b1 = sorted({n for n in p1 if any(sub_g.has_edge(n, m) or sub_g.has_edge(m, n) for m in p2)})
            b2 = sorted({n for n in p2 if any(sub_g.has_edge(n, m) or sub_g.has_edge(m, n) for m in p1)})
            new_boundary_nodes.extend([b1, b2])

        partitions.append(new_partitions)
        boundary_nodes.append(new_boundary_nodes)

    print(f"CRP preprocessing completed in {time.time() - start_time:.2f} seconds")
    return partitions, boundary_nodes


def save_results(city_name, partitions, boundary_nodes):
    """Save partitions and boundary nodes to files"""
    for data, filename in [
        (partitions, f'preprocessed/{city_name}/partitions.pkl'),
        (boundary_nodes, f'preprocessed/{city_name}/boundary_nodes.pkl')
    ]:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    config, _ = get_config()

    # Load road network graph
    with open(f'data/{config.city}/graph_sc.pkl', 'rb') as f:
        G = pickle.load(f)

    # Perform CRP preprocessing with fixed seed
    partitions, boundary_nodes = crp_preprocessing(G, seed=42)

    # Save results
    save_results(config.city, partitions, boundary_nodes)