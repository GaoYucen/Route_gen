import networkx as nx
import random
import numpy as np
from tqdm import tqdm
import pickle
import time
import sys
import os

sys.path.append('code')
from config import get_config

def calculate_via_path_length(G, via_path):
    """Calculate the total length of a path through via nodes"""
    path_length = 0
    for u, v in zip(via_path[:-1], via_path[1:]):
        path_length += G[u][v].get('weight', 1)
    return path_length


def generate_alternative_routes(G, s, t, partitions, boundary_nodes, num_routes=30, theta=0.9):
    """Generate alternative routes based on CRP algorithm"""
    # Get shortest paths
    forward_search = nx.shortest_path(G, source=s)
    backward_search = nx.shortest_path(G, target=t)
    shortest_path_length = nx.shortest_path_length(G, s, t, weight='weight')

    if shortest_path_length == 0:
        print(f'Warning: shortest path length is 0 for s:{s}, t:{t}')
        return [], []

    # Generate candidates
    candidates = []
    for level, (partition_level, boundary_level) in enumerate(zip(partitions, boundary_nodes)):
        for boundary in boundary_level:
            candidates.extend([node for node in boundary
                               if node in forward_search and node in backward_search
                               and calculate_via_path_length(G, forward_search[node] + backward_search[node][
                                                                                       1:]) / shortest_path_length <= 1.5])

    # Score and rank candidates
    scored_candidates = [
        (candidate, calculate_via_path_length(G, forward_search[candidate] + backward_search[candidate][1:]))
        for candidate in candidates]
    scored_candidates.sort(key=lambda x: x[1])

    # Select candidates
    selected_candidates = []
    selected_routes = []
    for candidate, _ in scored_candidates:
        via_path = forward_search[candidate] + backward_search[candidate][1:]
        path_edges = set(zip(via_path[:-1], via_path[1:]))

        if not any(len(path_edges & set(zip(route[:-1], route[1:]))) / min(len(via_path) - 1, len(route) - 1) > theta
                   for route in selected_routes):
            selected_candidates.append(candidate)
            selected_routes.append(via_path)

            if len(selected_routes) >= num_routes:
                break

    return selected_candidates, selected_routes


def process_trajectory_data(data, G, partitions, boundary_nodes, prefix, city_name):
    """Process trajectory data and save results"""
    candidate_list = []
    on_traj_flag_list = []

    # Generate candidates and flags
    for traj in tqdm(data, desc=f"Processing {prefix}"):
        s, t = traj[0], traj[-1]
        if s == t:
            continue

        alternative_candidates, _ = generate_alternative_routes(G, s, t, partitions, boundary_nodes)
        on_traj_flags = [1 if candidate in traj else 0 for candidate in alternative_candidates]

        candidate_list.append(alternative_candidates)
        on_traj_flag_list.append(on_traj_flags)

    # Calculate statistics
    non_empty_flags = [flags for flags in on_traj_flag_list if flags]
    if non_empty_flags:
        avg_ratio = np.mean([sum(flags) / len(flags) for flags in non_empty_flags])
        non_zero_ratio = np.mean([any(flags) for flags in on_traj_flag_list])

        print(f"\n{prefix} statistics:")
        print(f"On traj ratio: {avg_ratio:.4f}")
        print(f"On traj not 0 ratio: {non_zero_ratio:.4f}")

    # Save results
    for data, filename in [(candidate_list, f'preprocessed/{city_name}/{prefix}_candidate_list.pkl'),
                           (on_traj_flag_list, f'preprocessed/{city_name}/{prefix}_on_traj_flag_list.pkl')]:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def load_data(city_name, config):
    """Load and preprocess all required data"""
    # Load graph and partitions
    with open(f'data/{city_name}/graph_sc.pkl', 'rb') as f:
        G = pickle.load(f)

    with open(f'preprocessed/{city_name}/partitions.pkl', 'rb') as f:
        partitions = pickle.load(f)

    with open(f'preprocessed/{city_name}/boundary_nodes.pkl', 'rb') as f:
        boundary_nodes = pickle.load(f)

    # Load trajectory data
    datasets = {}
    for name in ['train', 'valid', 'test']:
        with open(f'data/{city_name}/{name}_data_small_sc.pkl', 'rb') as f:
            data = pickle.load(f)

        # Apply sample limits and extract trajectories
        if name == 'train':
            data = data[:config.num_train_samples]
        elif name == 'valid':
            data = data[:config.num_valid_samples]
        else:
            data = data[:config.num_test_samples]

        datasets[name] = [d[1] for d in data if d[1][0] != d[1][-1]]

    return G, partitions, boundary_nodes, datasets


if __name__ == "__main__":
    config, _ = get_config()

    # Load all data
    G, partitions, boundary_nodes, datasets = load_data(config.city, config)

    # Print dataset sizes
    for name, data in datasets.items():
        print(f"{name.capitalize()} data: {len(data)}")

    # Process all datasets
    for name, data in datasets.items():
        process_trajectory_data(data, G, partitions, boundary_nodes, name, config.city)