import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
import os
from tqdm import tqdm

sys.path.append('code')

from config import get_config

def load_dataset(city_name, prefix, config):
    """加载数据集和相关的候选节点列表"""
    with open(f'data/{city_name}/{prefix}_data_small_sc.pkl', 'rb') as f:
        trajectories = pickle.load(f)
        trajectories = [d[1] for d in trajectories]
        # 选择前config.num_{prefix}_samples
        trajectories = trajectories[:config.__dict__[f'num_{prefix}_samples']]

    with open(f'preprocessed/{city_name}/{prefix}_candidate_list.pkl', 'rb') as f:
        candidates = pickle.load(f)

    with open(f'preprocessed/{city_name}/{prefix}_on_traj_flag_list.pkl', 'rb') as f:
        flags = pickle.load(f)

    return trajectories, candidates, flags


def analyze_path_length(trajectories):
    """分析路径长度分布"""
    lengths = [len(path) for path in trajectories]
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths),
        'lengths': lengths
    }


def analyze_candidates(candidates, flags):
    """分析候选节点的特征"""
    candidate_counts = [len(c) for c in candidates]
    positive_ratios = [np.mean(f) if f else 0 for f in flags]
    return {
        'mean_candidates': np.mean(candidate_counts),
        'std_candidates': np.std(candidate_counts),
        'mean_positive_ratio': np.mean(positive_ratios),
        'std_positive_ratio': np.std(positive_ratios)
    }


def analyze_data_distribution(data_list, partitions):
    """分析起终点对的分布情况"""
    node_pairs = {}
    partition_pairs = {}

    # 获取所有节点的分区信息
    node_to_partition = {}
    for j, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = j

    for trajectory in data_list:
        if len(trajectory) < 2:
            continue

        s, t = trajectory[0], trajectory[-1]
        s_part = node_to_partition.get(s, -1)
        t_part = node_to_partition.get(t, -1)

        # 统计节点对
        pair = (s, t)
        node_pairs[pair] = node_pairs.get(pair, 0) + 1

        # 统计分区对
        part_pair = (s_part, t_part)
        partition_pairs[part_pair] = partition_pairs.get(part_pair, 0) + 1

    return node_pairs, partition_pairs

def plot_distributions(train_stats, val_stats, test_stats, save_dir):
    """绘制数据分布图"""
    plt.figure(figsize=(15, 5))

    # 路径长度分布
    plt.subplot(1, 2, 1)
    plt.hist(train_stats['path_length']['lengths'], bins=50, alpha=0.5, label='Train', density=True)
    plt.hist(val_stats['path_length']['lengths'], bins=50, alpha=0.5, label='Val', density=True)
    plt.hist(test_stats['path_length']['lengths'], bins=50, alpha=0.5, label='Test', density=True)
    plt.xlabel('Path Length')
    plt.ylabel('Density')
    plt.title('Path Length Distribution')
    plt.legend()

    # 候选节点数量分布
    plt.subplot(1, 2, 2)
    plt.boxplot([train_stats['candidate_counts'], val_stats['candidate_counts'],
                 test_stats['candidate_counts']], labels=['Train', 'Val', 'Test'])
    plt.ylabel('Number of Candidates')
    plt.title('Candidate Count Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distributions.png'))
    plt.close()

def calculate_distribution_overlap(dist1, dist2):
    """计算两个分布的重叠度"""
    keys = set(dist1.keys()) & set(dist2.keys())
    total_keys = set(dist1.keys()) | set(dist2.keys())
    return len(keys) / len(total_keys) if total_keys else 0

def analyze_distribution_differences(train_dist, val_dist, test_dist):
    """分析训练集、验证集和测试集的分布差异"""
    # 计算节点对和分区对的分布重叠
    node_pairs_train, part_pairs_train = train_dist
    node_pairs_val, part_pairs_val = val_dist
    node_pairs_test, part_pairs_test = test_dist

    # 计算各个数据集之间的重叠度
    overlaps = {
        'train_val': {
            'node_pairs': calculate_distribution_overlap(node_pairs_train, node_pairs_val),
            'partition_pairs': calculate_distribution_overlap(part_pairs_train, part_pairs_val)
        },
        'train_test': {
            'node_pairs': calculate_distribution_overlap(node_pairs_train, node_pairs_test),
            'partition_pairs': calculate_distribution_overlap(part_pairs_train, part_pairs_test)
        },
        'val_test': {
            'node_pairs': calculate_distribution_overlap(node_pairs_val, node_pairs_test),
            'partition_pairs': calculate_distribution_overlap(part_pairs_val, part_pairs_test)
        }
    }

    # 计算每个集合中的独特对数量
    unique_counts = {
        'train': {
            'node_pairs': len(set(node_pairs_train.keys())),
            'partition_pairs': len(set(part_pairs_train.keys()))
        },
        'val': {
            'node_pairs': len(set(node_pairs_val.keys())),
            'partition_pairs': len(set(part_pairs_val.keys()))
        },
        'test': {
            'node_pairs': len(set(node_pairs_test.keys())),
            'partition_pairs': len(set(part_pairs_test.keys()))
        }
    }

    return overlaps, unique_counts

def plot_distribution_differences(overlaps, unique_counts, save_dir):
    """绘制分布差异的可视化图表"""
    # 绘制重叠度热力图
    plt.figure(figsize=(15, 5))

    # 节点对重叠度热力图
    plt.subplot(1, 2, 1)
    node_overlap_matrix = np.array([
        [1.0, overlaps['train_val']['node_pairs'], overlaps['train_test']['node_pairs']],
        [overlaps['train_val']['node_pairs'], 1.0, overlaps['val_test']['node_pairs']],
        [overlaps['train_test']['node_pairs'], overlaps['val_test']['node_pairs'], 1.0]
    ])
    sns.heatmap(node_overlap_matrix,
                annot=True,
                fmt='.3f',
                xticklabels=['Train', 'Val', 'Test'],
                yticklabels=['Train', 'Val', 'Test'],
                cmap='YlOrRd')
    plt.title('Node Pairs Overlap')

    # 分区对重叠度热力图
    plt.subplot(1, 2, 2)
    part_overlap_matrix = np.array([
        [1.0, overlaps['train_val']['partition_pairs'], overlaps['train_test']['partition_pairs']],
        [overlaps['train_val']['partition_pairs'], 1.0, overlaps['val_test']['partition_pairs']],
        [overlaps['train_test']['partition_pairs'], overlaps['val_test']['partition_pairs'], 1.0]
    ])
    sns.heatmap(part_overlap_matrix,
                annot=True,
                fmt='.3f',
                xticklabels=['Train', 'Val', 'Test'],
                yticklabels=['Train', 'Val', 'Test'],
                cmap='YlOrRd')
    plt.title('Partition Pairs Overlap')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_overlap.png'))
    plt.close()

    # 绘制独特对数量的条形图
    plt.figure(figsize=(10, 5))
    datasets = ['train', 'val', 'test']
    node_pairs_counts = [unique_counts[d]['node_pairs'] for d in datasets]
    part_pairs_counts = [unique_counts[d]['partition_pairs'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    plt.bar(x - width/2, node_pairs_counts, width, label='Node Pairs')
    plt.bar(x + width/2, part_pairs_counts, width, label='Partition Pairs')

    plt.xlabel('Dataset')
    plt.ylabel('Count')
    plt.title('Unique Pairs Count')
    plt.xticks(x, datasets)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'unique_pairs_count.png'))
    plt.close()

def main():
    config, _ = get_config()

    # 创建保存目录
    save_dir = f'analyze/results/{config.city}'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    datasets = {}
    for prefix in ['train', 'valid', 'test']:
        trajectories, candidates, flags = load_dataset(config.city, prefix, config)
        datasets[prefix] = {
            'trajectories': trajectories,
            'candidates': candidates,
            'flags': flags
        }

    # 加载分区信息
    with open(f'preprocessed/{config.city}/partitions.pkl', 'rb') as f:
        partitions = pickle.load(f)[6]

    # 分析数据
    stats = {}
    for name, data in datasets.items():
        stats[name] = {
            'path_length': analyze_path_length(data['trajectories']),
            'candidates': analyze_candidates(data['candidates'], data['flags']),
            'candidate_counts': [len(c) for c in data['candidates']],
            'distribution': analyze_data_distribution(data['trajectories'], partitions)
        }

    # 打印统计信息
    for name, stat in stats.items():
        print(f"\n{name.upper()} Dataset Statistics:")
        print("Path Length:")
        print(f"  Mean: {stat['path_length']['mean']:.2f}")
        print(f"  Std: {stat['path_length']['std']:.2f}")
        print(f"  Min: {stat['path_length']['min']}")
        print(f"  Max: {stat['path_length']['max']}")
        print(f"  Median: {stat['path_length']['median']}")
        print("\nCandidates:")
        print(f"  Mean candidates: {stat['candidates']['mean_candidates']:.2f}")
        print(f"  Std candidates: {stat['candidates']['std_candidates']:.2f}")
        print(f"  Mean positive ratio: {stat['candidates']['mean_positive_ratio']:.4f}")
        print(f"  Std positive ratio: {stat['candidates']['std_positive_ratio']:.4f}")

    # 绘制分布图
    plot_distributions(stats['train'], stats['valid'], stats['test'], save_dir)

    # 保存详细统计信息
    with open(os.path.join(save_dir, 'statistics.pkl'), 'wb') as f:
        pickle.dump(stats, f)

        # 分析分布差异
        distributions = {
            'train': stats['train']['distribution'],
            'valid': stats['valid']['distribution'],
            'test': stats['test']['distribution']
        }

        overlaps, unique_counts = analyze_distribution_differences(
            distributions['train'],
            distributions['valid'],
            distributions['test']
        )

        # 打印分布差异统计
        print("\nDistribution Overlap Analysis:")
        for pair, metrics in overlaps.items():
            print(f"\n{pair}:")
            print(f"  Node pairs overlap: {metrics['node_pairs']:.4f}")
            print(f"  Partition pairs overlap: {metrics['partition_pairs']:.4f}")

        print("\nUnique Pairs Count:")
        for dataset, counts in unique_counts.items():
            print(f"\n{dataset}:")
            print(f"  Unique node pairs: {counts['node_pairs']}")
            print(f"  Unique partition pairs: {counts['partition_pairs']}")

        # 绘制分布差异图表
        plot_distribution_differences(overlaps, unique_counts, save_dir)

if __name__ == "__main__":
    main()