import networkx as nx
import random
import numpy as np
from tqdm import tqdm
import pickle

import networkx as nx
import random
import time

#%% CRP 预处理
def crp_preprocessing(G, L=6):
    partitions = []
    boundary_nodes = []
    nodes = list(G.nodes())
    # 初始分区
    initial_partition = [nodes]
    partitions.append(initial_partition)
    boundary_nodes.append([])

    start_time = time.time()
    for level in tqdm(range(L), desc="Processing levels"):
        new_partitions = []
        new_boundary_nodes = []
        for partition in tqdm(partitions[-1], desc=f"Level {level + 1} partitions", leave=False):
            # 简单的随机分区
            sub_graph = G.subgraph(partition)
            if len(partition) <= 1:
                new_partitions.append(partition)
                new_boundary_nodes.append([])
            else:
                split_point = len(partition) // 2
                random.shuffle(partition)
                p1 = partition[:split_point]
                p2 = partition[split_point:]
                new_partitions.extend([p1, p2])
                # 确定边界节点
                b1 = [n for n in p1 if any(nx.has_path(G, n, m) for m in p2)]
                b2 = [n for n in p2 if any(nx.has_path(G, n, m) for m in p1)]
                new_boundary_nodes.extend([b1, b2])
        partitions.append(new_partitions)
        boundary_nodes.append(new_boundary_nodes)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"CRP preprocessing completed in {total_time:.2f} seconds")

    return partitions, boundary_nodes

# #%% CRP 预处理
# # 自定义函数检查两个节点之间是否存在路径
# def has_path(adj_list, start, end):
#     visited = set()
#     stack = [start]
#     while stack:
#         node = stack.pop()
#         if node == end:
#             return True
#         if node not in visited:
#             visited.add(node)
#             stack.extend(adj_list.get(node, []))
#     return False
#
# def crp_preprocessing(G, L=6):
#     partitions = []
#     boundary_nodes = []
#     nodes = list(G.nodes())
#
#     # 构建邻接表
#     adj_list = {node: [] for node in G.nodes()}
#     for edge in G.edges():
#         source, target = edge
#         adj_list[source].append(target)
#
#     # 初始分区
#     initial_partition = [nodes]
#     partitions.append(initial_partition)
#     boundary_nodes.append([])
#
#     start_time = time.time()
#     for level in tqdm(range(L), desc="Processing levels"):
#         new_partitions = []
#         new_boundary_nodes = []
#         for partition in tqdm(partitions[-1], desc=f"Level {level + 1} partitions", leave=False):
#             if len(partition) <= 1:
#                 new_partitions.append(partition)
#                 new_boundary_nodes.append([])
#             else:
#                 split_point = len(partition) // 2
#                 random.shuffle(partition)
#                 p1 = partition[:split_point]
#                 p2 = partition[split_point:]
#                 new_partitions.extend([p1, p2])
#                 # 确定边界节点
#                 b1 = [n for n in p1 if any(has_path(adj_list, n, m) for m in p2)]
#                 b2 = [n for n in p2 if any(has_path(adj_list, n, m) for m in p1)]
#                 new_boundary_nodes.extend([b1, b2])
#         partitions.append(new_partitions)
#         boundary_nodes.append(new_boundary_nodes)
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"CRP preprocessing completed in {total_time:.2f} seconds")
#
#     return partitions, boundary_nodes

#%% 计算via_path_length
def calculate_via_path_length(G, via_path):
    via_path_length = 0
    for i in range(len(via_path) - 1):
        u = via_path[i]
        v = via_path[i + 1]
        # 获取边 (u, v) 的权重
        edge_weight = G[u][v].get('weight', 1)  # 如果没有指定权重，默认权重为 1
        via_path_length += edge_weight
    return via_path_length


#%%
# 基于 CRP 的替代路线生成
def generate_alternative_routes(G, s, t, partitions, boundary_nodes, num_routes=5, theta=0.75):
    # 候选生成
    forward_search = nx.shortest_path(G, source=s)
    backward_search = nx.shortest_path(G, target=t)
    shortest_path_length = nx.shortest_path_length(G, s, t, weight='weight')
    # print('shortest_path_length:', shortest_path_length)
    candidates = []
    for level in range(len(partitions)):
        for partition, boundary in zip(partitions[level], boundary_nodes[level]):
            for node in boundary:
                if node in forward_search and node in backward_search:
                    via_path = forward_search[node] + backward_search[node][1:]
                    stretch = calculate_via_path_length(G, via_path) / shortest_path_length
                    if stretch <= 1.5:
                        candidates.append(node)

    # print('candidates:', candidates)

    # 评分和排名
    scored_candidates = []
    for candidate in candidates:
        via_path = forward_search[candidate] + backward_search[candidate][1:]
        scored_candidates.append((candidate, calculate_via_path_length(G, via_path)))
    scored_candidates.sort(key=lambda x: x[1])

    # 选择
    selected_candidates = []
    selected_routes = []
    for candidate, _ in scored_candidates:
        via_path = forward_search[candidate] + backward_search[candidate][1:]
        overlap = False
        for existing_route in selected_routes:
            # 得到边表示
            common_edges = set(zip(via_path[:-1], via_path[1:])) & set(zip(existing_route[:-1], existing_route[1:]))
            overlap_ratio = len(common_edges) / min(len(via_path) - 1, len(existing_route) - 1)
            if overlap_ratio > theta:
                overlap = True
                break
        if not overlap:
            selected_candidates.append(candidate)
            selected_routes.append(via_path)
        if len(selected_routes) >= num_routes:
            break

    return selected_candidates, selected_routes

#%% 计算相似度
def similarity(traj1, traj2):
    return len(set(traj1).intersection(set(traj2)))

def precision_recall_f1_score(pred_path, orig_path):
    precision = len(set(pred_path) & set(orig_path)) / len(set(pred_path))
    recall = len(set(pred_path) & set(orig_path)) / len(set(orig_path))
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score

#%%
# 读取一个有向图路网
route_num = 10

#
# 读取link信息
def link_read_ds(file_name):
    link_df = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            [link_id, snode, enode, length] = line.split(',')
            # snode enode是起终点info
            link_df.append([int(link_id), int(snode), int(enode), float(length)])
    return link_df

link_df = link_read_ds('data/link_new.csv')

#%% 使用networkx构造路网
G = nx.DiGraph()
for link in tqdm(link_df):
    weight = round(link[3], 2)
    G.add_edge(link[1], link[2], weight=weight)

#%%
# CRP 预处理
partitions, boundary_nodes = crp_preprocessing(G)

#%% 存储分区和边界节点
with open('data/partitions.pkl', 'wb') as f:
    pickle.dump(partitions, f)
    f.close()

with open('data/boundary_nodes.pkl', 'wb') as f:
    pickle.dump(boundary_nodes, f)
    f.close()

#%% 读取前traj_small.txt
def travel_read_traj(file_name):
    traj_df = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            traj_df.append([int(item) for item in line.split(',')])

    return traj_df

traj_osmid_small = travel_read_traj('data/traj_osmid_small.txt')

#%% 取前100个轨迹
traj_osmid = traj_osmid_small[0:route_num]

# #%%
# traj = test_data[0]
# s = traj[0]
# t = traj[-1]
# # 生成替代路线
# alternative_candidates, alternative_routes = generate_alternative_routes(G, s, t, partitions, boundary_nodes)
#
# #%%
# print(f"源点: {s}, 终点: {t}")
# # 打印原始轨迹
# print("原始轨迹:")
# print(traj)
# print("替代路线:")
# for route in alternative_routes:
#     print(route)
#
# #%% 查看alternative_candidates是否在traj上
# for i, candidate in enumerate(alternative_candidates):
#     if candidate in traj:
#         print(i, candidate)
#
# #%%
# for route in alternative_routes:
#     similar = similarity(route, traj)
#     precision, recall, f1_score = precision_recall_f1_score(route, traj)
#     print(f"相似度: {similar}, 精确率: {precision}, 召回率: {recall}, F1 分数: {f1_score}")
#
# # Dijkstra最短路与traj的相似度
# dijk_path = nx.dijkstra_path(G, s, t, weight='weight')
# similar = similarity(dijk_path, traj)
# precision, recall, f1_score = precision_recall_f1_score(dijk_path, traj)
# print(f"相似度: {similar}, 精确率: {precision}, 召回率: {recall}, F1 分数: {f1_score}")

#%% 计算alternative_routes和traj的相似度
on_traj_list = []
f1_score_list = []
f1_score_dijk_list = []
for i in tqdm(range(0, len(traj_osmid))):
    traj = traj_osmid[i]
    s = traj[0]
    t = traj[-1]
    alternative_candidates, alternative_routes = generate_alternative_routes(G, s, t, partitions, boundary_nodes)
    # 计算alternative_candidates在traj上的比例
    count = 0
    f1_score_max = 0
    for idx, candidate in enumerate(alternative_candidates):
        if candidate in traj:
            # 计算F1分数
            _, _, f1_score = precision_recall_f1_score(alternative_routes[idx], traj)
            if f1_score > f1_score_max:
                f1_score_max = f1_score
            count += 1
    on_traj_list.append(count / len(alternative_candidates))
    f1_score_list.append(f1_score_max)
    # 计算Dijkstra最短路的F1分数
    dijk_path = nx.dijkstra_path(G, s, t, weight='weight')
    _, _, f1_score = precision_recall_f1_score(dijk_path, traj)
    f1_score_dijk_list.append(f1_score)

#%% on_traj的均值
print('On traj mean:', sum(on_traj_list) / len(on_traj_list))

#%% on_traj中不为0的比例
count = 0
for i in on_traj_list:
    if i != 0:
        count += 1
print('On traj not 0 ratio:', count / len(on_traj_list))

#%% 对比CRP和Dijkstra的F1分数
import matplotlib.pyplot as plt

plt.figure()
plt.plot(f1_score_list, label='CRP')
plt.plot(f1_score_dijk_list, label='Dijkstra')
plt.legend()
plt.xlabel('Trajectory Index')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison')
plt.show()
