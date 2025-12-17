import networkx as nx
import random
import numpy as np
from tqdm import tqdm
import pickle
import time

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
with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

#%% 读取分区和边界节点
with open('data/'+city_name+'/partitions.pkl', 'rb') as f:
    partitions = pickle.load(f)
    f.close()

with open('data/'+city_name+'/boundary_nodes.pkl', 'rb') as f:
    boundary_nodes = pickle.load(f)
    f.close()

#%% 读取轨迹数据
with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

# 提取轨迹数据
for i in range(len(test_data)):
    test_data[i] = (test_data[i][1])

# 取前100
test_data = test_data[:100]

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

#%% 计算alternative_candidates是否在traj上，生成训练数据，供后续MLP使用
candidate_list = []
on_traj_flag_list = []
for i in tqdm(range(0, len(test_data))):
    traj = test_data[i]
    s = traj[0]
    t = traj[-1]
    on_traj_flag = []
    alternative_candidates, alternative_routes = generate_alternative_routes(G, s, t, partitions, boundary_nodes)
    for idx, candidate in enumerate(alternative_candidates):
        if candidate in traj:
            on_traj_flag.append(1)
        else:
            on_traj_flag.append(0)
    candidate_list.append(alternative_candidates)
    on_traj_flag_list.append(on_traj_flag)

#%% on_traj_flag_list中1的比例
on_traj_ratio = []
for on_traj_flag in on_traj_flag_list:
    on_traj_ratio.append(sum(on_traj_flag) / len(on_traj_flag))
print('On traj ratio:', sum(on_traj_ratio) / len(on_traj_ratio))

#%% 保存
with open('data/'+city_name+'/candidate_list.pkl', 'wb') as f:
    pickle.dump(candidate_list, f)
    f.close()

with open('data/'+city_name+'/on_traj_flag_list.pkl', 'wb') as f:
    pickle.dump(on_traj_flag_list, f)
    f.close()



