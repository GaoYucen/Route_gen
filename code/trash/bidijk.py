#%%
# 导入一些必要的库
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt

#%% 双向Dijkstra
import heapq

k = 3

# def bidirectional_dijkstra(G, source, target, weight='weight'):
#     """双向Dijkstra算法"""
#     queue_from_source = []
#     heapq.heappush(queue_from_source, (0, source))
#     distance_from_source = {node: float('inf') for node in G}
#     distance_from_source[source] = 0
#     parents_from_source = {source: None}
#
#     queue_from_target = []
#     heapq.heappush(queue_from_target, (0, target))
#     distance_from_target = {node: float('inf') for node in G}
#     distance_from_target[target] = 0
#     parents_from_target = {target: None}
#
#     visited_from_source = set()
#     visited_from_target = set()
#
#     while queue_from_source and queue_from_target:
#         # 从源点方向扩展
#         dist_from_source, node_from_source = heapq.heappop(queue_from_source)
#         visited_from_source.add(node_from_source)
#         for neighbor, edge_attr in G[node_from_source].items():
#             if neighbor in visited_from_source:
#                 continue
#             new_dist = dist_from_source + edge_attr.get(weight, 1)
#             if new_dist < distance_from_source[neighbor]:
#                 distance_from_source[neighbor] = new_dist
#                 parents_from_source[neighbor] = node_from_source
#                 heapq.heappush(queue_from_source, (new_dist, neighbor))
#
#         # 从目标点方向扩展
#         dist_from_target, node_from_target = heapq.heappop(queue_from_target)
#         visited_from_target.add(node_from_target)
#         for neighbor, edge_attr in G[node_from_target].items():
#             if neighbor in visited_from_target:
#                 continue
#             new_dist = dist_from_target + edge_attr.get(weight, 1)
#             if new_dist < distance_from_target[neighbor]:
#                 distance_from_target[neighbor] = new_dist
#                 parents_from_target[neighbor] = node_from_target
#                 heapq.heappush(queue_from_target, (new_dist, neighbor))
#
#         # 检查是否找到碰撞点
#         for node in visited_from_source.intersection(visited_from_target):
#             total_dist = distance_from_source[node] + distance_from_target[node]
#             yield node, total_dist
#
# def find_k_collision_points(G, source, target, k, weight='weight'):
#     """找到k个不同的碰撞点"""
#     shortest_path_length = nx.shortest_path_length(G, source, target, weight=weight)
#     collision_points = {}
#     for node, total_dist in bidirectional_dijkstra(G, source, target, weight):
#         if total_dist > 1.1 * shortest_path_length and total_dist <= 1.5 * shortest_path_length:
#             collision_points[node] = total_dist
#         if len(collision_points) >= k:
#             break
#     # 将字典转换为列表并按total_dist排序
#     collision_points_list = sorted(collision_points.items(), key=lambda x: x[1])
#     return collision_points_list[:k]

def bidirectional_dijkstra(G, source, target, weight='weight'):
    """双向Dijkstra算法"""
    queue_from_source = []
    heapq.heappush(queue_from_source, (0, source))
    distance_from_source = {node: float('inf') for node in G}
    distance_from_source[source] = 0
    parents_from_source = {source: None}

    queue_from_target = []
    heapq.heappush(queue_from_target, (0, target))
    distance_from_target = {node: float('inf') for node in G}
    distance_from_target[target] = 0
    parents_from_target = {target: None}

    visited_from_source = set()
    visited_from_target = set()

    while queue_from_source and queue_from_target:
        # 从源点方向扩展
        dist_from_source, node_from_source = heapq.heappop(queue_from_source)
        visited_from_source.add(node_from_source)
        for neighbor, edge_attr in G[node_from_source].items():
            if neighbor in visited_from_source:
                continue
            new_dist = dist_from_source + edge_attr.get(weight, 1)
            if new_dist < distance_from_source[neighbor]:
                distance_from_source[neighbor] = new_dist
                parents_from_source[neighbor] = node_from_source
                heapq.heappush(queue_from_source, (new_dist, neighbor))

        # 从目标点方向扩展
        dist_from_target, node_from_target = heapq.heappop(queue_from_target)
        visited_from_target.add(node_from_target)
        for neighbor, edge_attr in G[node_from_target].items():
            if neighbor in visited_from_target:
                continue
            new_dist = dist_from_target + edge_attr.get(weight, 1)
            if new_dist < distance_from_target[neighbor]:
                distance_from_target[neighbor] = new_dist
                parents_from_target[neighbor] = node_from_target
                heapq.heappush(queue_from_target, (new_dist, neighbor))

        # 检查是否找到碰撞点
        for node in visited_from_source.intersection(visited_from_target):
            total_dist = distance_from_source[node] + distance_from_target[node]
            path_from_source = []
            current = node
            while current is not None:
                path_from_source.append(current)
                current = parents_from_source[current]
            path_from_source.reverse()

            path_from_target = []
            current = node
            while current is not None:
                path_from_target.append(current)
                current = parents_from_target[current]

            path = path_from_source + path_from_target[1:]  # 拼接路径，去掉重复的碰撞点
            yield node, total_dist, path

def find_k_collision_points(G, source, target, k, weight='weight'):
    """找到k个不同的碰撞点"""
    shortest_path_length = nx.shortest_path_length(G, source, target, weight=weight)
    collision_points = []
    for node, total_dist, path in bidirectional_dijkstra(G, source, target, weight):
        if total_dist > 1.1 * shortest_path_length and total_dist <= 1.5 * shortest_path_length:
            collision_points.append((node, total_dist, path))
        if len(collision_points) >= k:
            break
    # 按total_dist排序
    collision_points.sort(key=lambda x: x[1])
    return collision_points[:k]


#%% 读取graph_sc
with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

#%% 轨迹数据
with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

for i in range(len(test_data)):
    test_data[i] = (test_data[i][1])

test_data = test_data[:100]

#%%
# 计算两条轨迹的相似度，这里使用了一个简单的方法，即计算两条轨迹的交集的长度
# 也可以使用其他的方法，例如基于编辑距离或者循环神经网络等
def similarity(traj1, traj2):
    return len(set(traj1).intersection(set(traj2)))

def precision_recall_f1_score(pred_path, orig_path):
    precision = len(set(pred_path) & set(orig_path)) / len(set(pred_path))
    recall = len(set(pred_path) & set(orig_path)) / len(set(orig_path))
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score

#%%
k = 3
idx = 0
traj = test_data[idx]
source = traj[0]
target = traj[-1]
# 最短路和最短路长度
shortest_path = nx.shortest_path(G, source, target, weight='weight')
shortest_path_length = nx.shortest_path_length(G, source, target, weight='weight')
print('Shortest path length:', shortest_path_length)
top_k_collisions = find_k_collision_points(G, source, target, k)
for node, total_dist, path in top_k_collisions:
    print(f"碰撞点: {node}, 总长度: {total_dist}, 路径: {path}")


#%% 检查是否有碰撞点在最短路径上
for collision, total_dist in top_k_collisions:
    if collision in shortest_path:
        print('Collision point {} is on the shortest path'.format(collision))
    else:
        print('Collision point {} is not on the shortest path'.format(collision))

#%% 检查是否有碰撞点在轨迹上
for collision, total_dist in top_k_collisions:
    if collision in traj:
        print('Collision point {} is on the trajectory'.format(collision))
    else:
        print('Collision point {} is not on the trajectory'.format(collision))