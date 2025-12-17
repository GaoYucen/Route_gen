#%%
import networkx as nx
from tqdm import tqdm
import itertools
import time

route_num = 10

#%%
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
for link in link_df:
    weight = round(link[3], 2)
    G.add_edge(link[1], link[2], weight=weight)

#%% 读取前traj_small.txt
def travel_read_traj(file_name):
    traj_df = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            traj_df.append([int(item) for item in line.split(',')])

    return traj_df

traj_osmid_small = travel_read_traj('data/traj_osmid_small.txt')

#%% 取前100个轨迹
traj_osmid = traj_osmid_small[0:1]

#%% 评估指标
# 计算两条轨迹的相似度，这里使用了一个简单的方法，即计算两条轨迹的交集的长度
# 也可以使用其他的方法，例如基于编辑距离或者循环神经网络等
def similarity(traj1, traj2):
    return len(set(traj1).intersection(set(traj2)))

def precision_recall_f1_score(pred_path, orig_path):
    precision = len(set(pred_path) & set(orig_path)) / len(set(pred_path))
    recall = len(set(pred_path) & set(orig_path)) / len(set(orig_path))
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score

#%% 双向Dijkstra
import heapq

k = 3

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
    seen_nodes = set()  # 用于跟踪已经找到的碰撞点

    while queue_from_source and queue_from_target:
        # 从源点方向扩展
        dist_from_source, node_from_source = heapq.heappop(queue_from_source)
        if node_from_source in visited_from_source:
            continue
        visited_from_source.add(node_from_source)
        for start_node, neighbor, edge_attr in G.out_edges(node_from_source, data=True):
            if neighbor in visited_from_source:
                continue
            new_dist = dist_from_source + edge_attr.get(weight, 1)
            if new_dist < distance_from_source[neighbor]:
                distance_from_source[neighbor] = new_dist
                parents_from_source[neighbor] = node_from_source
                heapq.heappush(queue_from_source, (new_dist, neighbor))

        # 从目标点方向扩展
        dist_from_target, node_from_target = heapq.heappop(queue_from_target)
        if node_from_target in visited_from_target:
            continue
        visited_from_target.add(node_from_target)
        for neighbor, end_node, edge_attr in G.in_edges(node_from_target, data=True):
            if neighbor in visited_from_target:
                continue
            new_dist = dist_from_target + edge_attr.get(weight, 1)
            if new_dist < distance_from_target[neighbor]:
                distance_from_target[neighbor] = new_dist
                parents_from_target[neighbor] = node_from_target
                heapq.heappush(queue_from_target, (new_dist, neighbor))

        # 检查是否找到碰撞点
        for node in visited_from_source.intersection(visited_from_target):
            if node in seen_nodes:
                continue  # 如果节点已经在seen_nodes中，则跳过
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
            seen_nodes.update(path)  # 将路径上的所有节点添加到seen_nodes中

def find_k_collision_points(G, source, target, k, weight='weight'):
    """找到k个不同的碰撞点"""
    shortest_path_length = nx.shortest_path_length(G, source, target, weight=weight)
    # print('Shortest path length:', shortest_path_length)
    collision_points = []
    seen_coll_nodes = set()  # 用于跟踪已经找到的碰撞点
    count = 0
    for node, total_dist, path in bidirectional_dijkstra(G, source, target, weight):
        # print('Node:', node, 'Total distance:', total_dist, 'Path:', path)
        # count += 1
        if node not in seen_coll_nodes and total_dist > 1 * shortest_path_length and total_dist <= 1.2 * shortest_path_length:
            collision_points.append((node, total_dist, path))
            seen_coll_nodes.add(node)
            count += 1
            print('count:', count)
        if len(collision_points) >= k:
            break
        # if count > 20:
        #     break
    # 按total_dist排序
    collision_points.sort(key=lambda x: x[1])
    # 如果碰撞点数量小于k，重复最后一个碰撞点到k个
    # if len(collision_points) < k:
    #     for i in range(k - len(collision_points)):
    #         collision_points.append(collision_points[-1])
    return collision_points[:k]

#%% 评估
traj_osmid = traj_osmid_small[0:1]

k = 3
sim_list = []
precision_list = []
recall_list = []
f1_score_list = []
sim_mid_list = []
precision_mid_list = []
recall_mid_list = []
f1_score_mid_list = []
# 创造对于三个候选点记录结果的列表
similar_list_mid_1 = []
similar_list_mid_2 = []
similar_list_mid_3 = []
precision_list_mid_1 = []
precision_list_mid_2 = []
precision_list_mid_3 = []
recall_list_mid_1 = []
recall_list_mid_2 = []
recall_list_mid_3 = []
f1_score_list_mid_1 = []
f1_score_list_mid_2 = []
f1_score_list_mid_3 = []

# for idx in tqdm(range(len(traj_osmid))):
idx = 0
start_point = traj_osmid[idx][0]
end_point = traj_osmid[idx][-1]
mid_point = traj_osmid[idx][len(traj_osmid[idx])//2]
print('Original path:', traj_osmid[idx])
# 计算实走轨迹长度
real_path_length = 0
for i in range(len(traj_osmid[idx]) - 1):
    real_path_length += G[traj_osmid[idx][i]][traj_osmid[idx][i + 1]]['weight']
print('Real path length:', real_path_length)
# 添加traj_osmid[0]的中点到候选点中
# candidate_list.append(traj_osmid[idx][len(traj_osmid[idx])//2])
recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
print('Dijkstra')
print('Recommended trajectory:', recommended_trajectory)
# 计算推荐轨迹长度
recommended_path_length = 0
for i in range(len(recommended_trajectory) - 1):
    recommended_path_length += G[recommended_trajectory[i]][recommended_trajectory[i + 1]]['weight']
print('Recommended path length:', recommended_path_length)
sim = similarity(recommended_trajectory, traj_osmid[idx])
precision, recall, f1_score = precision_recall_f1_score(recommended_trajectory, traj_osmid[idx])
sim_list.append(sim)
precision_list.append(precision)
recall_list.append(recall)
f1_score_list.append(f1_score)

# print("Similarity:", sim)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
# print("Path:", recommended_trajectory)
# print("")

# 以候选点作为中间点，分别计算两段路径
recommended_trajectory_1 = nx.dijkstra_path(G, start_point, mid_point, weight='weight')
recommended_trajectory_2 = nx.dijkstra_path(G, mid_point, end_point, weight='weight')
recommended_trajectory_path = recommended_trajectory_1 + recommended_trajectory_2[1:]
print('Mid point')
print('Recommended trajectory:', recommended_trajectory_path)
sim = similarity(recommended_trajectory_path, traj_osmid[idx])
precision, recall, f1_score = precision_recall_f1_score(recommended_trajectory_path, traj_osmid[idx])
sim_mid_list.append(sim)
precision_mid_list.append(precision)
recall_mid_list.append(recall)
f1_score_mid_list.append(f1_score)
# print("Similarity:", sim)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
# print("Mid Point:", mid_point)
# print("Path:", recommended_trajectory_path)
# print("")

#%%
# traj_osmid = traj_osmid_small[3:4]
# print('Original path:', traj_osmid[0])
# idx = 0
# start_point = traj_osmid[idx][0]
# end_point = traj_osmid[idx][-1]
k = 2000

start_point = 9000023747051
end_point = 559961
top_k_collisions = find_k_collision_points(G, start_point, end_point, k)
# print('Top k collision points:', top_k_collisions)
candidate_list = [collision[0] for collision in top_k_collisions]
candidate_list_length = [collision[1] for collision in top_k_collisions]
candidate_list_path = [collision[2] for collision in top_k_collisions]

print('candidate_list:', candidate_list)

recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')

# 检验碰撞点是否在真实轨迹上
for candidate_node in candidate_list:
    # if candidate_node in traj_osmid[idx]:
    if candidate_node in recommended_trajectory:
        print('Collision point {} is on the shortest path'.format(candidate_node))

#%%
print(candidate_list_length[-1])

#%%
# 以候选点作为中间点，分别计算两段路径
for i in range(k):
    # recommended_trajectory_1 = nx.dijkstra_path(G, start_point, candidate_list[i], weight='weight')
    # recommended_trajectory_2 = nx.dijkstra_path(G, candidate_list[i], end_point, weight='weight')
    # recommended_trajectory_path = recommended_trajectory_1 + recommended_trajectory_2[1:]
    recommended_trajectory_path = candidate_list_path[i]
    print('Mid point', i + 1)
    print('Recommended trajectory:', recommended_trajectory_path)
    sim = similarity(recommended_trajectory_path, traj_osmid[idx])
    precision, recall, f1_score = precision_recall_f1_score(recommended_trajectory_path, traj_osmid[idx])
    if i == 0:
        similar_list_mid_1.append(sim)
        precision_list_mid_1.append(precision)
        recall_list_mid_1.append(recall)
        f1_score_list_mid_1.append(f1_score)
    elif i == 1:
        similar_list_mid_2.append(sim)
        precision_list_mid_2.append(precision)
        recall_list_mid_2.append(recall)
        f1_score_list_mid_2.append(f1_score)
    elif i == 2:
        similar_list_mid_3.append(sim)
        precision_list_mid_3.append(precision)
        recall_list_mid_3.append(recall)
        f1_score_list_mid_3.append(f1_score)
    # print("Similarity:", sim)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1_score)
    # print("Mid Point:", candidate_list[i])
    # print("Path:", recommended_trajectory_path)
    # print("")

# 打印结果
print("Dijkstra:")
print("Similarity:", sum(sim_list) / len(sim_list))
print("Precision:", sum(precision_list) / len(precision_list))
print("Recall:", sum(recall_list) / len(recall_list))
print("F1 Score:", sum(f1_score_list) / len(f1_score_list))
print("")
print("Mid Point:")
print("Similarity:", sum(sim_mid_list) / len(sim_mid_list))
print("Precision:", sum(precision_mid_list) / len(precision_mid_list))
print("Recall:", sum(recall_mid_list) / len(recall_mid_list))
print("F1 Score:", sum(f1_score_mid_list) / len(f1_score_mid_list))
print("")
print("Mid Point 1:")
print("Similarity:", sum(similar_list_mid_1) / len(similar_list_mid_1))
print("Precision:", sum(precision_list_mid_1) / len(precision_list_mid_1))
print("Recall:", sum(recall_list_mid_1) / len(recall_list_mid_1))
print("F1 Score:", sum(f1_score_list_mid_1) / len(f1_score_list_mid_1))
print("")
print("Mid Point 2:")
print("Similarity:", sum(similar_list_mid_2) / len(similar_list_mid_2))
print("Precision:", sum(precision_list_mid_2) / len(precision_list_mid_2))
print("Recall:", sum(recall_list_mid_2) / len(recall_list_mid_2))
print("F1 Score:", sum(f1_score_list_mid_2) / len(f1_score_list_mid_2))
print("")
print("Mid Point 3:")
print("Similarity:", sum(similar_list_mid_3) / len(similar_list_mid_3))
print("Precision:", sum(precision_list_mid_3) / len(precision_list_mid_3))
print("Recall:", sum(recall_list_mid_3) / len(recall_list_mid_3))
print("F1 Score:", sum(f1_score_list_mid_3) / len(f1_score_list_mid_3))

