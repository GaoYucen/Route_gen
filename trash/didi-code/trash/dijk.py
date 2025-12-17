#%%
import networkx as nx
from tqdm import tqdm
import itertools
import time

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

traj_osmid = travel_read_traj('data/traj_osmid_small.txt')

#%% 取前100个轨迹
traj_osmid = traj_osmid[:100]

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

def bidirectional_dijkstra_top_k(G, source, target, k):
    # 正向搜索队列和距离字典
    forward_queue = [(0, source)]
    forward_distances = {source: 0}
    forward_visited = set()
    # 反向搜索队列和距离字典
    reverse_queue = [(0, target)]
    reverse_distances = {target: 0}
    reverse_visited = set()
    # 存储前 k 个碰撞点及其距离
    top_k_collisions = []

    while forward_queue and reverse_queue:
        # 正向扩展
        forward_distance, forward_node = heapq.heappop(forward_queue)
        if forward_node in forward_visited:
            continue
        forward_visited.add(forward_node)
        # 检查是否已经到达目标节点
        if forward_node in reverse_visited:
            total_distance = forward_distance + reverse_distances[forward_node]
            heapq.heappush(top_k_collisions, (total_distance, forward_node))
            if len(top_k_collisions) > k:
                heapq.heappop(top_k_collisions)
        for neighbor in G.successors(forward_node):
            new_distance = forward_distance + G[forward_node][neighbor]['weight']
            if neighbor not in forward_distances or new_distance < forward_distances[neighbor]:
                forward_distances[neighbor] = new_distance
                heapq.heappush(forward_queue, (new_distance, neighbor))

        # 反向扩展
        reverse_distance, reverse_node = heapq.heappop(reverse_queue)
        if reverse_node in reverse_visited:
            continue
        reverse_visited.add(reverse_node)
        if reverse_node in forward_visited:
            total_distance = reverse_distance + forward_distances[reverse_node]
            heapq.heappush(top_k_collisions, (total_distance, reverse_node))
            if len(top_k_collisions) > k:
                heapq.heappop(top_k_collisions)
        for neighbor in G.predecessors(reverse_node):
            new_distance = reverse_distance + G[neighbor][reverse_node]['weight']
            if neighbor not in reverse_distances or new_distance < reverse_distances[neighbor]:
                reverse_distances[neighbor] = new_distance
                heapq.heappush(reverse_queue, (new_distance, neighbor))

    return [node for _, node in top_k_collisions]

#%% 评估
k = 3
sim_list = []
precision_list = []
recall_list = []
f1_score_list = []
sim_mid_list = []
precision_mid_list = []
recall_mid_list = []
f1_score_mid_list = []
for idx in tqdm(range(len(traj_osmid))):
    start_point = traj_osmid[idx][0]
    end_point = traj_osmid[idx][-1]
    mid_point = traj_osmid[idx][len(traj_osmid[idx])//2]
    # candidate_list = bidirectional_dijkstra_top_k(G, start_point, end_point, k)
    # 添加traj_osmid[0]的中点到候选点中
    # candidate_list.append(traj_osmid[idx][len(traj_osmid[idx])//2])
    recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
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
