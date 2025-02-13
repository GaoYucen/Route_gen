#%%
# 导入一些必要的库
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt

#%% 双向Dijkstra
import heapq

k = 3

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
# 根据起点和终点，使用networkx库中的函数，找到一条最短路径，表示推荐的轨迹
# 这里使用了一个基于权重的方法，即选择权重最大的边作为下一步
# 也可以使用其他的方法，例如基于随机选择或者贪心算法等
similar_list = []
precision_list = []
recall_list = []
f1_score_list = []
similar_list_mid = []
precision_list_mid = []
recall_list_mid = []
f1_score_list_mid = []
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

# 计算轨迹平均长度
traj_length_list = []
for traj in test_data:
    traj_length_list.append(len(traj))
print('Trajectory length mean:', sum(traj_length_list)/len(traj_length_list))

for i in range(0, len(test_data)):
    traj = test_data[i]
    start_point = traj[0]
    end_point = traj[-1]
    mid_point = traj[len(traj) // 2]
    # candidate_list = bidirectional_dijkstra_top_k(G, start_point, end_point, k)

    recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
    # mid_point = recommended_trajectory[len(recommended_trajectory) // 2]
    recommended_trajectory_1 = nx.dijkstra_path(G, start_point, mid_point, weight='weight')
    recommended_trajectory_2 = nx.dijkstra_path(G, mid_point, end_point, weight='weight')
    recommended_trajectory_path = recommended_trajectory_1 + recommended_trajectory_2[1:]

    # 计算推荐的轨迹和原始轨迹的相似度
    similar_value = similarity(recommended_trajectory, traj)
    similar_list.append(similar_value)
    precision, recall, f1_score = precision_recall_f1_score(recommended_trajectory, traj)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)

    # 计算推荐的轨迹和原始轨迹的相似度
    similar_mid_value = similarity(recommended_trajectory_path, traj)
    similar_list_mid.append(similar_mid_value)
    precision, recall, f1_score = precision_recall_f1_score(recommended_trajectory_path, traj)
    precision_list_mid.append(precision)
    recall_list_mid.append(recall)
    f1_score_list_mid.append(f1_score)

    # for mid in candidate_list:
    #     recommended_trajectory_1 = nx.dijkstra_path(G, start_point, mid, weight='weight')
    #     recommended_trajectory_2 = nx.dijkstra_path(G, mid, end_point, weight='weight')
    #     recommended_trajectory_path = recommended_trajectory_1 + recommended_trajectory_2[1:]
    #     similar_mid_value = similarity(recommended_trajectory_path, traj)
    #     precision, recall, f1_score = precision_recall_f1_score(recommended_trajectory_path, traj)
    #     if mid == candidate_list[0]:
    #         similar_list_mid_1.append(similar_mid_value)
    #         precision_list_mid_1.append(precision)
    #         recall_list_mid_1.append(recall)
    #         f1_score_list_mid_1.append(f1_score)
    #     elif mid == candidate_list[1]:
    #         similar_list_mid_2.append(similar_mid_value)
    #         precision_list_mid_2.append(precision)
    #         recall_list_mid_2.append(recall)
    #         f1_score_list_mid_2.append(f1_score)
    #     elif mid == candidate_list[2]:
    #         similar_list_mid_3.append(similar_mid_value)
    #         precision_list_mid_3.append(precision)
    #         recall_list_mid_3.append(recall)
    #         f1_score_list_mid_3.append(f1_score)

# 计算precision, recall和F1-Score
precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
f1_score = sum(f1_score_list) / len(f1_score_list)
print('Similarity mean:', sum(similar_list)/len(similar_list))
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

precision_mid = sum(precision_list_mid) / len(precision_list_mid)
recall_mid = sum(recall_list_mid) / len(recall_list_mid)
f1_score_mid = sum(f1_score_list_mid) / len(f1_score_list_mid)
print('Similarity (mid) mean:', sum(similar_list_mid)/len(similar_list_mid))
print("Precision (mid):", precision_mid)
print("Recall (mid):", recall_mid)
print("F1-Score (mid):", f1_score_mid)

# precision = sum(precision_list_mid_1) / len(precision_list_mid_1)
# recall = sum(recall_list_mid_1) / len(recall_list_mid_1)
# f1_score = sum(f1_score_list_mid_1) / len(f1_score_list_mid_1)
# print('Similarity (mid_1) mean:', sum(similar_list_mid_1)/len(similar_list_mid_1))
# print("Precision (mid_1):", precision)
# print("Recall (mid_1):", recall)
# print("F1-Score (mid_1):", f1_score)
#
# precision = sum(precision_list_mid_2) / len(precision_list_mid_2)
# recall = sum(recall_list_mid_2) / len(recall_list_mid_2)
# f1_score = sum(f1_score_list_mid_2) / len(f1_score_list_mid_2)
# print('Similarity (mid_2) mean:', sum(similar_list_mid_2)/len(similar_list_mid_2))
# print("Precision (mid_2):", precision)
# print("Recall (mid_2):", recall)
# print("F1-Score (mid_2):", f1_score)
#
# precision = sum(precision_list_mid_3) / len(precision_list_mid_3)
# recall = sum(recall_list_mid_3) / len(recall_list_mid_3)
# f1_score = sum(f1_score_list_mid_3) / len(f1_score_list_mid_3)
# print('Similarity (mid_3) mean:', sum(similar_list_mid_3)/len(similar_list_mid_3))
# print("Precision (mid_3):", precision)
# print("Recall (mid_3):", recall)
# print("F1-Score (mid_3):", f1_score)

#%% 绘制对比similarity_list和similarity_mid_list的柱状图
import matplotlib.pyplot as plt
import numpy as np
length = 50
x = np.arange(length)
width = 0.4
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2 + 1, similar_list[:length], width, label='Similarity')
rects2 = ax.bar(x + width/2 + 1, similar_list_mid[:length], width, label='Similarity (mid)')
ax.set_ylabel('Similarity')
ax.set_title('Similarity vs Similarity with mid point')
ax.legend()
fig.tight_layout()
plt.savefig('figure/similarity.pdf')


#%% 对于计算precision, recall和F1-Score