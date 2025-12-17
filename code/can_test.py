import pickle
import networkx as nx

import sys
import os

# 获取 code 文件夹的路径
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 code 文件夹路径添加到 sys.path
sys.path.append(code_dir)

from config import get_config

config, _ = get_config()

city_name = config.city

#%% 读取graph_sc
with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()


#%% 读取test_candidate_list和test_on_traj_flag_list
with open('preprocessed/'+city_name+'/test_candidate_list.pkl', 'rb') as f:
    test_candidate_list = pickle.load(f)
    f.close()
with open('preprocessed/'+city_name+'/test_on_traj_flag_list.pkl', 'rb') as f:
    test_on_traj_flag_list = pickle.load(f)
    f.close()

#%% 读取traj_data
with open('preprocessed/'+city_name+'/test_data_samples.pkl', 'rb') as f:
    traj_data = pickle.load(f)
    f.close()

# # 提取轨迹数据
# for i in range(len(traj_data)):
#     traj_data[i] = (traj_data[i][1])

# # 取前100
# traj_data = traj_data[800:900]

#%% f1-score
def f1_score(pred_path, orig_path):
    precision = len(set(pred_path) & set(orig_path)) / len(set(pred_path))
    recall = len(set(pred_path) & set(orig_path)) / len(set(orig_path))
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

#%% 对于每一个轨迹，对比Dijkstra和途经点CRP的f1-score
CRP_f1_list = []
Dijkstra_f1_list = []

for idx in range(len(traj_data)):
    traj = traj_data[idx]
    source = traj[0]
    target = traj[-1]
    # 获取所有途经点（test_candidate_list中test_on_traj_flag_list为1的位置）
    candidate_points = test_candidate_list[idx]
    flag_list = test_on_traj_flag_list[idx]
    waypoints = [pt for pt, flag in zip(candidate_points, flag_list) if flag == 1]

    # 筛选出len(waypoints) > 0的情况，对此进行统计，否则跳过
    if len(waypoints) == 0:
        continue

    shortest_path = nx.shortest_path(G, source, target, weight='weight')

    # 如果没有途经点，则只算Dijkstra
    if len(waypoints) == 0:
        # 取candidate_list中的第一个作为途经点
        waypoint = candidate_points[0]
        CRP_path_1 = nx.shortest_path(G, source, waypoint, weight='weight')
        CRP_path_2 = nx.shortest_path(G, waypoint, target, weight='weight')
        CRP_path = CRP_path_1 + CRP_path_2[1:]
    else:
        # 多途经点时，只取第一个
        waypoint = waypoints[0]
        CRP_path_1 = nx.shortest_path(G, source, waypoint, weight='weight')
        CRP_path_2 = nx.shortest_path(G, waypoint, target, weight='weight')
        CRP_path = CRP_path_1 + CRP_path_2[1:]
    
    CRP_f1 = f1_score(CRP_path, traj)
    Dijkstra_f1 = f1_score(shortest_path, traj)
    CRP_f1_list.append(CRP_f1)
    Dijkstra_f1_list.append(Dijkstra_f1)

#%% 计算平均f1-score
CRP_f1_avg = sum(CRP_f1_list) / len(CRP_f1_list)
Dijkstra_f1_avg = sum(Dijkstra_f1_list) / len(Dijkstra_f1_list)

print(f"CRP平均f1-score: {CRP_f1_avg}")
print(f"Dijkstra平均f1-score: {Dijkstra_f1_avg}")

#%% 画图对比
import matplotlib.pyplot as plt

plt.plot(CRP_f1_list, label='CRP')
plt.plot(Dijkstra_f1_list, label='Dijkstra')
plt.legend()
plt.show()


