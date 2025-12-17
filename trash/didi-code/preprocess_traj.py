#%%
import networkx as nx
from tqdm import tqdm

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

#%%
# 使用networkx构造路网
G = nx.DiGraph()
for link in link_df:
    # 将 link[3] 的权重值保留两位小数
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

#%%
traj_osmid_ok = []
for idx in tqdm(range(len(traj_osmid_small))):
    start_point = traj_osmid_small[idx][0]
    end_point = traj_osmid_small[idx][-1]
    # try:
    recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
    traj_osmid_ok.append(traj_osmid_small[idx])
    # except:
    #     continue

#%%
idx = 1
start_point = traj_osmid_small[idx][0]
end_point = traj_osmid_small[idx][-1]
recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
# recommended_trajectory = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
# recommended_trajectory = nx.astar_path(G, start_point, end_point, heuristic=None, weight='weight')

#%% 存储traj_osmid_ok
with open('data/traj_osmid_ok.txt', 'w') as f:
    for traj in traj_osmid_ok:
        f.write(','.join([str(item) for item in traj])+'\n')


