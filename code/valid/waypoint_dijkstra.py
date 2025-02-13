#%%
# 导入一些必要的库
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt

#%%
import geopandas as gpd
from shapely.geometry import LineString
node_df = gpd.read_file('data/'+city_name+'/map/nodes.shp')
edge_df = gpd.read_file('data/'+city_name+'/map/edges.shp')

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

#%%
mid_point_list = []

for i in range(0, len(test_data)):
    traj = test_data[i]
    start_point = traj[0]
    end_point = traj[-1]

    recommended_trajectory = nx.dijkstra_path(G, start_point, end_point, weight='weight')
    mid_point_list.append(recommended_trajectory[len(recommended_trajectory) // 2])

#%% 保存中间点
with open('data/'+city_name+'/mid_point_list_dijkstra.pkl', 'wb') as f:
    pickle.dump(mid_point_list, f)
    f.close()