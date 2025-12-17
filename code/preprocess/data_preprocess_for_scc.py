#%% 读取节点和边，构造图
import geopandas as gpd

import sys
import os

# 获取 code 文件夹的路径
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 code 文件夹路径添加到 sys.path
sys.path.append(code_dir)

from config import get_config

config, _ = get_config()

# 读取节点和边
city_name = config.city

node_df = gpd.read_file('data/'+city_name+'/map/nodes.shp')
edge_df = gpd.read_file('data/'+city_name+'/map/edges.shp')

# 构造图
import networkx as nx

G = nx.DiGraph()
for index, row in edge_df.iterrows():
    G.add_edge(row['u'], row['v'], weight=row['length'])

# 查看图是否是强连通图，如果不是，找出最大强连通子图
if not nx.is_strongly_connected(G):
    print('Graph is not strongly connected')
    scs = list(nx.strongly_connected_components(G))
    max_sc = max(scs, key=len)
    G = G.subgraph(max_sc).copy()
    print('Max sc extracted')

#%%
# 保存图
import pickle

with open('data/'+city_name+'/graph_sc.pkl', 'wb') as f:
    pickle.dump(G, f)
    f.close()

#%%
# 读取图
with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

#%%
# 读取数据集

import pickle

with open('data/'+city_name+'/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

with open('data/'+city_name+'/preprocessed_test_trips_small_osmid.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

with open('data/'+city_name+'/preprocessed_validation_trips_small_osmid.pkl', 'rb') as f:
    valid_data = pickle.load(f)
    f.close()

#%%

# 检查数据集中的节点是否都在图中，如果不在，则删除该轨迹，形成新的数据集

def check_data_in_graph(data):
    new_data = []
    for item in data:
        trip = item[1]
        if all([G.has_node(node) for node in trip]):
            new_data.append(item)
    print(f'Original data size: {len(data)}, new data size: {len(new_data)}')
    return new_data

train_data = check_data_in_graph(train_data)
test_data = check_data_in_graph(test_data)
valid_data = check_data_in_graph(valid_data)

#%%
# 保存新的数据集

with open('data/'+city_name+'/train_data_small_sc.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    f.close()

with open('data/'+city_name+'/test_data_small_sc.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    f.close()

with open('data/'+city_name+'/valid_data_small_sc.pkl', 'wb') as f:
    pickle.dump(valid_data, f)
    f.close()