#%% map数据
import geopandas as gpd

node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%%
print(node_df.head(5))
print(edge_df.iloc[0])
#
# #%% 用LineString解析node_df的geometry列
# print(node_df['geometry'].apply(lambda point: point.x).head(5))
#
#
# #%%
# print(edge_df.iloc[0]['geometry'].coords[-1][0])

#%% 计算node_df每个node的后继节点
from collections import defaultdict

node_nbrs = defaultdict(set)

for i in range(len(edge_df)):
    u = edge_df.iloc[i]['u']
    v = edge_df.iloc[i]['v']
    node_nbrs[u].add(v)

node_nbrs = dict(node_nbrs)

# 存储node_nbrs
import pickle

with open('data/chengdu_data/node_nbrs.pkl', 'wb') as f:
    pickle.dump(node_nbrs, f)
    f.close()
