#%% 读取graph_sc.pkl
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

# 读取节点和边
city_name = config.city

with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

#%% #%% 计算node_df每个node的后继节点
from collections import defaultdict

node_nbrs = defaultdict(set)

for i in range(len(G.nodes())):
    node = list(G.nodes())[i]
    node_nbrs[node] = set(G.successors(node))

node_nbrs = dict(node_nbrs)

# #%%
# print(G.nodes)
#
# #%% 计算node_df每个node的后继节点
# from collections import defaultdict
#
# node_nbrs = defaultdict(set)
#
# for i in range(len(edge_df)):
#     u = edge_df.iloc[i]['u']
#     v = edge_df.iloc[i]['v']
#     node_nbrs[u].add(v)
#
# node_nbrs = dict(node_nbrs)

#%% 存储node_nbrs
with open('data/'+city_name+'/node_nbrs_sc.pkl', 'wb') as f:
    pickle.dump(node_nbrs, f)
    f.close()



