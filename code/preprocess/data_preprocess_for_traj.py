# %% 读取轨迹数据
import pickle

# %% 边序列转化为点osmid序列
import geopandas as gpd

import sys
import os

# 获取 code 文件夹的路径
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 code 文件夹路径添加到 sys.path
sys.path.append(code_dir)

from config import get_config

config, _ = get_config()

city_name = config.city

# node_df = gpd.read_file('data/'+city_name+'/map/nodes.shp')
edge_df = gpd.read_file('data/'+city_name+'/map/edges.shp')

def edge2osmid(edges_seq):
    osmids = []
    for edge_idx in edges_seq:
        osmids.append(edge_df.iloc[edge_idx]['u'])
    osmids.append(edge_df.iloc[edges_seq[-1]]['v'])
    return osmids

name_list = ['train', 'test', 'validation']

for name in name_list:

    with open('data/'+city_name+'/preprocessed_'+name+'_trips_all.pkl', 'rb') as f:
        train_data_origin = pickle.load(f)
        f.close()

    train_data = []
    for idx, edges_seq, timestamps in train_data_origin:
        train_data.append([[idx[0], idx[1], timestamps[0], timestamps[1]], edge2osmid(edges_seq)])

    # %% 保存处理后的数据
    with open('data/'+city_name+'/preprocessed_'+name+'_trips_all_osmid.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        f.close()

