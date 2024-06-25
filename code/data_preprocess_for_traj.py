# %% 读取轨迹数据
import pickle

# %% 边序列转化为点osmid序列
import geopandas as gpd

# node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')


def edge2osmid(edges_seq):
    osmids = []
    for edge_idx in edges_seq:
        osmids.append(edge_df.iloc[edge_idx]['u'])
    osmids.append(edge_df.iloc[edges_seq[-1]]['v'])
    return osmids

name_list = ['train', 'test', 'validation']

for name in name_list:

    with open('data/chengdu_data/preprocessed_'+name+'_trips_small.pkl', 'rb') as f:
        train_data_origin = pickle.load(f)
        f.close()

    train_data = []
    for idx, edges_seq, timestamps in train_data_origin:
        train_data.append((idx, edge2osmid(edges_seq), timestamps))

    # %% 保存处理后的数据
    with open('data/chengdu_data/preprocessed_'+name+'_trips_small_osmid.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        f.close()