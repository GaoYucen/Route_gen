import networkx as nx
from tqdm import tqdm
import pandas as pd
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

#%% 读取轨迹
# 读取travel信息
def travel_read_ds(file_name):
    traj_df = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            columns = line.split('\t')
            # snode enode是起终点info
            traj_df.append(columns[7].split(','))
    return traj_df

# read travel
traj_df = travel_read_ds('data/traj.txt')

#%% 存储前1000条轨迹到traj_small.txt
with open('data/traj_small.txt', 'w') as f:
    for i in range(1000):
        f.write(','.join(traj_df[i])+'\n')

#%% 读取前traj_small.txt
def travel_read_traj(file_name):
    traj_df = []
    with open(file_name, 'r') as file_0:
        for line in tqdm(file_0):
            traj_df.append([int(item) for item in line.split(',')])

    return traj_df

traj_small = travel_read_traj('data/traj_small.txt')

#%% 读取边dict
edge_df = {}
with open('data/link_new.csv', 'r') as file_0:
    for line in tqdm(file_0):
        [link_id, snode, enode, length] = line.split(',')
        # snode enode是起终点info
        edge_df[int(link_id)] = [int(snode), int(enode), float(length)]

# 给edge_df添加列名['u', 'v', 'length']
edge_df = pd.DataFrame.from_dict(edge_df, orient='index')
edge_df.columns = ['u', 'v', 'length']

#%% 将边序列转化为节点序列
def edge2osmid(edges_seq):
    traj_osmid_small = []
    for i, edges_seq in enumerate(traj_small):
        try:
            osmids = []
            for edge_idx in edges_seq:
                osmids.append(edge_df.loc[edge_idx]['u'])
            osmids.append(edge_df.loc[edges_seq[-1]]['v'])
            osmids = list(map(int, osmids))
            traj_osmid_small.append(osmids)
        except KeyError:
            print('KeyError:', i)
    return traj_osmid_small

# def edge2osmid(edges_seq):
#     traj_osmid_small = []
#     for i, edges_seq in enumerate(traj_small):
#         try:
#             osmids = []
#             for edge_idx in edges_seq:
#                 # 如果edge_idx的最后一位是0，说明是u，否则是v
#                     if edge_idx % 2 == 0:
#                         osmids.append(edge_df.loc[edge_idx//10]['u'])
#                     else:
#                         osmids.append(edge_df.loc[edge_idx//10]['v'])
#
#             if edges_seq[-1] % 2 == 0:
#                 osmids.append(edge_df.loc[edges_seq[-1]//10]['u'])
#             else:
#                 osmids.append(edge_df.loc[edges_seq[-1]//10]['v'])
#             osmids = list(map(int, osmids))
#             traj_osmid_small.append(osmids)
#         except KeyError:
#             print('KeyError:', i)
#     return traj_osmid_small

traj_osmid_small = edge2osmid(traj_small)

#%% 存储前osmid到traj_small_osmid.txt
with open('data/traj_osmid_small.txt', 'w') as f:
    for i in range(len(traj_osmid_small)):
        f.write(','.join(map(str, traj_osmid_small[i]))+'\n')

# #%% 做边序列转化成点序列的连续性检验
# idx = 1
# edges_seq = traj_small[idx]
#
# try:
#     osmids = []
#     osmids_2 = []
#     for edge_idx in edges_seq:
#         osmids.append(edge_df.loc[edge_idx]['u'])
#     osmids.append(edge_df.loc[edges_seq[-1]]['v'])
#     osmids = list(map(int, osmids))
#     osmids_2.append(edge_df.loc[edges_seq[0]]['u'])
#     for edge_idx in edges_seq:
#         osmids_2.append(edge_df.loc[edge_idx]['v'])
#     osmids_2 = list(map(int, osmids_2))
# except KeyError:
#     print('KeyError')
#
# print('osmids:', osmids)
# print('osmids_2:', osmids_2)
#
# # 对比osmid和osmid_2是否一样，给出不一样的地方
# def compare_lists(osmid, osmid_2):
#     if len(osmid)!= len(osmid_2):
#         print("列表长度不同，osmid 长度为", len(osmid), "，osmid_2 长度为", len(osmid_2))
#         return
#
#     differences = []
#     for i in range(len(osmid)):
#         if osmid[i]!= osmid_2[i]:
#             differences.append((i, osmid[i], osmid_2[i]))
#
#     if differences:
#         print("以下位置的元素不同:")
#         for index, value1, value2 in differences:
#             print(f"索引 {index}: osmid 中的值为 {value1}，osmid_2 中的值为 {value2}")
#     else:
#         print("两个列表完全相同。")
#
#
# # 示例列表
# compare_lists(osmids, osmids_2)

