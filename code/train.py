#%% 读取轨迹数据
import pickle

with open('data/chengdu_data/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

with open('data/chengdu_data/preprocessed_validation_trips_small_osmid.pkl', 'rb') as f:
    val_data = pickle.load(f)
    f.close()

with open('data/chengdu_data/preprocessed_test_trips_small_osmid.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

# 构建数据集
for i in range(len(train_data)):
    train_data[i] = ([train_data[i][1][0], train_data[i][1][-1], train_data[i][2][0]], train_data[i][1])

for i in range(len(val_data)):
    val_data[i] = ([val_data[i][1][0], val_data[i][1][-1], val_data[i][2][0]], val_data[i][1])

for i in range(len(test_data)):
    test_data[i] = ([test_data[i][1][0], test_data[i][1][-1], test_data[i][2][0]], test_data[i][1])

# #%% 构造dataloader → 需要size相同
# import torch
# from torch.utils.data import DataLoader
#
# train_dataset = DataLoader(train_data, batch_size=32, shuffle=True)
# val_dataset = DataLoader(val_data, batch_size=32, shuffle=True)
# test_dataset = DataLoader(test_data, batch_size=32, shuffle=True)

#%% 读取节点嵌入
with open('data/chengdu_data/embeddings.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

#%% 读取node_nbrs
with open('data/chengdu_data/node_nbrs.pkl', 'rb') as f:
    node_nbrs = pickle.load(f)
    f.close()

#%% 读取config中定义的参数
# 将当前目录加上/code添加到目录中
import os
import sys
sys.path.append(os.getcwd() + '/code')
import config

params, _ = config.get_config()

#%% 读取模型




