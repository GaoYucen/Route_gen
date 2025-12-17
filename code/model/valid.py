#%%
import geopandas as gpd
import pickle
import numpy as np
import torch
from tqdm import tqdm
from haversine import haversine
from model import Model

import sys
import os

# 获取 code 文件夹的路径
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 code 文件夹路径添加到 sys.path
sys.path.append(code_dir)

from config import get_config

config, _ = get_config()

# # 添加路径
# import sys
# sys.path.append('code')
#
# from config import model_name
# print('model_name:', model_name)

# 边信息
node_df = gpd.read_file('data/'+city_name+'/map/nodes.shp')

# 读取数据集
with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

for i in range(len(test_data)):
    test_data[i] = (test_data[i][1])

# 取前100
test_data = test_data[:900]

# 读取节点嵌入
with open('data/'+city_name+'/node_embedding_sc.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

# 添加key为-1的embedding，指定dtype为float32
node_embeddings[-1] = np.array([0] * len(node_embeddings[288416374])).astype(np.float32)

# 读取node_nbrs
with open('data/'+city_name+'/node_nbrs_sc.pkl', 'rb') as f:
    node_nbrs = pickle.load(f)
    f.close()

# 确认node_nbrs的最大尺寸
max_nbrs = 0
for node in node_nbrs:
    if len(node_nbrs[node]) > max_nbrs:
        max_nbrs = len(node_nbrs[node])

# 将node_nbrs长度不到max_nbrs的补充到max_nbrs长度
for node in node_nbrs:
    node_nbrs[node] = list(node_nbrs[node])
    if len(node_nbrs[node]) < max_nbrs:
        node_nbrs[node] += [-1] * (max_nbrs - len(node_nbrs[node]))

# 训练
batch_size = 512

# # 指定cuda为device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 指定mps为device
device = torch.device('cpu')
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
# 打印当前使用的设备
print('device', device)
model = Model(embedding=node_embeddings, hidden_dim=config.hidden_dimen).to(device)

# 加载模型参数
model.load_state_dict(torch.load('param/rg.pth'))
model.eval()  # 设置模型为评估模式

# 准备测试数据
predicted_paths = []
original_paths = []
arrival_predicted_paths = []
arrival_original_paths = []

MAX_ITERS = 300

from collections import OrderedDict

with torch.no_grad():  # 不需要梯度计算，提高速度并减少内存消耗
    num = 0
    reach_num = 0
    for i in tqdm(range(0, len(test_data), batch_size)):
        all_paths = [item for item in test_data[i:i + batch_size]]
        gens = [[t[0]] for t in all_paths] # 起点
        pending = OrderedDict({i: None for i in range(len(all_paths))})
        # 长度为all_paths的长度的全0list
        arrival_list = OrderedDict({i: None for i in range(len(all_paths))})
        # for _ in tqdm(range(MAX_ITERS), desc="generating trips in lockstep", dynamic_ncols=True):
        for _ in range(MAX_ITERS):
            true_paths = [all_paths[i] for i in pending]
            current_temp = [gens[i][-1] for i in pending]
            # print('len(pending):', len(pending))
            # print('len(current_temp):', len(current_temp))
            # print('len(true_paths):', len(true_paths))
            curr = [c for c in current_temp for _ in node_nbrs[c]]
            starts = [t[0] for c, t in zip(current_temp, true_paths) for _ in node_nbrs[c]]
            pot_next = [nbr for c in current_temp for nbr in node_nbrs[c]]
            dests = [t[-1] for c, t in zip(current_temp, true_paths) for _ in node_nbrs[c]]

            curr_embed = torch.tensor(np.array([node_embeddings[node] for node in curr])).to(device)
            start_embed = torch.tensor(np.array([node_embeddings[node] for node in starts])).to(device)
            dest_embed = torch.tensor(np.array([node_embeddings[node] for node in dests])).to(device)
            nbr_embed = torch.tensor(np.array([node_embeddings[node] for node in pot_next])).to(device)
            input_embed = torch.cat((start_embed, curr_embed, dest_embed, nbr_embed), dim=1).to(device)

            unnormalized_confidence = model(input_embed)
            mask = torch.tensor([1 if pot_next[i] != -1 else 0 for i in range(len(pot_next))]).to(device).unsqueeze(1)
            unnormalized_confidence = unnormalized_confidence * mask
            chosen = torch.argmax(unnormalized_confidence.reshape(-1, max_nbrs), dim=1)
            chosen = chosen.detach().cpu().tolist()
            pending_trip_ids = list(pending.keys())

            for identity, choice_tmp in zip(pending_trip_ids, chosen):
                choice = node_nbrs[gens[identity][-1]][choice_tmp]
                # if choice in gens[identity] or choice == -1:
                if choice in gens[identity]:
                    del pending[identity]
                    del arrival_list[identity]
                    continue
                gens[identity].append(choice)
                if choice == all_paths[identity][-1] or haversine(node_df[node_df['osmid'] == choice][['y', 'x']].values[0], node_df[node_df['osmid'] == all_paths[identity][-1]][['y', 'x']].values[0])*1000 < config.threshold:
                    reach_num += 1
                    del pending[identity]
                    continue

            if len(pending) == 0:
                break

        predicted_paths.extend([gen for gen in gens])
        original_paths.extend([path for path in all_paths])
        arrival_predicted_paths.extend([gens[i] for i in arrival_list])
        arrival_original_paths.extend([all_paths[i] for i in arrival_list])


#%% 计算抵达率
print('reach_num:', reach_num)
# arrival_rate = sum([p[-1] == t[-1] for p, t in zip(predicted_paths, original_paths)]) / (len(original_paths)-num)
# print("Arrival Rate:", arrival_rate)
print('total_num:', len(original_paths))
print('reach rate:', reach_num / len(original_paths))

# 对于抵达的路径，计算precision和recall
precision_list = []
recall_list = []
for pred_path, orig_path in zip(arrival_predicted_paths, arrival_original_paths):
    # if pred_path[-1] == orig_path[-1]:
        # print("Predicted Path:", pred_path)
        # print("Original Path:", orig_path)
    precision = len(set(pred_path) & set(orig_path)) / len(set(pred_path))
    recall = len(set(pred_path) & set(orig_path)) / len(set(orig_path))
    precision_list.append(precision)
    recall_list.append(recall)

precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
print("Precision:", precision)
print("Recall:", recall)
# 算一下F1-Score
f1_score = 2 * precision * recall / (precision + recall)
print("F1-Score:", f1_score)