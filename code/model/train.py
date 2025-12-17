#%% 读取轨迹数据
import pickle
import numpy as np

from tqdm import tqdm
import torch
from model import Model
import random

import sys
import os

# 获取 code 文件夹的路径
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将 code 文件夹路径添加到 sys.path
sys.path.append(code_dir)

from config import get_config

config, _ = get_config()

with open('data/'+city_name+'/train_data_small_sc.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

with open('data/'+city_name+'/valid_data_small_sc.pkl', 'rb') as f:
    valid_data = pickle.load(f)
    f.close()

with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

# 构建数据集，只保留点序列数据
for i in range(len(train_data)):
    train_data[i] = (train_data[i][1])

for i in range(len(valid_data)):
    valid_data[i] = (valid_data[i][1])

for i in range(len(test_data)):
    test_data[i] = (test_data[i][1])

#%% 读取节点嵌入
with open('data/'+city_name+'/node_embedding_sc.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

#%% 添加key为-1的embedding，指定dtype为float32
node_embeddings[-1] = np.array([0] * len(node_embeddings[288416374])).astype(np.float32)

#%% 读取node_nbrs
with open('data/'+city_name+'/node_nbrs_sc.pkl', 'rb') as f:
    node_nbrs = pickle.load(f)
    f.close()

#%% 确认node_nbrs的最大尺寸
max_nbrs = 0
for node in node_nbrs:
    if len(node_nbrs[node]) > max_nbrs:
        max_nbrs = len(node_nbrs[node])

#%% 将node_nbrs长度不到max_nbrs的补充到max_nbrs长度
for node in node_nbrs:
    node_nbrs[node] = list(node_nbrs[node])
    if len(node_nbrs[node]) < max_nbrs:
        node_nbrs[node] += [-1] * (max_nbrs - len(node_nbrs[node]))

#%% 训练
num_epoches = 200
batch_size = 512

# 指定mps为device
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('device:', device)
model = Model(embedding=node_embeddings, hidden_dim=config.hidden_dimen).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# 早停参数
patience = 5  # 没有改善的连续epoch数
best_val_loss = float('inf')  # 最佳验证集损失
no_improve_epochs = 0  # 没有改善的连续epoch计数

for epoch in tqdm(range(num_epoches)):
    random.shuffle(train_data)
    epoch_loss = 0  # 用于累加每个batch的损失，记录当前epoch的总损失
    for i in range(0, len(train_data), batch_size):
        optimizer.zero_grad()
        batch = [item for item in train_data[i: i+batch_size]]          # item[1]就是一条路径
        curr = [item[j] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        start = [item[0] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        dest = [item[-1] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        nbr = [nbr for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]

        curr_embed = torch.tensor(np.array([node_embeddings[node] for node in curr]), device=device)
        start_embed = torch.tensor(np.array([node_embeddings[node] for node in start]), device=device)
        dest_embed = torch.tensor(np.array([node_embeddings[node] for node in dest]), device=device)
        nbr_embed = torch.tensor(np.array([node_embeddings[node] for node in nbr]), device=device)

        input_embed = torch.cat((start_embed, curr_embed, dest_embed, nbr_embed), dim=1).to(device)
        pred = model(input_embed)
        # 构造mask矩阵
        mask = torch.tensor([1 if nbr[i] != -1 else 0 for i in range(len(nbr))]).to(device).unsqueeze(1)
        # 将pred中对应nbr==-1的部分置为0
        pred = pred * mask
        pred = pred.view(-1, max_nbrs)
        # # 过softmax层
        # pred = torch.nn.functional.softmax(pred, dim=1)
        target = torch.tensor([node_nbrs[item[j]].index(item[j + 1]) for item in batch for j in range(len(item) - 1)]).to(device)
        loss = torch.nn.functional.cross_entropy(pred, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss_val = epoch_loss / (len(train_data) / batch_size)  # 当前epoch平均损失
    print(f'Epoch {epoch + 1} loss: {epoch_loss_val}')  # 打印当前epoch平均损失，除以batch数量得到平均

    # 检查是否需要更新最佳模型
    if epoch_loss_val < best_val_loss:
        best_val_loss = epoch_loss_val
        no_improve_epochs = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'param/rg_best_model.pth')
    else:
        no_improve_epochs += 1

    # 检查是否需要早停
    if no_improve_epochs >= patience:
        print(f'Early stopping triggered after {patience} epochs without improvement.')
        break

    # scheduler.step()

#存储模型数据
torch.save(model.state_dict(), 'param/rg.pth')
print('Model saved')