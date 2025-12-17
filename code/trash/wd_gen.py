#%% 读取轨迹数据
import pandas as pd
import numpy as np

trajs = []
with open('new_code/trajs/traj.txt', 'r') as f:
    for line in f.readlines():
        trajs.append([int(i) for i in line.split(',')])
    f.close()

trajs_num = 1000
trajs_all = trajs[0:trajs_num]
print(len(trajs))

#%% 统计起点数量和终点数量
start_count = {}
end_count = {}
for traj in trajs_all:
    start_count[traj[0]] = start_count.get(traj[0], 0) + 1
    end_count[traj[-1]] = end_count.get(traj[-1], 0) + 1
# 排序
start_count = sorted(start_count.items(), key=lambda x: x[1], reverse=True)
end_count = sorted(end_count.items(), key=lambda x: x[1], reverse=True)

#%% 取出最后5%的起点和终点形成list
start_list = [i[0] for i in start_count[int(len(start_count)*0.95):]]
end_list = [i[0] for i in end_count[int(len(end_count)*0.95):]]

#%% 计算不包含start_list和end_list的trajs
trajs = []
trajs_2 = []
for traj in trajs_all:
    if traj[0] not in start_list and traj[-1] not in end_list:
        trajs.append(traj)
    else:
        trajs_2.append(traj)

#%% 最长的traj
print(max([len(i) for i in trajs]))

#%% 统计trajs中有多少个不同的数字
links = set([j for i in trajs for j in i])
print(len(links))

#%% 统计每个数字出现的次数
link_count = {}
for i in links:
    link_count[i] = 0
for traj in trajs:
    for link in traj:
        link_count[link] += 1

#%% 找到出现次数最多的前100个数字
k = 150
link_topk = sorted(link_count.items(), key=lambda x: x[1], reverse=True)[:k]

# #%% 读取csm_all.txt
# csm_all = []
# with open('new_code/data/csm_all.txt', 'r') as f:
#     for line in f.readlines():
#         csm_all.append([int(i) for i in line.split(' ')])
#     f.close()

#%% links中最大的元素
max_link = max(links)
print(max_link)

#%% 针对trajs构造training data，输入为trajs的起点和终点及link_top100中的link，输出为0，1标签，0表示traj不经过该link，1表示traj经过该link
import random
import torch
from torch.utils.data import Dataset

data = []

for traj in trajs:
    start = traj[0]
    end = traj[-1]
    label = [0 for i in range(k)]
    for i, link in enumerate(link_topk):
        if link[0] in traj:
            label[i] = 1
    for i in range(k):
        data.append((start, end, link_topk[i][0], label[i]))

#%% 检测data中每100个数据第四列包含1的频率
cnt = 0
for i in range(0, len(data), k):
    if sum([data[j][3] for j in range(i, i+k)]) > 0:
        cnt += 1
print(cnt/len(trajs))

#%% 检测data中第四列是1的频率
cnt = 0
for i in data:
    if i[3] == 1:
        cnt += 1
print(cnt/len(data))

#%%
training_data = []
for i, item in enumerate(data):
    start_vec = np.zeros(max_link)
    end_vec = np.zeros(max_link)
    link_vec = np.zeros(max_link)
    start_vec[int(item[0])] = 1
    end_vec[int(item[1])] = 1
    link_vec[int(item[2])] = 1
    input = np.concatenate((start_vec, end_vec, link_vec))
    training_data.append([input, item[3]])

def get_batch(p1, p2):
    x_batch = np.zeros(((p2-p1), 3 * max_link))
    y_batch = np.zeros(((p2-p1),))
    z = 0
    for i in range(p1, p2):
        start_vec = np.zeros(max_link)
        end_vec = np.zeros(max_link)
        link_vec = np.zeros(max_link)
        start_vec[int(data[i][0])-1] = 1
        end_vec[int(data[i][1])-1] = 1
        link_vec[int(data[i][2])-1] = 1
        input = np.concatenate((start_vec, end_vec, link_vec))
        x_batch[z] = input
        y_batch[z] = data[i][3]
        z += 1
    return x_batch, y_batch

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

#%% 构造mlp分类器，输入为trajs的起点和终点及link_top100中的link，输出为0，1标签，0表示traj不经过该link，1表示traj经过该link
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def predict(self, data, batch_size):
        score = 0
        for i in tqdm(range(0, len(data), batch_size)):
            batch_x, batch_y = get_batch(i, i + batch_size)
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
            y_pred = self.forward(batch_x)
            y_pred = torch.squeeze(y_pred, 1)
            for j in range(len(batch_y)):
                if y_pred[j] > 0.5 and batch_y[j] == 1:
                    score += 1/k
                elif y_pred[j] <= 0.5 and batch_y[j] == 0:
                    score += 1/k
        return score

    def predict_2(self, data, batch_size):
        score = 0
        score_2 = 0
        max_cnt = 0
        for i in tqdm(range(0, len(data), batch_size)):
            batch_x, batch_y = get_batch(i, i + batch_size)
            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
            y_pred = self.forward(batch_x)
            y_pred = torch.squeeze(y_pred, 1)
            j = torch.argmax(y_pred)
            if batch_y[j] == 1:
                score += 1
            # 如果batch_y中有1，max_cnt加1
            if sum(batch_y) > 0:
                max_cnt += 1
            for j in range(len(batch_y)):
                if y_pred[j] > 0.5 and batch_y[j] == 1:
                    score_2 += 1/k
                elif y_pred[j] <= 0.5 and batch_y[j] == 0:
                    score_2 += 1/k
        return score, max_cnt, score_2
    #
    # def predict_proba(self, x):
    #     x = self.forward(x)
    #     return F.softmax(x, dim=1)
    #
    # def score(self, x, y):
    #     y_pred = self.predict(x)
    #     return torch.mean((y_pred == y).float())

    def fit(self, data, epochs, batch_size, lr, weight_decay):
        min_loss = 1000
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for i in tqdm(range(0, len(data), batch_size)):
                optimizer.zero_grad()
                batch_x, batch_y = get_batch(i, i+batch_size)
                batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
                batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
                y_pred = self.forward(batch_x)
                y_pred = torch.squeeze(y_pred, 1)
                loss = criterion(y_pred, batch_y)
                # if loss.item() < min_loss:
                #     min_loss = loss.item()
                #     torch.save(self.state_dict(), 'new_code/param/mlp_'+str(trajs_num)+'_'+str(k)+'.pth')
                loss.backward()
                optimizer.step()
            print("epoch: {}, loss: {}".format(epoch, loss.item()))

#%%
batch_size_tmp = k

#%%
mlp = MLP(3 * max_link, 100, 1).to(device)
# mlp.load_state_dict(torch.load('new_code/param/mlp_'+str(trajs_num)+'_'+str(k)+'.pth'))
# mlp.load_state_dict(torch.load('new_code/param/mlp_'+str(trajs_num)+'_'+str(k)+'.pth'))
mlp.fit(data, 20, batch_size_tmp, 0.01, 0.0001)
torch.save(mlp.state_dict(), 'new_code/param/mlp_'+str(trajs_num)+'_'+str(k)+'.pth')

#%% 读取模型参数
mlp = MLP(3 * max_link, 100, 1).to(device)
mlp.load_state_dict(torch.load('new_code/param/mlp_'+str(trajs_num)+'_'+str(k)+'.pth'))

#%%
print('start predict')
score, max_cnt, score_2 = mlp.predict_2(data, batch_size_tmp)
print(score, max_cnt, score/max_cnt, score_2)

# #%%
# print('start predict')
# score_2 = mlp.predict(data, batch_size_tmp)
# print(score_2)