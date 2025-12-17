import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 读取图和分区信息
# 读取数据
import networkx as nx
import pickle

# 读取一个有向图路网
with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
    G = pickle.load(f)
    f.close()

# 读取分区和边界节点
with open('data/'+city_name+'/partitions.pkl', 'rb') as f:
    partitions = pickle.load(f)
    f.close()

with open('data/'+city_name+'/boundary_nodes.pkl', 'rb') as f:
    boundary_nodes = pickle.load(f)
    f.close()

#%%
partitions = partitions[6]

#%%
# 定义嵌入维度
d = 64
m = 64

# 建立节点到索引的映射
node_to_index = {node: index for index, node in enumerate(G.nodes())}
num_nodes = len(G.nodes())
num_partitions = len(partitions)

# 生成节点和分区的嵌入
node_embeddings = torch.randn(num_nodes, d)
partition_embeddings = torch.randn(num_partitions, d)

# 查询编码模块
class QueryEncoder(nn.Module):
    def __init__(self, d, m):
        super(QueryEncoder, self).__init__()
        self.bert = nn.Sequential(
            nn.Linear(4 * d, 256),
            nn.ReLU(),
            nn.Linear(256, m)
        )

    def forward(self, s_emb, t_emb, s_partition_emb, t_partition_emb):
        input_emb = torch.cat([s_emb, t_emb, s_partition_emb, t_partition_emb], dim=1)
        query_emb = self.bert(input_emb)
        return query_emb

# 分类头模块
class ClassificationHead(nn.Module):
    def __init__(self, d, m):
        super(ClassificationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(m + d, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, query_emb, via_node_emb):
        input_emb = torch.cat([query_emb, via_node_emb], dim=1)
        output = self.mlp(input_emb)
        return output

# Via - node 预测模型
class ViaNodePredictionModel(nn.Module):
    def __init__(self, d, m):
        super(ViaNodePredictionModel, self).__init__()
        self.query_encoder = QueryEncoder(d, m)
        self.classification_head = ClassificationHead(d, m)

    def forward(self, s, t, s_partition, t_partition, via_node):
        s_index = node_to_index[s]
        t_index = node_to_index[t]
        via_node_index = node_to_index[via_node]
        s_emb = node_embeddings[s_index].unsqueeze(0)
        t_emb = node_embeddings[t_index].unsqueeze(0)
        s_partition_emb = partition_embeddings[s_partition].unsqueeze(0)
        t_partition_emb = partition_embeddings[t_partition].unsqueeze(0)
        via_node_emb = node_embeddings[via_node_index].unsqueeze(0)

        query_emb = self.query_encoder(s_emb, t_emb, s_partition_emb, t_partition_emb)
        output = self.classification_head(query_emb, via_node_emb)
        return output

#%%
# 生成训练数据
# 读取candidate_list和on_traj_flag_list数据
with open('data/'+city_name+'/candidate_list.pkl', 'rb') as f:
    candidate_list = pickle.load(f)
    f.close()

with open('data/'+city_name+'/on_traj_flag_list.pkl', 'rb') as f:
    on_traj_flag_list = pickle.load(f)
    f.close()

with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
    traj_data = pickle.load(f)
    f.close()

# 提取轨迹数据
for i in range(len(traj_data)):
    traj_data[i] = (traj_data[i][1])

# 取前100
traj_data = traj_data[:100]

train_data = []
for i in range(len(candidate_list)):
    s = traj_data[i][0]
    t = traj_data[i][-1]
    # 确认 s 和 t 所属的分区
    s_partition_index = None
    t_partition_index = None
    for j, partition in enumerate(partitions):
        if s in partition:
            s_partition_index = j
        if t in partition:
            t_partition_index = j
        if s_partition_index is not None and t_partition_index is not None:
            break
    for candidate, on_traj_flag in zip(candidate_list[i], on_traj_flag_list[i]):
        via_node = candidate
        label = 1 if on_traj_flag else 0
        train_data.append((s, t, s_partition_index, t_partition_index, via_node, label))

#%%
# 初始化模型
model = ViaNodePredictionModel(d, m)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for s, t, s_partition_index, t_partition_index, via_node, label in train_data:
        optimizer.zero_grad()
        output = model(s, t, s_partition_index, t_partition_index, via_node)
        label_tensor = torch.tensor([label], dtype=torch.long)
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data)}')

#%% 保存模型
torch.save(model.state_dict(), 'param/chengdu/via_node_prediction_model.pth')

#%% 测试模型
# 读取模型
model = ViaNodePredictionModel(d, m)
model.load_state_dict(torch.load('param/chengdu/via_node_prediction_model.pth'))
model.eval()  # 将模型设置为评估模式
correct = 0
total = len(train_data)

with torch.no_grad():  # 不计算梯度，节省计算资源
    for s, t, s_partition_index, t_partition_index, via_node, label in train_data:
        output = model(s, t, s_partition_index, t_partition_index, via_node)
        _, predicted = torch.max(output.data, 1)  # 获取预测的类别
        correct += (predicted.item() == label)  # 统计预测正确的样本数

accuracy = correct / total
print(f"模型在测试集上的准确率: {accuracy * 100:.2f}%")

#%%
test_data = []
for i in range(len(candidate_list)):
    s = traj_data[i][0]
    t = traj_data[i][-1]
    # 确认 s 和 t 所属的分区
    s_partition_index = None
    t_partition_index = None
    for j, partition in enumerate(partitions):
        if s in partition:
            s_partition_index = j
        if t in partition:
            t_partition_index = j
        if s_partition_index is not None and t_partition_index is not None:
            break
    via_node_list = []
    for candidate, on_traj_flag in zip(candidate_list[i], on_traj_flag_list[i]):
        via_node = candidate
        via_node_list.append(via_node)
    test_data.append((s, t, s_partition_index, t_partition_index, via_node_list))

#%%
print(candidate_list[0])
print(on_traj_flag_list[0])
print(test_data[0])

#%%
model.eval()  # 将模型设置为评估模式

selected_points = []
with torch.no_grad():  # 不计算梯度，节省计算资源
    for s, t, s_partition_index, t_partition_index, candidate_nodes in test_data:
        probabilities = []
        for via_node in candidate_nodes:
            output = model(s, t, s_partition_index, t_partition_index, via_node)
            # 假设输出是一个二维张量，形状为 (1, 2)，第二个元素是是正确途经点的概率
            probability = output[0][1].item()
            probabilities.append(probability)
        # 找到概率最大的候选点的索引
        max_index = probabilities.index(max(probabilities))
        # 根据索引选择对应的候选点
        selected_point = candidate_nodes[max_index]
        selected_points.append(selected_point)

print("每个轨迹数据中概率最大的候选途经点:", selected_points)

#%% 保存结果
with open('data/'+city_name+'/selected_points.pkl', 'wb') as f:
    pickle.dump(selected_points, f)
    f.close()

#%% 判断selected_points在traj的比例
selected_points_ratio = []
for selected_point, traj in zip(selected_points, traj_data):
    if selected_point in traj:
        selected_points_ratio.append(1)
    else:
        selected_points_ratio.append(0)

print('Selected points ratio:', sum(selected_points_ratio) / len(selected_points_ratio))



