import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import networkx as nx
import pickle

import sys
import os

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(code_dir)

from config import get_config

class AttentionLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by number of heads"

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.o_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask = None):      # x: seq_len
        query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Split the embedding into self.heads different pieces
        query = query.reshape(-1, self.heads, self.head_dim).transpose(-2, -3)
        key = key.reshape(-1, self.heads, self.head_dim).transpose(-2, -3)
        value = value.reshape(-1, self.heads, self.head_dim).transpose(-2, -3)  # (heads, value_len, head_dim)

        energy = torch.matmul(query, key.transpose(-1, -2))  # (heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)    # (heads, query_len, key_len)

        out = torch.matmul(attention, value)  # (heads, query_len, head_dim)
        out = out.transpose(-2, -3).reshape(-1, self.heads * self.head_dim)

        out = self.o_proj(out)          
        return out

class MlpLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MlpLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))
        return x

class ViaNodePredcitonModel(nn.Module):           # Via-Node-Prediction
    def __init__(self, config):
        super(ViaNodePredcitonModel, self).__init__()
        self.attn = AttentionLayer(
            embed_size = config["embedding_size"], 
            heads = config["num_heads"]
        )
        self.mlp = MlpLayer(
            input_size = config["node_embedding_size"], 
            hidden_size = config["hidden_size"], 
            output_size = config["node_embedding_size"]
        )
        self.norm1 = nn.LayerNorm(config["embedding_size"])
        self.norm2 = nn.LayerNorm(config["embedding_size"])

        with open('data/'+config["city_name"]+'/graph_sc.pkl', 'rb') as f:     # 读取一个有向图路网
            self.G = pickle.load(f)
            f.close()
        self.node_to_index = {node: index for index, node in enumerate(self.G.nodes())}

        self.node_embeddings = nn.Embedding(config["num_nodes"], config["node_embedding_size"])
        self.partition_embeddings = nn.Embedding(config["num_partitions"], config["node_embedding_size"])
        

        
    def compute_loss(self, logits, labels, T = 1.0):
        soft_targets = torch.nn.functional.softmax(labels / T, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(logits / T, dim=-1)
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / (soft_prob.size()[0]) * (T**2)
        return soft_targets_loss
    
    def forward(self, source, target, source_partition, target_partition, via_nodes, labels):
        num_via_nodes = len(via_nodes)

        source_index, target_index = self.node_to_index[source], self.node_to_index[target]
        via_node_indices = [self.node_to_index[via_node] for via_node in via_nodes]

        source_embedding, target_embedding = self.node_embeddings(torch.tensor(source_index)), self.node_embeddings(torch.tensor(target_index))
        via_node_embeddings = [self.node_embeddings(torch.tensor(via_node_index)) for via_node_index in via_node_indices]
        source_partition_embedding, target_partition_embedding = self.partition_embeddings(torch.tensor(source_partition)), self.partition_embeddings(torch.tensor(target_partition))

        all_embeddings = [source_embedding, target_embedding] + [source_partition_embedding, target_partition_embedding] + via_node_embeddings
        input_embedding = torch.stack(all_embeddings, dim=0)

        hidden_states = input_embedding

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = residual + self.attn(hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)      # (seq_len, node_embeddings)

        cosine_similarity = torch.nn.functional.cosine_similarity(hidden_states[-num_via_nodes:], input_embedding[-num_via_nodes:], dim=-1).unsqueeze(0)        # (1, num_via_nodes)
        logits = torch.nn.functional.softmax(cosine_similarity)
        loss = self.compute_loss(logits, labels)  

        return logits, loss

def get_network(city_name):              # 读取一个有向图路网, 分区和边界节点
    with open('data/'+city_name+'/graph_sc.pkl', 'rb') as f:
        G = pickle.load(f)
        f.close()
    
    with open('data/'+city_name+'/partitions.pkl', 'rb') as f:
        partitions = pickle.load(f)
        f.close()
    
    with open('data/'+city_name+'/boundary_nodes.pkl', 'rb') as f:
        boundary_nodes = pickle.load(f)
        f.close()

    return G, partitions[6], boundary_nodes

def get_train_data(city_name, num_samples):               # 读取candidate_list和on_traj_flag_list数据
    with open('data/'+city_name+'/candidate_list.pkl', 'rb') as f:
        candidate_list = pickle.load(f)
        f.close()

    with open('data/'+city_name+'/on_traj_flag_list.pkl', 'rb') as f:
        on_traj_flag_list = pickle.load(f)
        f.close()

    with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
        traj_data = pickle.load(f)
        f.close()
    
    for i in range(len(traj_data)):
        traj_data[i] = (traj_data[i][1])

    train_data = []
    for i in range(num_samples):
        source, target = traj_data[i][0], traj_data[i][-1]
        source_partition_index = None
        target_partition_index = None
        for j, partition in enumerate(partitions):
            if source in partition:
                source_partition_index = j
            if target in partition:
                target_partition_index = j
            if source_partition_index is not None and target_partition_index is not None:
                break

        via_nodes = []
        labels = []

        for candidate, on_traj_flag in zip(candidate_list[i], on_traj_flag_list[i]):
            via_node = candidate
            label = 1 if on_traj_flag else 0
            via_nodes.append(via_node)
            labels.append(label)
        
        labels = torch.tensor(labels).unsqueeze(0)
        if torch.sum(labels) == 0:
            continue
        labels = labels/torch.sum(labels)

        train_data.append((source, target, source_partition_index, target_partition_index, via_nodes, labels))

    return train_data

def get_test_data(city_name, start_sample_idx, end_sample_idx):  
    with open('data/'+city_name+'/candidate_list.pkl', 'rb') as f:
        candidate_list = pickle.load(f)
        f.close()

    with open('data/'+city_name+'/on_traj_flag_list.pkl', 'rb') as f:
        on_traj_flag_list = pickle.load(f)
        f.close()

    with open('data/'+city_name+'/test_data_small_sc.pkl', 'rb') as f:
        traj_data = pickle.load(f)
        f.close()
    
    for i in range(len(traj_data)):
        traj_data[i] = (traj_data[i][1])

    test_data = []
    for i in range(start_sample_idx, end_sample_idx):
        source, target = traj_data[i][0], traj_data[i][-1]
        source_partition_index = None
        target_partition_index = None
        for j, partition in enumerate(partitions):
            if source in partition:
                source_partition_index = j
            if target in partition:
                target_partition_index = j
            if source_partition_index is not None and target_partition_index is not None:
                break

        via_nodes = []
        labels = []

        for candidate, on_traj_flag in zip(candidate_list[i], on_traj_flag_list[i]):
            via_node = candidate
            label = 1 if on_traj_flag else 0
            via_nodes.append(via_node)
            labels.append(label)
        
        labels = torch.tensor(labels).unsqueeze(0)
        if torch.sum(labels) == 0:
            continue
        labels = labels/torch.sum(labels)

        test_data.append((source, target, source_partition_index, target_partition_index, via_nodes, labels))

    return test_data

# def cal_accuary(logits, labels):
#     logits, labels = logits.squeeze(0), labels.squeeze(0)
#     positive_indices = torch.nonzero(labels)
#     num_positive = len(positive_indices)
#     _, predicted_top_n = torch.topk(logits, num_positive)
#     correct_predictions = torch.sum(torch.isin(predicted_top_n, positive_indices)).item()
#     return correct_predictions / num_positive

def cal_accuary(logits, labels):
    logits, labels = logits.squeeze(0), labels.squeeze(0)
    positive_indices = torch.nonzero(labels)
    _, predicted = torch.max(logits, dim=0)
    if predicted in positive_indices:
        return 1
    else:
        return 0


if __name__ == "__main__":
    config, _ = get_config()

    # 读取一个有向图路网, 分区和边界节点
    G, partitions, boundary_nodes = get_network(city_name=config.city)      
    train_data = get_train_data(
        city_name=config.city, 
        num_samples=config.num_samples
    )
    test_data = get_test_data(
        city_name=config.city, 
        start_sample_idx=config.test_sample_start_index, 
        end_sample_idx=config.test_sample_end_index
    )

    model_config = {
        "embedding_size": 64, 
        "node_embedding_size": 64,
        "num_heads": 1,
        "hidden_size": 192,
        "num_nodes": len(G.nodes()),
        "num_partitions" : len(partitions),  
        "city_name": config.city,
    }
    
    model = ViaNodePredcitonModel(config=model_config)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    min_loss = float('inf')  # 初始化最小损失为正无穷
    best_model_state = None  # 用于存储最佳模型的状态
    no_improvement_count = 0  # 记录损失值没有改善的轮数

    for epoch in range(config.num_epochs):
        total_train_loss = 0
        total_train_acc = 0
        total_test_loss = 0
        total_test_acc = 0

        model.train()
        for data in train_data:
            source, target, source_partition_index, target_partition_index, via_nodes, labels = data
            optimizer.zero_grad()
            logits, loss = model(source, target, source_partition_index, target_partition_index, via_nodes, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_acc += cal_accuary(logits, labels)
        
        model.eval()
        for data in test_data:
            source, target, source_partition_index, target_partition_index, via_nodes, labels = data
            logits, loss = model(source, target, source_partition_index, target_partition_index, via_nodes, labels)
            total_test_loss += loss.item()
            total_test_acc += cal_accuary(logits, labels)

        train_loss, test_loss = total_train_loss / len(train_data), total_test_loss / len(test_data)
        train_acc, test_acc = total_train_acc / len(train_data), total_test_acc / len(test_data)

        print(f'Epoch {epoch + 1}/{config.num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%, '
              f'Val Loss: {test_loss:.4f}, Val Accuracy: {test_acc * 100:.2f}%')
        
        if train_loss < min_loss:
            min_loss = train_loss
            best_model_state = model.state_dict()  # 保存当前模型的状态
            no_improvement_count = 0  # 损失有改善，重置计数
        else:
            no_improvement_count += 1  # 损失没有改善，计数加1

        if no_improvement_count >= config.patience:
            print(f'Early stopping triggered at epoch {epoch + 1}.')
            break

    # 保存损失最小的模型参数
    if best_model_state is not None:
        torch.save(best_model_state, 'param/chengdu/via_node_prediction_model_kai.pth')

    
