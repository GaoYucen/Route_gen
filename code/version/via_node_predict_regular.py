import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from tqdm import tqdm
import os
from config import get_config
import matplotlib.pyplot as plt


class ViaNodeDataset(Dataset):
    def __init__(self, data, device):
        self.device = device
        # 直接使用已处理的数据，不需要额外处理
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# [保持 QueryEncoder, ClassificationHead, ViaNodePredictor 类不变]
class QueryEncoder(nn.Module):
    def __init__(self, node_dim, partition_dim, m, dropout=0.3):
        super().__init__()
        embed_dim = 128  # Fixed embedding dimension

        # Projections to match embed_dim
        self.node_projection = nn.Linear(node_dim, embed_dim)
        self.partition_projection = nn.Linear(partition_dim, embed_dim)

        # Attention with correct dimensions
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
            dropout=dropout
        )

        # Final projection layers
        self.bert = nn.Sequential(
            nn.Linear(4 * embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, m),
            nn.LayerNorm(m)
        )

        self.residual = nn.Linear(4 * embed_dim, m)

    def forward(self, s_idx, t_idx, s_part, t_part):
        # Project inputs to embed_dim
        s_idx = self.node_projection(s_idx)      # [batch, embed_dim]
        t_idx = self.node_projection(t_idx)      # [batch, embed_dim]
        s_part = self.partition_projection(s_part)  # [batch, embed_dim]
        t_part = self.partition_projection(t_part)  # [batch, embed_dim]

        # Concatenate and reshape for attention
        x = torch.cat([s_idx, t_idx, s_part, t_part], dim=1)  # [batch, 4*embed_dim]
        x = x.view(x.size(0), 4, -1)  # [batch, 4, embed_dim]

        # Self-attention
        attn_output, _ = self.attention(x, x, x)  # [batch, 4, embed_dim]

        # Flatten and process
        flat = attn_output.reshape(attn_output.size(0), -1)  # [batch, 4*embed_dim]
        main_output = self.bert(flat)
        residual_output = self.residual(flat)

        return main_output + residual_output


class ClassificationHead(nn.Module):
    def __init__(self, m, dropout=0.3):  # 增加dropout率
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(m, 256),
            nn.LayerNorm(256),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(x)

class ViaNodePredictor(nn.Module):
    def __init__(self, node_dim, partition_dim, m=128):
        super().__init__()
        self.query_encoder = QueryEncoder(node_dim, partition_dim, m)
        self.classification_head = ClassificationHead(m)
        self.node_embedding = nn.Embedding(node_dim, m)  # 添加节点嵌入层

    def forward(self, s_idx, t_idx, s_part, t_part, candidate_idx):
        # 获取候选点的嵌入
        candidate_emb = self.node_embedding(candidate_idx)

        # 获取查询编码
        query_encoded = self.query_encoder(s_idx, t_idx, s_part, t_part)

        # 将查询编码与候选点嵌入结合
        combined = query_encoded * candidate_emb

        # 通过分类头
        return self.classification_head(combined)

    def batch_forward(self, s_idx, t_idx, s_part, t_part, candidate_idx):
        # 已经是批处理格式，直接前向传播
        return self.forward(s_idx, t_idx, s_part, t_part, candidate_idx)

def train_model(model, train_loader, val_loader, device, num_epochs=100):
    optimizer = optim.AdamW(  # 使用AdamW优化器
        model.parameters(),
        lr=0.001,
        weight_decay=0.01  # 增加L2正则化
    )

    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    criterion = nn.BCEWithLogitsLoss()

    # 早停设置
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    best_model_path = 'best_model.pt'

    # 记录训练历史
    history = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            s_idx, t_idx, s_part, t_part, candidate_idx, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(s_idx, t_idx, s_part, t_part, candidate_idx)

            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            pred = (outputs > 0).float()
            train_correct += (pred == labels.unsqueeze(1)).sum().item()
            train_total += labels.numel()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                # 修改这里以匹配数据格式
                s_idx, t_idx, s_part, t_part, candidate_idx, labels = [b.to(device) for b in batch]

                outputs = model(s_idx, t_idx, s_part, t_part, candidate_idx)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

                pred = (outputs > 0).float()
                val_correct += (pred == labels.unsqueeze(1)).sum().item()
                val_total += labels.numel()

        # 计算指标
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    return model, history


def load_data(city_name):
    """Load all required data files"""
    data_files = {
        'graph': f'data/{city_name}/graph_sc.pkl',
        'partitions': f'preprocessed/{city_name}/partitions.pkl',
        'train_data': f'preprocessed/{city_name}/train_data_samples.pkl',
        'val_data': f'preprocessed/{city_name}/valid_data_samples.pkl',
        'test_data': f'preprocessed/{city_name}/test_data_samples.pkl',
        'train_candidate_list': f'preprocessed/{city_name}/train_candidate_list.pkl',
        'train_on_traj_flag_list': f'preprocessed/{city_name}/train_on_traj_flag_list.pkl',
        'val_candidate_list': f'preprocessed/{city_name}/valid_candidate_list.pkl',
        'val_on_traj_flag_list': f'preprocessed/{city_name}/valid_on_traj_flag_list.pkl',
        'test_candidate_list': f'preprocessed/{city_name}/test_candidate_list.pkl',
        'test_on_traj_flag_list': f'preprocessed/{city_name}/test_on_traj_flag_list.pkl'
    }

    data = {}
    for key, path in data_files.items():
        with open(path, 'rb') as f:
            data[key] = pickle.load(f)
    return data


def prepare_batch_data(raw_data, candidate_list, flag_list, partitions, graph, device):
    """将原始数据转换为batch格式，包含候选点信息"""
    # 创建节点到索引的映射
    node_list = sorted(list(graph.nodes()))
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # 创建分区映射
    node_to_partition = {}
    for j, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = j
    num_partitions = len(partitions)

    processed_data = []
    for i in range(len(raw_data)):
        trajectory = raw_data[i]
        candidates = candidate_list[i]
        flags = flag_list[i]

        if len(trajectory) < 2:
            continue

        s = trajectory[0]
        t = trajectory[-1]

        try:
            s_idx_num = node_to_index[s]
            t_idx_num = node_to_index[t]
            s_partition = node_to_partition.get(s, 0)
            t_partition = node_to_partition.get(t, 0)

            # 创建输入向量
            s_idx = torch.zeros(num_nodes, device=device)
            t_idx = torch.zeros(num_nodes, device=device)
            s_part = torch.zeros(num_partitions, device=device)
            t_part = torch.zeros(num_partitions, device=device)

            s_idx[s_idx_num] = 1
            t_idx[t_idx_num] = 1
            s_part[s_partition] = 1
            t_part[t_partition] = 1

            # 处理每个候选点
            for candidate, flag in zip(candidates, flags):
                try:
                    candidate_idx = node_to_index[candidate]
                    processed_data.append((
                        s_idx.clone(),
                        t_idx.clone(),
                        s_part.clone(),
                        t_part.clone(),
                        candidate_idx,  # 直接使用候选点的索引
                        torch.tensor(flag, dtype=torch.float, device=device)  # 标签
                    ))
                except KeyError:
                    continue

        except KeyError as e:
            print(f"Warning: Node {e} not found in graph, skipping trajectory {i}")
            continue

    if not processed_data:
        raise ValueError("No valid data after processing")

    print(f"Processed {len(processed_data)} valid samples")
    return processed_data


def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    binary_correct = 0  # 原有的二分类正确数
    binary_total = 0  # 原有的二分类总数
    sample_correct = 0  # 新增的样本正确数
    sample_total = 0  # 新增的样本总数
    predictions_dict = {}  # 存储每个样本的预测结果

    with torch.no_grad():
        for batch in data_loader:
            s_idx, t_idx, s_part, t_part, candidate_idx, labels = [b.to(device) for b in batch]
            outputs = model(s_idx, t_idx, s_part, t_part, candidate_idx)

            # 计算原有的二分类准确率
            pred = (outputs > 0).float()
            binary_correct += (pred == labels.unsqueeze(1)).sum().item()
            binary_total += labels.numel()

            # 对batch中的每个样本进行处理
            for i in range(len(s_idx)):
                # 使用节点向量的和作为样本标识符
                s_id = s_idx[i].nonzero().item()
                t_id = t_idx[i].nonzero().item()
                sample_key = (s_id, t_id)

                score = outputs[i].item()
                label = labels[i].item()
                candidate = candidate_idx[i].item()

                if sample_key not in predictions_dict:
                    predictions_dict[sample_key] = []
                predictions_dict[sample_key].append((score, label, candidate))

    binary_accuracy = binary_correct / binary_total if binary_total > 0 else 0

    # 计算样本级别的准确率
    correct_samples = 0
    total_samples = len(predictions_dict)

    for sample_predictions in predictions_dict.values():
        # 找到得分最高的候选点
        best_prediction = max(sample_predictions, key=lambda x: x[0])
        # 检查最高分候选点是否为真实via-node
        if best_prediction[1] == 1:  # 如果label是1
            correct_samples += 1

    sample_accuracy = correct_samples / total_samples if total_samples > 0 else 0

    return binary_accuracy, sample_accuracy


def plot_training_history(history, save_path):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # 加载配置
    config, _ = get_config()

    # 设置设备
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    print("Loading data...")
    data = load_data(config.city)

    # 准备数据集
    print("Preparing datasets...")

    print("Processing training data...")
    train_data = prepare_batch_data(
        data['train_data'],
        data['train_candidate_list'],
        data['train_on_traj_flag_list'],
        data['partitions'][6],
        data['graph'],
        device
    )

    print("Processing validation data...")
    val_data = prepare_batch_data(
        data['val_data'],
        data['val_candidate_list'],
        data['val_on_traj_flag_list'],
        data['partitions'][6],
        data['graph'],
        device
    )

    # 创建数据加载器
    train_dataset = ViaNodeDataset(train_data, device)
    val_dataset = ViaNodeDataset(val_data, device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 获取模型参数
    num_nodes = len(data['graph'].nodes())
    num_partitions = len(data['partitions'][6])

    # 创建模型
    print("Creating model...")
    d = num_nodes  # embedding dimension 修改为节点数量
    m = 128  # query dimension
    print(f"Model dimensions: d={d}, m={m}, num_nodes={num_nodes}")
    model = ViaNodePredictor(
        node_dim=num_nodes,
        partition_dim=num_partitions,
        m=m
    ).to(device)

    # 训练模型
    print("Training model...")
    model, history = train_model(model, train_loader, val_loader, device)

    # 绘制训练历史
    plot_training_history(history, f'results/{config.city}_training_history.png')

    node_list = sorted(list(data['graph'].nodes()))
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'node_to_index': node_to_index,
        'config': {
            'd': d,
            'm': m,
            'num_nodes': num_nodes,
            'num_partitions': num_partitions
        }
    }, f'param/{config.city}_final_model.pth')

    # 加载保存的模型参数
    print("Loading model parameters...")
    checkpoint_path = f'param/{config.city}_final_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model parameters loaded successfully")
    else:
        print("No saved model found, training new model...")
        model, history = train_model(model, train_loader, val_loader, device)

    # 评估测试集
    print("Evaluating on test set...")
    test_data = prepare_batch_data(
        data['test_data'],
        data['test_candidate_list'],
        data['test_on_traj_flag_list'],
        data['partitions'][6],
        data['graph'],  # 添加图数据
        device
    )
    test_dataset = ViaNodeDataset(test_data, device)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Important: set shuffle=False

    binary_acc, sample_acc = evaluate_model(model, test_loader, device)
    print(f'Binary Classification accuracy: {binary_acc * 100:.2f}%')
    print(f'Sample-level accuracy: {sample_acc * 100:.2f}%')

    # 生成并保存预测结果
    test_predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            s_idx, t_idx, s_part, t_part, candidate_idx, labels = [b.to(device) for b in batch]
            outputs = model(s_idx, t_idx, s_part, t_part, candidate_idx)
            predictions = (outputs > 0).float()
            test_predictions.extend(predictions.cpu().numpy())

    # 保存预测结果
    with open(f'preprocessed/{config.city}/test_predictions.pkl', 'wb') as f:
        pickle.dump(test_predictions, f)

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('param', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('preprocessed', exist_ok=True)

    # 运行主程序
    main()