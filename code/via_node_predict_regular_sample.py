import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
from tqdm import tqdm
import os
from config import get_config
import matplotlib.pyplot as plt


class RankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, valid_outputs, valid_labels):
        # 转换为float类型
        valid_outputs = valid_outputs.float()
        valid_labels = valid_labels.float()

        # 处理空情况
        if (valid_labels == 1).sum() == 0 or (valid_labels == 0).sum() == 0:
            return torch.tensor(0.0, device=valid_outputs.device, requires_grad=True)

        # 获取正负样本
        positive_mask = (valid_labels == 1)
        negative_mask = (valid_labels == 0)

        if not positive_mask.any() or not negative_mask.any():
            return torch.tensor(0.0, device=valid_outputs.device, requires_grad=True)

        positive_scores = valid_outputs[positive_mask]
        negative_scores = valid_outputs[negative_mask]

        # 计算每个正样本与所有负样本的loss
        positive_scores = positive_scores.unsqueeze(1)  # [P, 1]
        negative_scores = negative_scores.unsqueeze(0)  # [1, N]

        # 计算margin loss，添加数值稳定性检查
        loss = torch.clamp(self.margin - positive_scores + negative_scores, min=0.0)

        # 检查是否有有效的loss值
        if loss.numel() == 0:
            return torch.tensor(0.0, device=valid_outputs.device, requires_grad=True)

        # 计算平均loss
        loss = loss.mean()

        # 检查loss是否为nan
        if torch.isnan(loss):
            return torch.tensor(0.0, device=valid_outputs.device, requires_grad=True)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

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
    def __init__(self, node_dim, partition_dim, m=128, dropout=0.3):  # 增加dropout
        super().__init__()
        self.query_encoder = QueryEncoder(node_dim, partition_dim, m, dropout=dropout)
        self.classification_head = ClassificationHead(m, dropout=dropout)
        self.node_embedding = nn.Embedding(node_dim, m)
        self.dropout = nn.Dropout(dropout)  # 添加dropout层

    def forward(self, s_idx, t_idx, s_part, t_part, candidate_idx):
        query_encoded = self.query_encoder(s_idx, t_idx, s_part, t_part)
        candidate_emb = self.dropout(self.node_embedding(candidate_idx))  # 应用dropout
        query_encoded = self.dropout(query_encoded).unsqueeze(1)
        query_encoded = query_encoded.expand(-1, candidate_emb.size(1), -1)
        combined = query_encoded * candidate_emb
        scores = self.classification_head(combined).squeeze(-1)
        return scores


def train_model(model, train_loader, val_loader, device, num_epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # # 修改学习率调度器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.5,
    #     patience=5,
    #     verbose=True
    # )
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(gamma=2)
    # 使用新的损失函数
    criterion = RankingLoss(margin=1.0)

    best_val_acc = 0
    patience = 10
    patience_counter = 0
    history = {
        'train_acc': [], 'val_acc': [],
        'sample_train_acc': [], 'sample_val_acc': [],
        'train_loss': [], 'val_loss': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        sample_train_correct = 0
        sample_train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            s_idx, t_idx, s_part, t_part, candidate_indices, labels, valid_len = [b.to(device) for b in batch]

            optimizer.zero_grad()

            # 按batch处理所有候选点
            batch_size = s_idx.size(0)
            outputs = model(s_idx, t_idx, s_part, t_part, candidate_indices)

            # 只计算有效候选点的loss
            batch_loss = 0
            for i in range(batch_size):
                valid_outputs = outputs[i, :valid_len[i]]
                valid_labels = labels[i, :valid_len[i]]

                loss = criterion(valid_outputs, valid_labels)
                batch_loss += loss

                # 计算准确率（只考虑有效候选点）
                pred = (valid_outputs > 0).float()
                is_correct = (pred == valid_labels)
                train_correct += is_correct.sum().item()
                train_total += valid_len[i]

                # 计算sample accuracy
                best_idx = torch.argmax(valid_outputs)
                if valid_labels[best_idx] == 1:
                    sample_train_correct += 1
                sample_train_total += 1

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += batch_loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        sample_val_correct = 0
        sample_val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                s_idx, t_idx, s_part, t_part, candidate_indices, labels, valid_len = [b.to(device) for b in batch]
                batch_size = s_idx.size(0)
                outputs = model(s_idx, t_idx, s_part, t_part, candidate_indices)

                for i in range(batch_size):
                    valid_outputs = outputs[i, :valid_len[i]]
                    valid_labels = labels[i, :valid_len[i]]

                    loss = criterion(valid_outputs, valid_labels)
                    val_loss += loss.item()

                    # 计算准确率（只考虑有效候选点）
                    pred = (valid_outputs > 0).float()
                    is_correct = (pred == valid_labels)
                    val_correct += is_correct.sum().item()
                    val_total += valid_len[i]

                    # 计算sample accuracy
                    best_idx = torch.argmax(valid_outputs)
                    if valid_labels[best_idx] == 1:
                        sample_val_correct += 1
                    sample_val_total += 1

        train_acc = train_correct / train_total
        sample_train_acc = sample_train_correct / sample_train_total
        val_acc = val_correct / val_total
        sample_val_acc = sample_val_correct / sample_val_total

        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
        print(f"Sample-level Train Acc = {sample_train_acc:.4f}, Val Acc = {sample_val_acc:.4f}")
        print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if sample_val_acc > best_val_acc:
            best_val_acc = sample_val_acc
            torch.save(model.state_dict(), 'param/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(sample_val_acc)

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['sample_train_acc'].append(sample_train_acc)
        history['sample_val_acc'].append(sample_val_acc)
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))

    return model, history


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    sample_correct = 0
    sample_total = 0

    with torch.no_grad():
        for batch in data_loader:
            s_idx, t_idx, s_part, t_part, candidate_indices, labels, valid_len = [b.to(device) for b in batch]
            outputs = model(s_idx, t_idx, s_part, t_part, candidate_indices)

            batch_size = s_idx.size(0)
            for i in range(batch_size):
                valid_outputs = outputs[i, :valid_len[i]]
                valid_labels = labels[i, :valid_len[i]]

                # 计算二分类准确率
                pred = (valid_outputs > 0).float()
                correct += (pred == valid_labels).sum().item()
                total += valid_len[i]

                # 计算sample accuracy
                best_idx = torch.argmax(valid_outputs)
                if valid_labels[best_idx] == 1:
                    sample_correct += 1
                sample_total += 1

    binary_accuracy = correct / total if total > 0 else 0
    sample_accuracy = sample_correct / sample_total if sample_total > 0 else 0

    return binary_accuracy, sample_accuracy

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
    node_list = sorted(list(graph.nodes()))
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # 创建分区映射
    node_to_partition = {}
    for j, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = j
    num_partitions = len(partitions)

    MAX_CANDIDATES = 32  # padding到32个候选点
    processed_samples = []

    for i in range(len(raw_data)):
        trajectory = raw_data[i]
        candidates = candidate_list[i]
        flags = flag_list[i]

        if len(trajectory) < 2:
            continue

        s = trajectory[0]
        t = trajectory[-1]

        try:
            # 创建基础向量
            s_idx = torch.zeros(num_nodes, device=device)
            t_idx = torch.zeros(num_nodes, device=device)
            s_part = torch.zeros(num_partitions, device=device)
            t_part = torch.zeros(num_partitions, device=device)

            # 设置源点和目标点
            s_idx[node_to_index[s]] = 1
            t_idx[node_to_index[t]] = 1
            s_part[node_to_partition.get(s, 0)] = 1
            t_part[node_to_partition.get(t, 0)] = 1

            # 收集该样本的所有候选点
            candidate_indices = []
            valid_labels = []

            for candidate, flag in zip(candidates, flags):
                try:
                    candidate_idx = node_to_index[candidate]
                    candidate_indices.append(candidate_idx)
                    valid_labels.append(flag)
                except KeyError:
                    continue

            if not candidate_indices:  # 跳过没有有效候选点的样本
                continue

            # Padding到32
            num_valid = len(candidate_indices)
            padding_size = MAX_CANDIDATES - num_valid
            if padding_size > 0:
                candidate_indices.extend([candidate_indices[0]] * padding_size)  # 用第一个候选点填充
                valid_labels.extend([0] * padding_size)  # padding的标签为0

            # 转换为tensor，将标签转换为float类型
            candidate_tensor = torch.tensor(candidate_indices, device=device)
            labels_tensor = torch.tensor(valid_labels, dtype=torch.float, device=device)  # 添加dtype=torch.float

            processed_samples.append((
                s_idx,
                t_idx,
                s_part,
                t_part,
                candidate_tensor,
                labels_tensor,
                num_valid  # 记录实际有效的候选点数量
            ))

        except KeyError as e:
            print(f"Warning: Node {e} not found in graph, skipping trajectory {i}")
            continue

    if not processed_samples:
        raise ValueError("No valid data after processing")

    print(f"Processed {len(processed_samples)} valid samples")
    return processed_samples

def plot_training_history(history, save_path):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 5))

    # 转换tensor到numpy
    history_np = {
        key: [x.cpu().numpy() if torch.is_tensor(x) else x for x in values]
        for key, values in history.items()
    }

    # Binary Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history_np['train_acc'], label='Train Binary Acc')
    plt.plot(history_np['val_acc'], label='Val Binary Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Accuracy')
    plt.legend()

    # Sample Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history_np['sample_train_acc'], label='Train Sample Acc')
    plt.plot(history_np['sample_val_acc'], label='Val Sample Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Sample Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 3, 3)
    plt.plot(history_np['train_loss'], label='Train Loss')
    plt.plot(history_np['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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

    node_list = sorted(list(data['graph'].nodes()))
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # 训练模型
    print("Training model...")
    model, history = train_model(model, train_loader, val_loader, device, num_epochs=100)

    # 绘制训练历史
    plot_training_history(history, f'results/{config.city}_training_history.png')

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
            s_idx, t_idx, s_part, t_part, candidate_indices, labels, valid_len = [b.to(device) for b in batch]
            outputs = model(s_idx, t_idx, s_part, t_part, candidate_indices)

            batch_size = s_idx.size(0)
            for i in range(batch_size):
                valid_outputs = outputs[i, :valid_len[i]]
                valid_candidates = candidate_indices[i, :valid_len[i]]

                # 获取得分最高的候选点索引
                best_idx = torch.argmax(valid_outputs)
                selected_point = valid_candidates[best_idx].item()
                # selected_point需要转换为原始节点ID
                selected_point = node_list[selected_point]
                test_predictions.append(selected_point)

    # 打印一下预测结果
    print(f"Predictions for first 10 test samples: {test_predictions[:10]}")

    # 保存预测结果
    with open(f'preprocessed/{config.city}/test_selected_points.pkl', 'wb') as f:
        pickle.dump(test_predictions, f)

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('param', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('preprocessed', exist_ok=True)

    # 运行主程序
    main()