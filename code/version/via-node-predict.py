import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
import os
from tqdm import tqdm

sys.path.append('code')
from config import get_config

class QueryEncoder(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.attention = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.bert = nn.Sequential(
            nn.Linear(4 * d, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, m),
            nn.LayerNorm(m)
        )

    def forward(self, s_emb, t_emb, s_partition_emb, t_partition_emb):
        # Reshape inputs for attention
        # s_emb, t_emb shape: [batch_size, 1, d]
        combined = torch.cat([s_emb, t_emb], dim=1)  # [batch_size, 2, d]

        # Apply attention
        attn_output, _ = self.attention(combined, combined, combined)

        # Prepare input for BERT
        input_emb = torch.cat([
            attn_output[:, 0],  # source attention output
            attn_output[:, 1],  # target attention output
            s_partition_emb.squeeze(1),  # remove sequence dimension
            t_partition_emb.squeeze(1)  # remove sequence dimension
        ], dim=1)

        return self.bert(input_emb)


class ClassificationHead(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(m + d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, query_emb, via_node_emb):
        # Ensure via_node_emb is 2D if it's 3D
        if via_node_emb.dim() == 3:
            via_node_emb = via_node_emb.squeeze(1)  # Remove sequence dimension

        # Ensure query_emb is 2D
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(1)

        # Concatenate along feature dimension
        input_emb = torch.cat([query_emb, via_node_emb], dim=1)

        # Apply MLP
        output = self.mlp(input_emb)
        return output

class ViaNodePredictionModel(nn.Module):
    def __init__(self, d, m, num_nodes, num_partitions, G):
        super().__init__()
        self.query_encoder = QueryEncoder(d, m)
        self.classification_head = ClassificationHead(d, m)
        self.node_to_index = {node: index for index, node in enumerate(G.nodes())}
        self.node_embeddings = nn.Embedding(num_nodes, d)
        self.partition_embeddings = nn.Embedding(num_partitions, d)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        nn.init.xavier_uniform_(self.partition_embeddings.weight)

    def batch_forward(self, s_batch, t_batch, s_partition_batch, t_partition_batch, via_node_batch):
        # Get embeddings for all nodes at once
        s_emb = self.node_embeddings(s_batch).unsqueeze(1)  # [batch_size, 1, d]
        t_emb = self.node_embeddings(t_batch).unsqueeze(1)  # [batch_size, 1, d]
        via_node_emb = self.node_embeddings(via_node_batch)  # [batch_size, d]
        s_partition_emb = self.partition_embeddings(s_partition_batch).unsqueeze(1)  # [batch_size, 1, d]
        t_partition_emb = self.partition_embeddings(t_partition_batch).unsqueeze(1)  # [batch_size, 1, d]

        # Forward pass through encoder
        query_emb = self.query_encoder(s_emb, t_emb, s_partition_emb, t_partition_emb)  # [batch_size, m]

        # Forward pass through classification head
        return self.classification_head(query_emb, via_node_emb)  # [batch_size, 2]

def load_data(city_name):
    """Load all required data files"""
    data_files = {
        'graph': f'data/{city_name}/graph_sc.pkl',
        'partitions': f'preprocessed/{city_name}/partitions.pkl',
        'boundary_nodes': f'preprocessed/{city_name}/boundary_nodes.pkl',
        'train_candidate_list': f'preprocessed/{city_name}/train_candidate_list.pkl',
        'train_on_traj_flag_list': f'preprocessed/{city_name}/train_on_traj_flag_list.pkl',
        'val_candidate_list': f'preprocessed/{city_name}/valid_candidate_list.pkl',
        'val_on_traj_flag_list': f'preprocessed/{city_name}/valid_on_traj_flag_list.pkl',
        'test_candidate_list': f'preprocessed/{city_name}/test_candidate_list.pkl',
        'test_on_traj_flag_list': f'preprocessed/{city_name}/test_on_traj_flag_list.pkl',
        'train_data': f'preprocessed/{city_name}/train_data_samples.pkl',
        'val_data': f'preprocessed/{city_name}/valid_data_samples.pkl',
        'test_data': f'preprocessed/{city_name}/test_data_samples.pkl'
    }

    data = {}
    for key, path in data_files.items():
        with open(path, 'rb') as f:
            data[key] = pickle.load(f)
    return data

def prepare_dataset(traj_data, candidate_list, on_traj_flag_list, partitions):
    """Prepare dataset with pre-computed partition indices"""
    # Pre-compute node to partition mapping
    node_to_partition = {}
    for j, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = j

    dataset = []
    for i in range(len(candidate_list)):
        s = traj_data[i][0]
        t = traj_data[i][-1]

        # Use pre-computed partition indices
        s_partition_index = node_to_partition.get(s, 0)
        t_partition_index = node_to_partition.get(t, 0)

        dataset.append((s, t, s_partition_index, t_partition_index,
                       candidate_list[i], on_traj_flag_list[i]))

    return dataset


def train_model(model, train_data, val_data, num_epochs=50, lr=0.001, patience=5, batch_size=32, device='mps'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

    # Move model to device
    model = model.to(device)
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_instances = 0

        # Create batches
        for i in tqdm(range(0, len(train_data), batch_size), desc=f'Epoch {epoch + 1}/{num_epochs}'):
            batch = train_data[i:i + batch_size]
            optimizer.zero_grad()
            batch_loss = 0
            batch_correct = 0
            batch_instances = 0

            # Process each trajectory in batch
            for data in batch:
                s, t, s_partition, t_partition, via_nodes, labels = data

                # Process all via nodes at once
                via_node_tensor = torch.tensor([model.node_to_index[v] for v in via_nodes], device=device)
                s_idx = torch.tensor([model.node_to_index[s]], device=device).repeat(len(via_nodes))
                t_idx = torch.tensor([model.node_to_index[t]], device=device).repeat(len(via_nodes))
                s_part = torch.tensor([s_partition], device=device).repeat(len(via_nodes))
                t_part = torch.tensor([t_partition], device=device).repeat(len(via_nodes))

                # Forward pass
                outputs = model.batch_forward(s_idx, t_idx, s_part, t_part, via_node_tensor)
                targets = torch.tensor(labels, dtype=torch.long, device=device)
                batch_loss += criterion(outputs, targets)

                # Calculate accuracy
                probs = torch.softmax(outputs, dim=1)[:, 1]
                pred_idx = torch.argmax(probs).cpu().item()
                if labels[pred_idx] == 1:
                    batch_correct += 1
                batch_instances += 1

            # Backward pass for batch
            avg_loss = batch_loss / len(batch)
            avg_loss.backward()
            optimizer.step()

            total_loss += avg_loss.item()
            total_correct += batch_correct
            total_instances += batch_instances

        # Calculate training accuracy
        train_acc = total_correct / total_instances if total_instances > 0 else 0

        # Validation
        val_acc = evaluate_model(model, val_data, device)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {total_loss / len(train_data):.4f}, Train Acc: {train_acc * 100:.2f}%')
        print(f'Val Acc: {val_acc * 100:.2f}%')

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'param/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(val_acc)


def evaluate_model(model, data, device='mps', batch_size=32):
    """Evaluate model on given data using batch processing"""
    model.eval()
    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        # Create batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_correct = 0

            for s, t, s_partition, t_partition, via_nodes, labels in batch:
                # Process all via nodes at once
                via_node_tensor = torch.tensor([model.node_to_index[v] for v in via_nodes], device=device)
                s_idx = torch.tensor([model.node_to_index[s]], device=device).repeat(len(via_nodes))
                t_idx = torch.tensor([model.node_to_index[t]], device=device).repeat(len(via_nodes))
                s_part = torch.tensor([s_partition], device=device).repeat(len(via_nodes))
                t_part = torch.tensor([t_partition], device=device).repeat(len(via_nodes))

                # Forward pass
                outputs = model.batch_forward(s_idx, t_idx, s_part, t_part, via_node_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                pred_idx = torch.argmax(probs).cpu().item()

                if labels[pred_idx] == 1:
                    batch_correct += 1

            total_correct += batch_correct
            total_instances += len(batch)

        return total_correct / total_instances if total_instances > 0 else 0


def main():
    # Load configuration and data
    config, _ = get_config()
    data = load_data(config.city)

    # Check for MPS availability
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Prepare datasets
    train_data = prepare_dataset(
        [t for t in data['train_data']],
        data['train_candidate_list'],
        data['train_on_traj_flag_list'],
        data['partitions'][6]
    )

    val_data = prepare_dataset(
        [t for t in data['val_data']],
        data['val_candidate_list'],
        data['val_on_traj_flag_list'],
        data['partitions'][6]
    )

    test_data = prepare_dataset(
        [t for t in data['test_data']],
        data['test_candidate_list'],
        data['test_on_traj_flag_list'],
        data['partitions'][6]
    )

    # Prepare model parameters
    d = 64  # embedding dimension
    m = 64  # query dimension
    num_nodes = len(data['graph'].nodes())
    num_partitions = len(data['partitions'][6])

    # %%
    # Create and train model
    model = ViaNodePredictionModel(d, m, num_nodes, num_partitions, data['graph'])

    # Train model
    train_model(model, train_data, val_data, device=device)

    # %%
    # Load best model for inference
    model.load_state_dict(torch.load('param/best_model.pth'))
    model = model.to(device)
    model.eval()

    # Evaluate on test set
    test_acc = evaluate_model(model, test_data)
    print(f'Test accuracy: {test_acc * 100:.2f}%')

    # Make predictions on test set
    test_predictions = []
    with torch.no_grad():
        # Create batches
        batch_size = 32
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            batch_predictions = []

            for s, t, s_partition, t_partition, via_nodes, _ in tqdm(batch):
                # Process all via nodes at once
                via_node_tensor = torch.tensor([model.node_to_index[v] for v in via_nodes], device=device)
                s_idx = torch.tensor([model.node_to_index[s]], device=device).repeat(len(via_nodes))
                t_idx = torch.tensor([model.node_to_index[t]], device=device).repeat(len(via_nodes))
                s_part = torch.tensor([s_partition], device=device).repeat(len(via_nodes))
                t_part = torch.tensor([t_partition], device=device).repeat(len(via_nodes))

                # Forward pass
                outputs = model.batch_forward(s_idx, t_idx, s_part, t_part, via_node_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                # Get prediction
                pred_idx = torch.argmax(probs).cpu().item()
                selected_point = via_nodes[pred_idx]
                batch_predictions.append(selected_point)

            test_predictions.extend(batch_predictions)

    # Save test predictions
    with open(f'preprocessed/{config.city}/test_selected_points.pkl', 'wb') as f:
        pickle.dump(test_predictions, f)

if __name__ == "__main__":
    main()
