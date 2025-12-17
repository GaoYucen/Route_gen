import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding=None, hidden_dim=256):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.embedding_len = len(self.embedding[list(self.embedding.keys())[0]])
        self.fc1 = nn.Linear(self.embedding_len * 4, self.hidden_dim)
        # 归一化层
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, input_embed):
        x = self.fc1(input_embed)
        x = self.ln1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        return x