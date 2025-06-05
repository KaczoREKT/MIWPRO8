import torch
from torch import nn

class FeedForwardBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, in_dim)
        x = self.linear(x)           # → (batch_size, out_dim)
        x = torch.relu(x)
        x = self.dropout(x)
        return x