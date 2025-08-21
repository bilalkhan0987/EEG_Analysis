"""
model.py — CNN + Custom Transformer encoder for decoding Executive Functions.

Features
- Custom Multi-Head Attention (batch-first)
- Transformer encoder stack (pre-norm)
- CNN frontend (1D) to form token embeddings
- Sequence mean pooling → MLP head (length-agnostic)
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Batch-first multi-head self-attention.

    Expects inputs of shape (B, L, E).
    """

    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear projections: (B, L, E) -> (B, L, E)
        self.q_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.k_proj = nn.Linear(embed_size, embed_size, bias=False)
        self.v_proj = nn.Linear(embed_size, embed_size, bias=False)

        self.out_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        B, Lq, E = x_q.shape
        _, Lk, _ = x_k.shape
        # Projections
        Q = self.q_proj(x_q)  # (B, Lq, E)
        K = self.k_proj(x_k)  # (B, Lk, E)
        V = self.v_proj(x_v)  # (B, Lk, E)

        # Reshape to heads: (B, H, L, D)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (B, H, Lq, Lk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)
        context = torch.matmul(attn, V)  # (B, H, Lq, D)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.embed_size)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, embed_size: int, forward_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, forward_expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, forward_expansion, dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size: int, num_layers: int, num_heads: int, forward_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, E)
        for blk in self.layers:
            x = blk(x)
        return x


class CNNTransformer(nn.Module):
    """CNN → Transformer encoder → mean-pool → MLP classifier.

    Args:
        in_channels: input channels for Conv1d (e.g., 1 for mono sequence)
        hidden_channels: channels produced by CNN (and used as transformer embed size)
        num_layers: number of transformer blocks
        num_heads: attention heads
        num_classes: output classes
        dropout: dropout rate in transformer FFN and residuals
        forward_expansion: FFN expansion ratio
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
        forward_expansion: int = 4,
    ):
        super().__init__()
        # CNN frontend (reduce temporal length, increase channels)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # Transformer encoder operates on (B, L, E) where E = hidden_channels
        self.transformer = TransformerEncoder(
            embed_size=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, L_in)"""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 16, L/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, hidden, L/4)

        # Prepare for transformer: (B, L, E)
        x = x.transpose(1, 2)  # (B, L, E=hidden_channels)
        x = self.transformer(x)

        # Global average pooling over sequence length
        x = x.mean(dim=1)  # (B, E)
        logits = self.head(x)
        return logits
