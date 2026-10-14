"""Local PVA/TRA implementation; see THIRD_PARTY_NOTICES.md for provenance."""
from __future__ import annotations

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

class PMDSelfAttention(nn.Module):
    """Self-attention block matching PMDformer's PVA attention style."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        head_dim = x.shape[-1] // self.n_heads
        queries = self.query_projection(x).view(batch, length, self.n_heads, head_dim)
        keys = self.key_projection(x).view(batch, length, self.n_heads, head_dim)
        values = self.value_projection(x).view(batch, length, self.n_heads, head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) / sqrt(head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values).reshape(batch, length, -1)
        return self.out_projection(out), attn


class PMDTrendRestorationAttention(nn.Module):
    """PMDformer TRA attention: injects patch means into the value stream."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        head_dim = x.shape[-1] // self.n_heads
        queries = self.query_projection(x).view(batch, length, self.n_heads, head_dim)
        keys = self.key_projection(x).view(batch, length, self.n_heads, head_dim)
        values = self.value_projection(x).view(batch, length, self.n_heads, head_dim) + means.unsqueeze(2)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) / sqrt(head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values).reshape(batch, length, -1)
        return self.out_projection(out), attn


class PMDEncoderLayerPVA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = PMDSelfAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_x, attn = self.attention(x)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(F.gelu(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), attn


class PMDEncoderLayerTRA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = PMDTrendRestorationAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, means: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_x, attn = self.attention(x, means)
        x = x + self.dropout(new_x)
        y = self.norm1(x)
        y = self.dropout(F.gelu(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        return self.norm2(x + y), attn


