"""Shared attention primitives for DG-MPQ / HCMI-ViT (v8).

Adapted (not copied verbatim) from:
  - CrossAttention: CHPNet (github.com/NUIST-Videocoding/CHPNet)
    `models/SV_interaction.py`.
  - The same primitives were validated in the v7 project (MSQR_SHCMI_TAF).

Design rules (from 调整/改进1.md):
  - Every enhancement uses a residual form ``x' = x + gamma * Delta(x)``.
  - gamma / alpha / beta are initialized small or zero so new modules start
    close to the base CLIP representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """Simple MLP with GELU and dropout (ViT-style)."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention, query attends to key/value with optional mask.

    Adapted from CHPNet CrossAttention (models/SV_interaction.py) to support
    ``batch_first``, masked key/value, and returning the attended output
    (without internal residual; the caller adds the gamma-gated residual).
    """

    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=None,
                 dropout=0.0):
        super().__init__()
        context_dim = context_dim or query_dim
        dim_head = dim_head or (query_dim // heads)
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        # x: [B, Nq, Dq], context: [B, Nk, Dk]
        h = self.heads
        context = context if context is not None else x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        B, Nq, _ = q.shape
        _, Nk, _ = k.shape
        q = q.reshape(B, Nq, h, -1).permute(0, 2, 1, 3)   # [B, H, Nq, dh]
        k = k.reshape(B, Nk, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, Nk, h, -1).permute(0, 2, 1, 3)

        sim = (q @ k.transpose(-2, -1)) * self.scale     # [B, H, Nq, Nk]
        if mask is not None:
            # mask: [B, Nk], True = keep
            sim = sim.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = sim.softmax(dim=-1)
        out = attn @ v                                    # [B, H, Nq, dh]
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, -1)
        return self.to_out(out)


def masked_mean_pool(x, mask):
    """Masked mean pooling over the sequence dim.

    Args:
        x: [B, L, D]
        mask: [B, L] bool, True = keep
    Returns:
        [B, D]
    """
    mask_f = mask.unsqueeze(-1).to(x.dtype)   # [B, L, 1]
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom
