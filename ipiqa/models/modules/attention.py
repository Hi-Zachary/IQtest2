"""Shared attention primitives for MSQR / SHCMI.

Sources (adapted, not copied verbatim):
  - ChannelBlock / SpatialBlock / CrossBranchAttention: MS-SCANet
    (github.com/mithila442/MS-SCANet, ICASSP 2025) `ms_scanet.py`.
  - CrossAttention / MSA_T: CHPNet (github.com/NUIST-Videocoding/CHPNet)
    `models/SV_interaction.py`, `models/SV_MS.py`.

Design rules from AGIQA_MSQR_SHCMI_TAF_design.md:
  - Every enhancement uses a residual form ``x' = x + gamma * Delta(x)``.
  - gamma / alpha / beta are initialized to small or 0 values so the new
    modules start close to the base CLIP representation.
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


class ChannelAttention(nn.Module):
    """Channel recalibration on token sequences [B, N, D] (SE-style).

    Adaptation of MS-SCANet ChannelAttention (which squeezes per-patch with a
    1x1 conv over [B*N, D, 1, 1]); here we use a global sequence-mean channel
    gate, which is cheaper and stable on CLIP features.
    """

    def __init__(self, dim=256, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 8), dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, N, D]
        w = self.fc(x.mean(dim=1))          # [B, D]
        return x * w.unsqueeze(1)


class ChannelBlock(nn.Module):
    """norm -> channel attention (residual) -> norm -> MLP (residual)."""

    def __init__(self, dim=256, mlp_ratio=2.0, drop=0.1, reduction=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel_attn = ChannelAttention(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.channel_attn(self.norm(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialBlock(nn.Module):
    """norm -> global self-attention (residual) -> norm -> MLP (residual).

    MS-SCANet uses a WindowAttention whose code does not perform strict window
    partitioning; on CLIP's 16x16 / 8x8 grids the token count is small, so we
    use plain global self-attention and describe it as "Spatial Self-Attention"
    (see design doc section 4.4).
    """

    def __init__(self, dim=256, num_heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # x: [B, N, D]
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossScaleAttention(nn.Module):
    """Bidirectional cross-scale attention (Fine <-> Coarse).

    Adapted from MS-SCANet CrossBranchAttention. Returns the two attended
    residuals; the caller applies ``gamma``-gated residuals.
    """

    def __init__(self, dim=256, num_heads=4, drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_f = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_c = nn.Linear(dim, dim * 3, bias=False)
        self.proj_f = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

    def _split_heads(self, x, qkv):
        B, N, _ = x.shape
        qkv = qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv.unbind(0)

    def forward(self, fine, coarse):
        # fine: [B, Nf, D], coarse: [B, Nc, D]
        q_f, k_f, v_f = self._split_heads(fine, self.qkv_f)
        q_c, k_c, v_c = self._split_heads(coarse, self.qkv_c)

        # Fine attends to Coarse
        attn_fc = (q_f @ k_c.transpose(-2, -1)) * self.scale
        attn_fc = self.softmax(attn_fc)
        attn_fc = self.drop(attn_fc)
        out_f = (attn_fc @ v_c).transpose(1, 2).reshape(fine.shape[0], fine.shape[1], self.dim)
        out_f = self.proj_f(out_f)

        # Coarse attends to Fine
        attn_cf = (q_c @ k_f.transpose(-2, -1)) * self.scale
        attn_cf = self.softmax(attn_cf)
        attn_cf = self.drop(attn_cf)
        out_c = (attn_cf @ v_f).transpose(1, 2).reshape(coarse.shape[0], coarse.shape[1], self.dim)
        out_c = self.proj_c(out_c)

        return out_f, out_c


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

    def forward(self, x, context=None, mask=None, attn_bias=None):
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
        if attn_bias is not None:
            # attn_bias: [B, Nq, Nk]（或 [B,1,Nq,Nk]），broadcast over heads
            sim = sim + attn_bias.unsqueeze(1)
        if mask is not None:
            # mask: [B, Nk], True = keep
            sim = sim.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = sim.softmax(dim=-1)
        out = attn @ v                                    # [B, H, Nq, dh]
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, -1)
        return self.to_out(out)


class MSA_T(nn.Module):
    """Multi-kernel text enhancement on token sequences [B, L, D].

    Adapted from CHPNet MSA_T (models/SV_MS.py): Conv1d k=3/5/7 -> concat ->
    1x1 conv. Returns the enhancement ``Delta``; caller adds ``x + eta*Delta``.
    """

    def __init__(self, in_channels, out_channels, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3)
        self.conv1x1 = nn.Conv1d(in_channels * 3, out_channels, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: [B, L, D]
        xt = x.permute(0, 2, 1)          # [B, D, L]
        x1 = self.conv1(xt)
        x2 = self.conv2(xt)
        x3 = self.conv3(xt)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv1x1(x_cat)
        x_out = x_out.permute(0, 2, 1)   # [B, L, D]
        return self.gelu(x_out)


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
