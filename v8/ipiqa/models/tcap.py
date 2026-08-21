"""TCAP -- Text-Conditioned Attention Pooling (V9 alignment branch).

调整/改进5.md (V9): 替代 HCMI-ViT，更轻量、更直接的 image-text correspondence 建模。

核心思想：以文本全局表示 t0 作为 Query，从 ViT 高层 patch token 中检索
与 prompt 最相关的视觉区域，再构造显式 match/mismatch descriptor。

    V = visual_proj(H12_patch)            # [B,196,D] 高层语义视觉 token
    Q = t0.unsqueeze(1)                   # [B,1,D]   text 作 query
    z = CrossAttn(Q, V, V)                # [B,1,D]   prompt-relevant visual repr
    r = [z, t0, z*t0, |z-t0|]             # correspondence descriptor [B,4D]
    delta_a = align_mlp(r)                # [B,D]  alignment residual

只做 Text -> Image 单向检索（任务本质：给定 prompt 检查图像是否符合），
不做双向交互，避免 HCMI 的自由度/不稳定问题。
"""

import torch
import torch.nn as nn


class TCAP(nn.Module):
    def __init__(self, visual_width=768, dim=256, num_heads=4, drop=0.1):
        super().__init__()
        self.dim = dim

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_width),
            nn.Linear(visual_width, dim),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

        self.align_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self._last_attn = None

    def forward(self, h12, t0):
        # h12: [B, 196, width] (caller 已去掉 CLS)；t0: [B, D] global text anchor
        V = self.visual_proj(h12)                 # [B,196,D]
        Q = t0.unsqueeze(1)                       # [B,1,D]

        z, attn_w = self.attn(
            query=Q,
            key=V,
            value=V,
            need_weights=True,
            average_attn_weights=True,            # [B,1,196] over heads
        )                                         # z: [B,1,D]
        self._last_attn = attn_w.detach().float()

        z = z.squeeze(1)                          # [B,D]

        # explicit correspondence descriptor: relevance + semantics + match + mismatch
        r = torch.cat([
            z,
            t0,
            z * t0,
            torch.abs(z - t0),
        ], dim=-1)                                # [B,4D]

        return self.align_mlp(r)                  # [B,D]
