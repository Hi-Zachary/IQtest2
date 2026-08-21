"""QARD -- Quality-Aware Retrieval Decoder (V17 quality module).

调整/改进4.md (V17): DEIQT 的 Quality-Aware Decoder 思想（被其消融证明为主要增益来源），
但不复制 Attention Panel / 多评审者。核心机制：

    当前质量状态 v_q → dynamic quality query
        → 单层 cross-attention 从 H12 patch memory 主动检索与质量最相关的证据
        → residual query refinement (FFN)
        → delta_d

与 DG-MPQ 职责互补：
    DG-MPQ = bottom-up degradation aggregation（异常局部 -> 质量表征）
    QARD   = top-down quality-aware retrieval（质量状态 -> 读取补充证据）

三阶段：
    1. Patch Memory Construction   (H12 -> 256D memory)
    2. Dynamic Quality-Aware Retrieval (query = v_q; cross-attn)
    3. Residual Query Refinement   (attention residual + FFN + out)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QARD(nn.Module):
    def __init__(
        self,
        width=768,
        dim=256,
        num_heads=4,
        ffn_ratio=2,
        drop=0.0,
    ):
        super().__init__()
        self.dim = dim

        # Stage 1: patch memory
        self.patch_memory = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim),
            nn.GELU(),
        )

        # Stage 2: dynamic quality query
        self.query_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

        # Stage 3: decoder refinement
        hidden = int(dim * ffn_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_stats = None

    def forward(self, h12, v_q):
        # h12: [B,197,768]; v_q: [B,256] current quality representation
        # ---------- 1. Patch Memory ----------
        memory = self.patch_memory(h12[:, 1:, :])    # [B,196,256]

        # ---------- 2. Dynamic Quality Query + Cross-Attn Retrieval ----------
        q = self.query_proj(v_q).unsqueeze(1)        # [B,1,256]

        q_norm = self.norm_q(q)
        m_norm = self.norm_m(memory)

        attn_out, attn_weight = self.cross_attn(
            q_norm,
            m_norm,
            m_norm,
            need_weights=True,
            average_attn_weights=False,
        )                                            # attn_weight: [B,4,1,196]

        q = q + attn_out

        # ---------- 3. Residual Query Refinement ----------
        q = q + self.ffn(self.norm_ffn(q))
        delta_d = self.out(q.squeeze(1))             # [B,256]

        w = attn_weight.squeeze(2)                   # [B,4,196]
        entropy = (-w * torch.log(w + 1e-8)).sum(dim=-1)
        entropy_norm = entropy / math.log(w.shape[-1])

        self._last_stats = {
            "attn_entropy": round(entropy_norm.mean().item(), 4),
            "attn_max": round(w.max(dim=-1).values.mean().item(), 4),
            "decoder_norm": round(delta_d.norm(dim=-1).mean().item(), 4),
            "memory_norm": round(memory.norm(dim=-1).mean().item(), 4),
        }

        return delta_d
