"""DG-MPQ -- Deviation-Guided Multi-Level Patch Quality Module (v8 Module 1).

调整/改进1.md section 6. TeMu-IQA inspired multi-level patch representation +
v7 DMSQR local-global semantic deviation, adapted to AIGC-specific quality.

   P = (P3 + P6 + P9 + P12) / 4        # averaged multi-level patch tokens
   g = quality_global_proj(CLS_12)     # global quality anchor
   d_i = 1 - cos(p_i, g)               # local-global semantic deviation
   w_i = sigmoid(MLP([p_i, d_i]))      # deviation-guided patch weight
   q_local = sum(w_i * p_i) / sum(w_i) # patch-weighted aggregation
   delta_q = quality_out(q_local)      # quality residual feature [B, D]

The output is a residual feature (not a scalar); the caller injects it as
``v = v0 + tanh(lambda_q) * delta_q``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DgMpq(nn.Module):
    def __init__(self, width=768, dim=256):
        super().__init__()
        self.dim = dim

        # multi-level patch projection (layer 3/6/9/12)
        self.patch_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, dim),
            )
            for _ in range(4)
        ])

        # global quality anchor (last layer CLS)
        self.quality_global_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim),
        )

        # deviation-guided patch weight: [patch, deviation] -> importance
        self.patch_weight = nn.Sequential(
            nn.Linear(dim + 1, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 1),
        )

        # quality residual output (模块内部 LayerNorm 做 residual 尺度校准)
        self.quality_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_deviation = None
        self._last_weight = None

    def forward(self, h3, h6, h9, h12):
        # h3/h6/h9/h12: [B, 197, 768]; 丢弃 CLS，取 patch tokens
        P3 = self.patch_proj[0](h3[:, 1:, :])
        P6 = self.patch_proj[1](h6[:, 1:, :])
        P9 = self.patch_proj[2](h9[:, 1:, :])
        P12 = self.patch_proj[3](h12[:, 1:, :])
        P = (P3 + P6 + P9 + P12) / 4.0                # [B, 196, D]

        g = self.quality_global_proj(h12[:, 0, :])    # [B, D]

        # semantic deviation d_i = 1 - cos(p_i, g)
        p_n = F.normalize(P, dim=-1)
        g_n = F.normalize(g, dim=-1).unsqueeze(1)
        deviation = 1.0 - (p_n * g_n).sum(dim=-1, keepdim=True)   # [B, 196, 1]

        # deviation-guided patch weight
        weight_input = torch.cat([P, deviation], dim=-1)          # [B, 196, D+1]
        w = torch.sigmoid(self.patch_weight(weight_input))        # [B, 196, 1]

        # patch-weighted aggregation (no Good/Bad quality prompts)
        q_local = (w * P).sum(dim=1) / (w.sum(dim=1) + 1e-6)      # [B, D]

        delta_q = self.quality_out(q_local)                       # [B, D]

        self._last_deviation = deviation.detach().float()
        self._last_weight = w.detach().float()
        return delta_q
