"""MLPQ -- Multi-Level Patch Quality Modeling（V23-Final 干净版）。

调整/改进12.md Phase B：从 DgMpqAbl(use_multilevel=True, use_deviation=False) 收敛而来，
彻底删除 deviation 遗留：
    - quality_global_proj
    - _last_deviation
    - semantic deviation channel
    - patch_weight 输入从 dim+1 改为 dim

结构：
    P_l = W_l(LN(H_l)), l in {3,6,9,12}
    P = (P3+P6+P9+P12)/4
    w_i = sigma(f_w(P_i))                    # 质量感知 patch 加权
    q_local = sum_i w_i P_i / (sum_i w_i + eps)
    delta_v = f_q(q_local)                   # quality residual

与旧 no-deviation MLPQ 参数可迁移、function-equivalent（max_abs_diff < 1e-5）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQ(nn.Module):
    def __init__(self, width=768, dim=256):
        super().__init__()
        self.dim = dim

        # multi-level patch projection (layer 3/6/9/12)
        self.patch_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
            for _ in range(4)
        ])

        # quality-aware patch weighting (input = P only, no deviation)
        self.patch_weight = nn.Sequential(
            nn.Linear(dim, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 1),
        )

        # quality residual output
        self.quality_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_weight = None

    def forward(self, h3, h6, h9, h12):
        P3 = self.patch_proj[0](h3[:, 1:, :])
        P6 = self.patch_proj[1](h6[:, 1:, :])
        P9 = self.patch_proj[2](h9[:, 1:, :])
        P12 = self.patch_proj[3](h12[:, 1:, :])
        P = (P3 + P6 + P9 + P12) / 4.0        # [B,196,D]

        w = torch.sigmoid(self.patch_weight(P))   # [B,196,1]
        q_local = (w * P).sum(dim=1) / (w.sum(dim=1) + 1e-6)   # [B,D]
        delta_q = self.quality_out(q_local)       # [B,D]

        self._last_weight = w.detach().float()
        return delta_q
