"""DG-MPQ 因子消融版（V21）—— 只切换两个因素，结构与参数量完全一致。

调整/改进9.md (2×2 factorial):
    Factor A: Patch Representation
        Multi-Level: P = (P3+P6+P9+P12)/4
        Single-Level: P = P12
    Factor B: Semantic Deviation
        Deviation: d = 1 - cos(P, g)
        No-Deviation: d = 0

Variant:
    A: single + no-dev
    B: single + dev
    C: multi  + no-dev
    D: multi  + dev   (完整 DG-MPQ)

除实际信息流外，网络结构和参数量四组完全一致（可 strict=True 共享 init）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DgMpqAbl(nn.Module):
    def __init__(self, width=768, dim=256, use_multilevel=True, use_deviation=True):
        super().__init__()
        self.dim = dim
        self.use_multilevel = use_multilevel
        self.use_deviation = use_deviation

        # multi-level patch projection (layer 3/6/9/12)
        self.patch_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
            for _ in range(4)
        ])

        # global quality anchor
        self.quality_global_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim),
        )

        # deviation-guided patch weight (P + d)
        self.patch_weight = nn.Sequential(
            nn.Linear(dim + 1, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 1),
        )

        # quality residual output
        self.quality_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_deviation = None
        self._last_weight = None

    def forward(self, h3, h6, h9, h12):
        P3 = self.patch_proj[0](h3[:, 1:, :])
        P6 = self.patch_proj[1](h6[:, 1:, :])
        P9 = self.patch_proj[2](h9[:, 1:, :])
        P12 = self.patch_proj[3](h12[:, 1:, :])

        if self.use_multilevel:
            P = (P3 + P6 + P9 + P12) / 4.0        # [B,196,D]
        else:
            P = P12

        if self.use_deviation:
            g = self.quality_global_proj(h12[:, 0, :])            # [B,D]
            g_n = F.normalize(g, dim=-1).unsqueeze(1)
            d = 1.0 - (F.normalize(P, dim=-1) * g_n).sum(dim=-1)  # [B,196]
        else:
            d = torch.zeros(P.shape[0], P.shape[1], device=P.device, dtype=P.dtype)

        weight_input = torch.cat([P, d.unsqueeze(-1)], dim=-1)    # [B,196,D+1]
        w = torch.sigmoid(self.patch_weight(weight_input))        # [B,196,1]

        q_local = (w * P).sum(dim=1) / (w.sum(dim=1) + 1e-6)      # [B,D]
        delta_q = self.quality_out(q_local)                       # [B,D]

        self._last_deviation = d.detach().float()
        self._last_weight = w.detach().float()
        return delta_q
