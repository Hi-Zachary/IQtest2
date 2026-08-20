"""Shared regression heads for v8 (调整/改进1.md section 4/8).

Base anchors（保持 baseline 简洁，无额外 LayerNorm）:
    v0 = base_visual_proj(global_v)
    t0 = base_text_proj(global_t)

Shared fusion + dual regression heads:
    h = shared_fusion(concat[v, c])
    q = quality_head(h);  a = align_head(h)
"""

import torch.nn as nn


class DualTaskHeads(nn.Module):
    def __init__(self, projection_dim=512, dim=256, drop=0.1):
        super().__init__()
        self.base_visual_proj = nn.Sequential(
            nn.Linear(projection_dim, dim),
            nn.GELU(),
        )
        self.base_text_proj = nn.Sequential(
            nn.Linear(projection_dim, dim),
            nn.GELU(),
        )
        self.shared_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.quality_head = nn.Linear(dim, 1)
        self.align_head = nn.Linear(dim, 1)
