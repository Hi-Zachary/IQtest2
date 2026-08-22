"""DCAR -- Degradation-Conditioned Adaptive Regressor (V19).

调整/改进7.md: HyperIQA 的动态预测规则 + DG-MPQ 的显式 degradation profile。

    Degradation Profile Encoder:
        r3/r6/r9/r12 (prototypes [B,4,256]) -> shared projector -> 128D
        stats [B,4,3] + scale_weight [B,4]   -> 16D -> 32D
        deg_fuse(160D) -> z_d (128D)
    Semantic Content Encoder:
        global_v -> z_s (128D)
    HyperNet:
        c = [z_s; z_d] (256D) -> 128 -> [Δw(256), Δb(1)]
    Dynamic regression:
        Δq = (Δw^T LN(h_q))/sqrt(D) + Δb ;  q = q_base + Δq

HyperNet 末层 zero-init -> 初始 Δw=Δb=0 -> Full 初始函数 == B1。
"""

import math
import torch
import torch.nn as nn


class DCAR(nn.Module):
    def __init__(self, content_width, dim=256, hidden=128):
        super().__init__()
        self.dim = dim

        # ---- Degradation Profile Encoder ----
        self.prototype_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 32),
            nn.GELU(),
        )
        self.stat_proj = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
        )
        self.deg_fuse = nn.Sequential(
            nn.Linear(160, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # ---- Semantic Content Encoder ----
        self.content_proj = nn.Sequential(
            nn.LayerNorm(content_width),
            nn.Linear(content_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # ---- Degradation-Conditioned HyperNet ----
        self.hyper = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim + 1),
        )

        self.target_norm = nn.LayerNorm(dim)

        # Initial Full function == B1 (delta=0)
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

        self._last_stats = None

    def forward(self, global_v, h_q, deg_profile):
        prototype = deg_profile["prototype"]      # [B,4,D]
        stats = deg_profile["stats"]              # [B,4,3]
        scale_weight = deg_profile["scale_weight"]  # [B,4]

        # prototype: 4 levels -> shared proj -> 128D
        zp = self.prototype_proj(prototype)       # [B,4,32]
        zp = zp.reshape(zp.shape[0], -1)          # [B,128]

        # stats(12D) + scale_weight(4D) -> 16D -> 32D
        zs = torch.cat([stats.reshape(stats.shape[0], -1), scale_weight], dim=-1)  # [B,16]
        zs = self.stat_proj(zs)                   # [B,32]

        z_d = self.deg_fuse(torch.cat([zp, zs], dim=-1))  # [B,128]

        z_s = self.content_proj(global_v)         # [B,128]

        c = torch.cat([z_s, z_d], dim=-1)         # [B,256]
        params = self.hyper(c)                    # [B,257]
        delta_w = params[:, :-1]
        delta_b = params[:, -1:]

        z = self.target_norm(h_q)
        delta_q = (
            (delta_w * z).sum(dim=-1, keepdim=True)
            / math.sqrt(self.dim)
            + delta_b
        )

        with torch.no_grad():
            self._last_stats = {
                "dw_norm": delta_w.norm(dim=-1).mean().item(),
                "dw_batch_std": delta_w.std(dim=0).mean().item(),
                "db_abs": delta_b.abs().mean().item(),
                "delta_abs": delta_q.abs().mean().item(),
                "deg_descriptor_norm": z_d.norm(dim=-1).mean().item(),
                "prototype_norm": prototype.norm(dim=-1).mean().item(),
            }

        return delta_q
