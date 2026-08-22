"""DG-MPQ+ -- Deviation-Guided Multi-Level Patch Quality Modeling + DASF.

调整/改进7.md (V19): 原 DG-MPQ 的低风险扩展。
- 保留原 multi-level patch + local-global semantic deviation + deviation-guided
  patch weighting + quality residual 全部。
- 唯一结构扩展 DASF（Deviation-Aware Scale Fusion）：不再固定四层 25% 平均，
  而是按每层 deviation 统计（mu/sigma/top10%）用共享 MLP + softmax 自适应融合。
  DASF 的 scale-score 末层 zero-init -> 初始 alpha=1/4，与旧 DG-MPQ 完全一致。
- 额外输出 Multi-Level Degradation Profile（原型 r3/r6/r9/r12 + 层级统计 + alpha），
  供第二模块 DCAR 使用。

返回 (delta_q, deg_profile)：
    deg_profile = {
        "prototype":    [B,4,256]
        "stats":        [B,4,3]   (mu, sigma, tau per level)
        "scale_weight": [B,4]
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DgMpqPlus(nn.Module):
    def __init__(self, width=768, dim=256, use_dasf=True):
        super().__init__()
        self.dim = dim
        self.use_dasf = use_dasf

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

        # deviation-guided patch weight
        self.patch_weight = nn.Sequential(
            nn.Linear(dim + 1, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 1),
        )

        # DASF: deviation-aware scale fusion
        self.scale_score = nn.Sequential(
            nn.Linear(3, max(dim // 8, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 8, 8), 1),
        )
        nn.init.zeros_(self.scale_score[-1].weight)
        nn.init.zeros_(self.scale_score[-1].bias)

        # quality residual output
        self.quality_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_deviation = None
        self._last_weight = None
        self._last_alpha = None

    def forward(self, h3, h6, h9, h12):
        P = [self.patch_proj[l](hs[:, 1:, :])
             for l, hs in enumerate([h3, h6, h9, h12])]       # 4x [B,196,D]
        g = self.quality_global_proj(h12[:, 0, :])            # [B,D]
        g_n = F.normalize(g, dim=-1).unsqueeze(1)

        # per-level semantic deviation
        dev = []
        for Pl in P:
            pn = F.normalize(Pl, dim=-1)
            dev.append(1.0 - (pn * g_n).sum(dim=-1))          # [B,196]

        # ---- DASF: adaptive scale fusion ----
        stats = []
        for dl in dev:
            mu = dl.mean(dim=-1)
            sigma = dl.std(dim=-1)
            k = max(int(dl.shape[-1] * 0.10), 1)
            tau = torch.topk(dl, k=k, dim=-1).values.mean(dim=-1)
            stats.append(torch.stack([mu, sigma, tau], dim=-1))   # [B,3]
        stats = torch.stack(stats, dim=1)                     # [B,4,3]

        if self.use_dasf:
            a = self.scale_score(stats).squeeze(-1)           # [B,4]
            alpha = F.softmax(a, dim=-1)                      # [B,4]
        else:
            alpha = stats.new_full((stats.shape[0], 4), 0.25)

        P_fused = sum(alpha[:, l:l+1].unsqueeze(-1) * P[l] for l in range(4))  # [B,196,D]

        # ---- degradation prototype (per level) ----
        prototypes = []
        for dl, Pl in zip(dev, P):
            w = dl.unsqueeze(-1)                              # [B,196,1]
            r = (w * Pl).sum(dim=1) / (w.sum(dim=1) + 1e-6)   # [B,D]
            prototypes.append(r)
        prototypes = torch.stack(prototypes, dim=1)           # [B,4,D]

        # ---- deviation-guided patch weighting on fused P ----
        dev_fused = dev[0] + dev[1] + dev[2] + dev[3]         # 近似融合偏差 [B,196]
        d_f = 1.0 - (F.normalize(P_fused, dim=-1) * g_n).sum(dim=-1)  # [B,196]
        weight_input = torch.cat([P_fused, d_f.unsqueeze(-1)], dim=-1)  # [B,196,D+1]
        w_patch = torch.sigmoid(self.patch_weight(weight_input))        # [B,196,1]

        q_local = (w_patch * P_fused).sum(dim=1) / (w_patch.sum(dim=1) + 1e-6)  # [B,D]
        delta_q = self.quality_out(q_local)                   # [B,D]

        self._last_deviation = d_f.detach().float()
        self._last_weight = w_patch.detach().float()
        self._last_alpha = alpha.detach().float()

        deg_profile = {
            "prototype": prototypes.detach(),      # [B,4,D]
            "stats": stats.detach(),               # [B,4,3]
            "scale_weight": alpha.detach(),        # [B,4]
        }
        return delta_q, deg_profile
