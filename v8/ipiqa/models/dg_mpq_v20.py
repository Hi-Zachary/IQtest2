"""DG-MPQ V20 -- HMQE + DCGA（沿 multi-level + semantic deviation 核心深化）。

调整/改进8.md (V20):
  HMQE — Hierarchical Multi-Level Quality Encoding
    P_l = W_l(LN(H_l))；P_avg = mean(P_l)
    R_l = P_l - P_avg；R = [R3;R6;R9;R12]
    ΔP = hier_mixer(R)；P = P_avg + tanh(γ_h)ΔP   （γ_h=0 init -> P==P_avg）
  DCGA — Dual-Cue Deviation-Guided Aggregation
    d_sem  = 1-cos(P, g)                    （旧语义偏差，定义不变）
    d_hier = (1/4)Σ_l [1-cos(P_l, P)]       （跨层不一致）
    w = σ(f_w([P, d_sem, d_hier]))          （旧 D+1 输入，新增 d_hier 列，兼容 init 该列=0）
    q_μ = weighted mean
    q_σ = weighted dispersion
    q_stat = [q_μ; q_σ]；Δq = quality_out(q_stat)（Linear(2D,D) 兼容 init 后半=0）

兼容初始化：训练开始 == 旧 DG-MPQ 函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DgMpqV20(nn.Module):
    def __init__(self, width=768, dim=256, use_hmqe=True, use_hier_dev=True,
                 use_dispersion=True, init_scale=0.01):
        super().__init__()
        self.dim = dim
        self.use_hmqe = use_hmqe
        self.use_hier_dev = use_hier_dev
        self.use_dispersion = use_dispersion

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

        # HMQE: cross-level residual mixer
        self.hier_mixer = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gamma_h = nn.Parameter(torch.tensor(0.0))

        # DCGA: dual-cue patch weight (D + 1 + 1)
        self.patch_weight = nn.Sequential(
            nn.Linear(dim + 2, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 1),
        )

        # DCGA: quality residual (2D -> D)
        self.quality_out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_deviation = None
        self._last_weight = None
        self._last_d_hier = None
        self._last_q_mu = None
        self._last_q_sigma = None
        self._last_gamma_h = None

    def forward(self, h3, h6, h9, h12):
        P_l = [self.patch_proj[l](hs[:, 1:, :])
               for l, hs in enumerate([h3, h6, h9, h12])]      # 4x [B,196,D]
        P_avg = (P_l[0] + P_l[1] + P_l[2] + P_l[3]) / 4.0      # [B,196,D]

        # ---------- HMQE: cross-level residual ----------
        if self.use_hmqe:
            R = torch.cat([Pl - P_avg for Pl in P_l], dim=-1)  # [B,196,4D]
            dP = self.hier_mixer(R)                            # [B,196,D]
            P = P_avg + torch.tanh(self.gamma_h) * dP
            self._last_gamma_h = torch.tanh(self.gamma_h).item()
        else:
            P = P_avg

        g = self.quality_global_proj(h12[:, 0, :])             # [B,D]
        g_n = F.normalize(g, dim=-1).unsqueeze(1)

        # ---------- DCGA: dual-cue deviation ----------
        d_sem = 1.0 - (F.normalize(P, dim=-1) * g_n).sum(dim=-1)   # [B,196]

        # hierarchical deviation（始终计算；use_hier_dev=False 时置零，
        # 配合兼容 init 的 d_hier 列=0，E1 等效旧 SDGA）
        d_hier_raw = sum(
            1.0 - (F.normalize(Pl, dim=-1) * F.normalize(P, dim=-1)).sum(dim=-1)
            for Pl in P_l
        ) / 4.0                                                     # [B,196]
        d_hier = d_hier_raw if self.use_hier_dev else torch.zeros_like(d_hier_raw)

        w_input = torch.cat([P, d_sem.unsqueeze(-1), d_hier.unsqueeze(-1)], dim=-1)  # [B,196,D+2]
        self._last_d_hier = d_hier.detach().float()

        w = torch.sigmoid(self.patch_weight(w_input))           # [B,196,1]
        w_sum = w.sum(dim=1) + 1e-6

        # ---------- DCGA: central tendency + dispersion ----------
        q_mu = (w * P).sum(dim=1) / w_sum                       # [B,D]
        if self.use_dispersion:
            diff = (P - q_mu.unsqueeze(1)) ** 2
            q_sigma = torch.sqrt(
                (w * diff).sum(dim=1) / w_sum + 1e-6
            )                                                   # [B,D]
        else:
            q_sigma = torch.zeros_like(q_mu)

        q_stat = torch.cat([q_mu, q_sigma], dim=-1)             # [B,2D]
        delta_q = self.quality_out(q_stat)                      # [B,D]

        self._last_deviation = d_sem.detach().float()
        self._last_weight = w.detach().float()
        self._last_q_mu = q_mu.detach().float()
        self._last_q_sigma = q_sigma.detach().float()
        return delta_q
