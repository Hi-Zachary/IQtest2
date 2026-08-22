"""QARE -- Quality-Adaptive Representation Enhancement（V24 Candidate）。

调整/改进13.md：以 Frozen CLIP 单次 forward 为基础，Quality 分支内部做
quality-specific 低秩特征适配 + 分层质量证据聚合，替代 backbone Q/K LoRA。

Stage A — Quality-Specific Low-Rank Adaptation:
    layerwise: Z_l = LN(H_l^p); H~_l^p = H_l^p + (alpha/r) B_l A_l Z_l
    global:    Z_g = LN(g_v);   g~_v     = g_v   + (alpha/r) B_g A_g Z_g
    A: kaiming_uniform, B: zeros  -> 初始退化为恒等。

Stage B — Layerwise Quality Evidence Aggregation（继承 clean LQEA/MLPQ）:
    P_l = W_l(LN(H~_l));  P = mean(P_l)
    w_i = sigmoid(f_w(P_i));  q_e = sum_i w_i P_i / (sum_i w_i + eps)
    delta_v = f_q(q_e)

输出: (adapted_global, delta_v)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankFeatureAdapter(nn.Module):
    """B(alpha/r) A x 形式，A: kaiming, B: zeros -> 初始恒等。"""
    def __init__(self, in_features, rank=4, alpha=8):
        super().__init__()
        self.in_features = in_features
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(in_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        # x: [..., in]; delta = scaling * (x @ A.T) @ B.T = [..., in]
        return self.scaling * ((x @ self.A.T) @ self.B.T)


class QARE(nn.Module):
    def __init__(self, visual_width=768, global_dim=512, dim=256, rank=4, alpha=8):
        super().__init__()
        self.dim = dim

        # ---- Stage A: low-rank adaptation ----
        self.ln_layers = nn.ModuleList([nn.LayerNorm(visual_width) for _ in range(4)])
        self.layer_adapters = nn.ModuleList([
            LowRankFeatureAdapter(visual_width, rank, alpha) for _ in range(4)
        ])
        self.ln_global = nn.LayerNorm(global_dim)
        self.global_adapter = LowRankFeatureAdapter(global_dim, rank, alpha)

        # ---- Stage B: layerwise quality evidence aggregation (from clean LQEA/MLPQ) ----
        self.patch_proj = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(visual_width), nn.Linear(visual_width, dim))
            for _ in range(4)
        ])
        self.patch_weight = nn.Sequential(
            nn.Linear(dim, max(dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 8), 1),
        )
        self.quality_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_ratios = None
        self._last_weight = None

    def forward(self, global_v, h3, h6, h9, h12):
        layers = [h3, h6, h9, h12]
        adapted = []
        ratios = {}
        for i, (ln, ad, h) in enumerate(zip(self.ln_layers, self.layer_adapters, layers)):
            Hp = h[:, 1:, :]                        # [B,196,768]
            Z = ln(Hp)
            dH = ad(Z)
            Hs = Hp + dH
            adapted.append(Hs)
            ratios[f"h{[3,6,9,12][i]}"] = (
                (dH.norm(dim=-1).mean() / (Hp.norm(dim=-1).mean() + 1e-6)).item()
            )

        Zg = self.ln_global(global_v)
        dg = self.global_adapter(Zg)
        g_adapt = global_v + dg
        ratios["global"] = (
            (dg.norm(dim=-1).mean() / (global_v.norm(dim=-1).mean() + 1e-6)).item()
        )

        # ---- Stage B: aggregation ----
        P = sum(self.patch_proj[i](Hs) for i, Hs in enumerate(adapted)) / 4.0  # [B,196,D]
        w = torch.sigmoid(self.patch_weight(P))                               # [B,196,1]
        q_e = (w * P).sum(dim=1) / (w.sum(dim=1) + 1e-6)                       # [B,D]
        delta_v = self.quality_out(q_e)                                        # [B,D]

        self._last_ratios = ratios
        self._last_weight = w.detach().float()
        return g_adapt, delta_v
