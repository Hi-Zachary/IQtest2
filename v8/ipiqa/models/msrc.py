"""MSRC -- Multi-Level Semantic Relation Calibration (V14 alignment branch).

调整/改进1.md (V14): 替代 MSCM。三个子模块链：
    1. MSR  Multi-Level Semantic Refinement      (H9/H12 mean-pool -> proj -> residual adapter -> text-conditioned level gate)
    2. CMR  Cross-Modal Relation Modeling        (low-rank v/t projection -> multiplicative + difference relation + global CLIP cos)
    3. RCC  Relation Confidence Calibration      (channel-wise confidence gate on relation)

输入：H9/H12 [B,197,768]（内部自行丢 CLS）、t [B,256]、global_v/global_t [B,512]
输出：a_rel [B,256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, dim=256, bottleneck=64, init_scale=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        return x + torch.tanh(self.scale) * self.net(x)


class MSRC(nn.Module):
    def __init__(
        self,
        visual_width=768,
        dim=256,
        relation_rank=128,
        adapter_ratio=4,
        drop=0.0,
        init_scale=0.01,
    ):
        super().__init__()
        bottleneck = max(dim // adapter_ratio, 8)

        # ---------- 1. Multi-Level Semantic Refinement ----------
        self.proj9 = nn.Sequential(
            nn.LayerNorm(visual_width),
            nn.Linear(visual_width, dim),
        )
        self.proj12 = nn.Sequential(
            nn.LayerNorm(visual_width),
            nn.Linear(visual_width, dim),
        )
        self.adapter9 = ResidualAdapter(dim, bottleneck, init_scale)
        self.adapter12 = ResidualAdapter(dim, bottleneck, init_scale)
        self.level_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # ---------- 2. Cross-Modal Relation Modeling ----------
        self.rel_v = nn.Linear(dim, relation_rank)
        self.rel_t = nn.Linear(dim, relation_rank)
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_rank * 2 + 1, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(dim),
        )

        # ---------- 3. Relation Confidence Calibration ----------
        self.confidence = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self._last_stats = None

    def forward(self, h9, h12, t, global_v, global_t):
        # h9/h12: [B,197,768]；内部自行丢 CLS
        # ---------- MSR ----------
        p9 = h9[:, 1:, :].mean(dim=1)          # [B,768]
        p12 = h12[:, 1:, :].mean(dim=1)        # [B,768]

        v9 = self.adapter9(self.proj9(p9))     # [B,256]
        v12 = self.adapter12(self.proj12(p12)) # [B,256]

        gate = self.level_gate(torch.cat([v9, v12, t], dim=-1))   # [B,256]
        v_s = gate * v9 + (1.0 - gate) * v12   # [B,256]

        # ---------- CMR ----------
        z_v = self.rel_v(v_s)                  # [B,128]
        z_t = self.rel_t(t)                    # [B,128]
        r_mul = F.gelu(z_v) * F.gelu(z_t)      # [B,128] multiplicative relation
        r_diff = torch.abs(z_v - z_t)          # [B,128] difference relation

        gv = F.normalize(global_v, dim=-1)
        gt = F.normalize(global_t, dim=-1)
        s_global = (gv * gt).sum(dim=-1, keepdim=True)   # [B,1]

        r_raw = torch.cat([r_mul, r_diff, s_global], dim=-1)   # [B,257]
        r_c = self.relation_encoder(r_raw)     # [B,256]

        # ---------- RCC ----------
        conf = self.confidence(torch.cat([v_s, t, r_c, s_global], dim=-1))  # [B,256]
        r_hat = conf * r_c                     # [B,256]

        self._last_stats = {
            "level_gate_mean": round(gate.mean().item(), 4),
            "level_gate_std": round(gate.std().item(), 4),
            "confidence_mean": round(conf.mean().item(), 4),
            "confidence_std": round(conf.std().item(), 4),
            "global_cos": round(s_global.mean().item(), 4),
            "relation_norm": round(r_hat.norm(dim=-1).mean().item(), 4),
        }

        return r_hat
