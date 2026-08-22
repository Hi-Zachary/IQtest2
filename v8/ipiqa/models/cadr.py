"""CADR -- Content-Adaptive Dynamic Regressor (V18 quality module).

调整/改进6.md (V18): HyperIQA 启发（content-conditioned dynamic predictor
parameters），但轻量化适配 CLIP 框架。机制正交于 DG-MPQ：
    DG-MPQ = degradation representation（哪些局部退化被强调）
    CADR   = prediction-rule adaptation（不同内容如何映射退化证据到质量）

    conditioner  = global_v (CLIP 全局视觉语义 content)
    target input = h_q (DG-MPQ 增强 + 文本的质量表征)

    c = content_proj(global_v)
    [Δw(x), Δb(x)] = hyper(c)              # 样本专属回归参数
    Δq(x) = (Δw^T LN(h_q))/sqrt(D) + Δb
    q = q_base + Δq(x)                      # residual dynamic head

HyperNet 最后一层 zero-init：训练开始时 Δw=Δb=0，故 Full 初始函数 == B1。
不设 outer λ gate。参数量 ~0.1M。
"""

import math
import torch
import torch.nn as nn


class CADR(nn.Module):
    def __init__(self, content_width, dim=256, hidden=128):
        super().__init__()

        self.content_proj = nn.Sequential(
            nn.LayerNorm(content_width),
            nn.Linear(content_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.hyper = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim + 1),
        )

        self.target_norm = nn.LayerNorm(dim)

        # Initial Full function == B1 function (delta=0)
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)

        self._last_stats = None

    def forward(self, global_v, h_q):
        c = self.content_proj(global_v)
        params = self.hyper(c)

        delta_w = params[:, :-1]
        delta_b = params[:, -1:]

        z = self.target_norm(h_q)

        delta_q = (
            (delta_w * z).sum(dim=-1, keepdim=True)
            / math.sqrt(z.shape[-1])
            + delta_b
        )

        with torch.no_grad():
            self._last_stats = {
                "dw_norm": delta_w.norm(dim=-1).mean().item(),
                "dw_batch_std": delta_w.std(dim=0).mean().item(),
                "db_abs": delta_b.abs().mean().item(),
                "delta_abs": delta_q.abs().mean().item(),
                "content_norm": c.norm(dim=-1).mean().item(),
            }

        return delta_q
