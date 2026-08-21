"""DAPS -- Dual-Branch Adaptive Patch Scoring (V16 quality module).

调整/改进3.md (V16): MANIQA-style patch-wise subjective quality prediction，
但与 DG-MPQ（representation-level）职责不同——DAPS 是 prediction-level：
每个 patch 预测局部质量 latent score + 对整体主观评分的 importance（softmax），
加权得到 patch-wise quality，再结合 patch score 分布统计做 Global Calibration，
输出 scalar residual，注入 ``q = q_base + tanh(lambda_p) * delta_patch``。

三阶段：
    1. Patch Quality Encoding        (H12 patch -> 256D)
    2. Dual-Branch Score-Importance  (score + softmax importance -> weighted q_patch)
    3. Distribution-Aware Calibration(分布统计 + h_q -> scalar residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAPS(nn.Module):
    def __init__(
        self,
        width=768,
        dim=256,
        hidden=128,
    ):
        super().__init__()

        self.patch_encoder = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim),
            nn.GELU(),
        )

        self.score_branch = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.weight_branch = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.calibration = nn.Sequential(
            nn.Linear(dim + 6, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self._last_stats = None

    def forward(
        self,
        h12,
        h_q,
    ):
        # h12: [B,197,768]（完整传入，内部去 CLS）；h_q: [B,256] 当前 quality representation
        P = self.patch_encoder(h12[:, 1:, :])      # [B,196,256]

        score = self.score_branch(P)               # [B,196,1]

        weight_logits = self.weight_branch(P)
        weight = F.softmax(weight_logits, dim=1)   # [B,196,1]

        q_patch = (weight * score).sum(dim=1)      # [B,1]

        score_mean = score.mean(dim=1)             # [B,1]
        score_std = score.std(dim=1, unbiased=False)  # [B,1]
        score_min = score.min(dim=1).values        # [B,1]
        score_max = score.max(dim=1).values        # [B,1]
        weight_entropy = (-weight * torch.log(weight + 1e-8)).sum(dim=1)  # [B,1]

        descriptor = torch.cat([
            h_q,
            q_patch,
            score_mean,
            score_std,
            score_min,
            score_max,
            weight_entropy,
        ], dim=-1)                                 # [B,262]

        delta_q = self.calibration(descriptor)     # [B,1]

        self._last_stats = {
            "patch_score_mean": round(score_mean.mean().item(), 4),
            "patch_score_std": round(score_std.mean().item(), 4),
            "weight_entropy": round(weight_entropy.mean().item(), 4),
            "weight_max": round(weight.max(dim=1).values.mean().item(), 4),
            "q_patch": round(q_patch.mean().item(), 4),
            "delta_q": round(delta_q.abs().mean().item(), 4),
        }

        return delta_q
