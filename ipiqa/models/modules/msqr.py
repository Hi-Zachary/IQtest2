"""MSQR -- Multi-Scale Quality Refinement (主创新 1).

Adapted from MS-SCANet (ICASSP 2025, github.com/mithila442/MS-SCANet) onto a
single CLIP RN50 spatial feature:

    spatial [B, 2048, 16, 16]
      -> Conv1x1 (2048 -> D)          Fine map  [B, D, 16, 16]
      -> AvgPool                       Coarse map [B, D, 8, 8]
      -> Channel Attention (per scale)
      -> Spatial Attention  (per scale)
      -> Cross-Scale bidirectional attention (residual, gamma-gated)
      -> F_fine [B, 256, D], F_coarse [B, 64, D]

Only one CLIP image forward is used (design section 4.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.modules.attention import (
    ChannelBlock,
    SpatialBlock,
    CrossScaleAttention,
)


class MSQR(nn.Module):
    def __init__(self, in_channels=2048, dim=256, num_heads=4, mlp_ratio=2.0,
                 drop=0.1, use_channel_attention=True, use_spatial_attention=True,
                 use_cross_scale=True, gamma_init=0.0):
        super().__init__()
        self.dim = dim
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_cross_scale = use_cross_scale

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.GELU(),
        )

        if use_channel_attention:
            self.fine_channel = ChannelBlock(dim, mlp_ratio, drop)
            self.coarse_channel = ChannelBlock(dim, mlp_ratio, drop)
        if use_spatial_attention:
            self.fine_spatial = SpatialBlock(dim, num_heads, mlp_ratio, drop)
            self.coarse_spatial = SpatialBlock(dim, num_heads, mlp_ratio, drop)
        if use_cross_scale:
            self.cross_scale = CrossScaleAttention(dim, num_heads, drop)

        # gamma-gated residuals, small init so we start close to base CLIP
        self.gamma_f = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_c = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, feat):
        # feat: [B, 2048, H, W], H = W = 16 at 512 input
        f0 = self.proj(feat)                          # [B, D, H, W]
        B, C, H, W = f0.shape

        fine_map = f0
        coarse_map = F.avg_pool2d(f0, kernel_size=2)  # [B, D, H//2, W//2]

        fine = fine_map.flatten(2).transpose(1, 2)    # [B, H*W, D]
        coarse = coarse_map.flatten(2).transpose(1, 2)  # [B, H/2*W/2, D]

        # scale-internal enhancement
        if self.use_channel_attention:
            fine = self.fine_channel(fine)
            coarse = self.coarse_channel(coarse)
        if self.use_spatial_attention:
            fine = self.fine_spatial(fine)
            coarse = self.coarse_spatial(coarse)

        # cross-scale bidirectional interaction (residual)
        if self.use_cross_scale:
            delta_f, delta_c = self.cross_scale(fine, coarse)
            fine = fine + self.gamma_f * delta_f
            coarse = coarse + self.gamma_c * delta_c

        return fine, coarse


class QualityAwareMSQRSkip(nn.Module):
    """Visual skip from MSQR outputs (设计 15 节) + Quality-aware Token
    Aggregation (QTA, 改进2.md 第二节).

    v2:        F_fine -> MeanPool --+
                                   +--> concat -> Linear -> delta_v [B, D]
              F_coarse-> MeanPool --+

    QTA (use_qta=True): 不再对所有 token 平均，而是用 quality_gate 学习每个
    token 的质量权重（softmax），加权求和。解决 AIGC-IQA 局部异常敏感问题。
    use_qta=False 时退化为原始 mean pooling（B1/B3 消融用）。
    """

    def __init__(self, dim=256, drop=0.1, use_qta=True):
        super().__init__()
        self.use_qta = use_qta
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        if use_qta:
            self.quality_gate = nn.Sequential(
                nn.Linear(dim, max(dim // 2, 8)),
                nn.GELU(),
                nn.Linear(max(dim // 2, 8), 1),
            )
        self._last_fine_weight = None
        self._last_coarse_weight = None

    def _pool(self, tokens):
        # tokens: [B, N, D]
        if not self.use_qta:
            return tokens.mean(dim=1)  # [B, D]
        score = self.quality_gate(tokens)          # [B, N, 1]
        weight = torch.softmax(score, dim=1)       # [B, N, 1]
        return (weight * tokens).sum(dim=1)        # [B, D]

    def forward(self, fine, coarse):
        if self.use_qta:
            # 记录权重分布供日志
            fs = self.quality_gate(fine)
            cs = self.quality_gate(coarse)
            self._last_fine_weight = torch.softmax(fs, dim=1).detach().float()
            self._last_coarse_weight = torch.softmax(cs, dim=1).detach().float()

        v_f = self._pool(fine)      # [B, D]
        v_c = self._pool(coarse)    # [B, D]
        return self.fc(torch.cat([v_f, v_c], dim=-1))


# 向后兼容别名
MSQRVisualSkip = QualityAwareMSQRSkip
