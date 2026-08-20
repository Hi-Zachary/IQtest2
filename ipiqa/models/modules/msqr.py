"""MSQR / DMSQR -- Multi-Scale Quality Refinement (主创新 1).

Adapted from MS-SCANet (ICASSP 2025, github.com/mithila442/MS-SCANet) onto a
single CLIP RN50 spatial feature:

    spatial [B, 2048, 16, 16]
      -> Conv1x1 (2048 -> D)          Fine map  [B, D, 16, 16]
      -> AvgPool                       Coarse map [B, D, 8, 8]
      -> (DMSQR) semantic deviation reweight
      -> Channel Attention (per scale)
      -> Spatial Attention  (per scale)
      -> Cross-Scale bidirectional attention (residual, gamma-gated)
      -> F_fine [B, 256, D], F_coarse [B, 64, D]

v5 (改进4.md)：新增 **DMSQR（Distortion-aware MSQR）** semantic deviation
branch —— 用 CLIP global visual 作语义基准，对偏离全局语义的区域（局部生成
异常）以 sigmoid 权重增强（``tokens' = tokens * (1 + tanh(alpha)*weight)``），
``use_deviation=False`` 时退化为原始 MSQR（消融用）。

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
                 use_cross_scale=True, gamma_init=0.0,
                 use_deviation=False, global_dim=1024, alpha_init=0.0):
        super().__init__()
        self.dim = dim
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_cross_scale = use_cross_scale
        self.use_deviation = use_deviation

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.GELU(),
        )

        # DMSQR: semantic deviation branch（改进4.md 第 1 节）
        # 利用 CLIP 视觉特征与全局语义的不一致区域，增强 AIGC 异常区域感知。
        if use_deviation:
            self.global_proj = nn.Linear(global_dim, dim)
            self.deviation_mlp = nn.Sequential(
                nn.Linear(dim, max(dim // 2, 8)),
                nn.GELU(),
                nn.Linear(max(dim // 2, 8), 1),
            )
            self.alpha_dev = nn.Parameter(torch.tensor(float(alpha_init)))

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

        self._last_dev_weight = None

    def _deviation_reweight(self, tokens, global_v):
        """语义偏差重加权（DMSQR）：
            global_token = global_proj(global_v)
            deviation    = |tokens - global_token|
            weight       = sigmoid(MLP(deviation))
            tokens'      = tokens * (1 + alpha * weight)
        tokens: [B, N, D]; global_v: [B, global_dim]
        """
        global_token = self.global_proj(global_v).unsqueeze(1)   # [B, 1, D]
        deviation = (tokens - global_token).abs()                # [B, N, D]
        score = self.deviation_mlp(deviation)                    # [B, N, 1]
        weight = torch.sigmoid(score)                            # [B, N, 1]
        self._last_dev_weight = weight.detach().float()
        return tokens * (1.0 + torch.tanh(self.alpha_dev) * weight)

    def forward(self, feat, global_v=None):
        # feat: [B, 2048, H, W], H = W = 16 at 512 input
        f0 = self.proj(feat)                          # [B, D, H, W]
        B, C, H, W = f0.shape

        fine_map = f0
        coarse_map = F.avg_pool2d(f0, kernel_size=2)  # [B, D, H//2, W//2]

        fine = fine_map.flatten(2).transpose(1, 2)    # [B, H*W, D]
        coarse = coarse_map.flatten(2).transpose(1, 2)  # [B, H/2*W/2, D]

        # DMSQR: 语义偏差重加权（在 channel/spatial/cross-scale 之前）
        if self.use_deviation:
            if global_v is None:
                raise ValueError("DMSQR requires global_v")
            fine = self._deviation_reweight(fine, global_v)
            coarse = self._deviation_reweight(coarse, global_v)

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


class MSQRVisualSkip(nn.Module):
    """Visual skip from MSQR outputs (设计 15 节, v4 恢复):

        F_fine  -> MeanPool --+
                              +--> concat -> Linear -> delta_v [B, D]
        F_coarse-> MeanPool --+

    v3 的 Quality-aware Token Aggregation (QTA) 已在改进3.md 中删除——QTA 未带来
    quality 提升，恢复简单的 mean pooling。
    """

    def __init__(self, dim=256, drop=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, fine, coarse):
        v_f = fine.mean(dim=1)      # [B, D]
        v_c = coarse.mean(dim=1)    # [B, D]
        return self.fc(torch.cat([v_f, v_c], dim=-1))


# 向后兼容别名
QualityAwareMSQRSkip = MSQRVisualSkip
