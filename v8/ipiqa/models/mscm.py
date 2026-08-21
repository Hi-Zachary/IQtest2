"""MSCM -- Multi-Level Semantic Correspondence Modeling (V10 alignment branch).

调整/改进6.md (V10): 替代 HCMI / TCAP。不再学习跨模态特征重写或注意力检索，
而是直接在 CLIP 已训练好的视觉-文本共享空间上构造显式 patch-wise correspondence
maps，通过轻量卷积建模空间 correspondence，并融合多层局部统计与 CLIP 全局相似度。

三个层次：
    Level 1 (Patch):      H9/H12 patch token ↔ text 的 cosine similarity maps
    Level 2 (Spatial):    多尺度 similarity maps 经 3x3 卷积建模空间对应模式
    Level 3 (Global):     CLIP 原始 global similarity + 多层 map 统计量

输出 a_corr: [B, D]，直接进入 Alignment fusion（不做 residual gate）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCM(nn.Module):
    def __init__(self, visual_width=768, dim=256, spatial_dim=64, drop=0.1):
        super().__init__()
        self.dim = dim

        self.proj9 = nn.Sequential(
            nn.LayerNorm(visual_width),
            nn.Linear(visual_width, dim),
        )
        self.proj12 = nn.Sequential(
            nn.LayerNorm(visual_width),
            nn.Linear(visual_width, dim),
        )

        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, spatial_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        corr_dim = spatial_dim * 2 + 7   # f_mean(64) + f_max(64) + 6 stats + s_global(1)
        self.corr_proj = nn.Sequential(
            nn.Linear(corr_dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.LayerNorm(dim),
        )

        self._last_stats = None

    def forward(self, h9, h12, t0, global_v, global_t):
        # h9/h12: [B,196,width] (caller 已去 CLS)；t0/global_v/global_t: [B,D]/[B,P]
        B = h9.shape[0]

        V9 = F.normalize(self.proj9(h9), dim=-1)
        V12 = F.normalize(self.proj12(h12), dim=-1)
        T = F.normalize(t0, dim=-1)

        # Level 1: patch-wise semantic correspondence
        S9 = (V9 * T.unsqueeze(1)).sum(dim=-1)      # [B,196]
        S12 = (V12 * T.unsqueeze(1)).sum(dim=-1)    # [B,196]

        # Level 2: spatial correspondence (multi-level maps -> conv)
        S9_map = S9.view(B, 1, 14, 14)
        S12_map = S12.view(B, 1, 14, 14)
        S = torch.cat([S9_map, S12_map], dim=1)     # [B,2,14,14]
        Fcorr = self.spatial_fusion(S)              # [B,64,14,14]

        f_mean = Fcorr.mean(dim=(2, 3))             # [B,64]
        f_max = Fcorr.amax(dim=(2, 3))              # [B,64]

        # Level 3: multi-level map statistics
        s9_mean = S9.mean(dim=1, keepdim=True)
        s9_max = S9.amax(dim=1, keepdim=True)
        s9_min = S9.amin(dim=1, keepdim=True)
        s12_mean = S12.mean(dim=1, keepdim=True)
        s12_max = S12.amax(dim=1, keepdim=True)
        s12_min = S12.amin(dim=1, keepdim=True)

        # global CLIP similarity (原始 projection space，不经 256D 投影)
        gv = F.normalize(global_v, dim=-1)
        gt = F.normalize(global_t, dim=-1)
        s_global = (gv * gt).sum(dim=-1, keepdim=True)   # [B,1]

        corr = torch.cat([
            f_mean, f_max,
            s9_mean, s9_max, s9_min,
            s12_mean, s12_max, s12_min,
            s_global,
        ], dim=-1)                                       # [B,135]

        self._last_stats = {
            "s9_mean": round(s9_mean.mean().item(), 4),
            "s9_std": round(S9.std(dim=1).mean().item(), 4),
            "s12_mean": round(s12_mean.mean().item(), 4),
            "s12_std": round(S12.std(dim=1).mean().item(), 4),
            "s_global": round(s_global.mean().item(), 4),
            "f_max_mean": round(f_max.mean().item(), 4),
        }

        return self.corr_proj(corr)                      # [B,D]  a_corr
