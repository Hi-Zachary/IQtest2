"""HCMI-ViT -- Prompt-conditioned Hierarchical Cross-Modal Interaction for ViT
(v8 Module 2, V8-Slim Final).

调整/改进2.md + 改进3.md (V8-Slim Final):
  - 删除 discrepancy-guided attention bias（beta_align≈0.002 无贡献）；
  - 删除 MSA-T multi-kernel text enhancement（S2 显示删除后 A-SRCC 不降反升 +0.0012，
    省 ~1.18M 参数）；
  - 删除后不再使用 DP 前缀，改名为 HCMI-ViT。

Hierarchy 来自 ViT 语义深度：
    Detail-level   : Layer 6  patch tokens   (detail_proj)
    Semantic-level : Layer 12 patch tokens   (semantic_proj)

Pipeline (per level, bidirectional visual <-> prompt interaction):
    T = text_token_proj(text_tokens)                       [B,77,D]
    T = prompt_weight(T)                   # mask + scale preservation
    V' = V + alpha * CrossAttn(V, T);  T' = T + beta*CrossAttn(T, V)
    C_detail / C_semantic = MLP(mean_pool(V'), masked_mean_pool(T'))
    g_h = sigmoid(hierarchy_gate([C_detail, C_semantic]))
    delta_a = g_h * C_detail + (1-g_h) * C_semantic       # adaptive fusion
    delta_a = align_out_norm(delta_a)      # module-internal LayerNorm

Output is a cross-modal residual; caller injects ``c = t0 + tanh(lambda_a)*delta_a``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.attention import (
    CrossAttention,
    Mlp,
    masked_mean_pool,
)


class HcmiVit(nn.Module):
    def __init__(
            self,
            width=768,
            text_width=512,
            dim=256,
            num_heads=4,
            mlp_ratio=1.0,
            drop=0.1,
            use_prompt_weight=True,
            gamma_init=0.01,
    ):
        super().__init__()
        self.dim = dim
        self.use_prompt_weight = use_prompt_weight

        # ViT semantic-depth hierarchy projections
        self.detail_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim),
        )
        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, dim),
        )
        self.text_token_proj = nn.Linear(text_width, dim)

        # prompt weighting
        if use_prompt_weight:
            self.prompt_gate = nn.Sequential(
                nn.Linear(dim, max(dim // 2, 8)),
                nn.GELU(),
                nn.Linear(max(dim // 2, 8), 1),
            )

        # detail-level bidirectional cross-modal
        self.detail_cross_v = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        self.detail_cross_t = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        # semantic-level bidirectional cross-modal
        self.semantic_cross_v = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        self.semantic_cross_t = CrossAttention(dim, dim, heads=num_heads, dropout=drop)

        # per-level cross-modal representations (compact FFN, V8-Slim mlp_ratio=1)
        self.mlp_d = Mlp(dim * 2, int(dim * 2 * mlp_ratio), dim, drop=drop)
        self.mlp_s = Mlp(dim * 2, int(dim * 2 * mlp_ratio), dim, drop=drop)

        # adaptive hierarchy fusion gate
        self.hierarchy_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        # module-internal LayerNorm (residual scale calibration)
        self.align_out_norm = nn.LayerNorm(dim)

        self.alpha_d = nn.Parameter(torch.tensor(float(gamma_init)))
        self.beta_d = nn.Parameter(torch.tensor(float(gamma_init)))
        self.alpha_s = nn.Parameter(torch.tensor(float(gamma_init)))
        self.beta_s = nn.Parameter(torch.tensor(float(gamma_init)))

        self._last_hier_gate = None
        self._last_prompt_stats = None

    def forward(self, h6, h12, text_tokens, text_mask):
        # h6/h12: [B, N, width] patch tokens (caller 已去掉 CLS)
        detail_base = self.detail_proj(h6)        # [B, N, D]
        semantic_base = self.semantic_proj(h12)   # [B, N, D]
        T = self.text_token_proj(text_tokens)     # [B, L, D]

        # prompt-conditioned text weighting（mask + 保幅值）
        if self.use_prompt_weight:
            score = self.prompt_gate(T).squeeze(-1)               # [B, L]
            score = score.masked_fill(~text_mask, float("-inf"))
            probs = torch.softmax(score, dim=1)                   # [B, L]
            valid_len = text_mask.sum(dim=1, keepdim=True).clamp_min(1)
            weight = probs * valid_len                            # 均匀时 ≈1
            # 只统计 valid token 内的权重分布（改进2.md 第 9 节）
            vw = weight[text_mask]
            if vw.numel() > 0:
                # 熵用 valid token 内归一化的 softmax 概率（weight 不是概率分布）
                vp = probs[text_mask]
                vp = vp / vp.sum().clamp_min(1e-12)
                ent = -(vp * torch.log(vp.clamp_min(1e-12))).sum().item()
            else:
                ent = 0.0
            self._last_prompt_stats = {
                "valid_mean": round(vw.mean().item(), 4),
                "valid_std": round(vw.std().item(), 4),
                "max": round(vw.max().item(), 4),
                "entropy": round(ent, 4),
            }
            T = T * weight.unsqueeze(-1)

        # detail-level bidirectional interaction
        V_d = detail_base + self.alpha_d * self.detail_cross_v(
            detail_base, T, mask=text_mask)
        T_d = T + self.beta_d * self.detail_cross_t(T, detail_base)

        # semantic-level bidirectional interaction
        V_s = semantic_base + self.alpha_s * self.semantic_cross_v(
            semantic_base, T, mask=text_mask)
        T_s = T + self.beta_s * self.semantic_cross_t(T, semantic_base)

        # per-level cross-modal representations
        C_d = self.mlp_d(torch.cat([V_d.mean(dim=1), masked_mean_pool(T_d, text_mask)], dim=-1))
        C_s = self.mlp_s(torch.cat([V_s.mean(dim=1), masked_mean_pool(T_s, text_mask)], dim=-1))

        # adaptive hierarchy fusion
        g_h = torch.sigmoid(self.hierarchy_gate(torch.cat([C_d, C_s], dim=-1)))
        self._last_hier_gate = g_h.detach().float()
        delta_a = g_h * C_d + (1.0 - g_h) * C_s       # [B, D]
        delta_a = self.align_out_norm(delta_a)        # module-internal LN

        return delta_a
