"""DP-HCMI-ViT -- Discrepancy-aware Prompt-conditioned Hierarchical
Cross-Modal Interaction for ViT (v8 Module 2).

调整/改进1.md section 7. Keeps the v7 DP-HCMI machinery (validated in the
MSQR_SHCMI_TAF project), but redefines the hierarchy from manual spatial
fine/coarse tokens to ViT semantic-depth levels:

    Detail-level   : Layer 6  patch tokens   (detail_proj)
    Semantic-level : Layer 12 patch tokens   (semantic_proj)

Pipeline (per level, bidirectional visual <-> prompt interaction):
    T = text_token_proj(text_tokens)                       [B,77,D]
    T = prompt_weight(T)                   # mask + scale preservation
    T_e = T + eta * MSA_T(T)               # multi-kernel text enhancement
    bias = -tanh(beta_align) * (1 - cos(V, T_e))   # discrepancy bias
    V' = V + alpha * CrossAttn(V, T_e, bias);  T' = T_e + beta*CrossAttn(T_e, V)
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
    MSA_T,
    Mlp,
    masked_mean_pool,
)


class DpHcmiVit(nn.Module):
    def __init__(
            self,
            width=768,
            text_width=512,
            dim=256,
            num_heads=4,
            mlp_ratio=2.0,
            drop=0.1,
            use_multi_kernel=True,
            use_prompt_weight=True,
            use_align_bias=True,
            gamma_init=0.01,
            beta_init=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_multi_kernel = use_multi_kernel
        self.use_prompt_weight = use_prompt_weight
        self.use_align_bias = use_align_bias

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
        # discrepancy gate (single tanh, 禁止 double gating)
        if use_align_bias:
            self.beta_align = nn.Parameter(torch.tensor(float(beta_init)))

        # multi-kernel text enhancement
        if use_multi_kernel:
            self.msa_t = MSA_T(dim, dim, drop)
        self.eta = nn.Parameter(torch.tensor(float(gamma_init)))

        # detail-level bidirectional cross-modal
        self.detail_cross_v = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        self.detail_cross_t = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        # semantic-level bidirectional cross-modal
        self.semantic_cross_v = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        self.semantic_cross_t = CrossAttention(dim, dim, heads=num_heads, dropout=drop)

        # per-level cross-modal representations
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
        self._last_prompt_weight = None

    def _discrepancy(self, V, T):
        """D_ij = 1 - cos(V_i, T_j)，[B, Nv, Nt]。只返回 raw discrepancy，
        单次 gate 由 forward 统一乘 ``-tanh(beta_align)``。"""
        V_norm = F.normalize(V, dim=-1)
        T_norm = F.normalize(T, dim=-1)
        cos_sim = torch.matmul(V_norm, T_norm.transpose(1, 2))
        return 1.0 - cos_sim

    def forward(self, h6, h12, text_tokens, text_mask):
        # h6/h12: [B, N, width] patch tokens (caller 已去掉 CLS)
        detail_base = self.detail_proj(h6)        # [B, N, D]
        semantic_base = self.semantic_proj(h12)   # [B, N, D]
        T = self.text_token_proj(text_tokens)     # [B, L, D]

        # DP-HCMI: prompt-conditioned text weighting（mask + 保幅值）
        if self.use_prompt_weight:
            score = self.prompt_gate(T).squeeze(-1)               # [B, L]
            score = score.masked_fill(~text_mask, float("-inf"))
            weight = torch.softmax(score, dim=1)                  # [B, L]
            valid_len = text_mask.sum(dim=1, keepdim=True).clamp_min(1)
            weight = weight * valid_len                           # 均匀时 ≈1
            self._last_prompt_weight = weight.detach().float()
            T = T * weight.unsqueeze(-1)

        # multi-kernel text enhancement (residual)
        if self.use_multi_kernel:
            T_e = T + self.eta * self.msa_t(T)
        else:
            T_e = T

        # DP-HCMI: discrepancy-guided attention bias（单次 tanh gate）
        if self.use_align_bias:
            disc_d = self._discrepancy(detail_base, T_e)
            disc_s = self._discrepancy(semantic_base, T_e)
            bias_d = -torch.tanh(self.beta_align) * disc_d
            bias_s = -torch.tanh(self.beta_align) * disc_s
        else:
            bias_d = None
            bias_s = None

        # detail-level bidirectional interaction
        V_d = detail_base + self.alpha_d * self.detail_cross_v(
            detail_base, T_e, mask=text_mask, attn_bias=bias_d)
        T_d = T_e + self.beta_d * self.detail_cross_t(T_e, detail_base)

        # semantic-level bidirectional interaction
        V_s = semantic_base + self.alpha_s * self.semantic_cross_v(
            semantic_base, T_e, mask=text_mask, attn_bias=bias_s)
        T_s = T_e + self.beta_s * self.semantic_cross_t(T_e, semantic_base)

        # per-level cross-modal representations
        C_d = self.mlp_d(torch.cat([V_d.mean(dim=1), masked_mean_pool(T_d, text_mask)], dim=-1))
        C_s = self.mlp_s(torch.cat([V_s.mean(dim=1), masked_mean_pool(T_s, text_mask)], dim=-1))

        # adaptive hierarchy fusion
        g_h = torch.sigmoid(self.hierarchy_gate(torch.cat([C_d, C_s], dim=-1)))
        self._last_hier_gate = g_h.detach().float()
        delta_a = g_h * C_d + (1.0 - g_h) * C_s       # [B, D]
        delta_a = self.align_out_norm(delta_a)        # module-internal LN

        return delta_a
