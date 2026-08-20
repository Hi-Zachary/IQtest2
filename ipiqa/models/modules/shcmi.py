"""SHCMI / DP-HCMI -- Discrepancy-aware Prompt-conditioned HCMI (主创新 2).

Adapted from CHPNet (IEEE TBC 2026, github.com/NUIST-Videocoding/CHPNet)
``MSA_T`` + ``DualAttn`` logic, but:
  - visual backbone kept as MSQR/plain fine-coarse tokens (no second backbone),
  - prompt comes from CLIP text tokens (token-level hidden states + mask),
  - fine/coarse level each do bidirectional visual<->prompt interaction,
  - adaptive scale gate fuses the two cross-modal representations.

v6 (改进5.md)：在 SHCMI 之上增加 **DP-HCMI** 的两个机制（各自可开关，
``use_prompt_weight=False`` / ``use_align_bias=False`` 时退化为标准 SHCMI，
消融用）：
  1. Prompt-conditioned text weighting：prompt_gate 对 text tokens 学习
     softmax 权重，突出与质量/一致性相关的 prompt token。
  2. Discrepancy-aware attention bias：cross-attention 中
     ``attn_score -= beta * (1 - cos(V, T))``，显式建模图像与 prompt 的
     语义不一致——匹配区域增强、不匹配区域抑制（AIGC 条件一致性评价）。

v7 (改进6.md) 两处修复：
  1. **Prompt weighting 修复（第 5 节）**：softmax 前对 padding token
     ``masked_fill(-inf)``，且 ``weight *= valid_len``——所有有效 token 等权时
     weight≈1，不再把 CLIP token 整体幅值缩小 1/valid_length。
  2. **Discrepancy bias 修复（第 6 节）**：移除双重 ``tanh(beta_align)``。
     原实现 ``_align_bias`` 内部已经乘 ``-tanh(beta)``，forward 又乘一次
     ``tanh(beta)``，net 变成 ``-tanh(beta)^2 * D``，在 ``beta=0`` 时一阶梯度为
     0、无法学习。v7 只保留一次 gate：``bias = -tanh(beta_align) * D``。

v5 的 PCS-HCMI（alignment-guided bias）在本版升级为 discrepancy-guided。

Output semantics (v2): SHCMI outputs a **cross-modal residual** ``delta_c``;
the caller combines it as ``c = t0 + tanh(lambda_shcmi) * delta_c``.

Data flow (design sections 8-13):
    T0 [B, L, C_text] --Linear--> T [B, L, D]
    (DP) T = prompt_weight * T
    T_e = T + eta * MSA_T(T)
    fine :  V_f' = F_fine + alpha_f * CA(F_fine, T_e);  T_f' = T_e + beta_f * CA(T_e, F_fine)
    coarse: V_c' = F_coarse + alpha_c * CA(F_coarse, T_e); T_c' = T_e + beta_c * CA(T_e, F_coarse)
    C_f = MLP([MeanPool(V_f'); MaskedMeanPool(T_f')])
    C_c = MLP([MeanPool(V_c'); MaskedMeanPool(T_c')])
    g_s = Sigmoid(MLP([C_f; C_c]))
    F_cross = g_s * C_f + (1 - g_s) * C_c   == delta_c
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.modules.attention import (
    CrossAttention,
    MSA_T,
    Mlp,
    masked_mean_pool,
)


class SHCMI(nn.Module):
    def __init__(self, text_dim=512, dim=256, num_heads=4, mlp_ratio=2.0,
                 drop=0.1, gamma_init=0.01, use_multi_kernel=True,
                 use_prompt_weight=False, use_align_bias=False,
                 beta_init=0.0):
        super().__init__()
        self.dim = dim
        self.use_multi_kernel = use_multi_kernel
        self.use_prompt_weight = use_prompt_weight
        self.use_align_bias = use_align_bias

        self.text_proj = nn.Linear(text_dim, dim)

        # PCS-HCMI: prompt-conditioned text weighting
        if use_prompt_weight:
            self.prompt_gate = nn.Sequential(
                nn.Linear(dim, max(dim // 2, 8)),
                nn.GELU(),
                nn.Linear(max(dim // 2, 8), 1),
            )
        # PCS-HCMI: alignment-guided attention bias
        if use_align_bias:
            self.beta_align = nn.Parameter(torch.tensor(float(beta_init)))

        if use_multi_kernel:
            self.msa_t = MSA_T(dim, dim, drop)
        self.eta = nn.Parameter(torch.tensor(float(gamma_init)))

        # fine-level bidirectional cross-modal
        self.fine_cross_v = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        self.fine_cross_t = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        # coarse-level bidirectional cross-modal
        self.coarse_cross_v = CrossAttention(dim, dim, heads=num_heads, dropout=drop)
        self.coarse_cross_t = CrossAttention(dim, dim, heads=num_heads, dropout=drop)

        # per-scale cross-modal representations
        self.mlp_f = Mlp(dim * 2, int(dim * 2 * mlp_ratio), dim, drop=drop)
        self.mlp_c = Mlp(dim * 2, int(dim * 2 * mlp_ratio), dim, drop=drop)

        # adaptive scale gate
        self.scale_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.alpha_f = nn.Parameter(torch.tensor(float(gamma_init)))
        self.beta_f = nn.Parameter(torch.tensor(float(gamma_init)))
        self.alpha_c = nn.Parameter(torch.tensor(float(gamma_init)))
        self.beta_c = nn.Parameter(torch.tensor(float(gamma_init)))

        self._last_scale_gate = None
        self._last_prompt_weight = None

    def _discrepancy(self, V, T):
        """Discrepancy-aware attention bias（DP-HCMI, 改进6.md 第 6 节）。

        discrepancy matrix: D_ij = 1 - cos(V_i, T_j)，[B, Nv, Nt]
        只返回 raw discrepancy；单次 gate 由 forward 统一乘 ``-tanh(beta_align)``：
            匹配区域（cos 高 → D 小）几乎不减分；
            不匹配区域（cos 低 → D 大）被抑制。
        v7 修复：原实现 `_align_bias` 内部自带 -tanh(beta)，forward 又乘一次
        tanh(beta)，net 为 -tanh(beta)^2 * D，beta=0 时一阶梯度为 0。现只保留
        一次 gate。
        """
        V_norm = F.normalize(V, dim=-1)
        T_norm = F.normalize(T, dim=-1)
        cos_sim = torch.matmul(V_norm, T_norm.transpose(1, 2))   # [B, Nv, Nt]
        return 1.0 - cos_sim                                     # [B, Nv, Nt]

    def forward(self, fine, coarse, text_tokens, text_mask):
        # fine: [B, Nf, D], coarse: [B, Nc, D]
        # text_tokens: [B, L, C_text], text_mask: [B, L]
        T = self.text_proj(text_tokens)          # [B, L, D]

        # DP-HCMI: prompt-conditioned text weighting（v7 修复：mask + 保幅值）
        if self.use_prompt_weight:
            score = self.prompt_gate(T).squeeze(-1)              # [B, L]
            score = score.masked_fill(~text_mask, float("-inf"))  # padding 不参与 softmax
            weight = torch.softmax(score, dim=1)                 # [B, L]
            valid_len = text_mask.sum(dim=1, keepdim=True).clamp_min(1)
            weight = weight * valid_len                          # 均匀时 ≈1，不缩小幅值
            self._last_prompt_weight = weight.detach().float()
            T = T * weight.unsqueeze(-1)                         # [B, L, D]

        # text multi-kernel enhancement (residual)
        if self.use_multi_kernel:
            delta_t = self.msa_t(T)
            T_e = T + self.eta * delta_t
        else:
            T_e = T

        # DP-HCMI: discrepancy-guided attention bias（v7 只保留一次 tanh gate）
        if self.use_align_bias:
            disc_f = self._discrepancy(fine, T_e)
            disc_c = self._discrepancy(coarse, T_e)
            bias_f = -torch.tanh(self.beta_align) * disc_f
            bias_c = -torch.tanh(self.beta_align) * disc_c
        else:
            bias_f = None
            bias_c = None

        # fine-level bidirectional interaction
        V_f = fine + self.alpha_f * self.fine_cross_v(
            fine, T_e, mask=text_mask, attn_bias=bias_f)
        T_f = T_e + self.beta_f * self.fine_cross_t(T_e, fine)

        # coarse-level bidirectional interaction
        V_c = coarse + self.alpha_c * self.coarse_cross_v(
            coarse, T_e, mask=text_mask, attn_bias=bias_c)
        T_c = T_e + self.beta_c * self.coarse_cross_t(T_e, coarse)

        # per-scale cross-modal representations
        v_f = V_f.mean(dim=1)                     # [B, D]
        t_f = masked_mean_pool(T_f, text_mask)    # [B, D]
        C_f = self.mlp_f(torch.cat([v_f, t_f], dim=-1))

        v_c = V_c.mean(dim=1)                     # [B, D]
        t_c = masked_mean_pool(T_c, text_mask)    # [B, D]
        C_c = self.mlp_c(torch.cat([v_c, t_c], dim=-1))

        # adaptive scale fusion
        g_s = torch.sigmoid(self.scale_gate(torch.cat([C_f, C_c], dim=-1)))
        self._last_scale_gate = g_s.detach().float()
        delta_c = g_s * C_f + (1.0 - g_s) * C_c   # [B, D]

        return delta_c


