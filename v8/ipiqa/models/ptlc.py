"""PTLC -- Prompt-Token Late-interaction Calibration (V15 alignment module).

调整/改进2.md (V15): 不学习新的跨模态空间，直接复用冻结 CLIP 已训练的
token-patch shared space（visual_projection / text_projection -> 512D），
做双向 MaxSim late interaction + token reliability weighting + coverage /
weak-coverage / cross-level consistency 校准，输出轻量 alignment residual。

三阶段：
    1. Multi-Level Token Projection   (H9/H12 patches + text tokens -> 512D joint space)
    2. Bidirectional Prompt-Patch Late Interaction (T->I 与 I->T MaxSim)
    3. Coverage-Consistency Calibration (semantic coverage / bottom-k weak / consistency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PTLC(nn.Module):
    def __init__(
        self,
        text_width=512,
        dim=256,
        bottom_k=3,
    ):
        super().__init__()
        self.bottom_k = bottom_k

        # token reliability weighting
        self.token_weight = nn.Sequential(
            nn.Linear(text_width + 3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # PTLC descriptor -> alignment residual
        self.out_proj = nn.Sequential(
            nn.Linear(text_width + 6, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_stats = None

    def forward(
        self,
        v9,
        v12,
        text_tokens,
        text_mask,
        global_v,
        global_t,
    ):
        # v9/v12: [B,196,512]（已用 frozen visual_projection 映射 + 归一化）
        # text_tokens: [B,77,512]（已用 frozen text_projection 映射 + 归一化）
        V9 = F.normalize(v9, dim=-1)
        V12 = F.normalize(v12, dim=-1)
        T = F.normalize(text_tokens, dim=-1)

        valid = text_mask.bool().clone()
        valid[:, 0] = False            # 去掉 BOS；EOS 暂保留

        # ---------- Stage 2: Bidirectional Late Interaction ----------
        S9 = torch.einsum("btd,bpd->btp", T, V9)    # [B,77,196]
        S12 = torch.einsum("btd,bpd->btp", T, V12)  # [B,77,196]

        # text -> image MaxSim
        m9_t = S9.max(dim=-1).values                # [B,77]
        m12_t = S12.max(dim=-1).values              # [B,77]

        # image -> text MaxSim（只对 valid token）
        S9m = S9.masked_fill(~valid.unsqueeze(-1), -1e4)
        S12m = S12.masked_fill(~valid.unsqueeze(-1), -1e4)
        m9_v = S9m.max(dim=1).values                # [B,196]
        m12_v = S12m.max(dim=1).values              # [B,196]

        # multi-level consistency
        m_t = 0.5 * (m9_t + m12_t)                  # [B,77]
        d_t = torch.abs(m9_t - m12_t)               # [B,77]
        m_v = 0.5 * (m9_v + m12_v)                  # [B,196]

        # ---------- Stage 3: token reliability + coverage ----------
        token_feat = torch.cat([
            T,
            m9_t.unsqueeze(-1),
            m12_t.unsqueeze(-1),
            d_t.unsqueeze(-1),
        ], dim=-1)                                  # [B,77,515]

        token_logits = self.token_weight(token_feat).squeeze(-1)
        token_logits = token_logits.masked_fill(~valid, -1e4)
        token_w = F.softmax(token_logits, dim=-1)   # [B,77]

        # semantic coverage
        s_cov = (token_w * m_t).sum(dim=-1, keepdim=True)      # [B,1]

        # cross-level consistency
        s_cons = 1.0 - (token_w * d_t).sum(dim=-1, keepdim=True)  # [B,1]

        # token-aware semantic vector
        t_cov = torch.einsum("bt,btd->bd", token_w, T)          # [B,512]

        # bottom-k weak coverage（只对 valid token）
        weak_scores = []
        for b in range(m_t.size(0)):
            vals = m_t[b][valid[b]]
            k = min(self.bottom_k, vals.numel())
            if k == 0:
                weak = m_t.new_zeros(())
            else:
                weak = torch.topk(vals, k=k, largest=False).values.mean()
            weak_scores.append(weak)
        s_weak = torch.stack(weak_scores).unsqueeze(-1)          # [B,1]

        # visual-side coverage
        s_v_mean = m_v.mean(dim=-1, keepdim=True)                # [B,1]
        s_v_max = m_v.max(dim=-1, keepdim=True).values           # [B,1]

        # global CLIP cosine（原始 projection space）
        gv = F.normalize(global_v, dim=-1)
        gt = F.normalize(global_t, dim=-1)
        s_global = (gv * gt).sum(dim=-1, keepdim=True)           # [B,1]

        # ---------- PTLC descriptor ----------
        desc = torch.cat([
            t_cov,
            s_cov,
            s_weak,
            s_cons,
            s_v_mean,
            s_v_max,
            s_global,
        ], dim=-1)                                              # [B,518]

        delta_a = self.out_proj(desc)                            # [B,256]

        entropy = (-token_w * torch.log(token_w + 1e-8)).sum(dim=-1).mean()

        self._last_stats = {
            "coverage": round(s_cov.mean().item(), 4),
            "weak_coverage": round(s_weak.mean().item(), 4),
            "consistency": round(s_cons.mean().item(), 4),
            "global_cos": round(s_global.mean().item(), 4),
            "token_w_entropy": round(entropy.item(), 4),
            "token_w_max": round(token_w.max(dim=-1).values.mean().item(), 4),
        }

        return delta_a
