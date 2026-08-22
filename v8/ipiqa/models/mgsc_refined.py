"""MGSCRefined -- Multi-Granularity Semantic Correspondence（V23-Refined，精简版）。

调整/改进12.md Phase A：在 V23 MGSC 基础上删除冗余的 beta 标量融合与 scalar
residual，保留 global/local correspondence 完整链条：

    Global: s_g = cos(v_g, t_g)
    Local:  textual fragments (token chunks) -> fragment-patch TopK -> s_l
    Embedding: e_c = correspondence_proj([s_g, s_l])

不再显式计算 s_align，也不再输出用于回归的 scalar residual；
correspondence embedding 直接进入 alignment fusion。

命名规范：textual fragments / contiguous prompt token groups，不声称 noun phrase。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGSCRefined(nn.Module):
    def __init__(self, num_fragments=4, topk=3, dim=256):
        super().__init__()
        self.num_fragments = num_fragments
        self.topk = topk

        # Global-Local Correspondence Embedding（与原 sem_proj 结构一致，可迁移）
        self.correspondence_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self._last_stats = None

    def forward(self, h12, text_tokens, text_mask, global_v, global_t,
                visual_projection, text_projection):
        V = F.normalize(visual_projection(h12[:, 1:, :]), dim=-1)   # [B,196,512]
        T = F.normalize(text_projection(text_tokens), dim=-1)       # [B,77,512]

        # global
        gv = F.normalize(global_v, dim=-1)
        gt = F.normalize(global_t, dim=-1)
        s_g = (gv * gt).sum(dim=-1, keepdim=True)                   # [B,1]

        # ---- local: textual fragments -> fragment-patch TopK ----
        B = h12.shape[0]
        valid = text_mask.bool()
        valid_len = valid.sum(dim=-1).clamp_min(1)
        K = self.num_fragments
        frag_scores = []
        frag_weights = []
        for k in range(K):
            start = (valid_len * k) // K
            end = (valid_len * (k + 1)) // K
            idx = torch.arange(T.shape[1], device=T.device).unsqueeze(0)
            in_frag = (idx >= start.unsqueeze(-1)) & (idx < end.unsqueeze(-1)) & valid
            denom = in_frag.sum(dim=-1, keepdim=True).clamp_min(1)
            t_k = (T * in_frag.unsqueeze(-1).to(T.dtype)).sum(dim=1) / denom   # [B,512]
            t_k = F.normalize(t_k, dim=-1)
            sims = (t_k.unsqueeze(1) * V).sum(dim=-1)               # [B,196]
            tk = min(self.topk, V.shape[1])
            c_k = torch.topk(sims, k=tk, dim=-1).values.mean(dim=-1, keepdim=True)  # [B,1]
            frag_scores.append(c_k)
            a_k = (t_k * gt).sum(dim=-1, keepdim=True)              # [B,1]
            frag_weights.append(a_k)

        frag_scores = torch.cat(frag_scores, dim=-1)                # [B,K]
        frag_weights = torch.cat(frag_weights, dim=-1)              # [B,K]
        alpha = F.softmax(frag_weights, dim=-1)                     # [B,K]
        s_l = (alpha * frag_scores).sum(dim=-1, keepdim=True)       # [B,1]

        # ---- Global-Local Correspondence Embedding ----
        e_c = self.correspondence_proj(torch.cat([s_g, s_l], dim=-1))  # [B,256]

        with torch.no_grad():
            gap = s_g - s_l
            self._last_stats = {
                "global_sim_mean": round(s_g.mean().item(), 4),
                "global_sim_std": round(s_g.std().item(), 4),
                "local_sim_mean": round(s_l.mean().item(), 4),
                "local_sim_std": round(s_l.std().item(), 4),
                "frag_count": round(K, 1),
                "frag_w_entropy": round(
                    (-alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean().item(), 4),
                "gl_gap_mean": round(gap.mean().item(), 4),
                "gl_gap_std": round(gap.std().item(), 4),
            }

        return e_c, s_g, s_l
