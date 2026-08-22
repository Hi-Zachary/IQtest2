"""MGSC -- Multi-Granularity Semantic Correspondence (V23 alignment module).

调整/改进11.md: 同时建模全局与局部图文语义一致性。

    Global:  s_g = cos(global_v, global_t)
    Local:   prompt 片段 (token chunks) 与 image patch tokens 的 fragment-patch 相似度
             -> per-fragment Top-K matching -> fragment importance 加权 -> s_l
    Fusion:  s_align = sigmoid(eta)*s_g + (1-sigmoid(eta))*s_l
    输出:     s_g, s_l, s_align（供 alignment 分支使用）

使用冻结 CLIP 的 visual_projection / text_projection 映射到 512D joint space。
prompt 片段用 token 级 chunking（可复现、低成本、无需额外模型）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGSC(nn.Module):
    def __init__(self, num_fragments=4, topk=3, dim=256):
        super().__init__()
        self.num_fragments = num_fragments
        self.topk = topk

        # 把 s_align 标量扩展成 256D 供 alignment fusion 使用
        self.sem_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.eta = nn.Parameter(torch.tensor(0.0))   # beta = sigmoid(eta) -> 0.5

        self._last_stats = None

    def forward(self, h12, text_tokens, text_mask, global_v, global_t,
                visual_projection, text_projection):
        # h12: [B,197,768] patch tokens；text_tokens: [B,77,512] 已过 text model（未 projection）
        V = F.normalize(visual_projection(h12[:, 1:, :]), dim=-1)   # [B,196,512]
        T = F.normalize(text_projection(text_tokens), dim=-1)       # [B,77,512]

        # global
        gv = F.normalize(global_v, dim=-1)
        gt = F.normalize(global_t, dim=-1)
        s_g = (gv * gt).sum(dim=-1, keepdim=True)                   # [B,1]

        # ---- 局部：prompt 片段 (token chunk) 与 patch 匹配 ----
        B = h12.shape[0]
        valid = text_mask.bool()
        valid_len = valid.sum(dim=-1).clamp_min(1)                  # [B]

        # 把 valid token 分成 K 个连续片段，逐片 top-K patch 匹配
        K = self.num_fragments
        frag_scores = []
        frag_weights = []
        for k in range(K):
            # 每片段长度: valid_len / K（取整，最后一段补足）
            start = (valid_len * k) // K
            end = (valid_len * (k + 1)) // K
            idx = torch.arange(T.shape[1], device=T.device).unsqueeze(0)  # [1,77]
            in_frag = (idx >= start.unsqueeze(-1)) & (idx < end.unsqueeze(-1)) & valid
            # 片段 token 特征
            denom = in_frag.sum(dim=-1, keepdim=True).clamp_min(1)
            t_k = (T * in_frag.unsqueeze(-1).to(T.dtype)).sum(dim=1) / denom   # [B,512]
            t_k = F.normalize(t_k, dim=-1)
            # fragment->patch 相似度（只对片段内 token 的 top-k over patches）
            S_k = T * in_frag.unsqueeze(-1).to(T.dtype)             # [B,77,512]
            S_k = S_k.sum(dim=1) / denom                            # [B,512] = t_k
            sims = (t_k.unsqueeze(1) * V).sum(dim=-1)               # [B,196]
            tk = min(self.topk, V.shape[1])
            topk_vals = torch.topk(sims, k=tk, dim=-1).values.mean(dim=-1, keepdim=True)  # [B,1]
            frag_scores.append(topk_vals)
            # fragment importance: cos(t_k, t_g)
            a_k = (t_k * gt).sum(dim=-1, keepdim=True)              # [B,1]
            frag_weights.append(a_k)

        frag_scores = torch.cat(frag_scores, dim=-1)                # [B,K]
        frag_weights = torch.cat(frag_weights, dim=-1)              # [B,K]
        alpha = F.softmax(frag_weights, dim=-1)                     # [B,K]
        s_l = (alpha * frag_scores).sum(dim=-1, keepdim=True)       # [B,1]

        # ---- global-local fusion ----
        beta = torch.sigmoid(self.eta)
        s_align = beta * s_g + (1.0 - beta) * s_l                   # [B,1]
        s_embed = self.sem_proj(torch.cat([s_g, s_l], dim=-1))      # [B,256]

        with torch.no_grad():
            self._last_stats = {
                "global_sim_mean": round(s_g.mean().item(), 4),
                "local_sim_mean": round(s_l.mean().item(), 4),
                "local_sim_std": round(s_l.std().item(), 4),
                "beta": round(beta.item(), 4),
                "frag_w_entropy": round(
                    (-alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean().item(), 4),
                "frag_count": round(K, 1),
            }

        return s_align, s_embed
