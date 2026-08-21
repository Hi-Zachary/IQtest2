"""MSQRNetV15 -- Shared Visual LoRA + DG-MPQ(Q) + PTLC(A).

调整/改进2.md (V15): 迁移自 V14。DG-MPQ 完全保留；MSRC 替换为 PTLC
（Prompt-Token Late-interaction Calibration）。

关键：
    - 复用冻结 CLIP projection 把 H9/H12 patches + text tokens 映射到 512D joint space，
      不学习新的跨模态空间；双向 MaxSim late interaction + token 加权 coverage。
    - Alignment baseline 改回 2D [v_a, t_a]，PTLC 只作为 residual：
        h_a = h0_a + tanh(lambda_a) * delta_a   （lambda_a init 0.01，保护强 R0 baseline）
    - alignment_stopgrad=True：L_A 不更新 Visual LoRA（B1-Q ≈ Full-Q 可验证）。
    - PTLC always instantiate（use_ptlc 只控制 forward）。

V8~V14 历史代码（msrc/mscm/model_v12/model_v14）保留但不使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.dg_mpq import DgMpq
from ipiqa.models.ptlc import PTLC

from ipiqa.common.registry import registry


@registry.register_model("msqr_dgmpq_ptlc_v15")
class MSQRNetV15(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=2,
            dim=256,
            drop=0.0,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_dg_mpq=True,
            use_ptlc=True,
            ptlc_bottom_k=3,
            alignment_stopgrad=True,
            alignment_gate_init=0.01,
            freeze_visual=True,
            freeze_text=True,
            outer_gate_init=0.01,
            lora_lr_scale=1.0,
            module_lr_scale=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.use_lora = use_lora
        self.use_dg_mpq = use_dg_mpq
        self.use_ptlc = use_ptlc
        self.alignment_stopgrad = alignment_stopgrad
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.lora_lr_scale = lora_lr_scale
        self.module_lr_scale = module_lr_scale
        self.outer_gate_init = float(outer_gate_init)

        # ---------- backbone（共享单 LoRA） ----------
        self.backbone = CLIPViTBackbone(
            model_name=model_name,
            context_length=context_length,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_visual=freeze_visual,
            freeze_text=freeze_text,
        )

        # ===================== Quality branch（只受 L_Q） =====================
        self.quality_visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.quality_text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.quality_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(drop),
        )
        self.quality_head = nn.Linear(dim, 1)
        self.dg_mpq = DgMpq(width=self.backbone.visual_width, dim=dim)   # always
        self.lambda_q = nn.Parameter(torch.tensor(self.outer_gate_init))

        # ===================== Alignment branch（只受 L_A） =====================
        self.alignment_visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_base_fusion = nn.Sequential(   # 2D baseline: [v_a, t_a]
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(drop),
        )
        self.align_head = nn.Linear(dim, 1)
        self.lambda_a = nn.Parameter(torch.tensor(float(alignment_gate_init)))
        self.ptlc = PTLC(                        # always
            text_width=self.backbone.projection_dim,
            dim=dim,
            bottom_k=ptlc_bottom_k,
        )

        self._last_ratios = {}

    # ---------------- train/eval override ----------------
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_text:
            self.backbone.clip.text_model.eval()
        if self.freeze_visual:
            self.backbone.clip.vision_model.eval()
        return self

    # ------------------------------------------------------------------ #
    def forward(self, x, text):
        """x: [B,3,224,224]; text: list[str] -> [B,2] = (quality, alignment)."""
        input_ids, attention_mask = self.backbone.tokenize(text)
        input_ids = input_ids.to(x.device)
        attention_mask = attention_mask.to(x.device)

        feat = self.backbone(x, input_ids, attention_mask)
        global_v = feat["global_v"]
        global_t = feat["global_t"]
        hs = feat["vision_hidden"]

        t_q = self.quality_text_proj(global_t)
        t_a = self.alignment_text_proj(global_t)

        ratios = {}

        # ===================== Quality branch =====================
        v0_q = self.quality_visual_proj(global_v)
        if self.use_dg_mpq:
            delta_q = self.dg_mpq(hs[3], hs[6], hs[9], hs[12])
            q_scale = torch.tanh(self.lambda_q)
            v_q = v0_q + q_scale * delta_q
            ratios["raw_quality_ratio"] = (
                delta_q.norm(dim=-1).mean() / v0_q.norm(dim=-1).mean()
            ).item()
            ratios["quality_ratio"] = (
                (q_scale * delta_q).norm(dim=-1).mean() / v0_q.norm(dim=-1).mean()
            ).item()
        else:
            v_q = v0_q
        h_q = self.quality_fusion(torch.cat([v_q, t_q], dim=-1))
        q = self.quality_head(h_q)

        # ===================== Alignment branch（不用 v_q；第一轮 stop-gradient） =====================
        if self.alignment_stopgrad:
            global_v_a = global_v.detach()
            global_t_a = global_t.detach()
            h9_a = hs[9].detach()
            h12_a = hs[12].detach()
            text_tokens_a = feat["text_tokens"].detach()
        else:
            global_v_a = global_v
            global_t_a = global_t
            h9_a = hs[9]
            h12_a = hs[12]
            text_tokens_a = feat["text_tokens"]

        v_a = self.alignment_visual_proj(global_v_a)
        t_a = self.alignment_text_proj(global_t_a)
        h0_a = self.alignment_base_fusion(torch.cat([v_a, t_a], dim=-1))   # [B,256]

        if self.use_ptlc:
            # 复用冻结 CLIP projection -> 512D joint space
            V9 = self.backbone.clip.visual_projection(h9_a[:, 1:, :])      # [B,196,512]
            V12 = self.backbone.clip.visual_projection(h12_a[:, 1:, :])    # [B,196,512]
            T = self.backbone.clip.text_projection(text_tokens_a)          # [B,77,512]

            delta_a = self.ptlc(
                V9,
                V12,
                T,
                feat["text_mask"],
                global_v_a,
                global_t_a,
            )                                                            # [B,256]

            a_scale = torch.tanh(self.lambda_a)
            h_a = h0_a + a_scale * delta_a
            ratios["raw_align_ratio"] = (
                delta_a.norm(dim=-1).mean() / h0_a.norm(dim=-1).mean()
            ).item()
            ratios["align_ratio"] = (
                (a_scale * delta_a).norm(dim=-1).mean() / h0_a.norm(dim=-1).mean()
            ).item()
        else:
            h_a = h0_a

        a = self.align_head(h_a)

        self._last_ratios = ratios
        return torch.cat([q, a], dim=-1)

    # ---------------- logging ----------------
    def get_gate_log(self):
        out = dict(self._last_ratios)
        if self.use_dg_mpq:
            out["lambda_q"] = torch.tanh(self.lambda_q).item()
            dev = getattr(self.dg_mpq, "_last_deviation", None)
            if dev is not None:
                out["deviation_mean"] = round(dev.mean().item(), 4)
                out["deviation_std"] = round(dev.std().item(), 4)
            pw = getattr(self.dg_mpq, "_last_weight", None)
            if pw is not None:
                out["patch_w_mean"] = round(pw.mean().item(), 4)
                out["patch_w_std"] = round(pw.std().item(), 4)
        if self.use_ptlc:
            out["lambda_a"] = torch.tanh(self.lambda_a).item()
            st = getattr(self.ptlc, "_last_stats", None)
            if st is not None:
                out.update({f"ptlc_{k}": v for k, v in st.items()})
        return out

    # ------------------------------------------------------------------ #
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        lora_prefixes = ("backbone.clip.vision_model.",)
        lora_wd, lora_nowd, new_wd, new_nowd = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_nowd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
            if any(n.startswith(pfx) for pfx in lora_prefixes):
                (lora_nowd if is_nowd else lora_wd).append(p)
            else:
                (new_nowd if is_nowd else new_wd).append(p)
        return [
            {"params": lora_wd, "weight_decay": weight_decay,
             "lr_scale": self.lora_lr_scale * lr_scale},
            {"params": lora_nowd, "weight_decay": 0,
             "lr_scale": self.lora_lr_scale * lr_scale},
            {"params": new_wd, "weight_decay": weight_decay,
             "lr_scale": self.module_lr_scale * lr_scale},
            {"params": new_nowd, "weight_decay": 0,
             "lr_scale": self.module_lr_scale * lr_scale},
        ]

    def show_n_params(self, return_str=True):
        tot = sum(p.numel() for p in self.parameters())
        if return_str:
            return "{:.1f}M".format(tot / 1e6)
        return tot

    def trainable_summary(self):
        def cnt(prefix):
            return sum(p.numel() for n, p in self.named_parameters()
                       if n.startswith(prefix) and p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith("backbone.clip.vision_model.") and p.requires_grad)
        return {
            "total": self.show_n_params(),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "lora": lora,
            "dg_mpq": cnt("dg_mpq."),
            "ptlc": cnt("ptlc."),
            "quality_branch": cnt("quality_") + cnt("lambda_q"),
            "alignment_branch": cnt("alignment_") + cnt("lambda_a"),
            "text_trainable": sum(p.numel() for n, p in self.named_parameters()
                                  if n.startswith("backbone.clip.text_model.") and p.requires_grad),
        }

    @classmethod
    def from_config(cls, cfg):
        return cls(
            model_name=cfg.get('model_name', 'ckpt/clip-vit-base-patch16'),
            context_length=cfg.get('context_length', 77),
            output_dim=cfg.get('output_dim', 2),
            dim=cfg.get('dim', 256),
            drop=cfg.get('dropout_rate', 0.0),
            use_lora=cfg.get('use_lora', True),
            lora_r=cfg.get('lora_r', 4),
            lora_alpha=cfg.get('lora_alpha', 8),
            lora_dropout=cfg.get('lora_dropout', 0.0),
            use_dg_mpq=cfg.get('use_dg_mpq', True),
            use_ptlc=cfg.get('use_ptlc', True),
            ptlc_bottom_k=cfg.get('ptlc_bottom_k', 3),
            alignment_stopgrad=cfg.get('alignment_stopgrad', True),
            alignment_gate_init=cfg.get('alignment_gate_init', 0.01),
            freeze_visual=cfg.get('freeze_visual', True),
            freeze_text=cfg.get('freeze_text', True),
            outer_gate_init=cfg.get('outer_gate_init', 0.01),
            lora_lr_scale=cfg.get('lora_lr_scale', 1.0),
            module_lr_scale=cfg.get('module_lr_scale', 1.0),
        )
