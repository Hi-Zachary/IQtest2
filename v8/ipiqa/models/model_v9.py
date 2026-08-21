"""MSQRNetV9 -- Frozen CLIP ViT-B/16 + Visual LoRA + DG-MPQ + TCAP.

调整/改进5.md (V9): 用轻量 TCAP 替代 HCMI-ViT 作为 Alignment 分支；
DG-MPQ（Quality）保持不变；融合改为 task-specific fusion。

    B0    = Frozen CLIP baseline
    R0    = B0 + Visual LoRA
    B1    = R0 + DG-MPQ                    (Quality)
    B2    = R0 + TCAP                      (Alignment)
    Full  = R0 + DG-MPQ + TCAP

数据流（task-oriented routing + task-specific fusion）：
    v  = v0 + tanh(lambda_q) * DG-MPQ(H3,H6,H9,H12)
    c  = t0 + tanh(lambda_a) * TCAP(H12_patch, t0)
    h_q = quality_fusion([v, t0]);    q = quality_head(h_q)
    h_a = alignment_fusion([v, c]);   a = align_head(h_a)

Loss 纯 MSE(Q)+MSE(A)；LoRA/DG-MPQ/loss/LR 与 V8 保持一致。
V8（model_v8.py）与 HCMI（hcmi_vit.py）保留作回退，不在本链路使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.dg_mpq import DgMpq
from ipiqa.models.tcap import TCAP

from ipiqa.common.registry import registry


@registry.register_model("msqr_dgmpq_tcap_v9")
class MSQRNetV9(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=2,
            dim=256,
            num_heads=4,
            drop=0.1,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_dg_mpq=True,
            use_tcap=True,
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
        self.use_tcap = use_tcap
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.lora_lr_scale = lora_lr_scale
        self.module_lr_scale = module_lr_scale
        self.outer_gate_init = float(outer_gate_init)

        # ---------- backbone ----------
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

        # ---------- anchors ----------
        self.base_visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim),
            nn.GELU(),
        )
        self.base_text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim),
            nn.GELU(),
        )

        # ---------- task-specific fusion + dual heads（V9 不再共享同一 fusion） ----------
        self.quality_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.alignment_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.quality_head = nn.Linear(dim, 1)
        self.align_head = nn.Linear(dim, 1)

        # ---------- DG-MPQ residual branch ----------
        if use_dg_mpq:
            self.dg_mpq = DgMpq(width=self.backbone.visual_width, dim=dim)
            self.lambda_q = nn.Parameter(torch.tensor(self.outer_gate_init))
        else:
            self.dg_mpq = None
            self.lambda_q = None

        # ---------- TCAP residual branch ----------
        if use_tcap:
            self.tcap = TCAP(
                visual_width=self.backbone.visual_width,
                dim=dim,
                num_heads=num_heads,
                drop=drop,
            )
            self.lambda_a = nn.Parameter(torch.tensor(self.outer_gate_init))
        else:
            self.tcap = None
            self.lambda_a = None

        self._last_ratios = {}

    # ---------------- train/eval override（冻结 backbone 保持 eval） ----------------
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

        v0 = self.base_visual_proj(feat["global_v"])   # [B, D]
        t0 = self.base_text_proj(feat["global_t"])     # [B, D]
        v = v0
        c = t0
        hs = feat["vision_hidden"]
        ratios = {}

        # ===================== DG-MPQ residual =====================
        if self.use_dg_mpq:
            delta_q = self.dg_mpq(hs[3], hs[6], hs[9], hs[12])   # [B, D]
            q_scale = torch.tanh(self.lambda_q)
            v = v0 + q_scale * delta_q
            ratios["raw_quality_ratio"] = (
                delta_q.norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()
            ratios["quality_ratio"] = (
                (q_scale * delta_q).norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()

        # ===================== TCAP residual =====================
        if self.use_tcap:
            delta_a = self.tcap(hs[12][:, 1:, :], t0)            # [B, D]
            a_scale = torch.tanh(self.lambda_a)
            c = t0 + a_scale * delta_a
            ratios["raw_align_ratio"] = (
                delta_a.norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()
            ratios["align_ratio"] = (
                (a_scale * delta_a).norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()

        # ===================== task-specific routing + dual heads =====================
        h_q = self.quality_fusion(torch.cat([v, t0], dim=-1))    # [B, D]
        h_a = self.alignment_fusion(torch.cat([v, c], dim=-1))   # [B, D]
        q = self.quality_head(h_q)
        a = self.align_head(h_a)

        self._last_ratios = ratios
        return torch.cat([q, a], dim=-1)

    # ---------------- gate / ratio / TCAP attention logging ----------------
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
        if self.use_tcap:
            out["lambda_a"] = torch.tanh(self.lambda_a).item()
            aw = getattr(self.tcap, "_last_attn", None)
            if aw is not None:
                out["tcap_attn_mean"] = round(aw.mean().item(), 4)
                out["tcap_attn_max"] = round(aw.max().item(), 4)
                out["tcap_attn_entropy"] = round(
                    -(aw * torch.log(aw.clamp_min(1e-12))).mean().item(), 4
                )
        return out

    # ------------------------------------------------------------------ #
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        lora_prefixes = ("backbone.clip.vision_model.",)

        lora_wd, lora_nowd = [], []
        new_wd, new_nowd = [], []

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
        dg = cnt("dg_mpq.") + (self.lambda_q.numel() if self.lambda_q is not None else 0)
        tc = cnt("tcap.") + (self.lambda_a.numel() if self.lambda_a is not None else 0)
        head = cnt("quality_fusion.") + cnt("alignment_fusion.") + cnt("quality_head.") + cnt("align_head.")
        text = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith("backbone.clip.text_model.") and p.requires_grad)
        return {
            "total": self.show_n_params(),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "lora": lora,
            "dg_mpq": dg,
            "tcap": tc,
            "heads+fusions": head,
            "text_trainable": text,
        }

    @classmethod
    def from_config(cls, cfg):
        return cls(
            model_name=cfg.get('model_name', 'ckpt/clip-vit-base-patch16'),
            context_length=cfg.get('context_length', 77),
            output_dim=cfg.get('output_dim', 2),
            dim=cfg.get('dim', 256),
            num_heads=cfg.get('num_heads', 4),
            drop=cfg.get('dropout_rate', 0.1),
            use_lora=cfg.get('use_lora', True),
            lora_r=cfg.get('lora_r', 4),
            lora_alpha=cfg.get('lora_alpha', 8),
            lora_dropout=cfg.get('lora_dropout', 0.0),
            use_dg_mpq=cfg.get('use_dg_mpq', True),
            use_tcap=cfg.get('use_tcap', True),
            freeze_visual=cfg.get('freeze_visual', True),
            freeze_text=cfg.get('freeze_text', True),
            outer_gate_init=cfg.get('outer_gate_init', 0.01),
            lora_lr_scale=cfg.get('lora_lr_scale', 1.0),
            module_lr_scale=cfg.get('module_lr_scale', 1.0),
        )
