"""MSQRNetV23Refined -- MLPQ + MGSCRefined（精简版，Phase A）。

调整/改进12.md：在 V23 基础上精简：
    - MGSC 删除 beta 标量融合 + lambda_msc scalar residual
    - alignment: h_a = alignment_fusion([v_a, t_a, e_c]); a = align_head(h_a)
    - 其余（MLPQ=multi-level+learned weighting，no deviation；doublescore）不变

V23 原版（model_v23.py / mgsc.py）保留不使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.dg_mpq_abl import DgMpqAbl
from ipiqa.models.mgsc_refined import MGSCRefined

from ipiqa.common.registry import registry


@registry.register_model("msqr_mlpq_mgsc_v23_refined")
class MSQRNetV23Refined(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=2,
            dim=256,
            drop=0.1,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_mlpq=True,
            use_mgsc=True,
            mgsc_num_fragments=4,
            mgsc_topk=3,
            alignment_stopgrad=True,
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
        self.use_mlpq = use_mlpq
        self.use_mgsc = use_mgsc
        self.alignment_stopgrad = alignment_stopgrad
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.lora_lr_scale = lora_lr_scale
        self.module_lr_scale = module_lr_scale
        self.outer_gate_init = float(outer_gate_init)

        # backbone（共享单 LoRA）
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

        # ===================== Quality branch (MLPQ) =====================
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
        self.mlpq = DgMpqAbl(
            width=self.backbone.visual_width, dim=dim,
            use_multilevel=True, use_deviation=False,
        )                                                  # always instantiate
        self.lambda_q = nn.Parameter(torch.tensor(self.outer_gate_init))

        # ===================== Alignment branch (MGSC) =====================
        self.alignment_visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_fusion = nn.Sequential(   # 固定 3D: [v_a, t_a, s_embed]
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(drop),
        )
        self.align_head = nn.Linear(dim, 1)
        self.mgsc_refined = MGSCRefined(
            num_fragments=mgsc_num_fragments,
            topk=mgsc_topk,
            dim=dim,
        )                                                # always instantiate

        self._last_ratios = {}

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_text:
            self.backbone.clip.text_model.eval()
        if self.freeze_visual:
            self.backbone.clip.vision_model.eval()
        return self

    def forward(self, x, text):
        input_ids, attention_mask = self.backbone.tokenize(text)
        input_ids = input_ids.to(x.device)
        attention_mask = attention_mask.to(x.device)

        feat = self.backbone(x, input_ids, attention_mask)
        global_v = feat["global_v"]
        global_t = feat["global_t"]
        hs = feat["vision_hidden"]
        ratios = {}

        # ===================== Quality branch (MLPQ) =====================
        v0_q = self.quality_visual_proj(global_v)
        if self.use_mlpq:
            delta_q = self.mlpq(hs[3], hs[6], hs[9], hs[12])
            q_scale = torch.tanh(self.lambda_q)
            v_q = v0_q + q_scale * delta_q
            ratios["quality_ratio"] = (
                (q_scale * delta_q).norm(dim=-1).mean() / (v0_q.norm(dim=-1).mean() + 1e-6)
            ).item()
        else:
            v_q = v0_q
        t_q = self.quality_text_proj(global_t)
        h_q = self.quality_fusion(torch.cat([v_q, t_q], dim=-1))
        q = self.quality_head(h_q)

        # ===================== Alignment branch (MGSC) =====================
        if self.alignment_stopgrad:
            global_v_a = global_v.detach()
            global_t_a = global_t.detach()
            h12_a = hs[12].detach()
            text_tokens_a = feat["text_tokens"].detach()
        else:
            global_v_a = global_v
            global_t_a = global_t
            h12_a = hs[12]
            text_tokens_a = feat["text_tokens"]

        v_a = self.alignment_visual_proj(global_v_a)
        t_a = self.alignment_text_proj(global_t_a)

        if self.use_mgsc:
            e_c, s_g, s_l = self.mgsc_refined(
                h12_a, text_tokens_a, attention_mask.bool(),
                global_v_a, global_t_a,
                self.backbone.clip.visual_projection,
                self.backbone.clip.text_projection,
            )
            h_a = self.alignment_fusion(torch.cat([v_a, t_a, e_c], dim=-1))
            a = self.align_head(h_a)          # 无 scalar residual
        else:
            e_c = torch.zeros_like(v_a)
            h_a = self.alignment_fusion(torch.cat([v_a, t_a, e_c], dim=-1))
            a = self.align_head(h_a)

        self._last_ratios = ratios
        return torch.cat([q, a], dim=-1)

    def get_gate_log(self):
        out = dict(self._last_ratios)
        out["lambda_q"] = torch.tanh(self.lambda_q).item()
        if self.use_mlpq:
            pw = getattr(self.mlpq, "_last_weight", None)
            if pw is not None:
                out["patch_w_mean"] = round(pw.mean().item(), 4)
                out["patch_w_std"] = round(pw.std().item(), 4)
        if self.use_mgsc:
            st = getattr(self.mgsc_refined, "_last_stats", None)
            if st is not None:
                out.update({f"mgsc_{k}": v for k, v in st.items()})
        return out

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
        return "{:.1f}M".format(tot / 1e6) if return_str else tot

    def trainable_summary(self):
        def cnt(prefix):
            return sum(p.numel() for n, p in self.named_parameters()
                       if n.startswith(prefix) and p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith("backbone.clip.vision_model.") and p.requires_grad)
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "lora": lora,
            "mlpq": cnt("mlpq."),
            "mgsc": cnt("mgsc_refined."),
            "quality_branch": cnt("quality_") + cnt("lambda_q"),
            "alignment_branch": cnt("alignment_"),
        }

    @classmethod
    def from_config(cls, cfg):
        return cls(
            model_name=cfg.get('model_name', 'ckpt/clip-vit-base-patch16'),
            context_length=cfg.get('context_length', 77),
            output_dim=cfg.get('output_dim', 2),
            dim=cfg.get('dim', 256),
            drop=cfg.get('dropout_rate', 0.1),
            use_lora=cfg.get('use_lora', True),
            lora_r=cfg.get('lora_r', 4),
            lora_alpha=cfg.get('lora_alpha', 8),
            lora_dropout=cfg.get('lora_dropout', 0.0),
            use_mlpq=cfg.get('use_mlpq', True),
            use_mgsc=cfg.get('use_mgsc', True),
            mgsc_num_fragments=cfg.get('mgsc_num_fragments', 4),
            mgsc_topk=cfg.get('mgsc_topk', 3),
            alignment_stopgrad=cfg.get('alignment_stopgrad', True),
            freeze_visual=cfg.get('freeze_visual', True),
            freeze_text=cfg.get('freeze_text', True),
            outer_gate_init=cfg.get('outer_gate_init', 0.01),
            lora_lr_scale=cfg.get('lora_lr_scale', 1.0),
            module_lr_scale=cfg.get('module_lr_scale', 1.0),
        )
