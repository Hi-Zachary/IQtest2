"""LACEAIGIQA20K -- AIGIQA-20K 单 Overall MOS 模型（V26，no-stop joint backprop）。

调整/改进15.md:
    CLIP + Visual Q/K LoRA
        ├── LQEA (layerwise quality evidence aggregation) -> v_q
        └── CSAE (cross-scope alignment encoding)         -> e_a
    z = [v_q, t0, e_a] -> overall_fusion -> overall_head -> Overall MOS

全程无 stop-gradient / detach；标准 joint backprop。
AIGIQA-20K 单 Overall MOS，输出 [B,1]。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.mlpq import MLPQ
from ipiqa.models.mgsc_refined import MGSCRefined

from ipiqa.common.registry import registry


@registry.register_model("lace_aigiqa20k_v26")
class LACEAIGIQA20K(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=1,
            dim=256,
            drop=0.0,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_lqea=True,
            use_csae=True,
            mgsc_num_fragments=4,
            mgsc_topk=3,
            mgsc_mode="global_local",
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
        self.use_lqea = use_lqea
        self.use_csae = use_csae
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.lora_lr_scale = lora_lr_scale
        self.module_lr_scale = module_lr_scale
        self.outer_gate_init = float(outer_gate_init)

        # backbone（共享单 Q/K LoRA）
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

        # shared global projections
        self.visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )

        # LQEA（layerwise quality evidence aggregation = clean MLPQ）
        self.lqea = MLPQ(width=self.backbone.visual_width, dim=dim)
        self.quality_gate = nn.Parameter(torch.tensor(self.outer_gate_init))

        # CSAE（cross-scope alignment encoding = MGSCRefined global_local）
        self.csae = MGSCRefined(
            num_fragments=mgsc_num_fragments,
            topk=mgsc_topk,
            dim=dim,
            mgsc_mode=mgsc_mode,
        )

        # Overall MOS fusion + head
        self.overall_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.overall_head = nn.Linear(dim, 1)

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

        v0 = self.visual_proj(global_v)
        t0 = self.text_proj(global_t)

        # LQEA
        if self.use_lqea:
            delta_q = self.lqea(hs[3], hs[6], hs[9], hs[12])
            v_q = v0 + torch.tanh(self.quality_gate) * delta_q
        else:
            v_q = v0

        # CSAE
        if self.use_csae:
            e_a, s_g, s_r = self.csae(
                hs[12], feat["text_tokens"], attention_mask.bool(),
                global_v, global_t,
                self.backbone.clip.visual_projection,
                self.backbone.clip.text_projection,
            )
        else:
            e_a = torch.zeros_like(v0)

        # Overall MOS fusion
        h = self.overall_fusion(torch.cat([v_q, t0, e_a], dim=-1))
        mos = self.overall_head(h)

        self._last_ratios = {}
        return mos

    def get_gate_log(self):
        out = dict(self._last_ratios)
        out["quality_gate"] = torch.tanh(self.quality_gate).item()
        if self.use_lqea:
            pw = getattr(self.lqea, "_last_weight", None)
            if pw is not None:
                out["patch_w_mean"] = round(pw.mean().item(), 4)
                out["patch_w_std"] = round(pw.std().item(), 4)
        if self.use_csae:
            st = getattr(self.csae, "_last_stats", None)
            if st is not None:
                out.update({f"csae_{k}": v for k, v in st.items()})
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
            "lqea": cnt("lqea."),
            "csae": cnt("csae."),
            "proj": cnt("visual_proj.") + cnt("text_proj."),
            "overall": cnt("overall_fusion.") + cnt("overall_head.") + cnt("quality_gate"),
        }

    @classmethod
    def from_config(cls, cfg):
        return cls(
            model_name=cfg.get('model_name', 'ckpt/clip-vit-base-patch16'),
            context_length=cfg.get('context_length', 77),
            output_dim=cfg.get('output_dim', 1),
            dim=cfg.get('dim', 256),
            drop=cfg.get('dropout_rate', 0.0),
            use_lora=cfg.get('use_lora', True),
            lora_r=cfg.get('lora_r', 4),
            lora_alpha=cfg.get('lora_alpha', 8),
            lora_dropout=cfg.get('lora_dropout', 0.0),
            use_lqea=cfg.get('use_lqea', True),
            use_csae=cfg.get('use_csae', True),
            mgsc_num_fragments=cfg.get('mgsc_num_fragments', 4),
            mgsc_topk=cfg.get('mgsc_topk', 3),
            mgsc_mode=cfg.get('mgsc_mode', 'global_local'),
            freeze_visual=cfg.get('freeze_visual', True),
            freeze_text=cfg.get('freeze_text', True),
            outer_gate_init=cfg.get('outer_gate_init', 0.01),
            lora_lr_scale=cfg.get('lora_lr_scale', 1.0),
            module_lr_scale=cfg.get('module_lr_scale', 1.0),
        )
