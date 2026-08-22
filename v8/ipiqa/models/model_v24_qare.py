"""MSQRNetV24QARE -- Frozen CLIP + QARE + CSAE（V24 Candidate）。

调整/改进13.md：验证"单次 Frozen CLIP forward + Quality 分支专属低秩适配"
能否替代 backbone Q/K LoRA + stop-gradient。

    Frozen CLIP (single forward, no LoRA)
        ├── QARE (quality-specific low-rank adaptation + layerwise aggregation) -> Q
        └── CSAE (original CLIP features, global+regional) -> A

无 stop-gradient / detach（CLIP frozen + 两分支结构天然独立）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.qare import QARE
from ipiqa.models.mgsc_refined import MGSCRefined

from ipiqa.common.registry import registry


@registry.register_model("msqr_qare_csae_v24")
class MSQRNetV24QARE(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=2,
            dim=256,
            drop=0.1,
            use_lora=False,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            qare_rank=4,
            qare_alpha=8,
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
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.lora_lr_scale = lora_lr_scale
        self.module_lr_scale = module_lr_scale
        self.outer_gate_init = float(outer_gate_init)

        # Frozen CLIP backbone（use_lora=False）
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

        # ===================== Quality branch (QARE) =====================
        self.qare = QARE(
            visual_width=self.backbone.visual_width,
            global_dim=self.backbone.projection_dim,
            dim=dim,
            rank=qare_rank,
            alpha=qare_alpha,
        )
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
        self.gamma_q = nn.Parameter(torch.tensor(self.outer_gate_init))

        # ===================== Alignment branch (CSAE = MGSCRefined global_local) =====================
        self.alignment_visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_fusion = nn.Sequential(   # 3D: [v_a, t_a, e_c]
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(drop),
        )
        self.align_head = nn.Linear(dim, 1)
        self.csae = MGSCRefined(
            num_fragments=mgsc_num_fragments,
            topk=mgsc_topk,
            dim=dim,
            mgsc_mode=mgsc_mode,
        )

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

        feat = self.backbone(x, input_ids, attention_mask)   # 单次 forward
        global_v = feat["global_v"]
        global_t = feat["global_t"]
        hs = feat["vision_hidden"]

        # ===================== Quality (QARE) =====================
        g_adapt, delta_v = self.qare(global_v, hs[3], hs[6], hs[9], hs[12])
        v0_q = self.quality_visual_proj(g_adapt)
        v_q = v0_q + torch.tanh(self.gamma_q) * delta_v
        t_q = self.quality_text_proj(global_t)
        h_q = self.quality_fusion(torch.cat([v_q, t_q], dim=-1))
        q = self.quality_head(h_q)

        # ===================== Alignment (CSAE, raw frozen CLIP features) =====================
        e_c, s_g, s_l = self.csae(
            hs[12], feat["text_tokens"], attention_mask.bool(),
            global_v, global_t,
            self.backbone.clip.visual_projection,
            self.backbone.clip.text_projection,
        )
        v_a = self.alignment_visual_proj(global_v)
        t_a = self.alignment_text_proj(global_t)
        h_a = self.alignment_fusion(torch.cat([v_a, t_a, e_c], dim=-1))
        a = self.align_head(h_a)

        return torch.cat([q, a], dim=-1)

    def get_gate_log(self):
        out = dict(self._last_ratios)
        out["gamma_q"] = torch.tanh(self.gamma_q).item()
        r = getattr(self.qare, "_last_ratios", None)
        if r is not None:
            out.update({f"qare_{k}": v for k, v in r.items()})
        pw = getattr(self.qare, "_last_weight", None)
        if pw is not None:
            out["patch_w_mean"] = round(pw.mean().item(), 4)
            out["patch_w_std"] = round(pw.std().item(), 4)
        st = getattr(self.csae, "_last_stats", None)
        if st is not None:
            out.update({f"mgsc_{k}": v for k, v in st.items()})
        return out

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        new_wd, new_nowd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_nowd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
            (new_nowd if is_nowd else new_wd).append(p)
        return [
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
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "qare_adapters": cnt("qare.layer_adapters") + cnt("qare.global_adapter"),
            "qare_agg": cnt("qare.patch_proj") + cnt("qare.patch_weight") + cnt("qare.quality_out"),
            "quality_branch": cnt("quality_") + cnt("gamma_q"),
            "alignment_branch": cnt("alignment_") + cnt("csae."),
        }

    @classmethod
    def from_config(cls, cfg):
        return cls(
            model_name=cfg.get('model_name', 'ckpt/clip-vit-base-patch16'),
            context_length=cfg.get('context_length', 77),
            output_dim=cfg.get('output_dim', 2),
            dim=cfg.get('dim', 256),
            drop=cfg.get('dropout_rate', 0.1),
            use_lora=cfg.get('use_lora', False),
            lora_r=cfg.get('lora_r', 4),
            lora_alpha=cfg.get('lora_alpha', 8),
            lora_dropout=cfg.get('lora_dropout', 0.0),
            qare_rank=cfg.get('qare_rank', 4),
            qare_alpha=cfg.get('qare_alpha', 8),
            mgsc_num_fragments=cfg.get('mgsc_num_fragments', 4),
            mgsc_topk=cfg.get('mgsc_topk', 3),
            mgsc_mode=cfg.get('mgsc_mode', 'global_local'),
            freeze_visual=cfg.get('freeze_visual', True),
            freeze_text=cfg.get('freeze_text', True),
            outer_gate_init=cfg.get('outer_gate_init', 0.01),
            lora_lr_scale=cfg.get('lora_lr_scale', 1.0),
            module_lr_scale=cfg.get('module_lr_scale', 1.0),
        )
