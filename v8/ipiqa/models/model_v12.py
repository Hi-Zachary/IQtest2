"""MSQRNetV12 -- Shared Visual LoRA + Task-Specific Modules (DG-MPQ / MSCM).

调整/改进8.md (V12): 共享基础适配，隔离任务专属建模。

- Shared Visual LoRA（单套 LoRA，接受 L_Q + L_A，利用跨任务正迁移）
- DG-MPQ 只服务 Quality（只受 L_Q）
- MSCM 只服务 Alignment（只受 L_A）
- 独立 projection / fusion / head
- Alignment 分支不使用 DG-MPQ 增强后的 v_q，直接用共享 CLIP 特征建立自己的 v_a

数据流（一次共享视觉前向）：
    feat = backbone(image)                 # 共享 LoRA
    v0_q = quality_visual_proj(global_v);  delta_q = DG-MPQ(H3,H6,H9,H12)
    v_q = v0_q + tanh(lambda_q)*delta_q
    h_q = quality_fusion([v_q, t_q]);        q = quality_head(h_q)

    v_a = alignment_visual_proj(global_v)
    a_corr = MSCM(H9, H12, t_a, global_v, global_t)
    h_a = alignment_fusion([v_a, t_a, a_corr]);  a = align_head(h_a)

V12 默认 always-instantiate（模块开关只控制 forward 是否使用，参数布局在各消融间一致；
MSCM 关闭时 a_corr=zeros，保持 Alignment Fusion 3D 布局）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.dg_mpq import DgMpq
from ipiqa.models.mscm import MSCM

from ipiqa.common.registry import registry


@registry.register_model("msqr_dgmpq_mscm_v12")
class MSQRNetV12(BaseModel):
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
            use_mscm=True,
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
        self.use_mscm = use_mscm
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
        self.alignment_fusion = nn.Sequential(   # 固定 3D：始终 [v_a, t_a, a_corr]
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(drop),
        )
        self.align_head = nn.Linear(dim, 1)
        self.mscm = MSCM(                        # always
            visual_width=self.backbone.visual_width,
            dim=dim,
            drop=drop,
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

        # ===================== Alignment branch（不用 v_q） =====================
        v_a = self.alignment_visual_proj(global_v)
        if self.use_mscm:
            a_corr = self.mscm(
                hs[9][:, 1:, :],
                hs[12][:, 1:, :],
                t_a,
                global_v,
                global_t,
            )
        else:
            a_corr = torch.zeros_like(t_a)   # 保持 fusion 3D 布局
        h_a = self.alignment_fusion(torch.cat([v_a, t_a, a_corr], dim=-1))
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
        if self.use_mscm:
            st = getattr(self.mscm, "_last_stats", None)
            if st is not None:
                out.update({f"corr_{k}": v for k, v in st.items()})
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
            "mscm": cnt("mscm."),
            "quality_branch": cnt("quality_") + cnt("lambda_q"),
            "alignment_branch": cnt("alignment_"),
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
            use_mscm=cfg.get('use_mscm', True),
            freeze_visual=cfg.get('freeze_visual', True),
            freeze_text=cfg.get('freeze_text', True),
            outer_gate_init=cfg.get('outer_gate_init', 0.01),
            lora_lr_scale=cfg.get('lora_lr_scale', 1.0),
            module_lr_scale=cfg.get('module_lr_scale', 1.0),
        )
