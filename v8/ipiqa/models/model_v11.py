"""MSQRNetV11 -- Task-Isolated Dual-Branch Architecture.

调整/改进7.md (V11): 解决 V10 中 Quality/Alignment 梯度耦合导致的 negative transfer。

核心原则：
    Quality-specific 参数只能被 L_Q 更新；
    Alignment-specific 参数只能被 L_A 更新；
    唯一共享 = Frozen CLIP base weights (requires_grad=False)。

结构：
    Quality   : LoRA-Q -> [H3,H6,H9,H12] -> DG-MPQ -> v_q -> [v_q, t_q] -> Q fusion -> Q head
    Alignment : LoRA-A -> [H9,H12,global] -> MSCM  -> a_corr -> [v_a, t_a, a_corr] -> A fusion -> A head
    text      : frozen CLIP text 编码一次 -> global_t -> t_q/t_a（各自投影）

消融（V11 新定义，LoRA-Q/A 始终存在）：
    B0   = 无 LoRA、无模块
    R0   = LoRA-Q + LoRA-A
    B1   = R0 + DG-MPQ (仅 Quality)
    B2   = R0 + MSCM   (仅 Alignment)
    Full = R0 + DG-MPQ + MSCM

Loss = MSE(Q) + MSE(A)；第一版不加入 soft sharing / PCGrad 等。
V8/V9/V10 与 HCMI/TCAP/MSCM 代码全部保留。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.dual_lora_clip_vit import DualLoRACLIPViT
from ipiqa.models.dg_mpq import DgMpq
from ipiqa.models.mscm import MSCM

from ipiqa.common.registry import registry


@registry.register_model("msqr_dgmpq_mscm_v11")
class MSQRNetV11(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=2,
            dim=256,
            drop=0.1,
            use_dual_lora=True,
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
        self.use_dual_lora = use_dual_lora
        self.use_dg_mpq = use_dg_mpq
        self.use_mscm = use_mscm
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.lora_lr_scale = lora_lr_scale
        self.module_lr_scale = module_lr_scale
        self.outer_gate_init = float(outer_gate_init)

        # ---------- backbone（双 LoRA） ----------
        self.backbone = DualLoRACLIPViT(
            model_name=model_name,
            context_length=context_length,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            freeze_visual=freeze_visual,
            freeze_text=freeze_text,
        )

        # ===================== Quality branch（只被 L_Q 更新） =====================
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
        if use_dg_mpq:
            self.dg_mpq = DgMpq(width=self.backbone.visual_width, dim=dim)
            self.lambda_q = nn.Parameter(torch.tensor(self.outer_gate_init))
        else:
            self.dg_mpq = None
            self.lambda_q = None

        # ===================== Alignment branch（只被 L_A 更新） =====================
        self.alignment_visual_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        self.alignment_text_proj = nn.Sequential(
            nn.Linear(self.backbone.projection_dim, dim), nn.GELU(),
        )
        align_in = dim * 3 if use_mscm else dim * 2
        self.alignment_fusion = nn.Sequential(
            nn.Linear(align_in, dim), nn.GELU(), nn.Dropout(drop),
        )
        self.align_head = nn.Linear(dim, 1)
        if use_mscm:
            self.mscm = MSCM(
                visual_width=self.backbone.visual_width,
                dim=dim,
                drop=drop,
            )
        else:
            self.mscm = None

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

        # ---- shared frozen text（编码一次，各自投影） ----
        global_t = self.backbone.encode_text(input_ids, attention_mask)["global_t"]
        t_q = self.quality_text_proj(global_t)
        t_a = self.alignment_text_proj(global_t)

        # ===================== Quality branch（LoRA-Q） =====================
        feat_q = self.backbone.encode_image(x, lora_branch="quality")
        hs_q = feat_q["vision_hidden"]
        global_v_q = feat_q["global_v"]

        v0_q = self.quality_visual_proj(global_v_q)
        ratios = {}
        if self.use_dg_mpq:
            delta_q = self.dg_mpq(hs_q[3], hs_q[6], hs_q[9], hs_q[12])
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

        # ===================== Alignment branch（LoRA-A） =====================
        feat_a = self.backbone.encode_image(x, lora_branch="alignment")
        hs_a = feat_a["vision_hidden"]
        global_v_a = feat_a["global_v"]

        v_a = self.alignment_visual_proj(global_v_a)

        if self.use_mscm:
            a_corr = self.mscm(
                hs_a[9][:, 1:, :],
                hs_a[12][:, 1:, :],
                t_a,
                global_v_a,
                global_t,
            )
            h_a = self.alignment_fusion(torch.cat([v_a, t_a, a_corr], dim=-1))
        else:
            h_a = self.alignment_fusion(torch.cat([v_a, t_a], dim=-1))
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
        """按 branch 分组（LoRA-Q/quality 一组，LoRA-A/alignment 一组），
        weight decay 按维度拆分。梯度隔离由 forward 结构保证。"""
        def is_lora_a(name):
            return ("lora_A_a" in name) or ("lora_B_a" in name)
        def is_lora_q(name):
            return ("lora_A_q" in name) or ("lora_B_q" in name)
        def is_quality(name):
            return (name.startswith("quality_") or name.startswith("dg_mpq")
                    or name == "lambda_q" or is_lora_q(name))
        def is_nowd(p, name):
            return p.ndim < 2 or "bias" in name or "ln" in name or "bn" in name

        q_wd, q_nowd, a_wd, a_nowd = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if is_quality(n):
                (q_nowd if is_nowd(p, n) else q_wd).append(p)
            else:
                (a_nowd if is_nowd(p, n) else a_wd).append(p)

        return [
            {"params": q_wd, "weight_decay": weight_decay,
             "lr_scale": self.module_lr_scale * lr_scale},
            {"params": q_nowd, "weight_decay": 0,
             "lr_scale": self.module_lr_scale * lr_scale},
            {"params": a_wd, "weight_decay": weight_decay,
             "lr_scale": self.module_lr_scale * lr_scale},
            {"params": a_nowd, "weight_decay": 0,
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
        lora_q = self.backbone.lora_trainable_count("quality")
        lora_a = self.backbone.lora_trainable_count("alignment")
        q_branch = sum(p.numel() for n, p in self.named_parameters()
                       if (n.startswith("quality_") or n.startswith("dg_mpq") or n == "lambda_q") and p.requires_grad)
        a_branch = sum(p.numel() for n, p in self.named_parameters()
                       if (n.startswith("alignment_") or n.startswith("mscm")) and p.requires_grad)
        return {
            "total": self.show_n_params(),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "lora_q": lora_q,
            "lora_a": lora_a,
            "quality_branch": q_branch,
            "alignment_branch": a_branch,
            "dg_mpq": cnt("dg_mpq."),
            "mscm": cnt("mscm."),
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
            drop=cfg.get('dropout_rate', 0.1),
            use_dual_lora=cfg.get('use_dual_lora', True),
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
