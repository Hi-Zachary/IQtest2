"""MSQRNetV8 -- Frozen CLIP ViT-B/16 + Visual LoRA + DG-MPQ + HCMI-ViT.

调整/改进1.md (v8) + 调整/改进2.md (V8-Slim):

    B0 = Frozen CLIP ViT-B/16 baseline (no LoRA, no modules)
    R0 = B0 + Visual Q/K LoRA                     (strong reference)
    B1 = R0 + DG-MPQ                              (quality module)
    B2 = R0 + HCMI-ViT                            (alignment module)
    Full = R0 + DG-MPQ + HCMI-ViT

  - V8-Slim：删除 discrepancy-guided attention bias（beta_align≈0.002 无贡献），
    HCMI mlp_ratio=1（FFN 收窄），LoRA dropout=0（与 train() 强制 eval 的实际行为一致）。
  - 模块严格并行：DG-MPQ 与 HCMI-ViT 都直接从 backbone hidden states 出发，
    互不串行（B1/B2 与 Full 的模块输入完全一致，消除 hidden input change）。
  - 纯 MSE 双任务损失（quality + alignment）。
  - 每个模块输出 normalized residual，以 ``tanh(lambda)`` gated residual
    注入 anchor（v = v0 + tanh(lambda_q)*delta_q, c = t0 + tanh(lambda_a)*delta_a）。

Data flow:
    feat = backbone(images, input_ids, attention_mask)
    v0 = heads.base_visual_proj(feat.global_v);  t0 = heads.base_text_proj(feat.global_t)
    delta_q = dg_mpq(H3, H6, H9, H12)                    (use_dg_mpq)
    delta_a = hcmi(H6[:,1:], H12[:,1:], T, mask)         (use_hcmi)
    v = v0 + tanh(lambda_q)*delta_q;  c = t0 + tanh(lambda_a)*delta_a
    h = heads.shared_fusion(concat[v, c])
    q = heads.quality_head(h);  a = heads.align_head(h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.clip_vit_backbone import CLIPViTBackbone
from ipiqa.models.dg_mpq import DgMpq
from ipiqa.models.hcmi_vit import HcmiVit
from ipiqa.models.heads import DualTaskHeads

from ipiqa.common.registry import registry


@registry.register_model("msqr_dgmpq_hcmi_v8")
class MSQRNetV8(BaseModel):
    def __init__(
            self,
            model_name='ckpt/clip-vit-base-patch16',
            context_length=77,
            output_dim=2,
            dim=256,
            num_heads=4,
            mlp_ratio=1.0,
            drop=0.1,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            use_dg_mpq=True,
            use_hcmi=True,
            hcmi_use_multi_kernel=True,
            use_prompt_weight=True,
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
        self.use_hcmi = use_hcmi
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

        # ---------- anchors + heads ----------
        self.heads = DualTaskHeads(
            projection_dim=self.backbone.projection_dim,
            dim=dim,
            drop=drop,
        )

        # ---------- DG-MPQ residual branch ----------
        if use_dg_mpq:
            self.dg_mpq = DgMpq(
                width=self.backbone.visual_width,
                dim=dim,
            )
            self.lambda_q = nn.Parameter(
                torch.tensor(self.outer_gate_init)
            )
        else:
            self.dg_mpq = None
            self.lambda_q = None

        # ---------- HCMI-ViT residual branch ----------
        if use_hcmi:
            self.hcmi = HcmiVit(
                width=self.backbone.visual_width,
                text_width=self.backbone.text_width,
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                use_multi_kernel=hcmi_use_multi_kernel,
                use_prompt_weight=use_prompt_weight,
                gamma_init=self.outer_gate_init,
            )
            self.lambda_a = nn.Parameter(
                torch.tensor(self.outer_gate_init)
            )
        else:
            self.hcmi = None
            self.lambda_a = None

        self._last_ratios = {}

    # ---------------- train/eval override（冻结 backbone 保持 eval，无 BN 问题） ----------------
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_text:
            self.backbone.clip.text_model.eval()
        if self.freeze_visual:
            # LoRA 模式下 base 由 peft 冻结；统一保持 eval 以维持 frozen backbone 确定性
            self.backbone.clip.vision_model.eval()
        return self

    # ------------------------------------------------------------------ #
    def forward(self, x, text):
        """x: [B,3,224,224]; text: list[str] or str -> [B,2] = (quality, alignment)."""
        input_ids, attention_mask = self.backbone.tokenize(text)
        input_ids = input_ids.to(x.device)
        attention_mask = attention_mask.to(x.device)

        feat = self.backbone(x, input_ids, attention_mask)

        v0 = self.heads.base_visual_proj(feat["global_v"])   # [B, D]
        t0 = self.heads.base_text_proj(feat["global_t"])     # [B, D]
        v = v0
        c = t0
        ratios = {}

        # ===================== DG-MPQ residual =====================
        if self.use_dg_mpq:
            hs = feat["vision_hidden"]
            delta_q = self.dg_mpq(hs[3], hs[6], hs[9], hs[12])   # [B, D]
            q_scale = torch.tanh(self.lambda_q)
            v = v0 + q_scale * delta_q
            ratios["raw_quality_ratio"] = (
                delta_q.norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()
            ratios["quality_ratio"] = (
                (q_scale * delta_q).norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()

        # ===================== HCMI-ViT residual =====================
        if self.use_hcmi:
            hs = feat["vision_hidden"]
            delta_a = self.hcmi(
                hs[6][:, 1:, :],
                hs[12][:, 1:, :],
                feat["text_tokens"],
                feat["text_mask"],
            )                                            # [B, D]
            a_scale = torch.tanh(self.lambda_a)
            c = t0 + a_scale * delta_a
            ratios["raw_align_ratio"] = (
                delta_a.norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()
            ratios["align_ratio"] = (
                (a_scale * delta_a).norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()

        # ===================== shared fusion + dual heads =====================
        h = self.heads.shared_fusion(torch.cat([v, c], dim=-1))   # [B, D]
        q = self.heads.quality_head(h)
        a = self.heads.align_head(h)

        self._last_ratios = ratios
        return torch.cat([q, a], dim=-1)

    # ---------------- gate / residual / module logging ----------------
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
        if self.use_hcmi:
            out["lambda_a"] = torch.tanh(self.lambda_a).item()
            out["alpha_d"] = self.hcmi.alpha_d.item()
            out["beta_d"] = self.hcmi.beta_d.item()
            out["alpha_s"] = self.hcmi.alpha_s.item()
            out["beta_s"] = self.hcmi.beta_s.item()
            hg = getattr(self.hcmi, "_last_hier_gate", None)
            if hg is not None:
                out["hier_gate_mean"] = round(hg.mean().item(), 4)
            ps = getattr(self.hcmi, "_last_prompt_stats", None)
            if ps is not None:
                out.update({f"prompt_w_{k}": v for k, v in ps.items()})
        return out

    # ------------------------------------------------------------------ #
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """分组：LoRA 参数与新增模块/head 参数分开（可给不同 LR），
        weight decay 按维度/名字拆分。backbone base 全冻结，不进优化器。"""
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

        optim_params = [
            {"params": lora_wd, "weight_decay": weight_decay,
             "lr_scale": self.lora_lr_scale * lr_scale},
            {"params": lora_nowd, "weight_decay": 0,
             "lr_scale": self.lora_lr_scale * lr_scale},
            {"params": new_wd, "weight_decay": weight_decay,
             "lr_scale": self.module_lr_scale * lr_scale},
            {"params": new_nowd, "weight_decay": 0,
             "lr_scale": self.module_lr_scale * lr_scale},
        ]
        return [g for g in optim_params if len(g["params"]) > 0]

    # ---------------- parameter summary ----------------
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
        dp = cnt("hcmi.") + (self.lambda_a.numel() if self.lambda_a is not None else 0)
        head = cnt("heads.")
        text = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith("backbone.clip.text_model.") and p.requires_grad)
        base = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith("backbone.clip.vision_model.") and not p.requires_grad)
        return {
            "total": self.show_n_params(),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "lora": lora,
            "dg_mpq": dg,
            "hcmi": dp,
            "heads": head,
            "text_trainable": text,
            "visual_base_trainable": sum(
                p.numel() for n, p in self.named_parameters()
                if n.startswith("backbone.clip.vision_model.") and p.requires_grad and
                ("lora" not in n)
            ),
        }

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get('model_name', 'ckpt/clip-vit-base-patch16')
        context_length = cfg.get('context_length', 77)
        output_dim = cfg.get('output_dim', 2)
        dim = cfg.get('dim', 256)
        num_heads = cfg.get('num_heads', 4)
        mlp_ratio = cfg.get('mlp_ratio', 2.0)
        drop = cfg.get('dropout_rate', 0.1)
        use_lora = cfg.get('use_lora', True)
        lora_r = cfg.get('lora_r', 4)
        lora_alpha = cfg.get('lora_alpha', 8)
        lora_dropout = cfg.get('lora_dropout', 0.0)
        use_dg_mpq = cfg.get('use_dg_mpq', True)
        use_hcmi = cfg.get('use_hcmi', True)
        hcmi_use_multi_kernel = cfg.get('hcmi_use_multi_kernel', True)
        use_prompt_weight = cfg.get('use_prompt_weight', True)
        freeze_visual = cfg.get('freeze_visual', True)
        freeze_text = cfg.get('freeze_text', True)
        outer_gate_init = cfg.get('outer_gate_init', 0.01)
        lora_lr_scale = cfg.get('lora_lr_scale', 1.0)
        module_lr_scale = cfg.get('module_lr_scale', 1.0)

        return cls(
            model_name=model_name,
            context_length=context_length,
            output_dim=output_dim,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dg_mpq=use_dg_mpq,
            use_hcmi=use_hcmi,
            hcmi_use_multi_kernel=hcmi_use_multi_kernel,
            use_prompt_weight=use_prompt_weight,
            freeze_visual=freeze_visual,
            freeze_text=freeze_text,
            outer_gate_init=outer_gate_init,
            lora_lr_scale=lora_lr_scale,
            module_lr_scale=module_lr_scale,
        )
