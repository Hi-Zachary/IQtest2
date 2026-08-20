"""MSQRNet -- MSQR + SHCMI + TAF stitched onto a neutral CLIP baseline.

Ablation switches (all share this code, one model per variant):
    use_msqr / use_shcmi / use_taf

Data flow (design doc, one CLIP image forward only):

    Image --CLIP RN50--> spatial [B,2048,16,16]
      |--(use_msqr)--> MSQR --> F_fine [B,Nf,D], F_coarse [B,Nc,D]
      |                    `-> MSQRVisualSkip -> F_visual [B,D]
      `--(else)--> plain Conv1x1 + AvgPool -> fine/coarse tokens
    Prompt --CLIP text--> token-level T0 [B,L,512] (+ mask)
      `--(use_shcmi)--> SHCMI(fine, coarse, T0) -> F_cross [B,D]
    F_visual + F_cross
      |--(use_taf)--> TAF -> F_q -> Quality Head ; F_a -> Alignment Head
      `--(else)-----> concat -> shared MLP -> [quality, alignment]

When a module is off its stream falls back to the projected CLIP global
feature, so every B0-B4 variant remains a complete, trainable model.
"""

import os
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipiqa.models.base_model import BaseModel
from ipiqa.models.utils import interpolate_pos_embed, freeze_module
from ipiqa.models.modules.msqr import MSQR, MSQRVisualSkip
from ipiqa.models.modules.shcmi import SHCMI
from ipiqa.models.modules.taf import TAF

from ipiqa.common.registry import registry

CLIP_TEXT_WIDTH = 512   # RN50 text transformer width
CLIP_VISUAL_WIDTH = 1024  # RN50 attnpool output dim


def build_mlp(in_dim, out_dim, hidden_dim=None, drop=0.1, tanh=False):
    hidden_dim = hidden_dim or in_dim
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(drop),
        nn.Linear(hidden_dim, out_dim),
    ]
    if tanh:
        layers.insert(1, nn.Tanh())
    return nn.Sequential(*layers)


@registry.register_model("msqr_shcmi_taf")
class MSQRNet(BaseModel):
    def __init__(
            self,
            base_ckpt='',  # clip RN50 checkpoint
            input_resolution=512,
            output_dim=2,
            dim=256,
            num_heads=4,
            mlp_ratio=2.0,
            drop=0.1,
            use_msqr=True,
            use_shcmi=True,
            use_taf=True,
            msqr_use_channel=True,
            msqr_use_spatial=True,
            msqr_use_cross_scale=True,
            shcmi_use_multi_kernel=True,
            gamma_init=0.0,
            freeze_text=True,
            head_scale=None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.use_msqr = use_msqr
        self.use_shcmi = use_shcmi
        self.use_taf = use_taf

        clip_ckpt = clip.load(base_ckpt, device="cpu")[0]
        self.resnet50 = clip_ckpt.visual
        self.txt_model = clip_ckpt.transformer
        self.wte = clip_ckpt.token_embedding
        self.ln_final = clip_ckpt.ln_final
        self.txt_pos = clip_ckpt.positional_embedding
        self.text_projection = clip_ckpt.text_projection

        self.dtype = self.resnet50.conv1.weight.dtype

        # spatial feature path: keep attnpool separately, expose [B,2048,H,W]
        self.resnet50.attnpool.positional_embedding = nn.Parameter(
            interpolate_pos_embed(
                self.resnet50.attnpool.positional_embedding,
                input_resolution=input_resolution,
            )
        )
        self.attnpool = self.resnet50.attnpool
        self.resnet50.attnpool = nn.Identity()

        # ---------- visual stream ----------
        if use_msqr:
            self.msqr = MSQR(
                in_channels=2048, dim=dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop,
                use_channel_attention=msqr_use_channel,
                use_spatial_attention=msqr_use_spatial,
                use_cross_scale=msqr_use_cross_scale,
                gamma_init=gamma_init,
            )
            self.visual_skip = MSQRVisualSkip(dim, drop)
            self.f_visual_proj = None
        else:
            # plain fine/coarse tokens for SHCMI (design section 20)
            self.plain_proj = nn.Sequential(
                nn.Conv2d(2048, dim, kernel_size=1),
                nn.GELU(),
            )
            self.visual_skip = None
            self.f_visual_proj = nn.Linear(CLIP_VISUAL_WIDTH, dim)

        # ---------- cross-modal stream ----------
        if use_shcmi:
            self.shcmi = SHCMI(
                text_dim=CLIP_TEXT_WIDTH, dim=dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop, gamma_init=gamma_init,
                use_multi_kernel=shcmi_use_multi_kernel,
            )
            self.f_cross_proj = None
        else:
            self.shcmi = None
            self.f_cross_proj = nn.Linear(CLIP_VISUAL_WIDTH, dim)

        # ---------- fusion + heads ----------
        if use_taf:
            self.taf = TAF(dim, drop, gate_zero_init=True)
            self.quality_head = build_mlp(dim, 1, dim, drop)
            self.align_head = build_mlp(dim, 1, dim, drop)
            self.shared_head = None
        else:
            self.taf = None
            self.quality_head = None
            self.align_head = None
            self.shared_head = build_mlp(dim * 2, output_dim, dim * 2, drop)

        self.head_scale = head_scale

        if freeze_text:
            freeze_module(self.txt_model)
            freeze_module(self.wte)
            freeze_module(self.ln_final)
            freeze_module(self.txt_pos)
            freeze_module(self.text_projection)

    # ------------------------------------------------------------------ #
    def encode_text(self, text):
        """Return (token_features [B,L,C_text], mask [B,L], global [B,1024])."""
        text = clip.tokenize(text, context_length=77, truncate=True).cuda()
        mask = (text != 0)  # padding token id is 0
        x = self.wte(text).type(self.dtype)
        x = x + self.txt_pos.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _ = self.txt_model(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [B, L, 512]

        # global text = eot embedding @ text_projection
        eot = text.argmax(dim=-1)
        global_t = x[torch.arange(x.shape[0]), eot] @ self.text_projection
        return x, mask, global_t

    def build_plain_tokens(self, spatial):
        """fine/coarse tokens without any attention (used when MSQR is off)."""
        f0 = self.plain_proj(spatial)                     # [B, D, H, W]
        B, C, H, W = f0.shape
        fine = f0.flatten(2).transpose(1, 2)              # [B, H*W, D]
        coarse = F.avg_pool2d(f0, kernel_size=2).flatten(2).transpose(1, 2)
        return fine, coarse

    def forward(self, x, text):
        token_feat, text_mask, global_t = self.encode_text(text)

        spatial = self.resnet50(x)                        # [B,2048,16,16]
        global_v = self.attnpool(spatial)                 # [B,1024]

        # ----- visual stream -----
        if self.use_msqr:
            fine, coarse = self.msqr(spatial)
            f_visual = self.visual_skip(fine, coarse)     # [B, D]
        else:
            fine, coarse = self.build_plain_tokens(spatial)
            f_visual = self.f_visual_proj(global_v)       # [B, D]

        # ----- cross-modal stream -----
        if self.use_shcmi:
            f_cross = self.shcmi(fine, coarse, token_feat, text_mask)
        else:
            f_cross = self.f_cross_proj(global_t)         # [B, D]

        # ----- fusion + heads -----
        if self.use_taf:
            f_q, f_a = self.taf(f_visual, f_cross)
            q = self.quality_head(f_q)
            a = self.align_head(f_a)
            return torch.cat([q, a], dim=-1)
        else:
            feat = torch.cat([f_visual, f_cross], dim=-1)
            return self.shared_head(feat)

    # ------------------------------------------------------------------ #
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """Grouped LR: CLIP backbone 1x, new modules 10x, heads/gates high."""
        base_wd, base_nowd = [], []
        module_wd, module_nowd = [], []
        head_wd, head_nowd = [], []
        gate_params = []

        module_prefixes = ("msqr.", "shcmi.", "visual_skip.", "plain_proj.",
                           "f_visual_proj.", "f_cross_proj.", "taf.")
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = n.startswith("quality_head") or n.startswith("align_head") \
                or n.startswith("shared_head")
            is_nowd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
            if is_head:
                (head_nowd if is_nowd else head_wd).append(p)
            elif n.startswith("taf."):
                gate_params.append(p)  # gates, zero-init, no wd
            elif any(n.startswith(pfx) for pfx in module_prefixes):
                (module_nowd if is_nowd else module_wd).append(p)
            else:
                (base_nowd if is_nowd else base_wd).append(p)

        optim_params = [
            {"params": base_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": base_nowd, "weight_decay": 0, "lr_scale": lr_scale},
            {"params": module_wd, "weight_decay": weight_decay, "lr_scale": 10.0 * lr_scale},
            {"params": module_nowd, "weight_decay": 0, "lr_scale": 10.0 * lr_scale},
            {"params": head_wd, "weight_decay": weight_decay, "lr_scale": 10.0 * lr_scale},
            {"params": head_nowd, "weight_decay": 0, "lr_scale": 10.0 * lr_scale},
            {"params": gate_params, "weight_decay": 0, "lr_scale": 10.0 * lr_scale},
        ]
        if self.head_scale:
            for g in optim_params:
                if g["params"] is gate_params:
                    pass
            # scale head group further if head_scale is set
            optim_params = [
                {
                    "params": g["params"],
                    "weight_decay": g["weight_decay"],
                    "lr_scale": g["lr_scale"] * (self.head_scale if (
                        g["params"] is head_wd or g["params"] is head_nowd
                    ) else 1.0),
                }
                for g in optim_params
            ]
        optim_params = [g for g in optim_params if len(g["params"]) > 0]
        return optim_params

    @classmethod
    def from_config(cls, cfg):
        base_ckpt = cfg.get('base_ckpt', '../data/ckpt/clip/openai/resnet/RN50.pt')
        input_resolution = cfg.get("input_resolution", 512)
        output_dim = cfg.get("output_dim", 2)
        dim = cfg.get("dim", 256)
        num_heads = cfg.get("num_heads", 4)
        mlp_ratio = cfg.get("mlp_ratio", 2.0)
        drop = cfg.get("dropout_rate", 0.1)
        use_msqr = cfg.get("use_msqr", True)
        use_shcmi = cfg.get("use_shcmi", True)
        use_taf = cfg.get("use_taf", True)
        msqr_use_channel = cfg.get("msqr_use_channel", True)
        msqr_use_spatial = cfg.get("msqr_use_spatial", True)
        msqr_use_cross_scale = cfg.get("msqr_use_cross_scale", True)
        shcmi_use_multi_kernel = cfg.get("shcmi_use_multi_kernel", True)
        gamma_init = cfg.get("gamma_init", 0.0)
        freeze_text = cfg.get("freeze_text", True)
        head_scale = cfg.get("head_scale", None)

        model = cls(
            base_ckpt=base_ckpt,
            input_resolution=input_resolution,
            output_dim=output_dim,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            use_msqr=use_msqr,
            use_shcmi=use_shcmi,
            use_taf=use_taf,
            msqr_use_channel=msqr_use_channel,
            msqr_use_spatial=msqr_use_spatial,
            msqr_use_cross_scale=msqr_use_cross_scale,
            shcmi_use_multi_kernel=shcmi_use_multi_kernel,
            gamma_init=gamma_init,
            freeze_text=freeze_text,
            head_scale=head_scale,
        )
        return model
