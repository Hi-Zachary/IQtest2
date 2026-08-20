"""MSQRNet -- Frozen CLIP multimodal baseline + residual MSQR / SHCMI / TAF.

v2 架构（改进1.md）：

    B0 = Frozen CLIP multimodal baseline
    B1 = B0 + Residual MSQR
    B2 = B0 + Residual SHCMI
    B3 = B0 + MSQR + SHCMI
    B4 = B3 + Residual TAF

Nested 性质（严格消融）：
    关闭 MSQR            : B1 -> B0
    关闭 SHCMI           : B2 -> B0
    关闭 MSQR + SHCMI    : B3 -> B0
    TAF residual gate=0  : B4 -> B3

Base path（base_visual_proj / base_text_proj / shared_fusion / quality_head /
align_head）在 B0-B4 中永远存在，任何变体都共享同一套 regression head。

数据流（方案第 16 节）：
    spatial = CLIP RN50 (frozen)
    global_v = attnpool(spatial)
    global_t = text encoder (frozen)

    v0 = base_visual_proj(global_v)
    t0 = base_text_proj(global_t)

    v = v0 + tanh(lambda_msqr)  * delta_v      (use_msqr)
    c = t0 + tanh(lambda_shcmi) * delta_c      (use_shcmi)

    h = shared_fusion(concat[v, c])

    h_q = h + tanh(lambda_taf_q) * delta_q     (use_taf)
    h_a = h + tanh(lambda_taf_a) * delta_a

    q = quality_head(h_q); a = align_head(h_a)
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
            gamma_init=0.0,          # MSQR 内部 cross-scale gamma
            internal_gate_init=0.01,  # SHCMI 内部 eta/alpha/beta 初始值
            freeze_visual=True,
            freeze_text=True,
            module_lr_scale=1.0,     # 新模块 lr 倍数（FT-CLIP 用 10）
            head_scale=None,         # 主消融不用；保留兼容
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.use_msqr = use_msqr
        self.use_shcmi = use_shcmi
        self.use_taf = use_taf
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.module_lr_scale = module_lr_scale
        self.head_scale = head_scale

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

        # ====================== B0 base path (永远存在) ======================
        self.base_visual_proj = nn.Sequential(
            nn.Linear(CLIP_VISUAL_WIDTH, dim),
            nn.GELU(),
        )
        self.base_text_proj = nn.Sequential(
            nn.Linear(CLIP_VISUAL_WIDTH, dim),
            nn.GELU(),
        )
        self.shared_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.quality_head = nn.Linear(dim, 1)
        self.align_head = nn.Linear(dim, 1)

        # ====================== MSQR residual branch ======================
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
            self.lambda_msqr = nn.Parameter(torch.tensor(0.0))
        else:
            # Plain Multi-Scale Adapter：仅为 SHCMI 提供多尺度 token，不含任何
            # attention（B2 不能偷偷包含 MSQR）。
            self.plain_multiscale_adapter = nn.Sequential(
                nn.Conv2d(2048, dim, kernel_size=1),
                nn.GELU(),
            )
            self.visual_skip = None
            self.lambda_msqr = None

        # ====================== SHCMI residual branch ======================
        if use_shcmi:
            self.shcmi = SHCMI(
                text_dim=CLIP_TEXT_WIDTH, dim=dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop,
                gamma_init=internal_gate_init,
                use_multi_kernel=shcmi_use_multi_kernel,
            )
            self.lambda_shcmi = nn.Parameter(torch.tensor(0.0))
        else:
            self.shcmi = None
            self.lambda_shcmi = None

        # ====================== TAF residual branch ======================
        if use_taf:
            self.taf = TAF(dim, drop, gate_zero_init=True)
            self.lambda_taf_q = nn.Parameter(torch.tensor(0.0))
            self.lambda_taf_a = nn.Parameter(torch.tensor(0.0))
        else:
            self.taf = None
            self.lambda_taf_q = None
            self.lambda_taf_a = None

        # ====================== freeze ======================
        if freeze_visual:
            for p in self.resnet50.parameters():
                p.requires_grad = False
            for p in self.attnpool.parameters():
                p.requires_grad = False
        if freeze_text:
            freeze_module(self.txt_model)
            freeze_module(self.wte)
            freeze_module(self.ln_final)
            freeze_module(self.txt_pos)
            freeze_module(self.text_projection)

        self._last_ratios = {}

    # ---------------- train/eval override (冻结 BN 统计) ----------------
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual:
            self.resnet50.eval()
            self.attnpool.eval()
        if self.freeze_text:
            self.txt_model.eval()
        return self

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
        """fine/coarse tokens without any attention (Plain Multi-Scale Adapter)."""
        f0 = self.plain_multiscale_adapter(spatial)       # [B, D, H, W]
        B, C, H, W = f0.shape
        fine = f0.flatten(2).transpose(1, 2)              # [B, H*W, D]
        coarse = F.avg_pool2d(f0, kernel_size=2).flatten(2).transpose(1, 2)
        return fine, coarse

    def forward(self, x, text):
        # ===================== 1. Frozen CLIP backbone =====================
        spatial = self.resnet50(x)                        # [B,2048,16,16]
        global_v = self.attnpool(spatial)                 # [B,1024]
        text_tokens, text_mask, global_t = self.encode_text(text)

        # ===================== 2. B0 anchors =====================
        v0 = self.base_visual_proj(global_v)              # [B, D]
        t0 = self.base_text_proj(global_t)                # [B, D]
        v = v0
        c = t0

        ratios = {}

        # ===================== 3. visual tokens =====================
        if self.use_msqr:
            fine, coarse = self.msqr(spatial)
            delta_v = self.visual_skip(fine, coarse)      # [B, D]
            msqr_scale = torch.tanh(self.lambda_msqr)
            v = v0 + msqr_scale * delta_v
            ratios["msqr_ratio"] = (
                (msqr_scale * delta_v).norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()
        else:
            fine, coarse = self.build_plain_tokens(spatial)

        # ===================== 4. SHCMI residual =====================
        if self.use_shcmi:
            delta_c = self.shcmi(fine, coarse, text_tokens, text_mask)  # [B, D]
            shcmi_scale = torch.tanh(self.lambda_shcmi)
            c = t0 + shcmi_scale * delta_c
            ratios["shcmi_ratio"] = (
                (shcmi_scale * delta_c).norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()

        # ===================== 5. Shared representation =====================
        h = self.shared_fusion(torch.cat([v, c], dim=-1))  # [B, D]

        # ===================== 6. TAF residual =====================
        if self.use_taf:
            g_q, g_a = self.taf.compute_gates(v, c)
            mix_q = g_q * v + (1.0 - g_q) * c
            mix_a = g_a * v + (1.0 - g_a) * c
            delta_q = self.taf.quality_adapter(mix_q)
            delta_a = self.taf.align_adapter(mix_a)

            taf_q_scale = torch.tanh(self.lambda_taf_q)
            taf_a_scale = torch.tanh(self.lambda_taf_a)
            h_q = h + taf_q_scale * delta_q
            h_a = h + taf_a_scale * delta_a

            ratios["taf_q_ratio"] = (
                (taf_q_scale * delta_q).norm(dim=-1).mean() / h.norm(dim=-1).mean()
            ).item()
            ratios["taf_a_ratio"] = (
                (taf_a_scale * delta_a).norm(dim=-1).mean() / h.norm(dim=-1).mean()
            ).item()
        else:
            h_q = h
            h_a = h

        # ===================== 7. same task heads =====================
        q = self.quality_head(h_q)
        a = self.align_head(h_a)

        self._last_ratios = ratios
        return torch.cat([q, a], dim=-1)

    # ---------------- gate / residual logging ----------------
    def get_gate_log(self):
        """每 epoch 记录真实 gate 参数值与 residual ratio（方案 23/24 节）。"""
        out = dict(self._last_ratios)

        if self.use_msqr:
            out["lambda_msqr"] = torch.tanh(self.lambda_msqr).item()
            out["gamma_f"] = self.msqr.gamma_f.item()
            out["gamma_c"] = self.msqr.gamma_c.item()
        if self.use_shcmi:
            out["lambda_shcmi"] = torch.tanh(self.lambda_shcmi).item()
            out["eta"] = self.shcmi.eta.item()
            out["alpha_f"] = self.shcmi.alpha_f.item()
            out["beta_f"] = self.shcmi.beta_f.item()
            out["alpha_c"] = self.shcmi.alpha_c.item()
            out["beta_c"] = self.shcmi.beta_c.item()
            sg = getattr(self.shcmi, "_last_scale_gate", None)
            if sg is not None:
                out["scale_gate_mean"] = round(sg.mean().item(), 4)
                out["scale_gate_std"] = round(sg.std().item(), 4)
        if self.use_taf:
            out["lambda_taf_q"] = torch.tanh(self.lambda_taf_q).item()
            out["lambda_taf_a"] = torch.tanh(self.lambda_taf_a).item()
            gq, ga = getattr(self.taf, "_last_gates", (None, None))
            if gq is not None:
                out["g_q_mean"] = round(gq.mean().item(), 4)
                out["g_q_std"] = round(gq.std().item(), 4)
                out["g_a_mean"] = round(ga.mean().item(), 4)
                out["g_a_std"] = round(ga.std().item(), 4)
        return out

    # ------------------------------------------------------------------ #
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """简单分组：backbone（若可训练）1x，其余新层 module_lr_scale。

        主消融 B0-B4 全冻结 backbone，因此所有可训练参数都是新层，统一
        lr_scale = module_lr_scale（默认 1，即全 1e-4）。
        FT-CLIP 设 module_lr_scale=10 -> backbone 1x / new 10x。
        """
        backbone_prefixes = (
            "resnet50.", "attnpool.", "txt_model.", "wte.",
            "ln_final.", "txt_pos.", "text_projection.",
        )
        backbone_wd, backbone_nowd = [], []
        new_wd, new_nowd = [], []

        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_nowd = p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n
            if any(n.startswith(pfx) for pfx in backbone_prefixes):
                (backbone_nowd if is_nowd else backbone_wd).append(p)
            else:
                (new_nowd if is_nowd else new_wd).append(p)

        optim_params = [
            {"params": backbone_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": backbone_nowd, "weight_decay": 0, "lr_scale": lr_scale},
            {"params": new_wd, "weight_decay": weight_decay, "lr_scale": self.module_lr_scale * lr_scale},
            {"params": new_nowd, "weight_decay": 0, "lr_scale": self.module_lr_scale * lr_scale},
        ]
        return [g for g in optim_params if len(g["params"]) > 0]

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
        internal_gate_init = cfg.get("internal_gate_init", 0.01)
        freeze_visual = cfg.get("freeze_visual", True)
        freeze_text = cfg.get("freeze_text", True)
        module_lr_scale = cfg.get("module_lr_scale", 1.0)
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
            internal_gate_init=internal_gate_init,
            freeze_visual=freeze_visual,
            freeze_text=freeze_text,
            module_lr_scale=module_lr_scale,
            head_scale=head_scale,
        )
        return model
