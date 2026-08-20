"""MSQRNet -- Frozen CLIP multimodal baseline + DMSQR / DP-HCMI (v7).

v7（改进6.md）关键修复，目标解决 v6 的三个真实问题：
    1. 共享 Spatial Projection：Conv1x1 (2048->D) 从 MSQR 内部提取到 model 的
       ``spatial_proj``，B2 与 Full 的 DP-HCMI 拿到**完全相同**的 fine/coarse
       base tokens（B2 不再用独立的 plain_multiscale_adapter）；
    2. DP-HCMI 永远使用 base tokens，DMSQR 的 refinement 输出不再进入 DP-HCMI
       （消除 hidden input change）；
    3. 修复 prompt weighting（mask + 保幅值）与 discrepancy 双重 tanh 梯度死区。

架构（与 v6 相同）：

    B0 = Frozen CLIP multimodal baseline
    B1 = B0 + DMSQR        (Distortion-aware Multi-Scale Quality Refinement)
    B2 = B0 + DP-HCMI      (Discrepancy-aware Prompt-conditioned HCMI)
    Ours = B0 + DMSQR + DP-HCMI

TAF / QTA / AG / Consistency Loss 均已删除。创新点收敛为 2 个，各对应一个任务：
    - DMSQR → Quality：CLIP 视觉语义偏差感知 AIGC 局部生成缺陷
    - DP-HCMI → Alignment：显式建模图像与 prompt 的语义不一致（discrepancy）

Loss 保持纯 MSE（quality + alignment），不引入额外训练变量。

Nested 性质（严格消融）：
    关闭 DMSQR           : B1 -> B0
    关闭 DP-HCMI         : B2 -> B0
    关闭 DMSQR + DP-HCMI : Ours -> B0

Base path（spatial_proj / base_visual_proj / base_text_proj / shared_fusion /
quality_head / align_head）在 B0-B2 / Ours 中永远存在，任何变体都共享同一套
regression head。

数据流：
    spatial = CLIP RN50 (frozen)
    global_v = attnpool(spatial)
    global_t = text encoder (frozen)

    base_map = spatial_proj(spatial)                # 共享投影
    fine_base = flatten(base_map);  coarse_base = avg_pool2d(base_map, k=2)

    v0 = base_visual_proj(global_v)
    t0 = base_text_proj(global_t)

    fine_d, coarse_d = DMSQR(fine_base, coarse_base, global_v)   # (use_msqr)
    delta_v = visual_skip(fine_d, coarse_d)
    v = v0 + tanh(lambda_msqr) * delta_v

    delta_c = DP-HCMI(fine_base, coarse_base, text_tokens, text_mask)  # (use_shcmi)
    c = t0 + tanh(lambda_shcmi) * delta_c

    h = shared_fusion(concat[v, c])
    q = quality_head(h); a = align_head(h)
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

from ipiqa.common.registry import registry

CLIP_TEXT_WIDTH = 512   # RN50 text transformer width
CLIP_VISUAL_WIDTH = 1024  # RN50 attnpool output dim


@registry.register_model("msqr_shcmi")
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
            msqr_use_channel=True,
            msqr_use_spatial=True,
            msqr_use_cross_scale=True,
            shcmi_use_multi_kernel=True,
            use_deviation=False,     # DMSQR 语义偏差
            use_prompt_weight=False,  # DP-HCMI prompt 加权
            use_align_bias=False,     # DP-HCMI discrepancy bias
            gamma_init=0.0,          # MSQR 内部 cross-scale gamma
            internal_gate_init=0.01,  # SHCMI 内部 eta/alpha/beta 初始值
            outer_gate_init=0.01,    # v7.2: 外部门控 lambda_msqr/lambda_shcmi 初始值
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
        self.use_deviation = use_deviation
        self.use_prompt_weight = use_prompt_weight
        self.use_align_bias = use_align_bias
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text
        self.module_lr_scale = module_lr_scale
        self.head_scale = head_scale
        self.outer_gate_init = float(outer_gate_init)

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
        # v7: Shared Spatial Projection（改进6.md 第 2 节）—— Conv1x1 2048->D
        # 从 MSQR 内部提取为模型级共享投影，B0-B2 / Ours 完全一致，保证
        # B2 与 Full 的 DP-HCMI 输入（fine/coarse base tokens）严格相同。
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(2048, dim, kernel_size=1),
            nn.GELU(),
        )
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
        # v6: 简单 head（改进5.md 第 4 节，不引入 shared/task adapter）
        self.quality_head = nn.Linear(dim, 1)
        self.align_head = nn.Linear(dim, 1)

        # ====================== DMSQR residual branch ======================
        # v7.3（改进8.md）：LayerNorm 只放在 proposed module 内部
        # （MSQRVisualSkip / SHCMI 的 out_norm），不再对 anchor 或 model 层
        # delta 做 LayerNorm——避免 v7.2 里 anchor LN 强化 B0 的问题。
        if use_msqr:
            self.msqr = MSQR(
                dim=dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop,
                use_channel_attention=msqr_use_channel,
                use_spatial_attention=msqr_use_spatial,
                use_cross_scale=msqr_use_cross_scale,
                gamma_init=gamma_init,
                use_deviation=use_deviation,
                global_dim=CLIP_VISUAL_WIDTH,
            )
            self.visual_skip = MSQRVisualSkip(dim, drop)
            # v7.2: 外门控小正初始化（lambda=0 会抑制 branch 梯度，形成赢家通吃）
            self.lambda_msqr = nn.Parameter(torch.tensor(self.outer_gate_init))
        else:
            self.msqr = None
            self.visual_skip = None
            self.lambda_msqr = None

        # ====================== SHCMI residual branch ======================
        if use_shcmi:
            self.shcmi = SHCMI(
                text_dim=CLIP_TEXT_WIDTH, dim=dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop,
                gamma_init=internal_gate_init,
                use_multi_kernel=shcmi_use_multi_kernel,
                use_prompt_weight=use_prompt_weight,
                use_align_bias=use_align_bias,
            )
            # v7.2: 与 lambda_msqr 完全相同的初始值，避免人为偏向某个模块
            self.lambda_shcmi = nn.Parameter(torch.tensor(self.outer_gate_init))
        else:
            self.shcmi = None
            self.lambda_shcmi = None

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
        """v7: 共享 spatial projection -> fine/coarse base tokens（无任何 attention）。
        仅供 B2 / 参考使用，DMSQR 与 DP-HCMI 都从这一对 base tokens 出发。"""
        f0 = self.spatial_proj(spatial)       # [B, D, H, W]
        fine = f0.flatten(2).transpose(1, 2)              # [B, H*W, D]
        coarse = F.avg_pool2d(f0, kernel_size=2).flatten(2).transpose(1, 2)
        return fine, coarse

    def forward(self, x, text):
        """Return output [B, 2] = concat(quality, alignment)."""
        # ===================== 1. Frozen CLIP backbone =====================
        spatial = self.resnet50(x)                        # [B,2048,16,16]
        global_v = self.attnpool(spatial)                 # [B,1024]
        text_tokens, text_mask, global_t = self.encode_text(text)

        # ===================== 2. Shared base tokens + B0 anchors =====================
        # v7（改进6.md 第 2 节）：fine/coarse base tokens 来自共享 spatial_proj，
        # B2 / Full 的 DP-HCMI 输入严格相同。
        # v7.3（改进8.md 第 7 节）：anchor 恢复简单 base projection，不再加 LayerNorm。
        fine_base, coarse_base = self.build_plain_tokens(spatial)
        v0 = self.base_visual_proj(global_v)              # [B, D]
        t0 = self.base_text_proj(global_t)                # [B, D]
        v = v0
        c = t0

        ratios = {}

        # ===================== 3. DMSQR residual（refine base tokens） =====================
        if self.use_msqr:
            fine_d, coarse_d = self.msqr(fine_base, coarse_base, global_v)
            delta_v = self.visual_skip(fine_d, coarse_d)      # [B, D]（内部已 LayerNorm）
            msqr_scale = torch.tanh(self.lambda_msqr)
            v = v0 + msqr_scale * delta_v
            ratios["raw_msqr_ratio"] = (
                delta_v.norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()
            ratios["msqr_ratio"] = (
                (msqr_scale * delta_v).norm(dim=-1).mean() / v0.norm(dim=-1).mean()
            ).item()

        # ===================== 4. DP-HCMI residual（永远用 base tokens） =====================
        # v7（改进6.md 第 4 节）：DP-HCMI 不接收 DMSQR 的 refined tokens，
        # 只接收 fine_base / coarse_base，避免 hidden input change。
        if self.use_shcmi:
            delta_c = self.shcmi(fine_base, coarse_base, text_tokens, text_mask)  # [B, D]（内部已 LayerNorm）
            shcmi_scale = torch.tanh(self.lambda_shcmi)
            c = t0 + shcmi_scale * delta_c
            ratios["raw_shcmi_ratio"] = (
                delta_c.norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()
            ratios["shcmi_ratio"] = (
                (shcmi_scale * delta_c).norm(dim=-1).mean() / t0.norm(dim=-1).mean()
            ).item()

        # ===================== 5. Shared representation =====================
        h = self.shared_fusion(torch.cat([v, c], dim=-1))  # [B, D]

        # ===================== 6. same task heads =====================
        q = self.quality_head(h)
        a = self.align_head(h)

        self._last_ratios = ratios
        return torch.cat([q, a], dim=-1)

    # ---------------- gate / residual logging ----------------
    def get_gate_log(self):
        """每 epoch 记录真实 gate 参数值与 residual ratio。"""
        out = dict(self._last_ratios)

        if self.use_msqr:
            out["lambda_msqr"] = torch.tanh(self.lambda_msqr).item()
            out["gamma_f"] = self.msqr.gamma_f.item()
            out["gamma_c"] = self.msqr.gamma_c.item()
            if self.use_deviation:
                out["alpha_dev"] = torch.tanh(self.msqr.alpha_dev).item()
                dw = getattr(self.msqr, "_last_dev_weight", None)
                if dw is not None:
                    out["dev_w_mean"] = round(dw.mean().item(), 4)
                    out["dev_w_std"] = round(dw.std().item(), 4)
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
            if self.use_prompt_weight:
                pw = getattr(self.shcmi, "_last_prompt_weight", None)
                if pw is not None:
                    out["prompt_w_mean"] = round(pw.mean().item(), 4)
                    out["prompt_w_std"] = round(pw.std().item(), 4)
            if self.use_align_bias:
                out["beta_align"] = torch.tanh(self.shcmi.beta_align).item()
        return out

    # ------------------------------------------------------------------ #
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """简单分组：backbone（若可训练）1x，其余新层 module_lr_scale。

        主消融 B0-B3 / Ours 全冻结 backbone，因此所有可训练参数都是新层，统一
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
        use_deviation = cfg.get("use_deviation", False)
        use_prompt_weight = cfg.get("use_prompt_weight", False)
        use_align_bias = cfg.get("use_align_bias", False)
        msqr_use_channel = cfg.get("msqr_use_channel", True)
        msqr_use_spatial = cfg.get("msqr_use_spatial", True)
        msqr_use_cross_scale = cfg.get("msqr_use_cross_scale", True)
        shcmi_use_multi_kernel = cfg.get("shcmi_use_multi_kernel", True)
        gamma_init = cfg.get("gamma_init", 0.0)
        internal_gate_init = cfg.get("internal_gate_init", 0.01)
        outer_gate_init = cfg.get("outer_gate_init", 0.01)
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
            use_deviation=use_deviation,
            use_prompt_weight=use_prompt_weight,
            use_align_bias=use_align_bias,
            msqr_use_channel=msqr_use_channel,
            msqr_use_spatial=msqr_use_spatial,
            msqr_use_cross_scale=msqr_use_cross_scale,
            shcmi_use_multi_kernel=shcmi_use_multi_kernel,
            gamma_init=gamma_init,
            internal_gate_init=internal_gate_init,
            outer_gate_init=outer_gate_init,
            freeze_visual=freeze_visual,
            freeze_text=freeze_text,
            module_lr_scale=module_lr_scale,
            head_scale=head_scale,
        )
        return model

