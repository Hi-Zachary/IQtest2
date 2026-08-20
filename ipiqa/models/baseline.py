"""B0 -- Frozen CLIP multimodal baseline (改进1.md v2, section 3-4).

    Image  --Frozen CLIP RN50-->  global_v -> base_visual_proj -> v0
    Prompt --Frozen CLIP text-->  global_t -> base_text_proj   -> t0
    h0 = shared_fusion(concat[v0, t0])
    q  = quality_head(h0); a = align_head(h0)

只训练 base_visual_proj / base_text_proj / shared_fusion / quality_head /
align_head；CLIP visual encoder、text encoder、AttentionPool、text projection
全部冻结。

这是新的正式 B0，替代旧版 "fine-tuned CLIP baseline"（旧版保留为 FT-CLIP
强参考）。
"""

from ipiqa.models.model import MSQRNet
from ipiqa.common.registry import registry


@registry.register_model("msqr_baseline")
class MSQRBaseline(MSQRNet):
    def __init__(self, base_ckpt='', input_resolution=512, output_dim=2,
                 dim=256, drop=0.1, freeze_visual=True, freeze_text=True,
                 head_scale=None):
        super().__init__(
            base_ckpt=base_ckpt,
            input_resolution=input_resolution,
            output_dim=output_dim,
            dim=dim,
            drop=drop,
            use_msqr=False,
            use_shcmi=False,
            use_taf=False,
            freeze_visual=freeze_visual,
            freeze_text=freeze_text,
            head_scale=head_scale,
        )

    @classmethod
    def from_config(cls, cfg):
        return cls(
            base_ckpt=cfg.get('base_ckpt', '../data/ckpt/clip/openai/resnet/RN50.pt'),
            input_resolution=cfg.get("input_resolution", 512),
            output_dim=cfg.get("output_dim", 2),
            dim=cfg.get("dim", 256),
            drop=cfg.get("dropout_rate", 0.1),
            freeze_visual=cfg.get("freeze_visual", True),
            freeze_text=cfg.get("freeze_text", True),
            head_scale=cfg.get("head_scale", None),
        )
