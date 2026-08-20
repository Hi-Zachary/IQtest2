"""B0 -- truly neutral CLIP multimodal baseline (design section 2).

    Image  --CLIP RN50-->  Global Visual v
    Prompt --CLIP text-->  Global Text t
    concat [v; t] -> MLP -> [Quality, Alignment]

It deliberately avoids IP-IQA's Integral Prompt, TextAttentionPool2d, QA token
and Image2Prompt components, so every MSQR / SHCMI / TAF module forms a clean
increment over it.
"""

from ipiqa.models.model import MSQRNet
from ipiqa.common.registry import registry


@registry.register_model("msqr_baseline")
class MSQRBaseline(MSQRNet):
    def __init__(self, base_ckpt='', input_resolution=512, output_dim=2,
                 dim=256, drop=0.1, freeze_text=True, head_scale=None):
        super().__init__(
            base_ckpt=base_ckpt,
            input_resolution=input_resolution,
            output_dim=output_dim,
            dim=dim,
            drop=drop,
            use_msqr=False,
            use_shcmi=False,
            use_taf=False,
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
            freeze_text=cfg.get("freeze_text", True),
            head_scale=cfg.get("head_scale", None),
        )
