import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v10 import MSQRNetV10

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_mscm):
    return MSQRNetV10(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_mscm=use_mscm,
                      freeze_visual=True, freeze_text=True)

print("registry:", registry.list_models())

for name, kw in [("B1", dict(use_dg_mpq=True, use_mscm=False)),
                 ("B2-MSCM", dict(use_dg_mpq=False, use_mscm=True)),
                 ("Full-MSCM", dict(use_dg_mpq=True, use_mscm=True))]:
    m = build(**kw)
    s = m.trainable_summary()
    m.train()
    x = torch.randn(2, 3, 224, 224)
    text = ["a statue of a man in the park", "a tray of sushi on a table"]
    out = m(x, text)
    loss = nn.MSELoss()(out, torch.randn(2, 2))
    loss.backward()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    text_train = [n for n in trainable if n.startswith("backbone.clip.text_model.")]
    vis_base = [n for n in trainable if n.startswith("backbone.clip.vision_model.") and "lora" not in n]
    gl = m.get_gate_log()
    print(f"--- {name} ---")
    print(f"  summary: {json.dumps(s)}")
    print(f"  out={tuple(out.shape)} loss={loss.item():.4f}  text_trainable={len(text_train)} vis_base={len(vis_base)}")
    print(f"  gate keys: {sorted(gl.keys())}")
    assert not text_train and not vis_base
    if kw["use_mscm"]:
        assert "corr_s_global" in gl and "corr_s9_mean" in gl
    del m
    torch.cuda.empty_cache()

print("\nV10 SMOKE OK")
