import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v9 import MSQRNetV9

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_tcap):
    return MSQRNetV9(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_tcap=use_tcap,
                     freeze_visual=True, freeze_text=True)

print("registry:", registry.list_models())

for name, kw in [("B2-TCAP", dict(use_dg_mpq=False, use_tcap=True)),
                 ("Full-TCAP", dict(use_dg_mpq=True, use_tcap=True))]:
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
    print(f"--- {name} ---")
    print(f"  summary: {json.dumps(s)}")
    print(f"  out shape={tuple(out.shape)}  loss={loss.item():.4f}")
    print(f"  text trainable={len(text_train)}  visual_base trainable={len(vis_base)}")
    # TCAP attention logging
    gl = m.get_gate_log()
    print(f"  gate keys: {sorted(gl.keys())}")
    assert not text_train and not vis_base, "backbone must be frozen"
    assert "tcap_attn_entropy" in gl
    del m
    torch.cuda.empty_cache()

print("\nV9 SMOKE OK")
