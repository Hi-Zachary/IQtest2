import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v14 import MSQRNetV14

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_msrc):
    return MSQRNetV14(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_msrc=use_msrc,
                      alignment_stopgrad=True, drop=0.0,
                      freeze_visual=True, freeze_text=True)

print("registry:", registry.list_models())

for name, kw in [("R0", dict(use_dg_mpq=False, use_msrc=False)),
                 ("B1", dict(use_dg_mpq=True, use_msrc=False)),
                 ("B2-MSRC", dict(use_dg_mpq=False, use_msrc=True)),
                 ("Full-MSRC", dict(use_dg_mpq=True, use_msrc=True))]:
    m = build(**kw)
    s = m.trainable_summary()
    m.train()
    x = torch.randn(2, 3, 224, 224)
    text = ["a statue of a man in the park", "a tray of sushi on a table"]
    out = m(x, text)
    loss = nn.MSELoss()(out, torch.randn(2, 2))
    loss.backward()
    text_train = [n for n, p in m.named_parameters() if p.requires_grad and "text_model" in n]
    vis_base = [n for n, p in m.named_parameters() if p.requires_grad and "backbone.clip.vision_model" in n and "lora" not in n]
    print(f"--- {name} ---")
    print(f"  summary: {json.dumps(s)}")
    print(f"  out={tuple(out.shape)} loss={loss.item():.4f} text_train={len(text_train)} vis_base={len(vis_base)}")
    assert not text_train and not vis_base
    assert s["msrc"] > 0  # always instantiate
    del m
    torch.cuda.empty_cache()

# ---- 梯度隔离检查（方案第 37 节：stop-gradient 后 L_A 不应触及 Visual LoRA） ----
print("\n=== 梯度隔离检查 (Full-MSRC, alignment_stopgrad=True) ===")
m = build(True, True).train()
x = torch.randn(2, 3, 224, 224)
text = ["a statue of a man in the park", "a tray of sushi on a table"]
target_q, target_a = torch.randn(2), torch.randn(2)

# L_Q only
m.zero_grad(set_to_none=True)
out = m(x, text)
nn.MSELoss()(out[:, 0], target_q).backward()
lora_grad = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
                if "lora" in n and p.grad is not None) if any("lora" in n and p.grad is not None for n, p in m.named_parameters()) else 0
msrc_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for n, p in m.named_parameters() if n.startswith("msrc."))
print(f"  L_Q -> lora grad>0: {lora_grad>0}, MSRC grad==0: {not msrc_grad}")
assert lora_grad > 0 and not msrc_grad

# L_A only
m.zero_grad(set_to_none=True)
out = m(x, text)
nn.MSELoss()(out[:, 1], target_a).backward()
lora_grad = sum(p.grad.abs().sum().item() for n, p in m.named_parameters() if "lora" in n and p.grad is not None)
msrc_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for n, p in m.named_parameters() if n.startswith("msrc."))
print(f"  L_A -> MSRC grad>0: {msrc_grad}, lora grad==0: {lora_grad==0}")
assert msrc_grad and lora_grad == 0, "stop-gradient NOT clean: L_A touched Visual LoRA!"
print("  梯度隔离 OK (stop-gradient 干净)")
del m
torch.cuda.empty_cache()
print("\nV14 SMOKE OK")
