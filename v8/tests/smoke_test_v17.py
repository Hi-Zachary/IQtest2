import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v17 import MSQRNetV17

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_qard):
    return MSQRNetV17(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_qard=use_qard,
                      alignment_stopgrad=True, drop=0.0,
                      freeze_visual=True, freeze_text=True)

print("registry:", registry.list_models())

for name, kw in [("B1", dict(use_dg_mpq=True, use_qard=False)),
                 ("B2-QARD", dict(use_dg_mpq=False, use_qard=True)),
                 ("Full", dict(use_dg_mpq=True, use_qard=True))]:
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
    assert s["qard"] > 0
    del m
    torch.cuda.empty_cache()

print("\n=== 梯度隔离检查 ===")
m = build(True, True).train()  # Full
x = torch.randn(2, 3, 224, 224)
text = ["a statue of a man in the park", "a tray of sushi on a table"]
target_q, target_a = torch.randn(2), torch.randn(2)

def gz(mm, pred):
    return sum(p.grad.abs().sum().item() for n, p in mm.named_parameters() if pred(n) and p.grad is not None) > 0

m.zero_grad(set_to_none=True); out = m(x, text)
nn.MSELoss()(out[:, 0], target_q).backward()
print(f"  Full L_Q -> LoRA: {gz(m, lambda n: 'lora' in n)}, DG-MPQ: {gz(m, lambda n: n.startswith('dg_mpq.'))}, QARD: {gz(m, lambda n: n.startswith('qard.'))}, Align==0: {not gz(m, lambda n: n.startswith('alignment_'))}")
assert gz(m, lambda n: 'lora' in n) and gz(m, lambda n: n.startswith('dg_mpq.')) and gz(m, lambda n: n.startswith('qard.'))
assert not gz(m, lambda n: n.startswith('alignment_'))

m.zero_grad(set_to_none=True); out = m(x, text)
nn.MSELoss()(out[:, 1], target_a).backward()
print(f"  Full L_A -> Align: {gz(m, lambda n: n.startswith('alignment_'))}, LoRA==0: {not gz(m, lambda n: 'lora' in n)}, DG-MPQ==0: {not gz(m, lambda n: n.startswith('dg_mpq.'))}, QARD==0: {not gz(m, lambda n: n.startswith('qard.'))}")
assert gz(m, lambda n: n.startswith('alignment_'))
assert not gz(m, lambda n: 'lora' in n) and not gz(m, lambda n: n.startswith('dg_mpq.')) and not gz(m, lambda n: n.startswith('qard.'))

m2 = build(False, True).train()  # B2
m2.zero_grad(set_to_none=True); out = m2(x, text)
nn.MSELoss()(out[:, 0], target_q).backward()
print(f"  B2 L_Q -> DG-MPQ==0: {not gz(m2, lambda n: n.startswith('dg_mpq.'))}, QARD>0: {gz(m2, lambda n: n.startswith('qard.'))}")
assert not gz(m2, lambda n: n.startswith('dg_mpq.')) and gz(m2, lambda n: n.startswith('qard.'))
del m, m2
torch.cuda.empty_cache()
print("\nV17 SMOKE OK")
