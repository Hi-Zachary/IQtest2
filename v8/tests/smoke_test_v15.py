import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v15 import MSQRNetV15

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_ptlc):
    return MSQRNetV15(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_ptlc=use_ptlc,
                      alignment_stopgrad=True, drop=0.0,
                      freeze_visual=True, freeze_text=True)

print("registry:", registry.list_models())

for name, kw in [("R0", dict(use_dg_mpq=False, use_ptlc=False)),
                 ("B2-PTLC", dict(use_dg_mpq=False, use_ptlc=True)),
                 ("B1", dict(use_dg_mpq=True, use_ptlc=False)),
                 ("Full-PTLC", dict(use_dg_mpq=True, use_ptlc=True))]:
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
    assert s["ptlc"] > 0  # always instantiate
    del m
    torch.cuda.empty_cache()

# ---- 梯度隔离检查（方案第 40 节） ----
print("\n=== 梯度隔离检查 (Full-PTLC, alignment_stopgrad=True) ===")
m = build(True, True).train()
x = torch.randn(2, 3, 224, 224)
text = ["a statue of a man in the park", "a tray of sushi on a table"]
target_q, target_a = torch.randn(2), torch.randn(2)

def grad_sum(pred_fn):
    return sum(p.grad.abs().sum().item() for n, p in m.named_parameters() if pred_fn(n) and p.grad is not None)

m.zero_grad(set_to_none=True)
out = m(x, text)
nn.MSELoss()(out[:, 0], target_q).backward()
lora_q = grad_sum(lambda n: "lora" in n)
ptlc_q = grad_sum(lambda n: n.startswith("ptlc."))
print(f"  L_Q -> LoRA grad>0: {lora_q>0}, PTLC grad==0: {ptlc_q==0}")
assert lora_q > 0 and ptlc_q == 0

m.zero_grad(set_to_none=True)
out = m(x, text)
nn.MSELoss()(out[:, 1], target_a).backward()
lora_a = grad_sum(lambda n: "lora" in n)
ptlc_a = grad_sum(lambda n: n.startswith("ptlc."))
align_a = grad_sum(lambda n: n.startswith("alignment_") or n == "lambda_a")
print(f"  L_A -> PTLC grad>0: {ptlc_a>0}, alignment grad>0: {align_a>0}, LoRA grad==0: {lora_a==0}")
assert ptlc_a > 0 and align_a > 0 and lora_a == 0, "stop-gradient NOT clean!"
print("  梯度隔离 OK (stop-gradient 干净)")
del m
torch.cuda.empty_cache()
print("\nV15 SMOKE OK")
