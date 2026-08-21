import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v11 import MSQRNetV11

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_mscm):
    return MSQRNetV11(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_mscm=use_mscm,
                      freeze_visual=True, freeze_text=True)

print("registry:", registry.list_models())

# ---- Full-V11: 前向 + 参数 + LoRA 命名 ----
m = build(True, True)
print("--- Full-V11 summary ---")
print(json.dumps(m.trainable_summary()))
# 打印几个 LoRA 参数名看 adapter 命名
lora_names = [n for n, _ in m.named_parameters() if "lora" in n][:4]
print("lora sample names:", lora_names)
m.train()
x = torch.randn(2, 3, 224, 224)
text = ["a statue of a man in the park", "a tray of sushi on a table"]
out = m(x, text)
loss = nn.MSELoss()(out, torch.randn(2, 2))
loss.backward()
print(f"out={tuple(out.shape)} loss={loss.item():.4f}")
text_train = [n for n, p in m.named_parameters() if p.requires_grad and "text_model" in n]
print("text trainable:", len(text_train))
assert not text_train

# ---- 梯度隔离检查（方案第 33 节） ----
def param_by(pred):
    return {n: p for n, p in m.named_parameters() if pred(n) and p.requires_grad}
lora_q = param_by(lambda n: ("lora_A_q" in n) or ("lora_B_q" in n))
lora_a = param_by(lambda n: ("lora_A_a" in n) or ("lora_B_a" in n))

print("\n=== 梯度隔离检查 (Full-V11) ===")
m.zero_grad(set_to_none=True)
x = torch.randn(2, 3, 224, 224)
target_q = torch.randn(2)
target_a = torch.randn(2)

# L_Q only
out = m(x, text)
q_loss = nn.MSELoss()(out[:, 0], target_q)
q_loss.backward(retain_graph=True)
def has_grad(d, name):
    g = d[name].grad
    return (g is not None and g.abs().sum().item() > 0) if name in d else False
dg_ok = m.dg_mpq.quality_out[0].weight.grad is not None and m.dg_mpq.quality_out[0].weight.grad.abs().sum().item() > 0
loraq_ok = any(has_grad(lora_q, n) for n in lora_q)
mscm_ok = any(has_grad(param_by(lambda n: n.startswith("mscm.")), n) for n in param_by(lambda n: n.startswith("mscm.")))
loraa_ok = any(has_grad(lora_a, n) for n in lora_a)
print(f"  L_Q backward -> DG-MPQ grad>0: {dg_ok}, LoRA-Q grad>0: {loraq_ok}")
print(f"  L_Q backward -> MSCM grad==0: {not mscm_ok}, LoRA-A grad==0: {not loraa_ok}")
assert dg_ok and loraq_ok and not mscm_ok and not loraa_ok, "quality gradient leaked into alignment!"

m.zero_grad(set_to_none=True)
out = m(x, text)
a_loss = nn.MSELoss()(out[:, 1], target_a)
a_loss.backward()
mscm_ok = any(has_grad(param_by(lambda n: n.startswith("mscm.")), n) for n in param_by(lambda n: n.startswith("mscm.")))
loraa_ok = any(has_grad(lora_a, n) for n in lora_a)
dg_ok = m.dg_mpq.quality_out[0].weight.grad is not None and m.dg_mpq.quality_out[0].weight.grad.abs().sum().item() > 0
loraq_ok = any(has_grad(lora_q, n) for n in lora_q)
print(f"  L_A backward -> MSCM grad>0: {mscm_ok}, LoRA-A grad>0: {loraa_ok}")
print(f"  L_A backward -> DG-MPQ grad==0: {not dg_ok}, LoRA-Q grad==0: {not loraq_ok}")
assert mscm_ok and loraa_ok and not dg_ok and not loraq_ok, "alignment gradient leaked into quality!"
print("  梯度隔离 OK")

# ---- 显存测量（batch 64, AMP）----
print("\n=== Full-V11 显存 (batch 64, AMP) ===")
m2 = build(True, True).cuda().train()
x = torch.randn(64, 3, 224, 224).cuda()
text64 = [text[0]] * 64
opt = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=1e-4)
scaler = torch.cuda.amp.GradScaler()
torch.cuda.reset_peak_memory_stats()
for i in range(3):
    y = torch.randn(64, 2).cuda()
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        o = m2(x, text64)
        l = nn.MSELoss()(o, y)
    scaler.scale(l).backward(); scaler.step(opt); scaler.update()
print(f"  peak mem: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
del m, m2
torch.cuda.empty_cache()
print("\nV11 SMOKE OK")
