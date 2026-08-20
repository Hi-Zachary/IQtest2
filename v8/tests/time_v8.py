import sys, os, time, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ipiqa.models.model_v8 import MSQRNetV8

t0 = time.time()
m = MSQRNetV8(model_name='ckpt/clip-vit-base-patch16', use_lora=True, use_dg_mpq=True, use_hcmi=True,
              freeze_visual=True, freeze_text=True).cuda().train()
t_load = time.time() - t0
print(f"model load: {t_load:.1f}s")

text = ['a statue of a man in the park']*64
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

# 1 train epoch = 2384/64 = 37 iters
n_train = 37
torch.cuda.synchronize(); t0 = time.time()
for i in range(n_train):
    x = torch.randn(64,3,224,224).cuda(); y = torch.randn(64,2).cuda()
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        out = m(x, text); loss = nn.MSELoss()(out, y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize()
t_train = (time.time() - t0) / n_train
print(f"train iter: {t_train:.3f}s/iter  -> 1 train epoch (37 iters) = {t_train*37:.1f}s")

# 1 eval epoch = 598/64 = 10 iters (no_grad)
m.eval()
n_eval = 10
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    for i in range(n_eval):
        x = torch.randn(64,3,224,224).cuda()
        with torch.cuda.amp.autocast():
            out = m(x, text)
torch.cuda.synchronize()
t_eval = (time.time() - t0) / n_eval
print(f"eval iter: {t_eval:.3f}s/iter  -> 1 eval epoch (10 iters) = {t_eval*10:.1f}s")
