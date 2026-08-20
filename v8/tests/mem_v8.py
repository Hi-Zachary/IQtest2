import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(0)

from ipiqa.models.model_v8 import MSQRNetV8
m = MSQRNetV8(model_name="ckpt/clip-vit-base-patch16",
              use_lora=True, use_dg_mpq=True, use_dp_hcmi=True,
              freeze_visual=True, freeze_text=True).cuda().train()

text = ["a statue of a man in the park"] * 32
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
for i in range(5):
    x = torch.randn(32, 3, 224, 224).cuda()
    y = torch.randn(32, 2).cuda()
    opt.zero_grad()
    out = m(x, text)
    loss = nn.MSELoss()(out, y)
    loss.backward()
    opt.step()
peak = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"v8  Full  batch=32  input=224x224  peak_mem={peak:.0f} MiB")
