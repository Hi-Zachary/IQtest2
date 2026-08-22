import os, sys, json, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from ipiqa.models.model_v17 import MSQRNetV17
from ipiqa.datasets.agiqa_datasets import AIGIQA20K
from ipiqa.common.registry import registry

CKPT = "ckpt/clip-vit-base-patch16"

def build(use_dg_mpq, use_qard, single_score=True):
    return MSQRNetV17(model_name=CKPT, use_dg_mpq=use_dg_mpq, use_qard=use_qard,
                      alignment_stopgrad=True, single_score=single_score, drop=0.0,
                      freeze_visual=True, freeze_text=True)

print("=== 1. single_score 前向 + 梯度隔离 ===")
for name, kw in [("R0", dict(use_dg_mpq=False, use_qard=False)),
                 ("B1", dict(use_dg_mpq=True, use_qard=False)),
                 ("Full", dict(use_dg_mpq=True, use_qard=True))]:
    m = build(**kw).train()
    x = torch.randn(2, 3, 224, 224)
    text = ["a corgi", "a boat"]
    out = m(x, text)
    assert out.shape == (2, 1), f"{name}: out={out.shape}"
    loss = nn.MSELoss()(out, torch.randn(2, 1))
    loss.backward()
    def gz(pred):
        return sum(p.grad.abs().sum().item() for n, p in m.named_parameters() if pred(n) and p.grad is not None) > 0
    if name == "R0":
        assert gz(lambda n: 'lora' in n) and gz(lambda n: n.startswith('quality_'))
        assert not gz(lambda n: n.startswith('dg_mpq.')) and not gz(lambda n: n.startswith('qard.'))
        assert not gz(lambda n: n.startswith('alignment_'))
    elif name == "B1":
        assert gz(lambda n: n.startswith('dg_mpq.')) and not gz(lambda n: n.startswith('qard.'))
    else:
        assert gz(lambda n: n.startswith('dg_mpq.')) and gz(lambda n: n.startswith('qard.'))
        assert not gz(lambda n: n.startswith('alignment_'))
    print(f"  {name}: out={tuple(out.shape)} loss={loss.item():.4f} 梯度隔离 OK")
    del m
    torch.cuda.empty_cache()

print("\n=== 2. task 指标 (after_evaluation) ===")
from ipiqa.tasks.aigiqa_singlescore import AIGIQASingleScoreTask
task = AIGIQASingleScoreTask.setup_task()
# 合成 val_result: (loss, pred, label)
np.random.seed(0)
mos = np.random.uniform(1, 5, 100)
pred = mos + np.random.normal(0, 0.3, 100)
val_result = [(0.1, float(p), float(t)) for p, t in zip(pred, mos)]
metrics = task.after_evaluation(val_result, epoch=0)
print("  metrics:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}))
assert abs(metrics["main_score_official"] - metrics["main_score_raw"]) < 0.05  # poly map 近似
assert "main_score_official" in metrics and "srcc" in metrics and "plcc_mapped" in metrics

print("\n=== 3. AIGIQA20K dataset + collator ===")
# 构造微型 dummy 数据集（3 张图）
dummy_dir = "/tmp/aigiqa20k_dummy"
os.makedirs(f"{dummy_dir}/images", exist_ok=True)
import torchvision.transforms as T
for nm in ["DALLE2_0000.png", "SDXL_0001.png", "Midjourney_0002.png"]:
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(f"{dummy_dir}/images/{nm}")
meta = pd.DataFrame([{"name": "DALLE2_0000.png", "prompt": "a corgi", "mos": 3.8},
                     {"name": "SDXL_0001.png", "prompt": "a boat", "mos": 4.1},
                     {"name": "Midjourney_0002.png", "prompt": "a castle", "mos": 3.5}])
ds = AIGIQA20K(meta, T.Compose([T.Resize((224,224)), T.ToTensor()]), f"{dummy_dir}/images")
dl = DataLoader(ds, batch_size=2, collate_fn=ds.collator)
batch = next(iter(dl))
print("  batch keys:", list(batch.keys()))
print("  images:", tuple(batch["images"].shape), "text:", batch["text"], "score:", tuple(batch["score"].shape))
assert batch["images"].shape[0] == 2 and batch["score"].shape == (2, 1)
print("\nAIGIQA-20K SMOKE OK")
