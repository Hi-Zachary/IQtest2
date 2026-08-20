"""v8 smoke test: B0/R0/B1/B2/Full forward+backward, param counts, LoRA trainability,
module input independence (B1 DG-MPQ input == Full DG-MPQ input).

Run from v8 root:
    /root/autodl-tmp/CondaEnv/ipiqa/bin/python tests/smoke_test_v8.py
"""

import os
import sys
import json

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.common.registry import registry
from ipiqa.models.model_v8 import MSQRNetV8

CKPT = "ckpt/clip-vit-base-patch16"


def build(use_lora, use_dg_mpq, use_hcmi):
    m = MSQRNetV8(
        model_name=CKPT,
        use_lora=use_lora,
        use_dg_mpq=use_dg_mpq,
        use_hcmi=use_hcmi,
        freeze_visual=True,
        freeze_text=True,
    )
    return m


def param_counts(m):
    s = m.trainable_summary()
    return s


def check_shape(variant, m):
    m.eval()
    x = torch.randn(2, 3, 224, 224)
    text = ["a statue of a man in the park", "a tray of sushi on a table"]
    with torch.no_grad():
        out = m(x, text)
    assert out.shape == (2, 2), f"{variant}: bad output shape {out.shape}"
    return out


def check_train(variant, m):
    m.train()
    x = torch.randn(2, 3, 224, 224)
    text = ["a statue of a man in the park", "a tray of sushi on a table"]
    target = torch.randn(2, 2)
    out = m(x, text)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0, f"{variant}: no trainable grads"
    return float(loss.item())


def check_lora_trainable(m, expect_trainable):
    trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
    # text encoder must be frozen
    text_train = [n for n in trainable_names if n.startswith("backbone.clip.text_model.")]
    assert len(text_train) == 0, f"text encoder has trainable params: {text_train}"
    lora = [n for n in trainable_names if n.startswith("backbone.clip.vision_model.")]
    if expect_trainable:
        assert len(lora) > 0, "expected LoRA trainable params, none found"
        nonlora = [n for n in lora if "lora" not in n]
        assert len(nonlora) == 0, f"visual base should be frozen, trainable: {nonlora}"
    else:
        assert len(lora) == 0, f"no-LoRA variant has trainable visual params: {lora}"
    return len(lora)


def main():
    variants = [
        ("B0", dict(use_lora=False, use_dg_mpq=False, use_hcmi=False)),
        ("R0", dict(use_lora=True, use_dg_mpq=False, use_hcmi=False)),
        ("B1", dict(use_lora=True, use_dg_mpq=True, use_hcmi=False)),
        ("B2", dict(use_lora=True, use_dg_mpq=False, use_hcmi=True)),
        ("Full", dict(use_lora=True, use_dg_mpq=True, use_hcmi=True)),
    ]

    print("=== registry model list ===")
    print(registry.list_models())

    results = {}
    for name, kw in variants:
        m = build(**kw)
        p = param_counts(m)
        out = check_shape(name, m)
        loss = check_train(name, m)
        n_lora = check_lora_trainable(m, kw["use_lora"])
        print(f"\n--- {name} ---")
        print(f"  summary: {json.dumps(p)}")
        print(f"  forward shape={tuple(out.shape)}  train loss={loss:.4f}  trainable LoRA layers={n_lora}")
        results[name] = p
        del m
        torch.cuda.empty_cache()

    # ---- module input independence: B1 DG-MPQ input == Full DG-MPQ input ----
    print("\n=== module input independence (B1 == Full) ===")
    b1 = build(use_lora=True, use_dg_mpq=True, use_hcmi=False)
    full = build(use_lora=True, use_dg_mpq=True, use_hcmi=True)
    b1.eval(); full.eval()
    x = torch.randn(2, 3, 224, 224)
    text = ["a statue of a man in the park", "a tray of sushi on a table"]
    with torch.no_grad():
        ids, mask = full.backbone.tokenize(text)
        ids = ids.to(x.device); mask = mask.to(x.device)
        feat_b1 = b1.backbone(x, ids, mask)
        feat_full = full.backbone(x, ids, mask)
        for li in [3, 6, 9, 12]:
            d = (feat_b1["vision_hidden"][li] - feat_full["vision_hidden"][li]).abs().max().item()
            assert d < 1e-6, f"hidden state {li} differs between B1 and Full: {d}"
        d6 = (feat_b1["vision_hidden"][6][:, 1:, :] - feat_full["vision_hidden"][6][:, 1:, :]).abs().max().item()
        d12 = (feat_b1["vision_hidden"][12][:, 1:, :] - feat_full["vision_hidden"][12][:, 1:, :]).abs().max().item()
        print(f"  max|H6 patch diff| = {d6:.2e}   max|H12 patch diff| = {d12:.2e}")
    print("  module input independence OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
