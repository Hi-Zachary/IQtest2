"""Smoke test for MSQRNet (B4 full + each ablation variant).

Run from project root:
    python smoke_test.py
"""

import os
import sys
import torch

from ipiqa.models.model import MSQRNet
from ipiqa.models.baseline import MSQRBaseline

BASE_CKPT = "data/ckpt/clip/openai/resnet/RN50.pt"


def make_inputs(bs=2, size=512):
    x = torch.randn(bs, 3, size, size).cuda()
    text = ["a statue of a man", "a tray of sushi"]
    return x, text


def run_variant(name, **kwargs):
    print(f"=== {name} ===")
    model = MSQRNet(base_ckpt=BASE_CKPT, **kwargs).cuda().float()
    x, text = make_inputs()
    with torch.no_grad():
        out = model(x, text)
    assert out.shape == (2, 2), f"{name} output shape wrong: {out.shape}"
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  out={out.shape} trainable_params={nparams/1e6:.2f}M  OK")


if __name__ == "__main__":
    assert os.path.exists(BASE_CKPT), f"ckpt not found: {BASE_CKPT}"

    # B4 full
    run_variant("B4_full", use_msqr=True, use_shcmi=True, use_taf=True)
    # B3
    run_variant("B3_msqr_shcmi", use_msqr=True, use_shcmi=True, use_taf=False)
    # B2
    run_variant("B2_shcmi", use_msqr=False, use_shcmi=True, use_taf=False)
    # B1
    run_variant("B1_msqr", use_msqr=True, use_shcmi=False, use_taf=False)
    # B0 baseline
    b0 = MSQRBaseline(base_ckpt=BASE_CKPT).cuda().float()
    x, text = make_inputs()
    with torch.no_grad():
        out = b0(x, text)
    assert out.shape == (2, 2), f"B0 output shape wrong: {out.shape}"
    print(f"=== B0_baseline ===\n  out={out.shape}  OK")

    print("\nAll smoke tests passed.")
