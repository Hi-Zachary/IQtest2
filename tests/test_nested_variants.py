"""Nested / identity regression tests (改进1-5.md Step 0).

验证 Frozen B0-B2 / Ours 的严格嵌套关系。由于每个变体是独立模型实例（base
path 随机初始化不同），正确的验证方式是**在同一实例内**比较：

    1. 各变体 forward 输出 == 手动只走 base path（B0 计算图）的输出
       （模块外部门控 lambda 均初始化为 0）
    2. 给 lambda 赋值后，输出确实偏离 base path（模块真的能影响结果）
    3. Ours（DMSQR/DP-HCMI 开启，但 lambda=0）仍 == base path；lambda 非 0 时偏离

即证明：lambda=0 时任何模块都严格退化为纯 base path（B0）。

运行（项目根目录）：
    python -m tests.test_nested_variants
或：
    python tests/test_nested_variants.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipiqa.models.model import MSQRNet

BASE_CKPT = "data/ckpt/clip/openai/resnet/RN50.pt"


def build(use_msqr, use_shcmi, use_dev=False, use_pw=False, use_ab=False,
          freeze_visual=True):
    return MSQRNet(
        base_ckpt=BASE_CKPT,
        input_resolution=512,
        output_dim=2,
        use_msqr=use_msqr,
        use_shcmi=use_shcmi,
        use_deviation=use_dev,
        use_prompt_weight=use_pw,
        use_align_bias=use_ab,
        freeze_visual=freeze_visual,
        freeze_text=True,
        gamma_init=0.0,
        internal_gate_init=0.01,
    ).cuda().float().eval()


def make_inputs(bs=4):
    torch.manual_seed(0)
    x = torch.randn(bs, 3, 512, 512).cuda()
    text = ["a statue of a man", "a tray of sushi", "a red car", "a white cat"]
    return x, text


@torch.no_grad()
def base_path_output(model, x, text):
    """手动计算纯 B0 base path（不经过任何模块，含 v7.2 的 anchor LayerNorm）。"""
    spatial = model.resnet50(x)
    global_v = model.attnpool(spatial)
    _, _, global_t = model.encode_text(text)
    v0 = model.norm_v0(model.base_visual_proj(global_v))
    t0 = model.norm_t0(model.base_text_proj(global_t))
    h = model.shared_fusion(torch.cat([v0, t0], dim=-1))
    q = model.quality_head(h)
    a = model.align_head(h)
    return torch.cat([q, a], dim=-1)


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


def check(name, out_a, out_b, tol=1e-4):
    diff = max_abs_diff(out_a, out_b)
    status = "OK" if diff < tol else "FAIL"
    print(f"  [{status}] {name}: max_abs_diff = {diff:.2e} (tol={tol})")
    assert diff < tol, f"{name} exceeded tolerance: {diff}"


def main():
    assert os.path.exists(BASE_CKPT), f"ckpt not found: {BASE_CKPT}"
    x, text = make_inputs()

    # ---- B0: 全关，forward == base path ----
    b0 = build(False, False)
    with torch.no_grad():
        out_b0 = b0(x, text)
        out_base = base_path_output(b0, x, text)
    check("B0 forward == base path", out_b0, out_base)

    # ---- B1: DMSQR on；v7.2 默认 gate_init=0.01，显式归零后 == base path ----
    b1 = build(True, False, use_dev=True)
    assert abs(b1.lambda_msqr.item() - 0.01) < 1e-6, "lambda_msqr init should be 0.01"
    with torch.no_grad():
        b1.lambda_msqr.fill_(0.0)
        out_b1 = b1(x, text)
        out_base1 = base_path_output(b1, x, text)
    check("B1 (lambda_msqr=0) == base path", out_b1, out_base1)

    # ---- B2: DP-HCMI on；默认 gate_init=0.01，显式归零后 == base path ----
    b2 = build(False, True, use_pw=True, use_ab=True)
    assert abs(b2.lambda_shcmi.item() - 0.01) < 1e-6, "lambda_shcmi init should be 0.01"
    with torch.no_grad():
        b2.lambda_shcmi.fill_(0.0)
        out_b2 = b2(x, text)
        out_base2 = base_path_output(b2, x, text)
    check("B2 (lambda_shcmi=0) == base path", out_b2, out_base2)

    # ---- Ours: DMSQR+DP-HCMI on，双 lambda 归零 -> base path ----
    ours = build(True, True, use_dev=True, use_pw=True, use_ab=True)
    with torch.no_grad():
        ours.lambda_msqr.fill_(0.0)
        ours.lambda_shcmi.fill_(0.0)
        out_ours = ours(x, text)
        out_base_ours = base_path_output(ours, x, text)
    check("Ours (all lambdas=0) == base path", out_ours, out_base_ours)

    # ---- 模块确实能影响输出（lambda 非 0 时偏离 base path）----
    with torch.no_grad():
        ours.lambda_msqr.fill_(0.5)
        ours.lambda_shcmi.fill_(0.5)
        out_ours_on = ours(x, text)
    diff = max_abs_diff(out_ours_on, out_base_ours)
    print(f"  [check] Ours with lambdas=0.5 deviates from base: {diff:.3e}")
    assert diff > 1e-3, "Ours lambdas should change output when turned on"

    print("\nAll nested-variant identity tests passed.")


if __name__ == "__main__":
    main()
