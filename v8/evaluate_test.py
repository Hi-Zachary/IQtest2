"""在 AIGIQA-20K 官方 test (4K) 上评估 R0/B1/Full 三个 best-main-score checkpoint。"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from omegaconf import OmegaConf

from ipiqa.common.registry import registry
from ipiqa.datasets.agiqa_datasets import AIGIQA20K
from ipiqa.processors import load_processor

CKPT_DIR = "run"
CONFIGS = {
    "R0":   "configs/aigiqa20k/r0_v17.yaml",
    "B1":   "configs/aigiqa20k/b1_dgmpq_v17.yaml",
    "Full": "configs/aigiqa20k/full_dgmpq_qard_v17.yaml",
}


def build_test_loader(cfg):
    vis_root = cfg.dataset.vis_root
    test_meta = cfg.dataset.test_meta
    import pandas as pd
    test_info = pd.read_csv(test_meta)
    proc = load_processor(**cfg.dataset.transform_val)
    ds = AIGIQA20K(test_info, proc, vis_root)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=12,
                    collate_fn=ds.collator)
    return dl


def find_ckpt(tag, sub="_V17_20k"):
    import glob
    cands = sorted(glob.glob(f"{CKPT_DIR}/*_{tag}{sub}/checkpoint_best_main.pth"))
    assert cands, f"no best_main ckpt for {tag}{sub}"
    return cands[-1]


def evaluate(tag, config_path, sub="_V17_20k"):
    cfg = OmegaConf.load(config_path)
    model = registry.get_model_class(cfg.model.arch).from_config(cfg.model)
    ckpt = torch.load(find_ckpt(tag, sub), map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.cuda().eval()

    dl = build_test_loader(cfg)
    preds, mos = [], []
    with torch.no_grad():
        for samples in dl:
            x = samples["images"].cuda()
            text = samples["text"]
            y = samples["score"].numpy().reshape(-1)
            with torch.cuda.amp.autocast():
                out = model(x, text)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            mos.append(y)
    pred = np.concatenate(preds)
    mos = np.concatenate(mos)

    srcc = spearmanr(pred, mos)[0]
    plcc_raw = pearsonr(pred, mos)[0]
    krcc = kendalltau(pred, mos)[0]
    coef = np.polyfit(pred, mos, 3)
    plcc_mapped = pearsonr(np.polyval(coef, pred), mos)[0]
    main_raw = (abs(srcc) + abs(plcc_raw)) / 2
    main_off = (abs(srcc) + abs(plcc_mapped)) / 2
    mse = float(np.mean((pred - mos) ** 2))
    return dict(SRCC=srcc, PLCC_raw=plcc_raw, PLCC_mapped=plcc_mapped,
                KRCC=krcc, MSE=mse, MainScore_raw=main_raw,
                MainScore_official=main_off)


def main():
    import sys as _sys
    which = _sys.argv[1] if len(_sys.argv) > 1 else "v17"
    if which == "v18":
        variants = [
            ("B1", "configs/aigiqa20k/v18/b1_dgmpq.yaml", "_V18_20k"),
            ("B2_CADR", "configs/aigiqa20k/v18/b2_cadr.yaml", "_V18_20k"),
            ("Full", "configs/aigiqa20k/v18/full_dgmpq_cadr.yaml", "_V18_20k"),
        ]
    else:
        variants = [
            ("R0", "configs/aigiqa20k/r0_v17.yaml", "_V17_20k"),
            ("B1", "configs/aigiqa20k/b1_dgmpq_v17.yaml", "_V17_20k"),
            ("Full", "configs/aigiqa20k/full_dgmpq_qard_v17.yaml", "_V17_20k"),
        ]

    print(f"{'Variant':<9} | {'SRCC':>6} {'PLCC_raw':>8} {'PLCC_map':>8} {'KRCC':>6} {'MSE':>7} {'MainScore_off':>12}")
    print("-" * 70)
    results = {}
    for tag, cfgp, sub in variants:
        r = evaluate(tag, cfgp, sub)
        results[tag] = r
        print(f"{tag:<9} | {r['SRCC']:.4f} {r['PLCC_raw']:.4f} {r['PLCC_mapped']:.4f} "
              f"{r['KRCC']:.4f} {r['MSE']:.4f} {r['MainScore_official']:.4f}")
    print("-" * 70)
    if "R0" in results:
        dg = results["B1"]["MainScore_official"] - results["R0"]["MainScore_official"]
        print(f"TEST ΔDG   (B1-R0)   MainScore = {dg:+.4f}   SRCC = {results['B1']['SRCC']-results['R0']['SRCC']:+.4f}")
    if "B2_CADR" in results:
        dcadr = results["Full"]["MainScore_official"] - results["B1"]["MainScore_official"]
        print(f"TEST ΔCADR (Full-B1)  MainScore = {dcadr:+.4f}   SRCC = {results['Full']['SRCC']-results['B1']['SRCC']:+.4f}")
    else:
        dq = results["Full"]["MainScore_official"] - results["B1"]["MainScore_official"]
        print(f"TEST ΔQARD (Full-B1)  MainScore = {dq:+.4f}   SRCC = {results['Full']['SRCC']-results['B1']['SRCC']:+.4f}")


if __name__ == "__main__":
    main()
