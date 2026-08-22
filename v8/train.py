import os
import shutil
import json
from pathlib import Path

from tqdm import tqdm
import warnings

import argparse
from omegaconf import OmegaConf

import random
import numpy as np
import torch
import torch.distributed as dist

from ipiqa.common.dist_utils import (
    init_distributed_mode,
    main_process,
)
from trainer import Trainer
from ipiqa.processors import load_processor
from ipiqa.datasets.agiqa_datasets import AGIQA3k, AIGIQA20K
from ipiqa.common.registry import registry
from ipiqa.common.logger import setup_logger
from ipiqa.tasks import setup_task

from ipiqa.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
    ConstantLRScheduler,
)

import pandas as pd

warnings.filterwarnings('ignore')

def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_config(args):
    cfg_path = Path(args.cfg_path)
    assert cfg_path.suffix == '.yaml', 'config file must be .yaml file'
    config = OmegaConf.load(cfg_path)
    init_distributed_mode(config.run)
    return config

def get_transforms(config) -> dict:
    dataset_cfg = config.dataset

    transforms = {}
    transforms['train'] = load_processor(**dataset_cfg.transform_train)
    transforms['val'] = load_processor(**dataset_cfg.transform_val)

    return transforms

def get_datasets(config, transforms) -> dict:
    """AGIQA-3K / AIGCIQA2023 split, mirroring the validated protocol.

    - If `run.split_file` is set: load the fixed assignment (image-name ->
      train/val) from the json. Never falls back to random split.
    - Otherwise: 80/20 by content id (AGIQA-3K, 300 content groups) or by
      prompt (AIGCIQA2023), deterministically derived from `--seed`.
    """

    dataset_cfg = config.dataset
    split_file = config.run.get("split_file", None)

    # ===================== AIGIQA-20K：官方 train/val metadata =====================
    if dataset_cfg.get("name", None) == "aigiqa20k":
        train_info = pd.read_csv(dataset_cfg.train_meta)
        val_info = pd.read_csv(dataset_cfg.val_meta)
        return {
            "train": AIGIQA20K(train_info, transforms["train"], dataset_cfg.vis_root),
            "val": AIGIQA20K(val_info, transforms["val"], dataset_cfg.vis_root),
        }

    assignment = None
    if split_file:
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"split_file configured but not found: {split_file}"
            )
        with open(split_file) as f:
            split = json.load(f)
        assignment = split.get("assignment", None)

    data_info = dataset_cfg.data_path
    vis_root = dataset_cfg.vis_root
    data_info = pd.read_excel(data_info)

    if assignment is None:
        # Deterministic fallback: content-id / prompt grouped 80/20
        import re
        def content_key(name):
            m = re.findall(r'\d+', name.split(".")[0])
            return m[-1] if m else name
        groups = sorted({content_key(n) for n in data_info.iloc[:, 0]})
        rng = random.Random(int(config.run.get("seed", 42)))
        rng.shuffle(groups)
        n_train = int(0.8 * len(groups))
        train_groups = set(groups[:n_train])
        assignment = {
            n: ("train" if content_key(n) in train_groups else "val")
            for n in data_info.iloc[:, 0]
        }

    train_idx = [i for i, n in enumerate(data_info.iloc[:, 0])
                 if assignment.get(n, "train") == "train"]
    val_idx = [i for i, n in enumerate(data_info.iloc[:, 0])
               if assignment.get(n, "train") == "val"]

    datasets = {}
    datasets["train"] = AGIQA3k(data_info.iloc[train_idx], transforms['train'], vis_root)
    datasets['val'] = AGIQA3k(data_info.iloc[val_idx], transforms['val'], vis_root)

    return datasets

def get_model(config):
    model_cfg = config.model
    print(registry.list_models())
    model_cls = registry.get_model_class(model_cfg.arch)
    return model_cls.from_config(model_cfg)

def main(config):
    transforms = get_transforms(config)
    datasets = get_datasets(config, transforms)
    model = get_model(config)
    task = setup_task(config)
    job_id = now()

    trainer = Trainer(config, model, datasets, task, job_id)
    return trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_cv', type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)
    config = get_config(args)

    setup_logger()

    metric_lst = []
    results = {}
    for i in range(args.num_cv):
        metric_lst.append(main(config))

    print(metric_lst)

    # 键集合从实际返回的 metrics 推导（兼容 doublescore / singlescore）
    key_lst = list(metric_lst[0].keys())
    value_lst = [0] * len(key_lst)
    l = len(key_lst)

    for i in range(l):
        cur_key = key_lst[i]
        value_lst[i] = sum([metric[cur_key] for metric in metric_lst])
        results[cur_key] = value_lst[i] / args.num_cv

    print(results)
