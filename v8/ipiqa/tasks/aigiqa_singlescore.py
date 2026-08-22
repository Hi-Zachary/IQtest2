"""AIGIQA-20K single overall MOS task (改进5.md).

- 纯 MSE（第一轮不改 loss）
- Evaluation: SRCC / PLCC_raw / PLCC_mapped(3阶多项式回归) / KRCC / MSE
- MainScore_raw   = (|SRCC| + |PLCC_raw|) / 2
- MainScore_official = (|SRCC| + |PLCC_mapped|) / 2
- agg_metrics = MainScore_official（best_criterion=main_score 用）
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch
import torch.nn as nn
import logging

from ipiqa.tasks.base_task import BaseTask
from ipiqa.common.registry import registry
from ipiqa.common.dist_utils import is_dist_avail_and_initialized
from ipiqa.datasets.data_utils import prepare_sample
import torch.distributed as dist


@registry.register_task("aigiqa_singlescore")
class AIGIQASingleScoreTask(BaseTask):
    def __init__(self, train_fn, val_fn, **kwargs):
        super().__init__(train_fn=train_fn)
        self.val_fn = val_fn

    @classmethod
    def setup_task(cls, **kwargs):
        def iqa_loss(model, samples):
            x, y, text = samples['images'], samples['score'], samples['text']
            output = model(x, text)                      # [B,1]
            loss = nn.MSELoss()(output, y)
            return loss, {"loss": loss.detach().clone()}

        def iqa_loss_eval(model, samples):
            x, y, text = samples['images'], samples['score'], samples['text']
            output = model(x, text)
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(output, y)
            loss = loss.detach().cpu().numpy().reshape(-1).tolist()
            pred = output.detach().cpu().numpy().reshape(-1).tolist()
            label = y.detach().cpu().numpy().reshape(-1).tolist()
            return zip(loss, pred, label)

        return cls(train_fn=iqa_loss, val_fn=iqa_loss_eval)

    def evaluation(self, model, data_loader, cuda_enabled=True):
        results = []
        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)
        if is_dist_avail_and_initialized():
            dist.barrier()
        return results

    def after_evaluation(self, val_result, **kwargs):
        epoch = kwargs.get('epoch', None)
        pred = np.array([], dtype=np.float64)
        mos = np.array([], dtype=np.float64)
        losses = np.array([], dtype=np.float64)
        for info in val_result:
            losses = np.append(losses, info[0])
            pred = np.append(pred, info[1])
            mos = np.append(mos, info[2])

        srcc = spearmanr(pred, mos)[0]
        plcc_raw = pearsonr(pred, mos)[0]
        krcc = kendalltau(pred, mos)[0]

        # NTIRE official protocol: 3rd-order poly mapping before PLCC
        coef = np.polyfit(pred, mos, deg=3)
        pred_mapped = np.polyval(coef, pred)
        plcc_mapped = pearsonr(pred_mapped, mos)[0]

        main_score_raw = (abs(srcc) + abs(plcc_raw)) / 2
        main_score_official = (abs(srcc) + abs(plcc_mapped)) / 2
        mse = float(np.mean(losses))

        if epoch is not None:
            logging.info(
                "AIGIQA-20K: EPOCH[{}] -> SRCC {:.4f} PLCC_raw {:.4f} PLCC_mapped {:.4f} "
                "KRCC {:.4f} MainScore_official {:.4f}".format(
                    epoch, srcc, plcc_raw, plcc_mapped, krcc, main_score_official)
            )
        else:
            logging.info(
                "AIGIQA-20K: SRCC {:.4f} PLCC_raw {:.4f} PLCC_mapped {:.4f} "
                "KRCC {:.4f} MainScore_official {:.4f}".format(
                    srcc, plcc_raw, plcc_mapped, krcc, main_score_official)
            )

        metrics = {
            "agg_metrics": main_score_official,
            "main_score_raw": main_score_raw,
            "main_score_official": main_score_official,
            "srcc": srcc,
            "plcc_raw": plcc_raw,
            "plcc_mapped": plcc_mapped,
            "krcc": krcc,
            "mse": mse,
            "loss": mse,
        }
        return metrics

    def valid_step(self, model, samples):
        return self.val_fn(model, samples)
