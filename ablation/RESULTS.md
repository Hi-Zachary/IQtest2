# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / batch48 / init_lr 1e-4 / best-joint
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v2 Frozen）

| Model | Frozen CLIP | MSQR | SHCMI | TAF | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC | best-epoch(joint) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | ✓ | × | × | × | | | | | | |
| B1 | ✓ | ✓ | × | × | | | | | | |
| B2 | ✓ | × | ✓ | × | | | | | | |
| B3 | ✓ | ✓ | ✓ | × | | | | | | |
| B4 | ✓ | ✓ | ✓ | ✓ | | | | | | |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ |
|---|---|---|---|
| B1 − B0（MSQR） | | | |
| B2 − B0（SHCMI） | | | |
| B3 − B0 | | | |
| B4 − B3（TAF） | | | |
| **B4 − B0（总体）** | | | |

## 二、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | | | | |
| FT-CLIP | Fine-tuned | None | | | | |
| B4 | Frozen | MSQR+SHCMI+TAF | | | | |

## 三、Gate / Residual 观察

| Model | lambda_msqr | lambda_shcmi | lambda_taf_q/a | msqr_ratio | shcmi_ratio | taf_ratio |
|---|---:|---:|---:|---:|---:|---:|
| B1 | | — | — | | — | — |
| B2 | — | | — | — | | — |
| B3 | | | — | | | — |
| B4 | | | | | | |

> 若 lambda 始终≈0 且 ratio≈0 → 模块没真正进入训练；若 lambda 变化但指标不升 →
> 模块学到的是噪声。判断依据：`log.txt` 中每 epoch 的 gates/ratios 行。

## 四、运行记录（run 目录 ↔ 结果）

| 任务 | run 目录 | tag | 状态 |
|---|---|---|---|
| B0 | | B0_frozen | ⬜ |
| B1 | | B1_msqr | ⬜ |
| B2 | | B2_shcmi | ⬜ |
| B3 | | B3_msqr_shcmi | ⬜ |
| B4 | | B4_full | ⬜ |
| FT-CLIP | | FT_CLIP | ⬜ |
