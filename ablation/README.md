# Ablation（消融实验）

本目录集中管理 MSQR + SHCMI + TAF 的全部消融实验配置、运行脚本与结果记录。

> 协议见根目录 `PROTOCOL.md`（AGIQA-3K 统一 split3 / Frozen CLIP / 512 / batch48 / init_lr 1e-4）。

## 1. 消融矩阵（v2，改进1.md）

所有变体共享同一套代码与 base path，仅由 `use_msqr / use_shcmi / use_taf`
开关区分；CLIP visual/text 全冻结。

| 实验 | Frozen CLIP | MSQR | SHCMI | TAF | 配置（`configs/agiqa3k/`） | epoch |
|---|---|---:|---:|---:|---:|---:|
| B0 | ✓ | × | × | × | `b0_baseline.yaml` | 50 |
| B1 | ✓ | ✓ | × | × | `b1_msqr.yaml` | 50 |
| B2 | ✓ | × | ✓ | × | `b2_shcmi.yaml` | 50 |
| B3 | ✓ | ✓ | ✓ | × | `b3_msqr_shcmi.yaml` | 50 |
| B4 | ✓ | ✓ | ✓ | ✓ | `b4_full.yaml` | 100 |
| FT-CLIP（强参考） | 仅 text | × | × | × | `ft_clip_reference.yaml` | 100 |

Nested 关系（`tests/test_nested_variants.py` 验证）：
```
关闭 MSQR            : B1 → B0
关闭 SHCMI           : B2 → B0
关闭 MSQR + SHCMI    : B3 → B0
TAF residual gate=0  : B4 → B3
```

## 2. 目录说明

```text
ablation/
├── README.md              本文件（消融说明）
├── run_ablation.sh        一键串行跑 AGIQA-3K B0-B4（+可选 FT-CLIP）
├── RESULTS.md             消融结果记录表（跑完填）
└── configs/
    ├── agiqa3k/           AGIQA-3K 主消融（B0-B4 + FT-CLIP）
    └── aigciqa2023/       第二数据集验证（B0-B4）
```

## 3. 运行方式

从**项目根目录**执行（配置内相对路径依赖运行 cwd）：

```bash
# 一键串行（B0→B1→B2→B3→B4）
bash ablation/run_ablation.sh

# 或单独跑某一变体
python train.py --cfg-path ablation/configs/agiqa3k/b0_baseline.yaml --seed 42 --num_cv 1
python train.py --cfg-path ablation/configs/agiqa3k/b1_msqr.yaml     --seed 42 --num_cv 1
python train.py --cfg-path ablation/configs/agiqa3k/b4_full.yaml     --seed 42 --num_cv 1
python train.py --cfg-path ablation/configs/agiqa3k/ft_clip_reference.yaml --seed 42 --num_cv 1
```

## 4. 结构验证（先做，再跑长实验）

```bash
# Nested identity 测试：lambda=0 严格退化
python tests/test_nested_variants.py

# 2 epoch smoke：确认训练/显存/NaN/gate
# 改一个配置的 max_epoch=2 临时跑，或直接跑 B4 的 20-epoch 趋势
```

## 5. 判断标准（改进1.md 第 31 节）

- 理想：`B1 > B0`、`B2 > B0`、`B3 >= B1/B2`、`B4 >= B3`
- `B2 < B0` → 检查 SHCMI
- `B3 < B1/B2` → MSQR→SHCMI 耦合问题
- `B4 < B3` → TAF 问题
- 所有 `lambda ≈ 0` → 模块没真正进入训练（看 residual ratio 日志区分）

## 6. 结果记录

每轮结果填入 `RESULTS.md`，统一记录：Q-SRCC / Q-PLCC / Q-KROCC / A-SRCC / A-PLCC
（按 best-joint checkpoint）。
