# MSQR + SHCMI + TAF (AGIQA-3K / AIGCIQA2023)

缝合项目：在一个**真正中性的 CLIP 图文基线**之上构建两个主模块和一个轻量融合模块。

> 设计文档：`../AGIQA_MSQR_SHCMI_TAF_design (1).md`
> Donor 源码：`../SourceCode/MS-SCANet`、`../SourceCode/CHPNet`
> 数据：`../data`（本目录 `data` 为软链接）
> **固定实验协议：`PROTOCOL.md`**（AGIQA-3K 统一采用 split3 划分）

## 1. 方法一句话

**双尺度质量增强（MSQR）→ 双尺度图文交互（SHCMI）→ 双任务自适应融合（TAF）**

数据流：

```
CLIP RN50 spatial [B,2048,16,16]
   └─ MSQR (fine/coarse + channel/spatial + cross-scale) ─ F_fine/F_coarse
        ├─ MSQRVisualSkip ──────────────── F_visual [B,256]
        └─ SHCMI (multi-kernel text + fine/coarse↔prompt) ─ F_cross [B,256]
   F_visual + F_cross
        ├─ TAF ── F_q → Quality Head ; F_a → Alignment Head   (B4)
        └─ 否则 concat → shared MLP → [quality, alignment]     (B0-B3)
```

## 2. 模块 ↔ donor 源码映射

| 本实现 | 参考源码 | 文件 |
|---|---|---|
| Channel / Spatial Attention | MS-SCANet `ms_scanet.py` | `ipiqa/models/modules/attention.py` |
| Cross-Scale Attention | MS-SCANet `CrossBranchAttention` | `attention.py::CrossScaleAttention` |
| 文本多核增强 MSA_T | CHPNet `models/SV_MS.py::MSA_T` | `attention.py::MSA_T` |
| 双向跨模态交互 | CHPNet `models/SV_interaction.py` | `shcmi.py::SHCMI` |
| 任务感知融合思想 | CHPNet `models/Dynamic_regression.py` | `taf.py::TAF` |
| MSQR 主模块 | — | `modules/msqr.py` |
| 主模型（消融开关） | — | `models/model.py::MSQRNet` |
| 中性 CLIP 基线 B0 | — | `models/baseline.py` |

## 3. 目录结构

```text
MSQR_SHCMI_TAF/
├── train.py                # 统一训练入口（AGIQA-3K 与 AIGCIQA2023 通用）
├── trainer.py              # 训练循环（复用已验证的 LAVIS 风格 trainer）
├── smoke_test.py           # 冒烟测试：B0-B4 前向 + 训练路径
├── prepare_data.py         # 生成 mos_joint.xlsx
├── serial_train.sh         # 一键串行跑 AGIQA-3K B0-B4
├── requirements.txt / setup.py
├── clip/                   # 本地 CLIP (RN50) 实现
├── ipiqa/
│   ├── common/             # registry / dist_utils / logger / optims
│   ├── datasets/           # AGIQA3k dataset + collator
│   ├── processors/         # CLIP 图像预处理
│   ├── tasks/              # agiqa_doublescore（SRCC/PLCC/KROCC）
│   └── models/
│       ├── model.py        # MSQRNet（use_msqr/use_shcmi/use_taf）
│       ├── baseline.py     # MSQRBaseline (B0)
│       └── modules/        # attention / msqr / shcmi / taf
├── projects/
│   ├── agiqa3k/            # b0_baseline ~ b4_full 共 5 个 yaml
│   └── aigciqa2023/        # b0_baseline ~ b4_full 共 5 个 yaml
├── splits/                 # 固定划分 seed42 (AGIQA-3K + AIGCIQA2023)
├── data -> ../data         # 数据软链接（images/ckpt/annotations）
└── run/                    # 训练输出（log.txt / checkpoints）
```

## 4. 数据准备

数据已整理在 `../data`：

```text
data/
├── aigc_qa_3k/        AGIQA-3K 图像 + data.csv + mos_joint.xlsx
├── aigc_qa_2023/      AIGCIQA2023 图像(allimg) + mos_joint_aigciqa2023.xlsx
└── ckpt/clip/openai/resnet/RN50.pt   OpenAI CLIP RN50 权重
```

若需重新生成标注文件：

```bash
python prepare_data.py --csv data/aigc_qa_3k/data.csv --out data/aigc_qa_3k/mos_joint.xlsx
```

## 5. 训练

### 环境

使用已验证的 conda 环境：`/root/autodl-tmp/CondaEnv/ipiqa`（torch 2.3.1+cu121）。

### 一键串行（AGIQA-3K B0-B4）

```bash
bash serial_train.sh
```

### 单独运行某一变体

```bash
# AGIQA-3K
python train.py --cfg-path projects/agiqa3k/b0_baseline.yaml     --seed 42 --num_cv 1
python train.py --cfg-path projects/agiqa3k/b1_msqr.yaml         --seed 42 --num_cv 1
python train.py --cfg-path projects/agiqa3k/b2_shcmi.yaml        --seed 42 --num_cv 1
python train.py --cfg-path projects/agiqa3k/b3_msqr_shcmi.yaml   --seed 42 --num_cv 1
python train.py --cfg-path projects/agiqa3k/b4_full.yaml         --seed 42 --num_cv 1

# AIGCIQA2023
python train.py --cfg-path projects/aigciqa2023/b4_full.yaml     --seed 42 --num_cv 1
```

### 冒烟测试

```bash
python smoke_test.py        # B0-B4 前向 + 形状
# 训练路径小测（1 epoch, 224 输入）：
python train.py --cfg-path <临时 quick yaml> --seed 42 --num_cv 1
```

## 6. 消融矩阵（设计文档第 19 节）

| 实验 | MSQR | SHCMI | TAF | 说明 |
|---|---|---|---|---|
| B0 | × | × | × | Vanilla CLIP 图文基线 |
| B1 | ✓ | × | × | 验证视觉质量增强 |
| B2 | × | ✓ | × | 验证跨模态交互 |
| B3 | ✓ | ✓ | × | 验证两主模块耦合 |
| B4 | ✓ | ✓ | ✓ | Full |

所有消融共用一套代码，仅由 yaml 中 `use_msqr / use_shcmi / use_taf` 开关切换。
B0-B4 固定：同一 `split_file`、输入 512、100 epoch、warmup+cosine、AMP、batch 32、
`best_criterion: quality`（按 SRCC_qual + PLCC_qual 选 checkpoint）。

## 7. 训练协议要点

- 单阶段：从 CLIP RN50 初始化直接训练，不做两阶段 warm-start。
- 分组 LR：CLIP backbone 1×（1e-5），新增模块 10×，heads/gates 10×。
- 文本编码器默认冻结（`freeze_text: True`），prompt 仍作为任务输入参与交互。
- 稳定性设计：所有新增增强均为 `x' = x + γ·Δx`，`γ/α/β` 默认 0 初始化，
  训练初期接近基础 CLIP 表征；TAF gate 零初始化使初始 sigmoid≈0.5。

## 8. 输出

每次运行在 `run/<job_id>_<tag>/` 下生成：

- `log.txt`：每 epoch 的 train/val 全量指标（qual/align 的 PLCC/SROCC/KROCC）与 gates
- `checkpoint_best.pth`：按 best-quality 选出的 checkpoint
- `checkpoint_latest.pth`

## 9. 注意事项

- 本实现是 **MS-SCANet-inspired / CHPNet-inspired 的适配**，不是逐行复现原网络。
- MSQR 中使用的空间注意力是全局 Spatial Self-Attention（CLIP 16×16/8×8 token 数小，
  无严格 window partition），论文/答辩描述时应避免宣称“标准 Window Attention”。
- MS-SCANet 原论文未报告 AGIQA 结果，报告时应写“将传统 NR-IQA 多尺度质量建模适配到 AGIQA”。
