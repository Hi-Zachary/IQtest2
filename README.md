# MSQR + SHCMI + TAF (AGIQA-3K / AIGCIQA2023)

缝合项目：在 **Frozen CLIP 多模态基线** 之上构建两个主模块和一个轻量融合模块，
全部采用 **residual 公式**，构成严格 Nested 的消融链（B0→B1→B2→B3→B4）。

> 设计文档：`../AGIQA_MSQR_SHCMI_TAF_design (1).md`
> v2 方案：`../改进/改进1.md`
> Donor 源码：`../SourceCode/MS-SCANet`、`../SourceCode/CHPNet`
> 数据：`data/`（本目录内，已复制）
> **固定实验协议：`PROTOCOL.md`**（AGIQA-3K 统一采用 split3 划分）

## 1. 方法一句话

**双尺度质量增强（MSQR）→ 双尺度图文交互（SHCMI）→ 双任务自适应融合（TAF）**

```
Frozen CLIP RN50 spatial [B,2048,16,16]
   ├─ global_v ─ base_visual_proj ───────────── v0
   ├─ MSQR (fine/coarse + CA/SA/cross-scale) ── Δv
   ├─ Plain Multi-Scale Adapter (B2 无 MSQR 时) ─ fine/coarse tokens
   v = v0 + tanh(λm)·Δv

Frozen CLIP text tokens ─ global_t ─ base_text_proj ─ t0
   ├─ SHCMI (multi-kernel text + fine/coarse↔prompt) ─ Δc
   c = t0 + tanh(λs)·Δc

h = shared_fusion(concat[v, c])
   ├─ TAF residual: h_q = h + tanh(λq)·Δq ; h_a = h + tanh(λa)·Δa
   └─ q = quality_head(h_q) ; a = align_head(h_a)
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
| 主模型（消融开关+Nested） | — | `models/model.py::MSQRNet` |
| Frozen CLIP 基线 B0 | — | `models/baseline.py` |

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

### 单独运行某一变体

```bash
# AGIQA-3K（split3 固定协议）
python train.py --cfg-path projects/agiqa3k/b0_baseline.yaml      --seed 42 --num_cv 1  # 50ep Frozen
python train.py --cfg-path projects/agiqa3k/b1_msqr.yaml          --seed 42 --num_cv 1  # 50ep
python train.py --cfg-path projects/agiqa3k/b2_shcmi.yaml         --seed 42 --num_cv 1  # 50ep
python train.py --cfg-path projects/agiqa3k/b3_msqr_shcmi.yaml    --seed 42 --num_cv 1  # 50ep
python train.py --cfg-path projects/agiqa3k/b4_full.yaml          --seed 42 --num_cv 1  # 100ep
python train.py --cfg-path projects/agiqa3k/ft_clip_reference.yaml --seed 42 --num_cv 1 # 100ep 强参考

# AIGCIQA2023
python train.py --cfg-path projects/aigciqa2023/b4_full.yaml      --seed 42 --num_cv 1
```

### 冒烟测试

```bash
python tests/test_nested_variants.py   # Nested identity 验证（lambda=0 严格退化）
python smoke_test.py                   # B0-B4 前向 + 形状
```

## 6. 消融矩阵（设计文档第 19 节 / 改进1.md v2）

| 实验 | Frozen CLIP | MSQR | SHCMI | TAF | 说明 |
|---|---|---:|---:|---:|---:|---|
| B0 | ✓ | × | × | × | Frozen CLIP 图文基线（只训 proj/fusion/heads） |
| B1 | ✓ | ✓ | × | × | v = v0 + λm·ΔMSQR |
| B2 | ✓ | × | ✓ | × | c = t0 + λs·ΔSHCMI |
| B3 | ✓ | ✓ | ✓ | × | 双 residual |
| B4 | ✓ | ✓ | ✓ | ✓ | h 上 TAF residual |

所有消融共用一套代码，仅由 yaml 中 `use_msqr / use_shcmi / use_taf` 开关切换。
B0-B4 固定：同一 `split_file`、输入 512、warmup+cosine、AMP、batch 48、
`best_criterion: joint`（保存 best-joint 与 best-quality 两个 checkpoint）。

## 7. 训练协议要点

- 单阶段；CLIP visual/text 全冻结（`freeze_visual=True`/`freeze_text=True`）。
- 统一 LR = 1e-4（Frozen B0-B4 所有可训练层）；FT-CLIP 用 1e-5 + new 10×。
- 无 head_scale。
- 稳定性设计：所有新增模块均为 residual（`x' = x + tanh(λ)·Δx`），外层 λ 零初始化，
  训练初期严格等于 B0 base path；SHCMI 内部 gate 初始化 0.01（保证有梯度流）。
- 每 epoch 记录真实 gate 参数（lambda/gamma/alpha/beta/scale_gate/g_q/g_a）与
  residual ratio（||λΔ||/||base||），用于判断"模块没学到"还是"gate 没打开"。

## 8. 输出

每次运行在 `run/<job_id>_<tag>/` 下生成：

- `log.txt`：每 epoch 的 train/val 全量指标（qual/align 的 PLCC/SROCC/KROCC）与 gates/ratios
- `checkpoint_best_joint.pth`：按 joint 选出的 checkpoint（正式报告）
- `checkpoint_best.pth`：按 quality 选出的 checkpoint
- `checkpoint_latest.pth`

## 9. 注意事项

- 本实现是 **MS-SCANet-inspired / CHPNet-inspired 的适配**，不是逐行复现原网络。
- MSQR 中使用的空间注意力是全局 Spatial Self-Attention（CLIP 16×16/8×8 token 数小，
  无严格 window partition），论文/答辩描述时应避免宣称"标准 Window Attention"。
- MS-SCANet 原论文未报告 AGIQA 结果，报告时应写"将传统 NR-IQA 多尺度质量建模适配到 AGIQA"。
- 正式报告 B0-B4 时一律使用 Frozen CLIP；旧版 fine-tuned CLIP 只作为 FT-CLIP 强参考。
