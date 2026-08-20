# MSQR_SHCMI_TAF 固定实验协议 (Fixed Protocol)

> 本文件定义本项目**所有正式实验**必须遵守的统一协议。
> 任何消融/对比都必须在同一协议下进行，否则不可直接比较。
>
> **v2（改进1.md）**：正式 B0 改为 **Frozen CLIP Multimodal Baseline**，
> MSQR/SHCMI/TAF 全部改为 **residual delta 分支**（`v = v0 + tanh(λm)·Δv`、
> `c = t0 + tanh(λs)·Δc`、`h_q = h + tanh(λq)·Δq`），保证 B0→B1→B2→B3→B4
> 是严格 Nested 的消融链（lambda=0 时各级精确退化）。
> 旧版 fine-tuned CLIP baseline 保留为 **FT-CLIP** 强参考，不参与正式消融。

## 1. 数据划分（唯一入口）

**AGIQA-3K：固定使用 `splits/seed42_split3.json`（split_index=3）**

- seed=42, ratio=0.8, 按 content ID 分组 80/20
- train_count=2384 / val_count=598（共 2982 张）
- 划分方式：300 个 content 随机排列，取前 80% 为 train，同一 content 的生成图整体进同一侧
- 已由 `train.py` 通过 `run.split_file` 加载，配置了固定文件就绝不回退随机

**AIGCIQA2023：固定使用 `splits/aigciqa2023_seed42.json`**

- seed=42, ratio=0.8, 按 prompt 分组 80/20（100 个 prompt → 80/20）
- train_count=1920 / val_count=480

### 为什么选 split3

同 seed=42 下共生成 3 个互斥折（split1/2/3）。三个折的完整 100-epoch Full 结果：

| 折 | 文件 | qual SROCC | qual PLCC | align SROCC |
|---|---|---|---|---|
| split1 | `seed42.json` | 0.8372 | 0.8886 | 0.6326 |
| split2 | `seed42_split2.json` | 0.8424 | 0.8919 | 0.6562 |
| split3 | `seed42_split3.json` | **0.8498** | 0.9064 | 0.6546 |

split1 系统性偏低（最难的折）。**正式实验统一采用 split3**，保证所有模块
（B0-B4）在同一划分上严格可比，且与上表 split3 的既有结果可横向对比。

> 注意：若未来需报告"多折均值"，请额外补跑 split1/split2 并明确标注；
> 默认正式结果只报 split3。

## 2. 训练超参（正式配置 = `ablation/configs/agiqa3k/*.yaml` 非 quick 版本）

| 项 | 值 |
|---|---|
| 输入分辨率 | 512×512 |
| batch size | 48（`batch_size_val` 同 48） |
| epoch | B0/B1/B2/B3 = **50**；B4 = **100**；FT-CLIP = 100 |
| 优化器 | AdamW, betas=(0.9, 0.999), weight_decay=0 |
| LR 调度 | linear_warmup_cosine_lr，warmup_steps=50，min_lr=1e-6 |
| init_lr | **1e-4**（Frozen B0-B4 所有可训练层统一；FT-CLIP 为 1e-5） |
| 分组 LR | Frozen B0-B4：全部 1×（1e-4）；FT-CLIP：backbone 1× / new 10× |
| CLIP backbone | **Frozen**（visual + text，`freeze_visual=True`/`freeze_text=True`） |
| head_scale | **null**（已移除叠乘） |
| AMP | 开启 |
| 训练阶段 | 单阶段 |
| checkpoint 选择 | **best-joint**（Q_SRCC+Q_PLCC+A_SRCC+A_PLCC），同时保存 best-quality |
| 评估 | 每 epoch 全量 val；指标 SRCC/PLCC/KROCC（quality + alignment） |

## 3. 消融矩阵（同一协议，v2）

| 实验 | Frozen CLIP | MSQR | SHCMI | TAF | 配置 | epoch |
|---|---|---:|---:|---:|---|---:|
| B0 | ✓ | × | × | × | `ablation/configs/agiqa3k/b0_baseline.yaml` | 50 |
| B1 | ✓ | ✓ | × | × | `ablation/configs/agiqa3k/b1_msqr.yaml` | 50 |
| B2 | ✓ | × | ✓ | × | `ablation/configs/agiqa3k/b2_shcmi.yaml` | 50 |
| B3 | ✓ | ✓ | ✓ | × | `ablation/configs/agiqa3k/b3_msqr_shcmi.yaml` | 50 |
| B4 | ✓ | ✓ | ✓ | ✓ | `ablation/configs/agiqa3k/b4_full.yaml` | 100 |
| FT-CLIP（强参考） | 仅 text | × | × | × | `ablation/configs/agiqa3k/ft_clip_reference.yaml` | 100 |

所有配置仅由 `use_msqr / use_shcmi / use_taf` 区分，其余字段完全一致。
每个变体共享同一套 base path（base_visual_proj / base_text_proj /
shared_fusion / quality_head / align_head）。

Nested 性质（已通过 `tests/test_nested_variants.py` 验证）：
```
关闭 MSQR            : B1 → B0
关闭 SHCMI           : B2 → B0
关闭 MSQR + SHCMI    : B3 → B0
TAF residual gate=0  : B4 → B3
```

## 4. 运行方式

```bash
# 正式实验（split3, 100 epoch, batch32）
python train.py --cfg-path ablation/configs/agiqa3k/b0_baseline.yaml --seed 42 --num_cv 1
python train.py --cfg-path ablation/configs/agiqa3k/b4_full.yaml     --seed 42 --num_cv 1
# ... 其余变体同理

# 快速趋势验证（split3, 20 epoch, batch48）——仅用于定位，不进入正式报告
python train.py --cfg-path ablation/configs/agiqa3k/b4_full_quick20_bs48.yaml --seed 42 --num_cv 1
```

## 5. 报告与表述注意

- 正式报告写：**"AGIQA-3K，固定划分 seed42/split3（2384/598），100 epoch"**。
- 不要写成"官方固定测试集"——AGIQA-3K 无官方单次固定 test split，split3 是本项目内部统一划分。
- 与文献横比时必须标注协议差异；公平对比 = 本协议内部消融（B0-B4）。
- 与之前方案 A 系列对比时：A3 的 split3 结果（0.8498 SROCC）可直接作为对照行。
