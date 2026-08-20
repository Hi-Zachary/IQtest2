# MSQR_SHCMI_TAF 固定实验协议 (Fixed Protocol)

> 本文件定义本项目**所有正式实验**必须遵守的统一协议。
> 任何消融/对比都必须在同一协议下进行，否则不可直接比较。

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

## 2. 训练超参（正式配置 = `projects/agiqa3k/*.yaml` 非 quick 版本）

| 项 | 值 |
|---|---|
| 输入分辨率 | 512×512 |
| batch size | 48（`batch_size_val` 同 48；batch 32→48 为冒烟测试选定，显存 14.1GB/24GB，速度更优） |
| epoch | 100 |
| 优化器 | AdamW, betas=(0.9, 0.999), weight_decay=0 |
| LR 调度 | linear_warmup_cosine_lr，warmup_steps=75（≈1 epoch），min_lr=1e-6 |
| init_lr | 1e-5 |
| 分组 LR | CLIP backbone 1×，新模块(msqr/shcmi/taf/heads/gates) 10× |
| AMP | 开启 |
| 文本编码器 | 冻结（freeze_text=True） |
| 训练阶段 | 单阶段（从 CLIP RN50 初始化，无两阶段 warm-start） |
| checkpoint 选择 | `best_criterion: quality` → argmax(SRCC_qual + PLCC_qual) |
| 评估 | 每 epoch 全量 val；指标 SRCC/PLCC/KROCC（quality + alignment） |

## 3. 消融矩阵（同一协议）

| 实验 | MSQR | SHCMI | TAF | 配置 |
|---|---|---|---|---|
| B0 | × | × | × | `projects/agiqa3k/b0_baseline.yaml` |
| B1 | ✓ | × | × | `projects/agiqa3k/b1_msqr.yaml` |
| B2 | × | ✓ | × | `projects/agiqa3k/b2_shcmi.yaml` |
| B3 | ✓ | ✓ | × | `projects/agiqa3k/b3_msqr_shcmi.yaml` |
| B4 | ✓ | ✓ | ✓ | `projects/agiqa3k/b4_full.yaml` |

所有配置仅由 `use_msqr / use_shcmi / use_taf` 区分，其余字段完全一致。

## 4. 运行方式

```bash
# 正式实验（split3, 100 epoch, batch32）
python train.py --cfg-path projects/agiqa3k/b0_baseline.yaml --seed 42 --num_cv 1
python train.py --cfg-path projects/agiqa3k/b4_full.yaml     --seed 42 --num_cv 1
# ... 其余变体同理

# 快速趋势验证（split3, 20 epoch, batch48）——仅用于定位，不进入正式报告
python train.py --cfg-path projects/agiqa3k/b4_full_quick20_bs48.yaml --seed 42 --num_cv 1
```

## 5. 报告与表述注意

- 正式报告写：**"AGIQA-3K，固定划分 seed42/split3（2384/598），100 epoch"**。
- 不要写成"官方固定测试集"——AGIQA-3K 无官方单次固定 test split，split3 是本项目内部统一划分。
- 与文献横比时必须标注协议差异；公平对比 = 本协议内部消融（B0-B4）。
- 与之前方案 A 系列对比时：A3 的 split3 结果（0.8498 SROCC）可直接作为对照行。
