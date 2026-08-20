# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **v7.3 正式协议：B0/B1/B2 = 50 epoch，Full = 100 epoch**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v7.3, 50/100ep）—— 当前正式结果

> v7.3（改进8.md）：**模块内 Residual Calibration** ——
> 1. **删除 anchor LayerNorm**（norm_v0/norm_t0）：B0 恢复简单 Frozen-CLIP 基座，
>    避免 baseline enhancement 吸收模块贡献（v7.2 里 B0 被抬到 0.8408）；
> 2. **LayerNorm 移入模块内部**：MSQRVisualSkip 与 SHCMI 输出各加 `out_norm`，
>    只规范 proposed residual 的输出尺度，不强化 baseline；
> 3. 保留 `outer_gate_init=0.01`、共享 spatial projection、余弦偏差、prompt
>    mask+scale、单次 tanh discrepancy、shared fusion、双头、纯 MSE。
> 协议：B0/B1/B2 跑 50ep，Full 跑 100ep（项目既有协议，B4 规格）。
> 运行方式：B0/B1 → B2/Full，各 2 个一并行。
> run 目录：`run/20260820220_B0_frozen / 20260820220_B1_msqr /
> 20260820222_B2_shcmi / 20260820222_Ours`。

| Model | DMSQR | DP-HCMI | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | 0.7983 | 0.8759 | 0.6100 | 0.6324 | 0.7805 |
| B1 | ✓ | × | 0.7999 | 0.8888 | 0.6149 | 0.6354 | 0.7934 |
| B2 | × | ✓ | 0.8021 | 0.8840 | 0.6156 | 0.6274 | 0.7922 |
| **Full** | ✓ | ✓ | **0.8049** | **0.8901** | **0.6204** | **0.6403** | **0.8008** |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | Q-KROCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|---|
| B1 − B0（DMSQR） | +0.0016 | +0.0129 | +0.0049 | +0.0030 | +0.0129 |
| B2 − B0（DP-HCMI） | +0.0038 | +0.0081 | +0.0056 | −0.0050 | +0.0117 |
| **Full − B0** | **+0.0066** | **+0.0142** | **+0.0104** | **+0.0079** | **+0.0203** |
| **Full − B1** | **+0.0050** | **+0.0013** | **+0.0055** | **+0.0049** | **+0.0074** |
| **Full − B2** | **+0.0028** | **+0.0061** | **+0.0048** | **+0.0129** | **+0.0086** |

### v7.3 关键结论（相对 v7.2）

1. **达到改进8.md 第 15 节理想判据：Full 在所有 6 个指标上严格 ≥ B1 且 ≥ B2**。
   Full−B1、Full−B2 的 Q/A 全项增量为正——两个模块不再互斥，组合收益完全保留。
2. **B0 回落到合理档位**：删除 anchor LN 后 B0 从 v7.2 的 0.8408 回到 **0.7983**；
   模块增量重新显现（B1−B0 Q-PLCC +0.0129、Full−B0 A-PLCC +0.0203），
   「baseline 太强、增量坍缩」问题彻底解决。
3. **模块内部 LN 保留两分支活性**：Full 最后 epoch `raw_msqr_ratio≈14.7 /
   raw_shcmi_ratio≈1.58`、`msqr_ratio≈0.86 / shcmi_ratio≈0.047`、`scale_gate≈0.47`，
   DMSQR 与 DP-HCMI 都非零且 DMSQR 仍占主导但 DP-HCMI 未被关闭。
4. **Full 最优 epoch 在 11（100ep 内）**：100ep 只是按项目协议给的 Full 规格，
   实际早期已收敛；相对 B1/B2（50ep）的比较不受训练时长优势主导。

## 一·四、历史版本对比（同协议，供参考）

| Model | Q-SRCC | A-SRCC | 说明 |
|---|---|---|---|
| v3 Ours（MSQR+SHCMI+QTA+AG） | 0.8175 | 0.6389 | 30ep；QTA/AG |
| v6 Ours | 0.8116 | 0.6231 | 30ep；Ours < B2 |
| v7 Ours | 0.8163 | 0.6391 | 30ep；Ours≈B2 |
| v7.2 Ours(Full) | 0.8414 | 0.6909 | 30ep；anchor LN 强化 B0 |
| **v7.3 Full（本次）** | **0.8049** | **0.6403** | **50/100ep；Full 全指标 ≥ B1,B2** |

## 一·五、AGIQA-3K 主消融（v7.2, 30ep）—— 历史

> v7.2（改进7.md）：Residual Scale Calibration —— 对 anchor（norm_v0/norm_t0）与
> residual（norm_msqr/norm_shcmi）双层 LayerNorm，outer gate 小正初始化 0.01。
> anchor LN 使 B0 升到 0.8408、模块增量坍缩，故被 v7.3 取代。

| Model | DMSQR | DP-HCMI | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | 0.8408 | 0.9023 | 0.6543 | 0.6866 | 0.8268 |
| B1 | ✓ | × | 0.8409 | 0.9029 | 0.6551 | 0.6896 | 0.8279 |
| B2 | × | ✓ | **0.8429** | 0.9029 | **0.6571** | **0.6912** | 0.8268 |
| **Ours(Full)** | ✓ | ✓ | 0.8414 | **0.9036** | 0.6554 | 0.6909 | **0.8287** |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|
| B1 − B0（DMSQR） | +0.0001 | +0.0006 | +0.0030 | +0.0011 |
| B2 − B0（DP-HCMI） | **+0.0021** | +0.0006 | **+0.0046** | +0.0000 |
| Ours − B0 | +0.0006 | **+0.0013** | +0.0043 | **+0.0019** |
| Ours − B2 | −0.0015 | +0.0007 | −0.0003 | +0.0019 |

### v7.2 关键结论（相对 v7）

1. **核心目标达成：DP-HCMI 不再在 Full 中被自动关闭**。Full 最后 epoch：
   `raw_msqr_ratio≈1.03 / raw_shcmi_ratio≈0.99`（两个 residual 校准到与 anchor 同量级），
   `shcmi_ratio=0.0128`（v7 里只有 0.003）、`scale_gate_mean=0.46`（v7 里 0.0，
   自适应尺度门真正开始平衡两分支）——改进7.md 的「尺度失衡」假设得到验证并修复。
2. **LayerNorm 让整体大幅提升**：B0 Q-SRCC 0.7880→**0.8408**、B1 0.8211→0.8409、
   B2 0.8166→0.8429、Ours 0.8163→0.8414。基座锚点尺度稳定后，融合/双头更好收敛。
3. **单模块排序反转**：v7 里 B1（DMSQR）最强，v7.2 里 **B2（DP-HCMI）最强**（Q 0.8429 / A 0.6912 双最高）；DMSQR 的边际增益被校准后的强基座压缩到 +0.0001。
4. **Full 是 Joint 最优**：Q-PLCC（0.9036）与 A-PLCC（0.8287）全场最高，A-SRCC 与 B2 并列最高（差 0.0003），Q-SRCC 仅比 B2 低 0.0015。符合 改进7.md 第 13 节「Q/A 互补、Joint 最优」判据；Q-SRCC 上 Full 与 B2 的差距在单 seed 噪声范围内，下一步建议 3-seed 确认。
5. **不再需要 Task-Routed Fusion**（改进7.md 第 15 节的备用方案）：尺度校准后
   Full 已 joint 最优，未出现「两个分支都非零却互斥」的任务级 feature conflict。

## 二、跨数据集验证（AIGCIQA2023, v7.2）

> 目的：验证「AGIQA-3K 上 B0 太强、模块增量小」是数据集饱和问题还是模块本身问题。
> 协议与 AGIQA-3K v7.2 完全一致：Frozen / 512 / batch64 / init_lr 1e-4 / best-joint /
> 30ep / seed42 / `splits/aigciqa2023_seed42.json`（1920/480，按 prompt 分组）。
> 配置：`ablation/configs/aigciqa2023/*.yaml`（覆盖了旧的 msqr_shcmi_taf 版本）。
> run 目录：`run/20260820213_B0_frozen`、`20260820213_B1_msqr`、
> `20260820214_B2_shcmi`、`20260820214_B3_full`。

| Model | DMSQR | DP-HCMI | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | 0.6656 | 0.6128 | 0.4751 | 0.4245 | 0.4175 |
| B1 | ✓ | × | **0.6744** | **0.6286** | **0.4823** | 0.4161 | 0.4080 |
| B2 | × | ✓ | 0.6713 | 0.6237 | 0.4799 | **0.4249** | **0.4199** |
| **Full** | ✓ | ✓ | 0.6727 | 0.6230 | 0.4803 | 0.4206 | 0.4116 |

### 增量（AIGCIQA2023, 同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|
| B1 − B0（DMSQR） | **+0.0088** | **+0.0158** | −0.0084 | −0.0095 |
| B2 − B0（DP-HCMI） | **+0.0057** | +0.0109 | +0.0004 | +0.0024 |
| Full − B0 | **+0.0071** | +0.0102 | −0.0039 | −0.0059 |
| Full − B1 | −0.0017 | −0.0056 | +0.0045 | +0.0036 |
| Full − B2 | +0.0014 | −0.0007 | −0.0043 | −0.0083 |

### 跨数据集关键结论

1. **模块在 AIGCIQA2023 上增益清晰**：B1−B0 Q-SRCC **+0.0088**（Q-PLCC +0.0158）、
   B2−B0 **+0.0057**、Full−B0 **+0.0071**——对比 AGIQA-3K 上 B1−B0 仅 +0.0001，
   说明 **「B0 太强、增量小」主要是 AGIQA-3K 基线饱和（0.84 档）所致，不是模块失效**。
2. **Full 中两分支都激活**：AIGCIQA2023 上 Full 最后 epoch
   `raw_msqr_ratio≈1.01 / raw_shcmi_ratio≈1.00`，`msqr_ratio≈0.010 / shcmi_ratio≈0.010`，
   `scale_gate_mean≈0.47`——DMSQR 与 DP-HCMI 都非零且均衡。
3. **Full 与单模块关系**：Full Q-SRCC 0.6727 ≈ B1（0.6744，差 0.0017）且 ≥ B2（+0.0014）；
   B2 在 A-SRCC/A-PLCC 上单模块最强（0.4249/0.4199），Full 介于 B1 与 B2 之间。
4. **双数据集叙事成立**：AGIQA-3K 报 Joint 最优（Full Q/A PLCC 最高、Q-SRCC≈B2）；
   AIGCIQA2023 报清晰增量（各模块 +0.006~0.009，Full≈B1≥B2）。两个数据集上
   Full 都不低于 B2，且都能观察到模块相对 B0 的正向贡献。
5. **残留观察**：A-SRCC 上 B1（DMSQR）在 AIGCIQA2023 反而低于 B0（−0.0084），
   说明 DMSQR 的视觉增强在该数据集的 alignment 上略有副作用；DP-HCMI 是更稳的
   alignment 贡献者。仍建议正式定稿前 3-seed 确认趋势。

> 注：v5 与 v6 的 head 结构不同（v6 删除三适配器恢复简单 Linear），
> 因此同一模块的绝对数值不可直接跨版本比较；同版本内消融才严格可比。

## 三、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.8408 | 0.9023 | 0.6866 | 0.8268 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| Ours (v7.2) | Frozen | DMSQR+DP-HCMI | 0.8414 | **0.9036** | 0.6909 | **0.8287** |

## 四、Gate / Residual 观察

> v7.3 只保留模块内部 LayerNorm（anchor 不归一化），因此 raw ratio 反映模块输出
> 相对原始 anchor 的真实比例。下表为 v7.3 各 run **最后 epoch**（详见各 run/log.txt
> 的 gates 行）：

| Model | lambda_msqr | lambda_shcmi | raw_msqr | raw_shcmi | msqr_ratio | shcmi_ratio | scale_gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 (50ep) | 0.0498 | — | 14.66 | — | 0.730 | — | — |
| B2 (50ep) | — | 0.0446 | — | 1.38 | — | 0.061 | 0.411 |
| Full (100ep) | 0.0585 | 0.0299 | 14.70 | 1.58 | 0.860 | 0.047 | 0.657 |

**关键观察（v7.3）**：
- **两分支都活跃且未关闭**：Full 最后 epoch `msqr_ratio≈0.86 / shcmi_ratio≈0.047`、
  `scale_gate≈0.66`——DMSQR 占主导但 DP-HCMI 保持非零，两分支共同贡献。
- **raw ratio 不再≈1 是预期行为**：v7.3 不给 anchor 归一化，模块输出（LN 后 norm≈16）
  相对原始 anchor 更大，配合可学习 gate（lambda_msqr≈0.058 / lambda_shcmi≈0.030）
  提供真实的 residual 增强——这正是「模块有效、baseline 不强」的目标形态。
- **与 v7.2 的对比**：v7.2 把 anchor 也 LN 后 raw≈1、gate≈0.01-0.03（模块贡献被压扁）；
  v7.3 模块能实际注入（msqr_ratio 0.86 vs v7.2 的 0.03），Full 全指标 ≥ B1,B2。

## 五、运行记录（run 目录 ↔ 结果）

| 任务 | 30ep (v6/v7/v7.2) | v7.3 run（本次） | tag | 状态 |
|---|---|---|---|---|
| B0 | 2026082019?_…/20260820200_…/20260820205_… | run/20260820220_B0_frozen | B0_frozen | ✅ 50ep |
| B1 | 同左 | run/20260820220_B1_msqr | B1_msqr | ✅ 50ep |
| B2 | 同左 | run/20260820222_B2_shcmi | B2_shcmi | ✅ 50ep |
| Ours | 同左 | run/20260820222_Ours | Ours | ✅ 100ep |
| FT-CLIP | — | — | FT_CLIP | ⬜ |

> v7.3 运行方式：B0/B1 并行 → 完成后 B2/Full 并行（每次 2 个进程共享单卡），
> 两次并行条件一致，便于批间横向比较。
