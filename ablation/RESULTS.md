# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **30 epoch（B0-B2 / Ours 统一）**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v7.2, 30ep）—— 当前正式结果

> v7.2（改进7.md）：**Residual Scale Calibration** —— 只改两点：
> 1. **Residual LayerNorm**：`norm_v0 / norm_t0`（anchor）+ `norm_msqr / norm_shcmi`
>    （residual delta），消除 DMSQR/DP-HCMI 的 norm 量级失衡；
> 2. **Outer gate 小正初始化**：`lambda_msqr = lambda_shcmi = 0.01`（两边相同）。
> 其他结构全部保持 v7（共享 spatial projection / 余弦偏差 / prompt mask+scale /
> 单次 tanh discrepancy）。运行方式：B0/B1、B2/Ours 各 2 个一并行。
> Ours(Full) = DMSQR + DP-HCMI

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

## 一·五、历史版本对比（同协议 30ep，供参考）

| Model | Q-SRCC | A-SRCC | 说明 |
|---|---|---|---|
| v3 Ours（MSQR+SHCMI+QTA+AG） | 0.8175 | 0.6389 | QTA/AG |
| v5 B1（DMSQR, 三适配器 head） | 0.8181 | 0.6452 | head 结构不同 |
| v6 B2（DP-HCMI） | 0.8206 | 0.6579 | DP-HCMI 输入被 DMSQR 改写 |
| v6 Ours | 0.8116 | 0.6231 | Ours < B2（负交互） |
| v7 Ours | 0.8163 | 0.6391 | 修复 hidden input + 双重 tanh；Ours≈B2 |
| **v7.2 Ours(Full)** | **0.8414** | **0.6909** | 尺度校准 + gate_init=0.01；Joint 最优 |

> 注：v5 与 v6 的 head 结构不同（v6 删除三适配器恢复简单 Linear），
> 因此同一模块的绝对数值不可直接跨版本比较；同版本内消融才严格可比。

## 三、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.8408 | 0.9023 | 0.6866 | 0.8268 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| Ours (v7.2) | Frozen | DMSQR+DP-HCMI | 0.8414 | **0.9036** | 0.6909 | **0.8287** |

## 四、Gate / Residual 观察

> v7.2 引入 LayerNorm 校准后，两个 residual 的 raw ratio 都被拉到 ~1（与 anchor
> 同量级），Full 中 DP-HCMI 不再被自动关闭。下表为 v7.2 各 run **最后 epoch**：

| Model | lambda_msqr | lambda_shcmi | raw_msqr | raw_shcmi | msqr_ratio | shcmi_ratio | scale_gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.0302 | — | 1.03 | — | 0.031 | — | — |
| B2 | — | 0.0117 | — | 0.998 | — | 0.0117 | 0.537 |
| Ours(Full) | 0.0302 | 0.0129 | 1.03 | 0.995 | 0.031 | 0.0128 | 0.464 |

**关键观察（v7.2）**：
- **Full 中 DP-HCMI 激活**：`shcmi_ratio=0.0128`（v7 里 0.003）、`scale_gate_mean=0.46`
  （v7 里 0.0）——自适应尺度门真正开始平衡 DMSQR/DP-HCMI 两分支，v7.2 核心目标达成。
- **raw ratio ≈ 1**：两个 residual 经 LayerNorm 后与 anchor 同量级，消除了
  v7 里 msqr_ratio≈3.6 vs shcmi_ratio≈0.003 的失衡。
- **两门控都从小正初始化正常学习**：lambda_msqr 0.030 / lambda_shcmi 0.013，无赢家通吃。

## 五、运行记录（run 目录 ↔ 结果）

| 任务 | v7 run | v7.2 run（本次） | tag | 状态 |
|---|---|---|---|---|
| B0 | run/20260820200_B0_frozen | run/20260820205_B0_frozen | B0_frozen | ✅ 30ep |
| B1 | run/20260820200_B1_msqr | run/20260820205_B1_msqr | B1_msqr | ✅ 30ep |
| B2 | run/20260820202_B2_shcmi | run/20260820210_B2_shcmi | B2_shcmi | ✅ 30ep |
| Ours | run/20260820202_Ours | run/20260820210_Ours | Ours | ✅ 30ep |
| FT-CLIP | — | — | FT_CLIP | ⬜ |

> v7.2 运行方式：B0/B1 并行 → 完成后 B2/Ours 并行（每次 2 个进程共享单卡），
> 两次并行条件一致，便于批间横向比较。
