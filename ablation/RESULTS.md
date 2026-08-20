# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **30 epoch（B0-B2 / Ours 统一）**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v6, 30ep）—— 当前正式结果

> v6（改进5.md）：TAF / QTA / AG / Consistency Loss 已全部删除，纯 MSE
> Ours = DMSQR + DP-HCMI
> DMSQR = Distortion-aware MSQR（语义偏差增强）；DP-HCMI = Discrepancy-aware
> Prompt-conditioned HCMI（prompt 加权 + 视觉-文本 discrepancy 引导）

| Model | DMSQR | DP-HCMI | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | 0.7847 | 0.8656 | 0.5983 | 0.6437 | 0.7698 |
| B1 | ✓ | × | 0.8107 | 0.8876 | 0.6212 | 0.6286 | 0.8013 |
| B2 | × | ✓ | **0.8206** | **0.8945** | **0.6325** | **0.6579** | **0.8114** |
| **Ours** | ✓ | ✓ | 0.8116 | 0.8851 | 0.6229 | 0.6231 | 0.7930 |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|
| B1 − B0（DMSQR） | **+0.0260** | +0.0220 | −0.0151 | +0.0315 |
| B2 − B0（DP-HCMI） | **+0.0359** | +0.0289 | **+0.0142** | +0.0416 |
| Ours − B0 | **+0.0269** | +0.0195 | −0.0206 | +0.0232 |
| Ours − B2 | −0.0090 | −0.0094 | −0.0348 | −0.0184 |

## 二、历史版本对比（同协议 30ep，供参考）

| Model | Q-SRCC | A-SRCC | 说明 |
|---|---|---|---|
| v3 Ours（MSQR+SHCMI+QTA+AG） | 0.8175 | 0.6389 | QTA/AG |
| v5 B1（DMSQR, 三适配器 head） | 0.8181 | 0.6452 | head 结构不同 |
| v6 B2（DP-HCMI） | **0.8206** | **0.6579** | 当前最优单模块 |
| v6 Ours（DMSQR+DP-HCMI） | 0.8116 | 0.6231 | 本版 |

> 注：v5 与 v6 的 head 结构不同（v6 删除三适配器恢复简单 Linear），
> 因此同一模块的绝对数值不可直接跨版本比较；同版本内消融才严格可比。

## 三、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.7847 | 0.8656 | 0.6437 | 0.7698 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| Ours | Frozen | DMSQR+DP-HCMI | 0.8116 | 0.8851 | 0.6231 | 0.7930 |

## 四、Gate / Residual 观察（v6 最后 epoch）

| Model | lambda_msqr | lambda_shcmi | alpha_dev | beta_align | msqr_ratio | shcmi_ratio |
|---|---:|---:|---:|---:|---:|---:|
| B1 | ~0.01 | — | 见 log | — | 见 log | — |
| B2 | — | ~−0.01 | — | 见 log | — | 见 log |
| Ours | 见 log | 见 log | 见 log | 见 log | 见 log | 见 log |

**关键观察**：
- **DP-HCMI 是当前最强模块**（B2：Q-SRCC +0.036 且 A-SRCC +0.014 双升）——
  discrepancy map（`attn_score -= beta*(1-cos)`）比 v5 的 alignment bias 更有效。
- **Ours（双模块叠加）低于 B2**（Q −0.009，A −0.035）：DMSQR + DP-HCMI 存在
  负交互，DMSQR 的视觉偏差增强干扰了 DP-HCMI 的跨模态 alignment 建模。
  需排查：两个模块的 lambda 平衡 / 输入 token 竞争。
- 与 v5 的 B3 观察一致（两模块同时开时联合增益不明显）。

## 五、运行记录（run 目录 ↔ 结果）

| 任务 | run 目录 | tag | 状态 |
|---|---|---|---|
| B0 | run/2026082019?_B0_frozen | B0_frozen | ✅ 30ep |
| B1 | run/2026082019?_B1_msqr | B1_msqr | ✅ 30ep |
| B2 | run/2026082019?_B2_shcmi | B2_shcmi | ✅ 30ep |
| Ours | run/2026082019?_Ours | Ours | ✅ 30ep |
| FT-CLIP | — | FT_CLIP | ⬜ |
