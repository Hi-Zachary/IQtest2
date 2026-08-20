# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **30 epoch（B0-B4 统一）**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v2 Frozen, 30ep）

| Model | Frozen CLIP | MSQR | SHCMI | TAF | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | ✓ | × | × | × | 0.7847 | 0.8656 | 0.5983 | 0.6437 | 0.7698 |
| B1 | ✓ | ✓ | × | × | 0.8216 | 0.8926 | 0.6343 | 0.6200 | 0.7989 |
| B2 | ✓ | × | ✓ | × | 0.8066 | 0.8875 | 0.6176 | 0.6564 | 0.8091 |
| B3 | ✓ | ✓ | ✓ | × | **0.8239** | **0.8957** | **0.6368** | 0.6299 | 0.8034 |
| B4 | ✓ | ✓ | ✓ | ✓ | 0.8132 | 0.8935 | 0.6263 | 0.6272 | 0.8023 |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ |
|---|---|---|---|
| B1 − B0（MSQR） | **+0.0369** | +0.0270 | −0.0237 |
| B2 − B0（SHCMI） | **+0.0219** | +0.0219 | +0.0128 |
| B3 − B0 | **+0.0392** | +0.0301 | −0.0138 |
| B4 − B3（TAF） | −0.0107 | −0.0022 | −0.0027 |
| **B4 − B0（总体）** | +0.0285 | +0.0279 | −0.0164 |

## 二、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.7847 | 0.8656 | 0.6437 | 0.7698 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| B4 | Frozen | MSQR+SHCMI+TAF | 0.8132 | 0.8935 | 0.6272 | 0.8023 |

## 三、Gate / Residual 观察（最后 epoch）

| Model | lambda_msqr | lambda_shcmi | lambda_taf_q/a | msqr_ratio | shcmi_ratio | taf_ratio |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.0096 | — | — | 3.38 | — | — |
| B2 | — | −0.0149 | — | — | 0.48 | — |
| B3 | 0.0080 | −0.0075 | — | 2.13 | 0.25 | — |
| B4 | 0.0084 | 0.0068 | −0.0032 / −0.0034 | 1.97 | 0.17 | 0.0011 / 0.0013 |

**关键观察**：
- B1/B3 的 `msqr_ratio` 较大（2-3.4）→ MSQR 残差幅值可观，是 Q 提升主力。
- **B4 的 TAF 残差几乎未激活**（`taf_q_ratio≈0.001`，`lambda_taf_q≈-0.003`，`g_q≈0.49`）：
  这正是 **B4 < B3** 的直接原因——TAF 在 30 epoch / Frozen 下 gate 没学开，
  却引入了额外随机初始化参数（gate_q/gate_a + 两个 adapter），小幅拖累。
  下一步建议：TAF 换成更简单的 residual（或调大内部 gate 初始化、延长 epoch）。

## 四、运行记录（run 目录 ↔ 结果）

| 任务 | run 目录 | tag | 状态 |
|---|---|---|---|
| B0 | run/2026082016?_B0_frozen | B0_frozen | ✅ 30ep |
| B1 | run/2026082016?_B1_msqr | B1_msqr | ✅ 30ep |
| B2 | run/2026082016?_B2_shcmi | B2_shcmi | ✅ 30ep |
| B3 | run/2026082016?_B3_msqr_shcmi | B3_msqr_shcmi | ✅ 30ep |
| B4 | run/2026082016?_B4_full | B4_full | ✅ 30ep |
| FT-CLIP | — | FT_CLIP | ⬜ |
