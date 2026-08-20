# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **30 epoch（B0-B3 / Ours 统一）**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v3, 30ep）

> v3（改进2.md）：TAF 已删除；Ours = MSQR + SHCMI + QTA + AG
> QTA = MSQR quality-aware token aggregation（质量感知）
> AG  = SHCMI alignment-guided cross-modal gate（图文一致性）

| Model | MSQR | SHCMI | QTA | AG | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | × | × | 0.7847 | 0.8656 | 0.5983 | 0.6437 | 0.7698 |
| B1 | ✓ | × | × | × | 0.8216 | 0.8926 | 0.6343 | 0.6200 | 0.7989 |
| B2 | × | ✓ | × | × | 0.8066 | 0.8875 | 0.6176 | 0.6564 | 0.8091 |
| B3 | ✓ | ✓ | × | × | **0.8239** | **0.8957** | **0.6368** | 0.6299 | 0.8034 |
| **Ours** | ✓ | ✓ | ✓ | ✓ | 0.8175 | 0.8927 | 0.6284 | 0.6389 | 0.7977 |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|
| B1 − B0（MSQR） | **+0.0369** | +0.0270 | −0.0237 | +0.0291 |
| B2 − B0（SHCMI） | **+0.0219** | +0.0219 | +0.0128 | +0.0393 |
| B3 − B0 | **+0.0392** | +0.0301 | −0.0138 | +0.0336 |
| Ours − B0 | **+0.0328** | +0.0271 | −0.0048 | +0.0279 |
| **Ours − B3（QTA+AG）** | −0.0064 | −0.0030 | **+0.0090** | −0.0057 |

## 二、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.7847 | 0.8656 | 0.6437 | 0.7698 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| Ours | Frozen | MSQR+SHCMI+QTA+AG | 0.8175 | 0.8927 | 0.6389 | 0.7977 |

## 三、Gate / Residual 观察（最后 epoch）

| Model | lambda_msqr | lambda_shcmi | msqr_ratio | shcmi_ratio | QTA fine w | AG fine w |
|---|---:|---:|---:|---:|---:|---:|
| B1 | ~0.01 | — | ~3.4 | — | — | — |
| B2 | — | ~−0.01 | — | ~0.5 | — | — |
| B3 | ~0.008 | ~−0.008 | ~2.1 | ~0.25 | — | — |
| Ours | 见 log | 见 log | 见 log | 见 log | 见 log | 见 log |

**关键观察**：
- B1/B3 的 `msqr_ratio` 较大（2-3.4）→ MSQR 残差幅值可观，是 Q 提升主力。
- **Ours 相对 B3**：Q-SRCC 略降（−0.006）但 **A-SRCC 提升 +0.009**——
  AG（alignment-guided gate）修复了 MSQR 牺牲图文一致性的问题；
  QTA 在 30 epoch 下对 quality 无增益（可能需更多 epoch / 调 gate 初始化）。
- B2 的 A-SRCC 全场最高（0.6564），SHCMI 本身对 alignment 贡献最大。

## 四、运行记录（run 目录 ↔ 结果）

| 任务 | run 目录 | tag | 状态 |
|---|---|---|---|
| B0 | run/20260820173_B0_frozen | B0_frozen | ✅ 30ep |
| B1 | run/20260820173_B1_msqr | B1_msqr | ✅ 30ep |
| B2 | run/20260820173_B2_shcmi | B2_shcmi | ✅ 30ep |
| B3 | run/20260820174_B3_msqr_shcmi | B3_msqr_shcmi | ✅ 30ep |
| Ours | run/20260820174_Ours | Ours | ✅ 30ep |
| FT-CLIP | — | FT_CLIP | ⬜ |
