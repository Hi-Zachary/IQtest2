# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **30 epoch（B0-B3 / Ours 统一）**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v5, 30ep）—— 当前正式结果

> v5（改进4.md）：TAF / QTA / AG 已删除
> Ours = DMSQR + PCS-HCMI + Shared-Task Consistency Loss
> DMSQR = MSQR + 语义偏差增强（Distortion-aware）；PCS-HCMI = SHCMI + prompt 加权 + alignment bias

| Model | DMSQR | PCS-HCMI | Cons | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | × | 0.7748 | 0.8621 | 0.5879 | 0.6432 | 0.7816 |
| B1 | ✓ | × | × | **0.8181** | **0.8903** | **0.6309** | **0.6452** | 0.8045 |
| B2 | × | ✓ | × | 0.7932 | 0.8834 | 0.6040 | 0.6406 | **0.8070** |
| B3 | ✓ | ✓ | × | 0.8147 | 0.8883 | 0.6262 | 0.6386 | 0.8036 |
| **Ours** | ✓ | ✓ | ✓ | 0.8152 | 0.8888 | 0.6253 | 0.6397 | 0.8034 |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|
| B1 − B0（DMSQR） | **+0.0433** | +0.0282 | +0.0020 | +0.0229 |
| B2 − B0（PCS-HCMI） | **+0.0184** | +0.0213 | −0.0027 | +0.0254 |
| B3 − B0 | **+0.0399** | +0.0262 | −0.0047 | +0.0220 |
| Ours − B0 | **+0.0403** | +0.0267 | −0.0035 | +0.0218 |
| **Ours − B3（Consistency）** | +0.0005 | +0.0005 | +0.0012 | −0.0002 |

## 二、历史版本对比（同协议 30ep，供参考）

| Model | Q-SRCC | A-SRCC | 说明 |
|---|---|---|---|
| v3 B3（MSQR+SHCMI） | 0.8239 | 0.6299 | 旧基线 |
| v3 Ours（+QTA+AG） | 0.8175 | 0.6389 | QTA 无增益，AG 提 A |
| v5 B3（DMSQR+PCS-HCMI） | 0.8147 | 0.6386 | 本版 |
| **v5 Ours** | **0.8152** | 0.6397 | 本版 |

> 注：v3 与 v5 的 B0/B1/B2 数值略有差异，因为 v3/v4 的 head 结构与 v5 不同
> （v5 增加 shared_adapter + 三适配器），B0 从 0.7847 变为 0.7748。同版本内消融才严格可比。

## 三、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.7748 | 0.8621 | 0.6432 | 0.7816 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| Ours | Frozen | DMSQR+PCS-HCMI+Cons | 0.8152 | 0.8888 | 0.6397 | 0.8034 |

## 四、Gate / Residual 观察（v5 最后 epoch）

| Model | lambda_msqr | lambda_shcmi | alpha_dev | beta_align | msqr_ratio | shcmi_ratio |
|---|---:|---:|---:|---:|---:|---:|
| B1 | ~0.01 | — | — | — | 见 log | — |
| B2 | — | ~−0.01 | — | — | — | 见 log |
| B3 | 见 log | 见 log | — | — | 见 log | 见 log |
| Ours | 见 log | 见 log | 见 log | 见 log | 见 log | 见 log |

**关键观察**：
- **DMSQR 是最强单一模块**（B1 Q-SRCC +0.043），且 A-SRCC 不降反升（+0.002）——语义偏差增强既提质量又不牺牲一致性。
- **PCS-HCMI 单独较弱**（B2 +0.018）但 A-PLCC 全场最高（0.8070）——对 alignment 贡献最大。
- **Consistency loss 30ep 下贡献甚微**（Ours−B3：Q +0.0005 / A +0.0012），需更长 epoch 验证。
- Frozen 协议下 30ep 基本收敛（参考 v2 100ep 曲线：30ep 后不再上涨），故 30ep 值≈最终值。

## 五、运行记录（run 目录 ↔ 结果）

| 任务 | run 目录 | tag | 状态 |
|---|---|---|---|
| B0 | run/2026082018?_B0_frozen | B0_frozen | ✅ 30ep |
| B1 | run/2026082018?_B1_msqr | B1_msqr | ✅ 30ep |
| B2 | run/2026082018?_B2_shcmi | B2_shcmi | ✅ 30ep |
| B3 | run/2026082018?_B3_msqr_shcmi | B3_msqr_shcmi | ✅ 30ep |
| Ours | run/2026082018?_Ours | Ours | ✅ 30ep |
| FT-CLIP | — | FT_CLIP | ⬜ |
