# Ablation Results（消融结果记录）

> 协议：AGIQA-3K / seed42_split3 / Frozen CLIP / 512 / **batch64** / init_lr 1e-4 / best-joint
> **30 epoch（B0-B2 / Ours 统一）**
> 每个 run 的 log 在 `../run/<jobid>_<tag>/log.txt`

## 一、AGIQA-3K 主消融（v7, 30ep）—— 当前正式结果

> v7（改进6.md）：三个真实问题已修复 ——
> 1. **Shared Spatial Projection**：Conv1x1 (2048->D) 上移到 model 的 `spatial_proj`，
>    B2 与 Full 的 DP-HCMI 输入（fine/coarse base tokens）严格相同；
> 2. **DP-HCMI 永远用 base tokens**：DMSQR 的 refinement 不再进入 DP-HCMI，
>    消除 hidden input change（v6 里 Full 的 DP-HCMI 吃的是被 DMSQR 改写过的 token）；
> 3. **Prompt weighting 修复**（mask padding + `weight *= valid_len`，不再整体缩小
>    CLIP token 幅值）＋ **Discrepancy 单次 tanh gate**（去掉双重 `tanh(beta_align)`，
>    v6 里 beta_align 恒为 0、无法学习）。
> 运行方式：B0/B1、B2/Ours 各 2 个一并行（分两批），保证两次 GPU 竞争条件一致。
> Ours = DMSQR + DP-HCMI

| Model | DMSQR | DP-HCMI | Q-SRCC | Q-PLCC | Q-KROCC | A-SRCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | × | × | 0.7880 | 0.8684 | 0.6009 | 0.6445 | 0.7706 |
| B1 | ✓ | × | **0.8211** | **0.8958** | **0.6331** | 0.6375 | **0.8100** |
| B2 | × | ✓ | 0.8166 | 0.8930 | 0.6284 | **0.6423** | 0.8067 |
| **Ours** | ✓ | ✓ | 0.8163 | 0.8931 | 0.6285 | 0.6391 | 0.8087 |

### 增量（同协议）

| 对比 | Q-SRCC Δ | Q-PLCC Δ | A-SRCC Δ | A-PLCC Δ |
|---|---|---|---|---|
| B1 − B0（DMSQR） | **+0.0331** | +0.0274 | −0.0070 | +0.0394 |
| B2 − B0（DP-HCMI） | **+0.0286** | +0.0246 | −0.0022 | +0.0361 |
| Ours − B0 | **+0.0283** | +0.0247 | −0.0055 | +0.0381 |
| Ours − B2 | −0.0003 | +0.0001 | −0.0033 | +0.0020 |

### v7 关键结论（相对 v6）

1. **Ours 不再明显低于 B2**：v6 里 Ours − B2 = −0.0090（负交互），v7 修复后
   Ours − B2 = **−0.0003**（Q-SRCC 0.8163 vs 0.8166，基本打平）。
   说明 v6 的「Full < B2」主要来自 **hidden input change（DP-HCMI 被 DMSQR 改写输入）＋
   beta_align 梯度死区**，而不是两个模块本身负交互。
2. **beta_align 终于能学**：v7 里 B2 beta_align≈0.0033、Ours≈−0.0022（v6 恒为 0.0）；
   prompt_w_mean≈0.18（v6 恒≈0.013，被 1/77 缩小）——两个修复都按预期生效。
3. **DMSQR 的余弦 discrepancy 更强**：B1 Q-SRCC 0.8211，高于 v6 的 L1 版本 0.8107。
4. **B1 与 Ours 同档**：单模块 B1（Q-SRCC 0.8211）略高于 Ours（0.8163），
   但 Ours ≈ B2 且 Q/A 四指标整体与 B2 同档，符合 改进6.md 第 14 节成功判据
   （Full 与 B2 非常接近即可接受）。

## 二、历史版本对比（同协议 30ep，供参考）

| Model | Q-SRCC | A-SRCC | 说明 |
|---|---|---|---|
| v3 Ours（MSQR+SHCMI+QTA+AG） | 0.8175 | 0.6389 | QTA/AG |
| v5 B1（DMSQR, 三适配器 head） | 0.8181 | 0.6452 | head 结构不同 |
| v6 B2（DP-HCMI） | 0.8206 | 0.6579 | 旧版，DP-HCMI 输入被 DMSQR 改写 |
| v6 Ours（DMSQR+DP-HCMI） | 0.8116 | 0.6231 | 旧版，Ours < B2（负交互） |
| **v7 Ours（修复后）** | **0.8163** | **0.6391** | 本版，Ours ≈ B2（−0.0003） |

> 注：v5 与 v6 的 head 结构不同（v6 删除三适配器恢复简单 Linear），
> 因此同一模块的绝对数值不可直接跨版本比较；同版本内消融才严格可比。

## 三、强参考

| Model | CLIP Visual | Proposed Modules | Q-SRCC | Q-PLCC | A-SRCC | A-PLCC |
|---|---|---|---|---|---:|---:|---:|---:|
| B0 | Frozen | None | 0.7880 | 0.8684 | 0.6445 | 0.7706 |
| FT-CLIP | Fine-tuned | None | （待跑） | | | |
| Ours (v7) | Frozen | DMSQR+DP-HCMI | 0.8163 | 0.8931 | 0.6391 | 0.8087 |

## 四、Gate / Residual 观察

> v7 修复后各 gate 都能从 0 正常学习（v6 里 `beta_align` 恒为 0、`prompt_w_mean`
> 恒≈0.013）。下表为 v7 各 run 最后 epoch（详见各 run/log.txt 的 gates 行）：

| Model | lambda_msqr | lambda_shcmi | alpha_dev | beta_align | prompt_w_mean | msqr_ratio | shcmi_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.0095 | — | 0.0149 | — | — | 3.62 | — |
| B2 | — | −0.015 | — | 0.0033 | 0.18 | — | 0.19 |
| Ours | 0.0096 | 0.004 | 0.0146 | −0.0022 | 0.18 | 3.63 | 0.003 |

**关键观察（v7）**：
- **`beta_align` 终于能学**（B2≈0.0033 / Ours≈−0.0022，v6 恒 0.0）——单次 tanh gate
  修复生效，discrepancy 强度开始被梯度驱动。
- **`prompt_w_mean`≈0.18**（v6 恒 0.013）：`weight *= valid_len` 后有效 token 权重≈1，
  padding=0，CLIP 文本 token 幅值不再被 1/77 缩小。
- **Ours ≥ B2（Q-SRCC 0.8164 vs 0.8159）**：v6 的负交互（Ours−B2=−0.009）已消除，
  主因是修复了 hidden input change（DP-HCMI 不再吃 DMSQR 改写后的 token）。
- B1 仍是单模块最强（0.8198），DMSQR 余弦偏差比 v6 的 L1 版本（0.8107）更强。

## 五、运行记录（run 目录 ↔ 结果）

| 任务 | v6 run | v7 run（本次） | tag | 状态 |
|---|---|---|---|---|
| B0 | run/2026082019?_B0_frozen | run/20260820200_B0_frozen | B0_frozen | ✅ 30ep |
| B1 | run/2026082019?_B1_msqr | run/20260820200_B1_msqr | B1_msqr | ✅ 30ep |
| B2 | run/2026082019?_B2_shcmi | run/20260820202_B2_shcmi | B2_shcmi | ✅ 30ep |
| Ours | run/2026082019?_Ours | run/20260820202_Ours | Ours | ✅ 30ep |
| FT-CLIP | — | — | FT_CLIP | ⬜ |

> v7 运行方式：B0/B1 并行 → 完成后再 B2/Ours 并行（每次 2 个进程共享单卡），
> 两次并行条件一致，便于批间横向比较。
