# IQtest v8：CLIP ViT-B/16 + Visual LoRA + DG-MPQ + DP-HCMI-ViT

缝合方案落地（`../调整/改进1.md`）。在 Frozen CLIP ViT-B/16 主干上：
Visual Q/K LoRA 做 backbone adaptation，两个主创新模块并行：
- **DG-MPQ**（Module 1，Quality）：multi-level ViT patch（L3/6/9/12）+ local-global
  semantic deviation 引导的 patch weighting → normalized quality residual
- **DP-HCMI-ViT**（Module 2，Alignment）：detail-level（L6）/ semantic-level（L12）
  与真实 generation prompt 的双向跨模态交互 + discrepancy-guided attention bias
  + adaptive hierarchy fusion → normalized cross-modal residual

两模块从 backbone hidden states 并行出发（互不串行），以 `tanh(lambda)` gated
residual 注入 anchor：`v = v0 + tanh(λq)·Δq`，`c = t0 + tanh(λa)·Δa`。

## 环境

已验证 conda 环境：`/root/autodl-tmp/CondaEnv/ipiqa`（torch 2.3.1+cu121）
已安装：`transformers==4.44.2`、`peft==0.12.0`（与 torch 2.3.1 兼容版本）。

## 目录

```text
v8/
├── train.py                     # 训练入口（复用 v7 框架）
├── trainer.py                   # 训练循环（best-joint / gate 日志）
├── serial_train.sh              # 一键串行 B0→R0→B1→B2→Full
├── ipiqa/
│   ├── common|datasets|processors|tasks   # 复用的 LAVIS 风格框架
│   └── models/
│       ├── clip_vit_backbone.py  # Frozen CLIP ViT + Visual LoRA (q/k, r4 a8)
│       ├── dg_mpq.py             # Module 1
│       ├── dp_hcmi_vit.py        # Module 2
│       ├── heads.py              # anchors + shared fusion + dual heads
│       └── model_v8.py           # MSQRNetV8（消融开关 use_lora/dg_mpq/dp_hcmi）
├── ckpt/clip-vit-base-patch16/   # HF openai/clip-vit-base-patch16 权重
├── configs/                      # b0_frozen / r0_lora / b1_dgmpq / b2_dphcmi / full
├── splits/seed42_split3.json     # 固定划分（沿用 v7 协议）
├── data -> ../../data            # 数据软链接
└── run/                          # 训练输出（log.txt / checkpoints）
```

参考源码在 `../参考源码/`：NR_IQA_AGM（LoRA 配置 r4/a8/dropout0.05/qk，已核对一致）、
CLIP-AGIQA、LoDa（Plan B 备用）、及已有 CHPNet（DP-HCMI donor）。

## 主消融（调整/改进1.md 第 10 节）

| Model | Visual LoRA | DG-MPQ | DP-HCMI | 作用 |
|---|---:|---:|---:|---|
| B0 | × | × | × | Frozen CLIP baseline |
| R0 | ✓ | × | × | LoRA-CLIP strong reference |
| B1 | ✓ | ✓ | × | Quality module |
| B2 | ✓ | × | ✓ | Alignment module |
| Full | ✓ | ✓ | ✓ | Final |

## 运行

```bash
# 单独跑某一变体
python train.py --cfg-path configs/b0_frozen.yaml  --seed 42 --num_cv 1
python train.py --cfg-path configs/r0_lora.yaml   --seed 42 --num_cv 1
python train.py --cfg-path configs/b1_dgmpq.yaml  --seed 42 --num_cv 1
python train.py --cfg-path configs/b2_dphcmi.yaml --seed 42 --num_cv 1
python train.py --cfg-path configs/full.yaml      --seed 42 --num_cv 1

# 一键串行
bash serial_train.sh

# 冒烟测试
python tests/smoke_test_v8.py
```

## 协议要点（第 11 节）

- 224×224 / 30ep / batch 64 / AdamW / init_lr 1e-4（LoRA 与主模块同 1e-4，沿用 v7 batch64 协议）
- LoRA r=4 α=8 dropout=0.05，target q_proj/k_proj；text 全冻结、visual base 冻结
- 纯 MSE(Q)+MSE(A)，无额外 loss
- split3 seed42，best_criterion=joint
- 每 epoch 记录 lambda_q/lambda_a/beta_align/hier_gate/deviation/patch_w/
  prompt_w/raw_ratio/ratio，用于判断模块是否训练、分支是否 collapse

## 关键检查点

1. **R0 > B0 ？**（LoRA 是否真的训练：requires_grad / 梯度 / qkv 被替换）
2. **模块独立性**：B1 的 DG-MPQ 输入 == Full 的 DG-MPQ 输入（已由 smoke test 验证）
3. **分支活性**：Full 中 quality_ratio / align_ratio 均非 0
4. **判据**：R0>B0, B1>R0, B2>R0, Full ≥ 最好单模块；不追求每指标第一
