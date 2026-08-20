#!/bin/bash
# 一键串行跑 AGIQA-3K 主消融（v2 Frozen 协议）
# 统一: seed42_split3 / 512 / batch48 / init_lr 1e-4 / best_criterion=joint
# 用法: bash ablation/run_ablation.sh   （从项目根目录执行）
# 可选: INCLUDE_FT=1 bash ablation/run_ablation.sh  （额外跑 FT-CLIP 强参考）
set -e
cd "$(dirname "$0")/.."

PY=${PY:-/root/autodl-tmp/CondaEnv/ipiqa/bin/python}

run_task() {
    echo "=================================================="
    echo "[$(date '+%m-%d %H:%M')] $1"
    echo "=================================================="
    $PY train.py --cfg-path "$2" --seed 42 --num_cv 1 || exit 1
}

run_task "B0 Frozen baseline"   "ablation/configs/agiqa3k/b0_baseline.yaml"
run_task "B1 +DMSQR"            "ablation/configs/agiqa3k/b1_msqr.yaml"
run_task "B2 +DP-HCMI"          "ablation/configs/agiqa3k/b2_shcmi.yaml"
run_task "Ours (all)"           "ablation/configs/agiqa3k/ours.yaml"

if [ "${INCLUDE_FT:-0}" = "1" ]; then
    run_task "FT-CLIP reference" "ablation/configs/agiqa3k/ft_clip_reference.yaml"
fi

echo "[$(date '+%m-%d %H:%M')] All AGIQA-3K ablations done."
