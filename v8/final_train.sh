#!/bin/bash
# 最终消融实验: B0/R0/B1/B2 = 50ep, Full = 100ep (V8-Slim Final, AGIQA-3K split3 seed42)
#
# 分批调度:
#   默认 MODE=3_2 : 先 B0 R0 B1 (3 并行, 各 50ep) -> 等 GPU 空闲 -> 再 B2 Full (2 并行, 50/100ep)
#   MODE=2_2_1    : 先 B0 R0 -> B1 B2 -> Full (显存压力大时用)
#
# 用法:
#   bash final_train.sh            # 3+2
#   MODE=2_2_1 bash final_train.sh # 2+2+1
#
# 每个变体日志: run/<tag>.log (控制台) + run/<job_id>_<tag>/log.txt (每 epoch 指标/gates)
# 显存参考 (batch64/224/AMP): B0~1.5G R0~2G B1~2.5G B2~4G Full~5.4G (RTX3090 24G)
#   -> 3+2 批次峰值 ~6G / ~9.5G, 均安全

PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
MODE=${MODE:-3_2}

declare -A CFG=(
  [B0]="configs/b0_frozen.yaml"
  [R0]="configs/r0_lora.yaml"
  [B1]="configs/b1_dgmpq.yaml"
  [B2]="configs/b2_hcmi.yaml"
  [Full]="configs/full.yaml"
)
declare -A TAG=(
  [B0]="B0_frozen" [R0]="R0_lora" [B1]="B1_dgmpq" [B2]="B2_hcmi" [Full]="Full"
)

mkdir -p run

# ---- 等 GPU 空闲: 没有 train.py 进程且显存使用低于阈值 ----
wait_gpu_idle() {
  echo "[$(date '+%m-%d %H:%M')] waiting for GPU idle ..."
  while :; do
    np=$(pgrep -f "train.py --cfg-path" | wc -l)
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "$np" -eq 0 ] && [ "${mem:-9999}" -lt 500 ]; then
      echo "[$(date '+%m-%d %H:%M')] GPU idle (used=${mem}MiB)"
      return 0
    fi
    sleep 15
  done
}

run_batch() {
  for v in "$@"; do
    echo "[$(date '+%m-%d %H:%M')] start $v -> run/${TAG[$v]}.log"
    $PY train.py --cfg-path "${CFG[$v]}" --seed 42 --num_cv 1 > "run/${TAG[$v]}.log" 2>&1 &
  done
  wait
}

if [ "$MODE" = "2_2_1" ]; then
  echo "=== MODE 2+2+1 ==="
  run_batch B0 R0
  wait_gpu_idle
  run_batch B1 B2
  wait_gpu_idle
  run_batch Full
else
  echo "=== MODE 3+2 ==="
  run_batch B0 R0 B1
  wait_gpu_idle
  run_batch B2 Full
fi

echo "[$(date '+%m-%d %H:%M')] ALL FINAL EXPERIMENTS DONE"
