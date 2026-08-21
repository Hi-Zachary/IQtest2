#!/bin/bash
# 并行跑 v8 主消融 (AGIQA-3K, split3 seed42, 30ep)
# 调整/改进1.md 第 10/17 节
#
# 用法:
#   bash parallel_train.sh                    # 并行跑全部 5 个变体 (最多 MAX_JOBS 个同时)
#   bash parallel_train.sh B0 R0 B1           # 只并行跑指定变体
#   MAX_JOBS=2 bash parallel_train.sh         # 限制同时最多 2 个
#
# 每个变体输出到 run/<job_id>_<tag>/, 各自日志写到 run/<tag>.log
# 显存参考: v8 Full (batch64/224, AMP) ~5.4GB + 模型/上下文 ~1.5GB ≈ 7GB/进程,
#           3090 24GB 最多同时 3 个; 默认 MAX_JOBS=3

PY=/root/autodl-tmp/CondaEnv/ipiqa/bin/python
MAX_JOBS=${MAX_JOBS:-3}
RUNS=${@:-B0 R0 B1 B2 Full}

declare -A CFG=(
  [B0]="configs/b0_frozen.yaml"
  [R0]="configs/r0_lora.yaml"
  [B1]="configs/b1_dgmpq.yaml"
  [B2]="configs/b2_hcmi.yaml"
  [Full]="configs/full.yaml"
)

mkdir -p run

for v in $RUNS; do
  cfg=${CFG[$v]}
  if [ -z "$cfg" ]; then echo "unknown variant: $v"; exit 1; fi

  # 并发控制：正在跑的进程数 >= MAX_JOBS 时等待空位
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 5
  done

  echo "[$(date '+%m-%d %H:%M')] start $v -> run/$v.log"
  $PY train.py --cfg-path "$cfg" --seed 42 --num_cv 1 > "run/$v.log" 2>&1 &
done

echo "waiting for all jobs ..."
wait
echo "[$(date '+%m-%d %H:%M')] all done. logs in run/*.log"
