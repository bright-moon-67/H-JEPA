#!/bin/bash
# Push-T 全矩阵评测队列：7 模型 × 3 seeds，由 parallel_eval.py 每次 4 worker 并行
# 基准组（pusht_hjepa_l3 seed42）在跑完后自动接力剩余的 20 组
set -u
export HJEPA_HOME=/root/rivermind-data/hjepa_home
cd /root/rivermind-data/H-JEPA/h_jepa || exit 1
PY=/opt/conda/envs/hjepa/bin/python

echo "[$(date +%F' '%T)] 队列启动，等待基准任务(pusht_hjepa_l3 seed42)结束..."
while pgrep -f "parallel_eval.py --config-name pusht_l3" >/dev/null 2>&1; do
  sleep 20
done
echo "[$(date +%F' '%T)] 基准任务已结束，开始剩余 20 组"

declare -A CFG=(
  [pusht_hjepa_l2]=pusht_l2 [pusht_hjepa_l3]=pusht_l3 [pusht_hjepa_l4]=pusht_l4
  [pusht_hwm_l2]=pusht_l2   [pusht_hwm_l3]=pusht_l3   [pusht_hwm_l4]=pusht_l4
  [pusht_lewm]=pusht_flat
)

FAILED=0
for M in pusht_hjepa_l2 pusht_hjepa_l3 pusht_hjepa_l4 pusht_hwm_l2 pusht_hwm_l3 pusht_hwm_l4 pusht_lewm; do
  for S in 42 43 44; do
    if [ "$M" = "pusht_hjepa_l3" ] && [ "$S" = "42" ]; then continue; fi
    CKPT="$HJEPA_HOME/ckpts/pusht/$M/seed$S/${M}_object.ckpt"
    OUT="eval_${M#pusht_}_s${S}_par"
    echo "[$(date +%F' '%T)] START $M seed$S config=${CFG[$M]} out=$OUT"
    if ! "$PY" parallel_eval.py --config-name "${CFG[$M]}" --policy "$CKPT" \
        --workers 4 --batch-size 13 --seed "$S" --out "$OUT" \
        >> "/tmp/peval_${M}_s${S}.log" 2>&1; then
      echo "[$(date +%F' '%T)] FAIL $M seed$S (见 /tmp/peval_${M}_s${S}.log)"
      FAILED=$((FAILED+1))
    else
      echo "[$(date +%F' '%T)] DONE $M seed$S"
    fi
  done
done
echo "[$(date +%F' '%T)] 队列结束，失败 $FAILED 组"
