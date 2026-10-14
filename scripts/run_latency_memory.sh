#!/usr/bin/env bash
# Jiayi plots: per-agent latency (LatentMAS vs ours) + batch-size memory.
# Usage on pod:
#   bash scripts/run_latency_memory.sh smoke
#   bash scripts/run_latency_memory.sh full     # GSM8K n=50, K=40, B=16 evict
set -u
REPO=/workspace/latentmas-baseline
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$REPO" || exit 2
source /root/venv/bin/activate
source /workspace/env_native.sh 2>/dev/null || true

MODE=${1:-full}
TAG=${TAG:-gsm8k_b16}
OUT=/workspace/latentmas-baseline/artifacts/exp_latency_mem/${TAG}
LOG=/workspace/latency_mem_${TAG}.log
SENT=/workspace/latency_mem_${TAG}.sentinel

if [[ "$MODE" == "smoke" ]]; then
  python scripts/exp_agent_latency_memory.py --smoke --task gsm8k \
    --out_dir artifacts/exp_latency_mem/smoke
  exit $?
fi

mkdir -p "$OUT"
rm -f "$SENT"
echo "[run] full latency+memory -> $LOG"
nohup python -u scripts/exp_agent_latency_memory.py \
  --mode both --task gsm8k --n 50 --k 40 \
  --relay_budget 16 --sink 4 --compress_mode evict \
  --judger_budget 768 \
  --batch_grid 1,2,4,8,16,24,32,48,64 \
  --out_dir "$OUT" \
  > "$LOG" 2>&1 &
echo $! > /workspace/latency_mem_${TAG}.pid
echo "[run] pid=$(cat /workspace/latency_mem_${TAG}.pid) log=$LOG"
echo "[run] when done: grep EXP_LATENCY_MEM_DONE $LOG"
