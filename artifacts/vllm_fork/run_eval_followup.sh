#!/bin/bash
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface TMPDIR=/root/tmp
export PATH=/root/venv111/bin:$PATH PYTHONPATH=/workspace/latentmas-hybrid
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /workspace/latentmas-hybrid
OUT=/workspace/fork_runs
echo "FOLLOWUP_START $(date -u)" | tee -a "$OUT/eval_followup.log"

run() {
  local arm=$1 task=$2 n=$3 model=$4 conc=$5 max_new=$6
  local tag="${model##*/}"
  local dir="$OUT/${arm}_${task}_n${n}_c${conc}_${tag}_v2"
  mkdir -p "$dir"
  echo "=== START $arm $task n=$n c=$conc $(date -u) ===" | tee -a "$OUT/eval_followup.log"
  python -u scripts/eval_vllm_fork.py --arm "$arm" --task "$task" --n "$n" --model "$model" \
    --concurrency "$conc" --greedy --latent-steps 10 --max-new-tokens "$max_new" \
    --gpu-memory-utilization 0.45 --max-model-len 16384 --out "$dir" \
    2>&1 | tee "$dir/run.log"
  echo "=== DONE $arm exit:${PIPESTATUS[0]} $(date -u) ===" | tee -a "$OUT/eval_followup.log"
}

run fork_stock gsm8k 50 Qwen/Qwen3-14B 8 2048
run fork_tail gsm8k 50 Qwen/Qwen3-14B 8 2048
run fork_stock gsm8k 50 Qwen/Qwen3-14B 16 2048
run single gsm8k 50 Qwen/Qwen3-14B 1 2048
run fork_stock math500 50 Qwen/Qwen3-14B 1 2048
echo "FOLLOWUP_COMPLETE $(date -u)" | tee -a "$OUT/eval_followup.log"
