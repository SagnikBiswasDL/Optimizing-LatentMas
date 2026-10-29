#!/bin/bash
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface TMPDIR=/root/tmp
export PATH=/root/venv111/bin:$PATH
export PYTHONPATH=/workspace/latentmas-hybrid
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /workspace/latentmas-hybrid
OUT=/workspace/fork_runs
mkdir -p "$OUT"
echo "QUEUE_START $(date -u)" | tee -a "$OUT/eval_queue.log"

run() {
  local arm=$1 task=$2 n=$3 model=$4 conc=$5 max_new=$6
  local tag="${model##*/}"
  local dir="$OUT/${arm}_${task}_n${n}_c${conc}_${tag}"
  mkdir -p "$dir"
  echo "=== START $arm $task n=$n c=$conc max_new=$max_new model=$model $(date -u) ===" | tee -a "$OUT/eval_queue.log"
  python -u scripts/eval_vllm_fork.py --arm "$arm" --task "$task" --n "$n" --model "$model" \
    --concurrency "$conc" --greedy --latent-steps 10 --max-new-tokens "$max_new" \
    --gpu-memory-utilization 0.45 --max-model-len 16384 --out "$dir" \
    2>&1 | tee "$dir/run.log"
  local ec=${PIPESTATUS[0]}
  echo "=== DONE $arm exit:$ec $(date -u) ===" | tee -a "$OUT/eval_queue.log"
  return 0
}

run fork_stock gsm8k 20 Qwen/Qwen3-4B 1 512
run fork_stock gsm8k 50 Qwen/Qwen3-14B 1 2048
run hf gsm8k 50 Qwen/Qwen3-14B 1 2048
run fork_stock gsm8k 50 Qwen/Qwen3-14B 8 2048
run fork_tail gsm8k 50 Qwen/Qwen3-14B 8 2048
run single gsm8k 50 Qwen/Qwen3-14B 8 2048
run hybrid gsm8k 30 Qwen/Qwen3-14B 1 2048
run fork_stock aime 60 Qwen/Qwen3-14B 1 8192
run hf aime 60 Qwen/Qwen3-14B 1 8192
run fork_stock gsm8k 50 Qwen/Qwen3-14B 16 2048
run fork_stock gsm8k 50 Qwen/Qwen3-14B 32 2048

echo "QUEUE_COMPLETE $(date -u)" | tee -a "$OUT/eval_queue.log"
