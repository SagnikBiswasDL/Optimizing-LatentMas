#!/bin/bash
# Continuation eval queue after AIME drift diagnosis.
set -uo pipefail
export HF_HOME=/workspace/.cache/huggingface TMPDIR=/root/tmp
export PATH=/root/venv111/bin:$PATH PYTHONPATH=/workspace/latentmas-hybrid
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /workspace/latentmas-hybrid
OUT=/workspace/fork_runs
mkdir -p "$OUT"
LOG="$OUT/continuation_queue.log"
echo "CONTINUATION_START $(date -u)" | tee -a "$LOG"

run() {
  local arm=$1 task=$2 n=$3 model=$4 conc=$5 max_new=$6
  local tag="${model##*/}"
  local dir="$OUT/${arm}_${task}_n${n}_c${conc}_${tag}_cont"
  mkdir -p "$dir"
  echo "=== START $arm $task n=$n c=$conc max_new=$max_new $(date -u) ===" | tee -a "$LOG"
  python -u scripts/eval_vllm_fork.py --arm "$arm" --task "$task" --n "$n" --model "$model" \
    --concurrency "$conc" --greedy --latent-steps 10 --max-new-tokens "$max_new" \
    --gpu-memory-utilization 0.45 --max-model-len 16384 --out "$dir" \
    2>&1 | tee "$dir/run.log"
  local ec=${PIPESTATUS[0]}
  echo "=== DONE $arm exit:$ec $(date -u) ===" | tee -a "$LOG"
  return 0
}

# 0) 14B AIME latent cosine (sdpa vs eager vs fork) — blocking precursor
echo "=== START diag_cosine_14b_aime $(date -u) ===" | tee -a "$LOG"
python -u scripts/diag_latent_cosine.py \
  --model Qwen/Qwen3-14B --task aime --n 3 --latent-steps 10 \
  --out "$OUT/diag_latent_cosine_14b_aime.json" \
  2>&1 | tee "$OUT/diag_cosine_14b_aime.log"
echo "=== DONE diag_cosine_14b_aime exit:${PIPESTATUS[0]} $(date -u) ===" | tee -a "$LOG"

# 1) Concurrency fill for fork (wave-batched) + tail + single C=1 already have some
run fork_stock gsm8k 50 Qwen/Qwen3-14B 60 2048
run fork_tail gsm8k 50 Qwen/Qwen3-14B 16 2048
run fork_tail gsm8k 50 Qwen/Qwen3-14B 32 2048
run fork_tail gsm8k 50 Qwen/Qwen3-14B 60 2048
run single gsm8k 50 Qwen/Qwen3-14B 1 2048
# single at C>1 stays serial in harness; still record wall for completeness at C=8
run single gsm8k 50 Qwen/Qwen3-14B 8 2048

# 2) MATH-500 n=200 primary comparison
run fork_stock math500 200 Qwen/Qwen3-14B 1 2048
run hf math500 200 Qwen/Qwen3-14B 1 2048

# 3) GSM8K n=200 primary comparison
run fork_stock gsm8k 200 Qwen/Qwen3-14B 1 2048
run hf gsm8k 200 Qwen/Qwen3-14B 1 2048

# 4) text_mas + hybrid remaining
run text_mas gsm8k 50 Qwen/Qwen3-14B 1 2048
run hybrid gsm8k 50 Qwen/Qwen3-14B 1 2048
run hybrid math500 50 Qwen/Qwen3-14B 1 2048
run text_mas math500 50 Qwen/Qwen3-14B 1 2048
run text_mas aime 60 Qwen/Qwen3-14B 1 8192
run hybrid aime 60 Qwen/Qwen3-14B 1 8192
run single aime 60 Qwen/Qwen3-14B 1 8192
run hf math500 50 Qwen/Qwen3-14B 1 2048

# 5) fork MATH/AIME concurrency spot checks
run fork_stock math500 50 Qwen/Qwen3-14B 8 2048
run fork_stock gsm8k 50 Qwen/Qwen3-14B 16 2048

echo "CONTINUATION_COMPLETE $(date -u)" | tee -a "$LOG"
