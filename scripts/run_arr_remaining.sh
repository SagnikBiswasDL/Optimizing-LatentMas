#!/usr/bin/env bash
# Remaining ARR suite (Jiayi list + reviewer K=10 vs K=40+top-k).
# Paper decode: temp=0.6 top_p=0.95 sequential K=40 evict keep-B.
# Restarts skip a job if report.json already exists.
set -u
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/latentmas-baseline
source /root/venv/bin/activate
source /workspace/env_native.sh
PY=/root/venv/bin/python
LOG=/workspace/arr_suite.log
echo "[arr] START $(date)" | tee -a "$LOG"

run_job () {
  local tag=$1; shift
  local out=$1; shift
  if [[ -f "$out/report.json" ]]; then
    echo "[arr] SKIP $tag (exists) $(date)" | tee -a "$LOG"
    return 0
  fi
  echo "[arr] START $tag $(date)" | tee -a "$LOG"
  mkdir -p "$out"
  if "$PY" -u scripts/exp_agent_latency_memory.py --out_dir "$out" "$@"; then
    echo "[arr] OK $tag $(date)" | tee -a "$LOG"
  else
    echo "[arr] FAIL $tag exit=$? $(date)" | tee -a "$LOG"
  fi
}

SEEDS=(42 123 7)
COMMON=(--mode latency --k 40 --temperature 0.6 --top_p 0.95 --sink 4 --compress_mode evict --with_k10)

for seed in "${SEEDS[@]}"; do
  run_job "gsm8k_n100_b64_s${seed}" \
    "artifacts/exp_latency_mem/arr_gsm8k_n100_b64_s${seed}" \
    "${COMMON[@]}" --task gsm8k --n 100 --relay_budget 64 --judger_budget 2048 --seed "$seed"

  run_job "math_n100_b128_s${seed}" \
    "artifacts/exp_latency_mem/arr_math_n100_b128_s${seed}" \
    "${COMMON[@]}" --task math --n 100 --relay_budget 128 --judger_budget 4096 --seed "$seed"

  run_job "gpqa_n100_b128_s${seed}" \
    "artifacts/exp_latency_mem/arr_gpqa_n100_b128_s${seed}" \
    "${COMMON[@]}" --task gpqa --n 100 --relay_budget 128 --judger_budget 8192 --seed "$seed"
done

# AIME is small but long; 3 seeds still ok (30 items).
for seed in "${SEEDS[@]}"; do
  run_job "aime24_b128_s${seed}" \
    "artifacts/exp_latency_mem/arr_aime24_b128_s${seed}" \
    "${COMMON[@]}" --task aime2024 --n 30 --relay_budget 128 --judger_budget 8192 --seed "$seed"
  run_job "aime25_b128_s${seed}" \
    "artifacts/exp_latency_mem/arr_aime25_b128_s${seed}" \
    "${COMMON[@]}" --task aime2025 --n 30 --relay_budget 128 --judger_budget 8192 --seed "$seed"
done

# Coding: HumanEval+ executes in the original repo; here we still get per-agent
# latency. LiveCodeBench is best-effort (may fail to download).
run_job "humaneval_n50_b64_s42" \
  "artifacts/exp_latency_mem/arr_humaneval_n50_b64_s42" \
  "${COMMON[@]}" --task humanevalplus --n 50 --relay_budget 64 --judger_budget 2048 --seed 42

run_job "lcb_n40_b64_s42" \
  "artifacts/exp_latency_mem/arr_lcb_n40_b64_s42" \
  "${COMMON[@]}" --task livecodebench --n 40 --relay_budget 64 --judger_budget 2048 --seed 42

echo "[arr] DONE $(date)" | tee -a "$LOG"
echo ARR_SUITE_DONE | tee -a "$LOG"
