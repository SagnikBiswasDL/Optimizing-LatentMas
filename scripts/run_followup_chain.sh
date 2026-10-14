#!/usr/bin/env bash
# Follow-up chain after GSM8K B=16 showed a Judger verbosity tax.
# 1) B=64: thicker scaffold — does Judger time recover at ~10x memory cut?
# 2) B=16 + Judger ASC: does steering cancel the tax?
# 3) MATH500 n=30 B=128: hard-task operating point (latency only).
set -u
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/latentmas-baseline
source /root/venv/bin/activate
source /workspace/env_native.sh
PY=/root/venv/bin/python
ASC=artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt
LOG=/workspace/followup_chain.log
echo "[chain] START $(date)" | tee "$LOG"

run () {
  local tag=$1; shift
  echo "[chain] START $tag $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_agent_latency_memory.py "$@"; then
    echo "[chain] OK $tag $(date)" | tee -a "$LOG"
  else
    echo "[chain] FAIL $tag exit=$? $(date)" | tee -a "$LOG"
  fi
}

run gsm8k_b64 \
  --mode both --task gsm8k --n 50 --k 40 \
  --relay_budget 64 --sink 4 --compress_mode evict \
  --judger_budget 768 \
  --out_dir artifacts/exp_latency_mem/gsm8k_b64

run gsm8k_b16_asc \
  --mode latency --task gsm8k --n 50 --k 40 \
  --relay_budget 16 --sink 4 --compress_mode evict \
  --judger_budget 768 \
  --asc_vector "$ASC" --asc_coef 40 \
  --out_dir artifacts/exp_latency_mem/gsm8k_b16_asc

run math_b128 \
  --mode latency --task math --n 30 --k 40 \
  --relay_budget 128 --sink 4 --compress_mode evict \
  --judger_budget 2048 \
  --out_dir artifacts/exp_latency_mem/math_b128

echo "[chain] DONE $(date)" | tee -a "$LOG"
echo FOLLOWUP_CHAIN_DONE | tee -a "$LOG"
