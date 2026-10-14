#!/usr/bin/env bash
# Paper protocol: K=40, Judger temp=0.6 / top-p=0.95, GSM8K max_new=2048.
# Naive eviction (sink + key-norm top-k) vs full LatentMAS on the SAME upstream.
set -u
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/latentmas-baseline
source /root/venv/bin/activate
source /workspace/env_native.sh
PY=/root/venv/bin/python
LOG=/workspace/paper_proto_compress.log
echo "[paper] START $(date)" | tee "$LOG"

run () {
  local tag=$1; shift
  echo "[paper] START $tag $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_agent_latency_memory.py "$@"; then
    echo "[paper] OK $tag $(date)" | tee -a "$LOG"
  else
    echo "[paper] FAIL $tag exit=$? $(date)" | tee -a "$LOG"
  fi
}

# Operating point that looked free under greedy, now under THEIR sampling.
run gsm8k_k40_t06_b64 \
  --mode latency --task gsm8k --n 50 --k 40 \
  --temperature 0.6 --top_p 0.95 --judger_budget 2048 \
  --relay_budget 64 --sink 4 --compress_mode evict \
  --seed 42 \
  --out_dir artifacts/exp_latency_mem/paper_gsm8k_k40_t06_b64

# Aggressive crush — does the tax survive sampling?
run gsm8k_k40_t06_b16 \
  --mode latency --task gsm8k --n 50 --k 40 \
  --temperature 0.6 --top_p 0.95 --judger_budget 2048 \
  --relay_budget 16 --sink 4 --compress_mode evict \
  --seed 42 \
  --out_dir artifacts/exp_latency_mem/paper_gsm8k_k40_t06_b16

echo "[paper] DONE $(date)" | tee -a "$LOG"
echo PAPER_PROTO_DONE | tee -a "$LOG"
