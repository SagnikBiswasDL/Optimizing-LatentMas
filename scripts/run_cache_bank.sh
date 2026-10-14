#!/usr/bin/env bash
# Build a task-banked KV prefix file, then eval GSM8K without running silent agents.
# No eviction. Restarts skip a stage if its report already exists.
#
# On the pod:
#   tmux new-session -d -s bank \
#     'bash scripts/run_cache_bank.sh > /workspace/cache_bank.log 2>&1'
set -uo pipefail

ROOT="${ROOT:-/workspace/latentmas-baseline}"
cd "$ROOT"
# shellcheck disable=SC1091
source env_native.sh 2>/dev/null || true

MODEL="${MODEL:-Qwen/Qwen3-14B}"
K="${K:-40}"
N="${N:-50}"
BUDGET="${BUDGET:-2048}"
# Paper sampling: TEMP=0.6 TOP_P=0.95
TEMP="${TEMP:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
OUT="${OUT:-artifacts/cache_bank/qwen3-14b_k${K}}"

mkdir -p "$OUT"

echo "=== cache_bank $(date) model=$MODEL k=$K n=$N temp=$TEMP ==="

if [[ ! -f "$OUT/build_report.json" ]]; then
  python scripts/exp_cache_bank.py --mode build \
    --model_name "$MODEL" --k "$K" --seed "$SEED" \
    --tasks gsm8k,medqa --n_donors 4 --donor_split train \
    --out_dir "$OUT"
else
  echo "[bank] skip build (build_report.json exists)"
fi

EVAL_GSM="$OUT/eval_gsm8k_s${SEED}"
if [[ ! -f "$EVAL_GSM/report.json" ]]; then
  python scripts/exp_cache_bank.py --mode eval \
    --model_name "$MODEL" --k "$K" --seed "$SEED" \
    --task gsm8k --n "$N" --bank "$OUT/bank.pt" \
    --arms none,bank,wrong_bank,synth --wrong_bank_key medqa \
    --judger_budget "$BUDGET" --temperature "$TEMP" --top_p "$TOP_P" \
    --out_dir "$EVAL_GSM"
else
  echo "[bank] skip gsm8k eval"
fi

EVAL_MED="$OUT/eval_medqa_s${SEED}"
if [[ ! -f "$EVAL_MED/report.json" ]]; then
  python scripts/exp_cache_bank.py --mode eval \
    --model_name "$MODEL" --k "$K" --seed "$SEED" \
    --task medqa --n "$N" --bank "$OUT/bank.pt" \
    --arms none,bank,wrong_bank,synth --wrong_bank_key gsm8k \
    --judger_budget "$BUDGET" --temperature "$TEMP" --top_p "$TOP_P" \
    --out_dir "$EVAL_MED"
else
  echo "[bank] skip medqa eval"
fi

echo "=== cache_bank DONE $(date) ==="
echo "BANK_CHAIN_DONE"
