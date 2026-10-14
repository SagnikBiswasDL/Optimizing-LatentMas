#!/usr/bin/env bash
# Math lock: one REAL train relay (DonorScaffold). Never average KV.
#   bash scripts/run_math_harden.sh gsm8k   # n=150, 8 correct donors, heldout pick
#   bash scripts/run_math_harden.sh aime    # AIME2025 donor → AIME2024
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 2
if [[ -f /workspace/env_native.sh ]]; then
  source /root/venv/bin/activate 2>/dev/null || true
  source /workspace/env_native.sh 2>/dev/null || true
fi
export HF_HOME=${HF_HOME:-${REPO}/.cache/huggingface}

MODE=${1:-gsm8k}

run() {
  local tag=$1
  shift
  local out=${REPO}/artifacts/exp_synth_latency/${tag}
  mkdir -p "$out"
  echo "[run] $tag -> $out"
  python -u scripts/exp_synth_latency.py --mode latency --out_dir "$out" "$@"
}

if [[ "$MODE" == "gsm8k" ]]; then
  run gsm8k_donor_n150 \
    --task gsm8k --n 150 --k 10 --stat_n 8 --select_n 32 --filter_correct \
    --scaffold donor --donor_select heldout --generate_bs 20 \
    --judger_budget 1024 --temperature 0.0 --top_p 1.0
  exit $?
fi

if [[ "$MODE" == "aime" ]]; then
  # Same contest family: freeze one AIME 2025 relay, eval AIME 2024.
  run aime24_donor_from25 \
    --task aime2024 --donor_task aime2025 --stat_split train \
    --n 30 --k 10 --stat_n 8 --select_n 12 \
    --scaffold donor --donor_select heldout --generate_bs 8 \
    --judger_budget 4096 --temperature 0.0 --top_p 1.0
  exit $?
fi

echo "usage: $0 gsm8k|aime" >&2
exit 2
