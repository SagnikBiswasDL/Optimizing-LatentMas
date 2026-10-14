#!/usr/bin/env bash
# LatentMAS vs type-level SynthScaffold at the paper/fork generate_bs (20).
# Not a batch-size sweep.
#   bash scripts/run_synth_latency.sh smoke
#   bash scripts/run_synth_latency.sh medqa
#   bash scripts/run_synth_latency.sh gsm8k
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 2
if [[ -f /workspace/env_native.sh ]]; then
  source /root/venv/bin/activate 2>/dev/null || true
  source /workspace/env_native.sh 2>/dev/null || true
fi

MODE=${1:-medqa}
export HF_HOME=${HF_HOME:-${REPO}/.cache/huggingface}

if [[ "$MODE" == "smoke" ]]; then
  python -u scripts/exp_synth_latency.py --smoke --task medqa \
    --out_dir artifacts/exp_synth_latency/smoke
  exit $?
fi

TASK=$MODE
TAG=${TAG:-${TASK}_k10_bs20}
OUT=${REPO}/artifacts/exp_synth_latency/${TAG}
mkdir -p "$OUT"
echo "[run] $TASK generate_bs=20 -> $OUT"
python -u scripts/exp_synth_latency.py \
  --mode latency --task "$TASK" --n 40 --k 10 --stat_n 8 \
  --generate_bs 20 \
  --judger_budget 1024 --temperature 0.0 --top_p 1.0 \
  --out_dir "$OUT"
echo "[run] grep EXP_SYNTH_LATENCY_DONE when done"
