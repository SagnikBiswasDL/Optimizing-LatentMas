#!/usr/bin/env bash
# AIME localization: which silent-agent writes the Judger actually uses,
# then role-budgeted eviction. Restart-safe. GPU from `collect` onward.
#
#   bash scripts/run_aime_localize.sh smoke
#   bash scripts/run_aime_localize.sh focus     # items 0,1,2,4,10,18
#   bash scripts/run_aime_localize.sh full      # n=30
#   bash scripts/run_aime_localize.sh qwen      # Real decode at Qwen thinking sampler
#   bash scripts/run_aime_localize.sh aime25    # same table on AIME 2025
#
# Env: CACHE= path to math1k/cache.pt for isolated_frozen (Jiayi line 4).
#      SEAL_VECTOR= gsm8k L28 vector for isolated_frozen_seal.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 2
if [[ -f /workspace/env_native.sh ]]; then
  source /root/venv/bin/activate 2>/dev/null || true
  source /workspace/env_native.sh 2>/dev/null || true
  PY=${PY:-/root/venv/bin/python}
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY=${PY:-"$REPO/.venv/bin/python"}
else
  PY=${PY:-python}
fi
export HF_HOME=${HF_HOME:-${REPO}/.cache/huggingface}

ROOT_DIR=${ROOT_DIR:-${REPO}/artifacts/aime_localize}
CACHE=${CACHE:-${REPO}/artifacts/math_ladder/math1k/cache.pt}
SEAL_VECTOR=${SEAL_VECTOR:-${REPO}/artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt}
K=${K:-10}
SEED=${SEED:-42}
LOG=${LOG:-${ROOT_DIR}/localize.log}
mkdir -p "$ROOT_DIR"

run_py () {
  echo "[localize] $* $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_aime_localize.py --out_dir "$ROOT_DIR" --seed "$SEED" --k "$K" "$@"; then
    echo "[localize] OK $(date)" | tee -a "$LOG"
  else
    echo "[localize] FAIL exit=$? $(date)" | tee -a "$LOG"
    return 1
  fi
}

smoke () {
  run_py --mode smoke
}

focus () {
  local idx=${INDICES:-0,1,2,4,10,18}
  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192
  run_py --mode isolated --task aime2024 --indices "$idx" --n 6 --judger_budget 8192
  if [[ -f "$CACHE" ]]; then
    run_py --mode isolated_frozen --task aime2024 --indices "$idx" --n 6 \
      --judger_budget 8192 --cache "$CACHE"
  else
    echo "[localize] no frozen cache at $CACHE — skip isolated_frozen" | tee -a "$LOG"
  fi
  run_py --mode report
}

full () {
  run_py --mode collect --task aime2024 --n 30 --indices 0-29 --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192
  run_py --mode isolated --task aime2024 --n 30 --indices 0-29 --judger_budget 8192
  if [[ -f "$CACHE" ]]; then
    run_py --mode isolated_frozen --task aime2024 --n 30 --indices 0-29 \
      --judger_budget 8192 --cache "$CACHE"
  fi
  run_py --mode report
}

qwen () {
  # Jiayi: Qwen3 thinking sampler, not greedy. Decode Real tapes only.
  run_py --mode views --task aime2024 --judger_budget 8192 \
    --view_arms real --temperature 0.6 --top_p 0.95 --top_k 20
  run_py --mode report
}

aime25 () {
  local out="${ROOT_DIR}_aime25"
  mkdir -p "$out"
  local save=$ROOT_DIR
  ROOT_DIR=$out
  LOG=$out/localize.log
  run_py --mode collect --task aime2025 --n 30 --indices 0-29 --judger_budget 8192
  run_py --mode views --task aime2025 --judger_budget 8192 \
    --view_arms none,real,c1,c2,c3,evict_seg
  run_py --mode report
  ROOT_DIR=$save
}

stage=${1:-focus}
case "$stage" in
  smoke) smoke ;;
  focus) focus ;;
  full) full ;;
  qwen) qwen ;;
  aime25) aime25 ;;
  report) run_py --mode report ;;
  collect) run_py --mode collect --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" ;;
  views) run_py --mode views ;;
  isolated) run_py --mode isolated --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" ;;
  *) echo "usage: $0 smoke|focus|full|qwen|aime25|report" >&2; exit 2 ;;
esac
echo "[localize] DONE $stage $(date)" | tee -a "$LOG"
cat "${ROOT_DIR}/CHECKIN.md" 2>/dev/null || true
