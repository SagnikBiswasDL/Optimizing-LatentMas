#!/usr/bin/env bash
# Math ladder on the pod. Precompute is once; evals are restart-safe.
#
#   bash scripts/run_math_ladder.sh build          # 1000 MATH-train tapes + Mean-Replay
#   bash scripts/run_math_ladder.sh gate           # frozen vs Real vs None on GSM8K/MATH/AIME
#   bash scripts/run_math_ladder.sh ttc            # residual K=2,5 if the gate said so
#   bash scripts/run_math_ladder.sh evict          # light eviction on the kept arm
#   bash scripts/run_math_ladder.sh all
#
# GPU needed from `build` onward. Local CPU can only run tests/test_math_ladder.py.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 2
if [[ -f /workspace/env_native.sh ]]; then
  source /root/venv/bin/activate 2>/dev/null || true
  source /workspace/env_native.sh 2>/dev/null || true
  PY=${PY:-/root/venv/bin/python}
else
  PY=${PY:-python}
fi
export HF_HOME=${HF_HOME:-${REPO}/.cache/huggingface}

CACHE_DIR=${CACHE_DIR:-${REPO}/artifacts/math_ladder/math1k}
CACHE=${CACHE:-${CACHE_DIR}/cache.pt}
STAT_N=${STAT_N:-1000}
K=${K:-10}
SEED=${SEED:-42}
LOG=${LOG:-${REPO}/artifacts/math_ladder/ladder.log}
mkdir -p "$(dirname "$LOG")" "$CACHE_DIR"

run_job () {
  local tag=$1; shift
  local out=$1; shift
  if [[ -f "$out/report.json" ]]; then
    echo "[ladder] SKIP $tag (exists) $(date)" | tee -a "$LOG"
    return 0
  fi
  echo "[ladder] START $tag $(date)" | tee -a "$LOG"
  mkdir -p "$out"
  if "$PY" -u scripts/exp_math_ladder.py --out_dir "$out" --seed "$SEED" --k "$K" "$@"; then
    echo "[ladder] OK $tag $(date)" | tee -a "$LOG"
  else
    echo "[ladder] FAIL $tag exit=$? $(date)" | tee -a "$LOG"
    return 1
  fi
}

stage=${1:-gate}

build () {
  if [[ -f "$CACHE" ]]; then
    echo "[ladder] SKIP build (cache exists: $CACHE)" | tee -a "$LOG"
    return 0
  fi
  echo "[ladder] BUILD MATH-train n=${STAT_N} -> $CACHE_DIR $(date)" | tee -a "$LOG"
  "$PY" -u scripts/exp_math_ladder.py --mode build \
    --stat_n "$STAT_N" --k "$K" --seed "$SEED" \
    --out_dir "$CACHE_DIR"
}

gate () {
  [[ -f "$CACHE" ]] || { echo "missing $CACHE — run: $0 build" >&2; exit 2; }
  run_job gate_gsm8k "${REPO}/artifacts/math_ladder/gate_gsm8k" \
    --mode eval --cache "$CACHE" --task gsm8k --n 100 \
    --arms none,frozen,real --generate_bs 20 --judger_budget 1024
  run_job gate_math "${REPO}/artifacts/math_ladder/gate_math" \
    --mode eval --cache "$CACHE" --task math --n 100 \
    --arms none,frozen,real --generate_bs 20 --judger_budget 2048
  run_job gate_aime "${REPO}/artifacts/math_ladder/gate_aime24" \
    --mode eval --cache "$CACHE" --task aime2024 --n 30 \
    --arms none,frozen,real --generate_bs 1 --judger_budget 8192
}

ttc () {
  [[ -f "$CACHE" ]] || { echo "missing $CACHE — run: $0 build" >&2; exit 2; }
  run_job ttc_gsm8k "${REPO}/artifacts/math_ladder/ttc_gsm8k" \
    --mode eval --cache "$CACHE" --task gsm8k --n 100 \
    --k_ttc 0,2,5 --arms frozen,frozen_k2,frozen_k5,real \
    --generate_bs 20 --judger_budget 1024
  run_job ttc_math "${REPO}/artifacts/math_ladder/ttc_math" \
    --mode eval --cache "$CACHE" --task math --n 100 \
    --k_ttc 0,2,5 --arms frozen,frozen_k2,frozen_k5,real \
    --generate_bs 20 --judger_budget 2048
  run_job ttc_aime "${REPO}/artifacts/math_ladder/ttc_aime24" \
    --mode eval --cache "$CACHE" --task aime2024 --n 30 \
    --k_ttc 0,2,5 --arms frozen,frozen_k2,frozen_k5,real \
    --generate_bs 1 --judger_budget 8192
}

aime_signal () {
  [[ -f "$CACHE" ]] || { echo "missing $CACHE — run: $0 build" >&2; exit 2; }
  # New arms only. Gate already has none / frozen / real on the same 30 items.
  run_job aime_signal "${REPO}/artifacts/math_ladder/aime_signal" \
    --mode eval --cache "$CACHE" --task aime2024 --n 30 \
    --k_ttc 2,5 --arms frozen_k2,frozen_k5,k2,k5 \
    --generate_bs 4 --judger_budget 8192
}

evict () {
  [[ -f "$CACHE" ]] || { echo "missing $CACHE — run: $0 build" >&2; exit 2; }
  local kttc=${K_TTC:-0}
  local frozen_arm="frozen"
  local evict64="frozen_evict64"
  local evict128="frozen_evict128"
  if [[ "$kttc" != "0" ]]; then
    frozen_arm="frozen_k${kttc}"
    evict64="frozen_k${kttc}_evict64"
    evict128="frozen_k${kttc}_evict128"
  fi
  run_job evict_gsm8k "${REPO}/artifacts/math_ladder/evict_gsm8k" \
    --mode eval --cache "$CACHE" --task gsm8k --n 100 \
    --k_ttc "$kttc" --evict_budget 0,64 \
    --generate_bs 20 --judger_budget 1024 \
    --arms "${frozen_arm},${evict64},real"
  run_job evict_math "${REPO}/artifacts/math_ladder/evict_math" \
    --mode eval --cache "$CACHE" --task math --n 100 \
    --k_ttc "$kttc" --evict_budget 0,64,128 \
    --generate_bs 20 --judger_budget 2048 \
    --arms "${frozen_arm},${evict64},${evict128},real"
}

case "$stage" in
  build) build ;;
  gate) gate ;;
  ttc) ttc ;;
  evict) evict ;;
  aime_signal) aime_signal ;;
  all) build && gate && ttc && evict ;;
  *) echo "usage: $0 build|gate|ttc|evict|aime_signal|all" >&2; exit 2 ;;
esac
echo "[ladder] DONE $stage $(date)" | tee -a "$LOG"
