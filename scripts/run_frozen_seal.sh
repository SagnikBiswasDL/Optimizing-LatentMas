#!/usr/bin/env bash
# Lock-in: Frozen MATH-1k + Judger SEAL coef 40. Kill on accuracy drop.
#
#   bash scripts/run_frozen_seal.sh
#   bash scripts/run_frozen_seal.sh gsm8k
#   bash scripts/run_frozen_seal.sh math
#   bash scripts/run_frozen_seal.sh aime6
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 2
if [[ -f /workspace/env_native.sh ]]; then
  source /root/venv/bin/activate 2>/dev/null || true
  source /workspace/env_native.sh 2>/dev/null || true
  PY=${PY:-/root/venv/bin/python}
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY=${PY:-$REPO/.venv/bin/python}
else
  PY=${PY:-python}
fi
export HF_HOME=${HF_HOME:-${REPO}/.cache/huggingface}

ROOT_DIR=${ROOT_DIR:-${REPO}/artifacts/frozen_seal}
CACHE=${CACHE:-${REPO}/artifacts/math_ladder/math1k/cache.pt}
VEC=${VEC:-${REPO}/artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt}
COEF=${COEF:-40}
SEED=${SEED:-42}
SKIP_EXISTING=${SKIP_EXISTING:-1}
LOG=${LOG:-${ROOT_DIR}/frozen_seal.log}
mkdir -p "$ROOT_DIR"

run_py () {
  echo "[fseal] $* $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_frozen_seal.py --root_dir "$ROOT_DIR" --cache "$CACHE" \
      --seal_vector "$VEC" --seal_coef "$COEF" --seed "$SEED" "$@"; then
    echo "[fseal] OK $(date)" | tee -a "$LOG"
  else
    echo "[fseal] FAIL exit=$? $(date)" | tee -a "$LOG"
    return 1
  fi
}

need_cache () {
  [[ -f "$CACHE" ]] || { echo "missing $CACHE" >&2; exit 2; }
  [[ -f "$VEC" ]] || { echo "missing $VEC" >&2; exit 2; }
}

recommend () {
  local g="${ROOT_DIR}/gate.json"
  [[ -f "$g" ]] || { echo running; return 0; }
  "$PY" -c "import json; print(json.load(open('$g')).get('recommend','running'))"
}

refuse_if_stop () {
  local rec
  rec=$(recommend)
  if [[ "$rec" == "stop" ]]; then
    echo "[fseal] gate says stop" | tee -a "$LOG"
    cat "${ROOT_DIR}/CHECKIN.md" 2>/dev/null | tee -a "$LOG" || true
    return 2
  fi
}

skip_report () {
  local out=$1
  if [[ "$SKIP_EXISTING" == "1" && -f "$out/report.json" ]]; then
    echo "[fseal] SKIP $out" | tee -a "$LOG"
    return 0
  fi
  return 1
}

eval_task () {
  local tag=$1 task=$2 n=$3 bs=$4 budget=$5 idx=${6:-}
  need_cache
  refuse_if_stop || return 2
  local out="${ROOT_DIR}/${tag}"
  if skip_report "$out"; then
    run_py --mode gate
    return 0
  fi
  mkdir -p "$out"
  local extra=()
  if [[ -n "$idx" ]]; then
    extra+=(--indices "$idx")
  fi
  run_py --mode eval --task "$task" --n "$n" --generate_bs "$bs" \
    --judger_budget "$budget" --out_dir "$out" "${extra[@]}"
}

ladder () {
  eval_task gsm8k_n40 gsm8k 40 4 1024 || return 1
  refuse_if_stop || return 2
  eval_task math_n40 math 40 4 2048 || return 1
  refuse_if_stop || return 2
  eval_task aime6 aime2024 6 1 8192 "0,1,2,3,4,5" || return 1
  refuse_if_stop || return 2
  eval_task aime12 aime2024 12 1 8192 "0-11" || return 1
  refuse_if_stop || return 2
  eval_task aime30 aime2024 30 1 8192 "0-29" || return 1
  run_py --mode gate
}

stage=${1:-ladder}
case "$stage" in
  gsm8k) eval_task gsm8k_n40 gsm8k 40 4 1024 ;;
  math) eval_task math_n40 math 40 4 2048 ;;
  aime6) eval_task aime6 aime2024 6 1 8192 "0,1,2,3,4,5" ;;
  aime12) eval_task aime12 aime2024 12 1 8192 "0-11" ;;
  aime30) eval_task aime30 aime2024 30 1 8192 "0-29" ;;
  gate) run_py --mode gate ;;
  ladder|all) ladder ;;
  *) echo "usage: $0 gsm8k|math|aime6|aime12|aime30|gate|ladder" >&2; exit 2 ;;
esac
echo "[fseal] DONE $stage $(date)" | tee -a "$LOG"
