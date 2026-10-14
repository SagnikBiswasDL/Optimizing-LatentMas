#!/usr/bin/env bash
# Frozen MATH-1k + one live K=10 agent. Gated. No training. No PCA.
#
#   bash scripts/run_one_role.sh           # checkpoints 1→5, kill on fail
#   bash scripts/run_one_role.sh smoke
#   bash scripts/run_one_role.sh math
#   bash scripts/run_one_role.sh aime6
#   bash scripts/run_one_role.sh aime12
#   bash scripts/run_one_role.sh aime30
#   bash scripts/run_one_role.sh gate
#
# Env:
#   FORCE_ROLE=planner   skip MATH pick, use this role on AIME
#   CACHE=...            path to math1k/cache.pt
#   SKIP_EXISTING=1      default; set 0 to rerun a stage
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

ROOT_DIR=${ROOT_DIR:-${REPO}/artifacts/one_role}
CACHE_DIR=${CACHE_DIR:-${REPO}/artifacts/math_ladder/math1k}
CACHE=${CACHE:-${CACHE_DIR}/cache.pt}
K=${K:-10}
SEED=${SEED:-42}
SKIP_EXISTING=${SKIP_EXISTING:-1}
FORCE_ROLE=${FORCE_ROLE:-}
LOG=${LOG:-${ROOT_DIR}/one_role.log}
mkdir -p "$ROOT_DIR"

run_py () {
  echo "[one_role] $* $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_one_role.py --root_dir "$ROOT_DIR" --cache "$CACHE" --seed "$SEED" --k "$K" "$@"; then
    echo "[one_role] OK $(date)" | tee -a "$LOG"
  else
    echo "[one_role] FAIL exit=$? $(date)" | tee -a "$LOG"
    return 1
  fi
}

need_cache () {
  if [[ -f "$CACHE" ]]; then
    echo "[one_role] cache ok: $CACHE" | tee -a "$LOG"
    return 0
  fi
  echo "[one_role] missing $CACHE — building Mean-Replay (~17 min)" | tee -a "$LOG"
  bash "$REPO/scripts/run_math_ladder.sh" build
}

skip_report () {
  local out=$1
  if [[ "$SKIP_EXISTING" == "1" && -f "$out/report.json" ]]; then
    echo "[one_role] SKIP $out (report exists)" | tee -a "$LOG"
    return 0
  fi
  return 1
}

recommend () {
  local g="${ROOT_DIR}/gate.json"
  if [[ ! -f "$g" ]]; then
    echo "running"
    return 0
  fi
  "$PY" -c "import json; print(json.load(open('$g')).get('recommend','running'))"
}

best_arm () {
  if [[ -n "$FORCE_ROLE" ]]; then
    echo "frozen_${FORCE_ROLE}_k${K}"
    return 0
  fi
  local g="${ROOT_DIR}/gate.json"
  if [[ ! -f "$g" ]]; then
    echo "frozen_planner_k${K}"
    return 0
  fi
  "$PY" -c "import json; print(json.load(open('$g')).get('best_arm') or json.load(open('$g')).get('math',{}).get('best_arm') or 'frozen_planner_k${K}')"
}

refuse_if_stop () {
  local rec
  rec=$(recommend)
  if [[ "$rec" == "stop" ]]; then
    echo "[one_role] gate says stop — not spending more GPU" | tee -a "$LOG"
    cat "${ROOT_DIR}/CHECKIN.md" 2>/dev/null | tee -a "$LOG" || true
    return 2
  fi
  return 0
}

smoke () {
  need_cache
  local out="${ROOT_DIR}/smoke"
  if skip_report "$out"; then
    local ok
    ok=$("$PY" -c "import json; print(json.load(open('$out/report.json')).get('ok', False))")
    if [[ "$ok" != "True" ]]; then
      echo "[one_role] smoke report exists but ok=$ok — stop" | tee -a "$LOG"
      return 2
    fi
    run_py --mode gate
    return 0
  fi
  mkdir -p "$out"
  run_py --mode smoke --out_dir "$out" --judger_budget 256
}

math20 () {
  need_cache
  refuse_if_stop || return 2
  local out="${ROOT_DIR}/math_n20"
  if skip_report "$out"; then
    run_py --mode gate
    return 0
  fi
  mkdir -p "$out"
  run_py --mode eval --task math --n 20 --generate_bs 4 --judger_budget 2048 \
    --out_dir "$out" \
    --arms frozen,frozen_planner_k${K},frozen_refiner_k${K},frozen_critic_k${K}
}

aime_stage () {
  local n=$1
  local tag=$2
  local idx=$3
  need_cache
  refuse_if_stop || return 2
  local arm
  arm=$(best_arm)
  echo "[one_role] AIME n=${n} frozen + ${arm}" | tee -a "$LOG"
  local out="${ROOT_DIR}/${tag}"
  if skip_report "$out"; then
    run_py --mode gate
    return 0
  fi
  mkdir -p "$out"
  run_py --mode eval --task aime2024 --n "$n" --indices "$idx" \
    --generate_bs 1 --judger_budget 8192 \
    --out_dir "$out" --arms "frozen,${arm}"
}

ladder () {
  smoke || return 1
  refuse_if_stop || return 2
  math20 || return 1
  refuse_if_stop || return 2
  aime_stage 6 aime6 "0,1,2,3,4,5" || return 1
  refuse_if_stop || return 2
  aime_stage 12 aime12 "0,1,2,3,4,5,6,7,8,9,10,11" || return 1
  refuse_if_stop || return 2
  aime_stage 30 aime30 "0-29" || return 1
  run_py --mode gate
  echo "[one_role] ladder finished. SEAL is a separate stage after aime30 GO." | tee -a "$LOG"
}

stage=${1:-ladder}
case "$stage" in
  smoke) smoke ;;
  math|math20) math20 ;;
  aime6) aime_stage 6 aime6 "0,1,2,3,4,5" ;;
  aime12) aime_stage 12 aime12 "0,1,2,3,4,5,6,7,8,9,10,11" ;;
  aime30) aime_stage 30 aime30 "0-29" ;;
  gate|checkin) run_py --mode gate ;;
  ladder|all) ladder ;;
  seal)
    echo "[one_role] SEAL refused until aime30 GO. Do not pass --seal_coef yet." | tee -a "$LOG"
    exit 2
    ;;
  *) echo "usage: $0 smoke|math|aime6|aime12|aime30|gate|ladder" >&2; exit 2 ;;
esac
echo "[one_role] DONE $stage $(date)" | tee -a "$LOG"
