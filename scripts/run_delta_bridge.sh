#!/usr/bin/env bash
# DeltaBridge campaign on the pod. Restart-safe. Writes CHECKIN.md after
# every stage so you can decide whether the next GPU hours are worth it.
#
#   bash scripts/run_delta_bridge.sh              # collect → fit → MATH → GSM8K → AIME
#   bash scripts/run_delta_bridge.sh collect
#   bash scripts/run_delta_bridge.sh fit
#   bash scripts/run_delta_bridge.sh math
#   bash scripts/run_delta_bridge.sh aime
#   bash scripts/run_delta_bridge.sh gate          # refresh CHECKIN.md
#   bash scripts/run_delta_bridge.sh train        # only if gate says go
#
# Env:
#   STOP_BEFORE_AIME=1   stop after MATH/GSM8K (recommended first check-in)
#   CACHE=...             path to math1k/cache.pt
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

ROOT_DIR=${ROOT_DIR:-${REPO}/artifacts/delta_bridge}
CACHE_DIR=${CACHE_DIR:-${REPO}/artifacts/math_ladder/math1k}
CACHE=${CACHE:-${CACHE_DIR}/cache.pt}
STAT_N=${STAT_N:-1000}
K=${K:-10}
SEED=${SEED:-42}
LOG=${LOG:-${ROOT_DIR}/delta_bridge.log}
STOP_BEFORE_AIME=${STOP_BEFORE_AIME:-0}
mkdir -p "$ROOT_DIR"

run_py () {
  echo "[delta] $* $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_delta_bridge.py --root_dir "$ROOT_DIR" --cache "$CACHE" --seed "$SEED" --k "$K" "$@"; then
    echo "[delta] OK $(date)" | tee -a "$LOG"
  else
    echo "[delta] FAIL exit=$? $(date)" | tee -a "$LOG"
    return 1
  fi
}

need_cache () {
  if [[ -f "$CACHE" ]]; then
    echo "[delta] cache ok: $CACHE" | tee -a "$LOG"
    return 0
  fi
  echo "[delta] missing $CACHE — building Mean-Replay (~17 min)" | tee -a "$LOG"
  bash "$REPO/scripts/run_math_ladder.sh" build
}

collect () {
  need_cache
  run_py --mode collect --out_dir "$ROOT_DIR" --stat_n "$STAT_N"
}

fit () {
  run_py --mode fit --out_dir "$ROOT_DIR"
}

eval_math () {
  local out="${ROOT_DIR}/eval_math"
  if [[ -f "$out/report.json" ]]; then
    echo "[delta] SKIP math (report exists)" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$out"
  run_py --mode eval --task math --n 100 --judger_budget 2048 \
    --out_dir "$out" --arms zero,oracle_full,oracle_r,shuffled
}

eval_gsm8k () {
  local out="${ROOT_DIR}/eval_gsm8k"
  if [[ -f "$out/report.json" ]]; then
    echo "[delta] SKIP gsm8k (report exists)" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$out"
  run_py --mode eval --task gsm8k --n 100 --judger_budget 1024 \
    --out_dir "$out" --arms zero,oracle_full,oracle_r,shuffled
}

eval_aime () {
  local out="${ROOT_DIR}/eval_aime24"
  if [[ -f "$out/report.json" ]]; then
    echo "[delta] SKIP aime (report exists)" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$out"
  run_py --mode eval --task aime2024 --n 30 --judger_budget 8192 \
    --out_dir "$out" --arms zero,oracle_full,oracle_r,shuffled
}

gate () {
  run_py --mode gate --out_dir "$ROOT_DIR"
  echo "----- CHECKIN -----" | tee -a "$LOG"
  cat "$ROOT_DIR/CHECKIN.md" | tee -a "$LOG"
}

math_protocol_ok () {
  "$PY" - "$ROOT_DIR/eval_math/report.json" <<'PY'
import json, sys
p = sys.argv[1]
if not __import__("os").path.isfile(p):
    sys.exit(0)
r = json.load(open(p))
z = (r.get("arms") or {}).get("zero") or {}
acc = z.get("acc")
if acc is None:
    sys.exit(0)
if acc < 0.60:
    print(f"[delta] MATH zero acc={acc:.3f} < 0.60 — protocol broken, stop before AIME", flush=True)
    sys.exit(2)
print(f"[delta] MATH zero acc={acc:.3f} — protocol ok", flush=True)
PY
}

train () {
  local g="${ROOT_DIR}/gate.json"
  if [[ -f "$g" ]]; then
    rec=$("$PY" -c "import json; print(json.load(open('$g')).get('recommend',''))")
    if [[ "$rec" != "go" ]]; then
      echo "[delta] refuse train: recommend=$rec (need go)" | tee -a "$LOG"
      return 2
    fi
  fi
  run_py --mode train_coef --out_dir "$ROOT_DIR"
}

eval_predicted () {
  local out="${ROOT_DIR}/eval_predicted_aime24"
  if [[ -f "$out/report.json" ]]; then
    echo "[delta] SKIP predicted aime (report exists)" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$out"
  run_py --mode eval --task aime2024 --n 30 --judger_budget 8192 \
    --out_dir "$out" --arms zero,predicted
}

oracle () {
  collect
  fit
  eval_math
  gate
  math_protocol_ok || { gate; exit 2; }
  eval_gsm8k
  gate
  if [[ "$STOP_BEFORE_AIME" == "1" ]]; then
    echo "[delta] STOP_BEFORE_AIME=1 — check CHECKIN.md then rerun: $0 aime" | tee -a "$LOG"
    exit 0
  fi
  eval_aime
  gate
}

stage=${1:-oracle}
case "$stage" in
  collect) collect; gate ;;
  fit) fit; gate ;;
  math) need_cache; eval_math; gate ;;
  gsm8k) need_cache; eval_gsm8k; gate ;;
  aime) need_cache; eval_aime; gate ;;
  train) train; gate ;;
  predicted) eval_predicted; gate ;;
  gate|checkin) gate ;;
  oracle|all) oracle ;;
  *) echo "usage: $0 collect|fit|math|gsm8k|aime|train|predicted|gate|oracle" >&2; exit 2 ;;
esac
echo "[delta] DONE $stage $(date)" | tee -a "$LOG"
