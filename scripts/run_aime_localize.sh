#!/usr/bin/env bash
# AIME localization: which silent-agent writes the Judger actually uses,
# then role-budgeted eviction. Restart-safe. GPU from `collect` onward.
#
#   bash scripts/run_aime_localize.sh smoke
#   bash scripts/run_aime_localize.sh quick     # ~20 min: Real arm + token-budget curve
#   bash scripts/run_aime_localize.sh focus     # items 0,1,2,4,10,18, all arms (~3h)
#   bash scripts/run_aime_localize.sh full      # n=30
#   bash scripts/run_aime_localize.sh qwen      # Real decode at Qwen thinking sampler
#   bash scripts/run_aime_localize.sh aime25    # same table on AIME 2025
#
# Env: CACHE= path to math1k/cache.pt for isolated_frozen (Jiayi line 4).
#      SEAL_VECTOR= gsm8k L28 vector for isolated_frozen_seal.
#      TAPE_DIR= local disk for big KV tapes (network mounts corrupt them).
#      PERSIST_DIR= durable dir for small artifacts; also where DONE_<stage>
#        lands. Poll that sentinel to know when it is safe to stop the pod.
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$REPO" || exit 2
# Interpreter: first venv that actually has torch wins. Pod layout has moved
# between /root/venv and /workspace/venv across migrations, so probe instead
# of hardcoding.
if [[ -z ${PY:-} ]]; then
  for cand in /workspace/venv/bin/python /root/venv/bin/python "$REPO/.venv/bin/python" \
              "$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)"; do
    if [[ -x $cand ]] && "$cand" -c 'import torch' >/dev/null 2>&1; then PY=$cand; break; fi
  done
fi
if [[ -z ${PY:-} ]]; then echo "[localize] no python with torch found" >&2; exit 2; fi
[[ -f /workspace/env_native.sh ]] && source /workspace/env_native.sh 2>/dev/null || true
cd "$REPO" || exit 2
echo "[localize] PY=$PY"
export HF_HOME=${HF_HOME:-${REPO}/.cache/huggingface}

ROOT_DIR=${ROOT_DIR:-${REPO}/artifacts/aime_localize}
# /workspace on the pod is a network volume (MooseFS). Large torch.save writes
# corrupt there, so tapes go to container-local disk; but container-local disk is
# wiped when the pod stops, so the small artifacts get mirrored back (see
# PERSIST_DIR). Treat /workspace as networked when it is a different device to /.
WS_IS_NETWORK=0
if [[ -d /workspace ]]; then
  if [[ $(df -P /workspace / 2>/dev/null | awk 'NR>1{print $1}' | sort -u | wc -l) -gt 1 ]]; then
    WS_IS_NETWORK=1
  fi
fi
if [[ -z ${TAPE_DIR:-} ]]; then
  if [[ $WS_IS_NETWORK -eq 1 ]]; then
    TAPE_DIR=/root/aime_localize_tapes
  else
    TAPE_DIR=${ROOT_DIR}/tapes
  fi
fi
export TAPE_DIR
mkdir -p "$TAPE_DIR"
CACHE=${CACHE:-${REPO}/artifacts/math_ladder/math1k/cache.pt}
SEAL_VECTOR=${SEAL_VECTOR:-${REPO}/artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt}
K=${K:-10}
SEED=${SEED:-42}
LOG=${LOG:-${ROOT_DIR}/localize.log}
mkdir -p "$ROOT_DIR"

# Mirror the small artifacts somewhere that outlives the container. Pod-local
# disk is wiped when the pod stops, and that is how a finished sweep got lost.
# Set PERSIST_DIR to a network-volume path; tapes are deliberately not mirrored.
if [[ -z ${PERSIST_DIR:-} ]] && [[ $WS_IS_NETWORK -eq 1 ]] && \
   [[ $ROOT_DIR != /workspace/* ]]; then
  PERSIST_DIR=/workspace/aime_localize_results
fi
PERSIST_DIR=${PERSIST_DIR:-}
export PERSIST_DIR
echo "[localize] ROOT_DIR=$ROOT_DIR TAPE_DIR=$TAPE_DIR PERSIST_DIR=${PERSIST_DIR:-none}"
persist () {
  [[ -n $PERSIST_DIR ]] || return 0
  mkdir -p "$PERSIST_DIR" 2>/dev/null || return 0
  # small files only: rows/times/texts/reports. Many small writes are fine on
  # the network mount; it is large seeking writes that corrupt.
  rsync -a --exclude 'tapes' --exclude '*.pt' "$ROOT_DIR"/ "$PERSIST_DIR"/ 2>/dev/null \
    || cp -r "$ROOT_DIR"/* "$PERSIST_DIR"/ 2>/dev/null || true
}

run_py () {
  echo "[localize] $* $(date)" | tee -a "$LOG"
  if "$PY" -u scripts/exp_aime_localize.py --out_dir "$ROOT_DIR" --tape_dir "$TAPE_DIR" \
      --persist_dir "$PERSIST_DIR" --seed "$SEED" --k "$K" "$@"; then
    echo "[localize] OK $(date)" | tee -a "$LOG"
    persist
  else
    echo "[localize] FAIL exit=$? $(date)" | tee -a "$LOG"
    persist
    return 1
  fi
}

# Loud, machine-checkable completion marker. The pod cannot stop itself
# (runpodctl has no API key here), and an idle H200 burns credits, so the
# sentinel lands on the durable volume where it can be polled from outside.
finish () {
  local stage=$1
  local dest=${PERSIST_DIR:-$ROOT_DIR}
  mkdir -p "$dest" 2>/dev/null || true
  {
    echo "stage=$stage"
    echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "elapsed_s=$SECONDS"
  } > "$dest/DONE_${stage}" 2>/dev/null || true
  {
    echo ""
    echo "=================================================================="
    echo "  RUN COMPLETE: $stage   (elapsed ${SECONDS}s)"
    echo "  >>> STOP THE POD NOW — nothing further will run. <<<"
    echo "  sentinel: $dest/DONE_${stage}"
    echo "=================================================================="
  } | tee -a "$LOG"
}

smoke () {
  run_py --mode smoke
}

# Cheap-first: the only arm that targets the 99.4% (Judger decode). ~20 min on
# an H200 — collect is seconds, 6 real decodes dominate, budget replay is CPU.
quick () {
  local idx=${INDICES:-0,1,2,4,10,18}
  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192 --view_arms real
  run_py --mode report
  run_py --mode budget --task aime2024 --budget_arms real
  run_py --mode loops --task aime2024 --budget_arms real
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
  run_py --mode budget --task aime2024 --budget_arms real,c3,none
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

stage=${1:-quick}
case "$stage" in
  smoke) smoke ;;
  quick) quick ;;
  focus) focus ;;
  full) full ;;
  qwen) qwen ;;
  aime25) aime25 ;;
  report) run_py --mode report ;;
  budget) run_py --mode budget --task aime2024 --budget_arms "${BUDGET_ARMS:-real}" ;;
  collect) run_py --mode collect --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" ;;
  views) run_py --mode views ;;
  isolated) run_py --mode isolated --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" ;;
  *) echo "usage: $0 quick|smoke|focus|full|qwen|aime25|budget|report" >&2; exit 2 ;;
esac
echo "[localize] DONE $stage $(date)" | tee -a "$LOG"
persist
cat "${ROOT_DIR}/CHECKIN.md" 2>/dev/null || true
cat "${ROOT_DIR}/BUDGET.md" 2>/dev/null || true
finish "$stage"
