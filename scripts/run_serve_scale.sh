#!/usr/bin/env bash
# Stock vLLM vs tail-only preemption, serving LatentMAS on AIME 2024+2025.
#
#   bash scripts/run_serve_scale.sh smoke   # ~5 min: forced preemption, both modes
#   bash scripts/run_serve_scale.sh full    # stock at 8,16,32,60 then tail at 16,32,60
#   bash scripts/run_serve_scale.sh all     # smoke, then full only if smoke passes
#
# Env: OUT= results dir (default /workspace/serve_scale on a pod).
#      LEVEL_TIMEOUT_S= cap per concurrency level (default 3600).
#      STOCK_LEVELS= / TAIL_LEVELS= override the concurrency lists.
# Writes DONE_<stage> into OUT when finished. The pod cannot stop itself.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 2
if [[ -z ${PY:-} ]]; then
  for cand in /workspace/venv/bin/python /root/venv/bin/python "$(command -v python3)"; do
    if [[ -x $cand ]] && "$cand" -c 'import vllm' >/dev/null 2>&1; then PY=$cand; break; fi
  done
fi
[[ -n ${PY:-} ]] || { echo "[serve] no python with vllm" >&2; exit 2; }
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
OUT=${OUT:-/workspace/serve_scale}
LEVEL_TIMEOUT_S=${LEVEL_TIMEOUT_S:-3600}
STOCK_LEVELS=${STOCK_LEVELS:-8,16,32,60}
TAIL_LEVELS=${TAIL_LEVELS:-16,32,60}
mkdir -p "$OUT"
LOG=$OUT/run.log
echo "[serve] PY=$PY OUT=$OUT $(git log --oneline -1)" | tee -a "$LOG"
"$PY" -c "import vllm;print('[serve] vllm', vllm.__version__)" 2>&1 | tee -a "$LOG"

run () {
  echo "[serve] $* $(date)" | tee -a "$LOG"
  "$PY" -u scripts/serve_scale.py --out "$OUT" "$@" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

# The smoke passes only if tail mode actually pinned and later reused blocks,
# and every request in both modes finished.
smoke () {
  run --mode stock --smoke || return 1
  run --mode tail --smoke || return 1
  "$PY" - "$OUT" <<'EOF' 2>&1 | tee -a "$LOG"
import json, os, sys
out = sys.argv[1]
ok = True
for mode in ("stock", "tail"):
    s = json.load(open(os.path.join(out, f"summary_{mode}_smoke.json")))["levels"][-1]
    ev = s["events"]
    print(f"[smoke] {mode}: finished {s['n_finished']}/{s['n']} preempt={ev['preempt']} "
          f"resume={ev['resume']} recompute={ev['recompute_tokens']} hit={ev['hit_tokens']} "
          f"trimmed_blocks={ev['trimmed_blocks']} wall={s['wall_s']:.1f}s")
    ok &= s["n_finished"] == s["n"]
    ok &= ev["preempt"] > 0
    if mode == "tail":
        ok &= ev["pinned_blocks_at_preempt"] > 0 and ev["hit_tokens"] > 0
print("[smoke] PASS" if ok else "[smoke] FAIL")
sys.exit(0 if ok else 1)
EOF
  return "${PIPESTATUS[0]}"
}

full () {
  run --mode stock --concurrency "$STOCK_LEVELS" --level-timeout-s "$LEVEL_TIMEOUT_S" || return 1
  run --mode tail --concurrency "$TAIL_LEVELS" --level-timeout-s "$LEVEL_TIMEOUT_S" \
    --embeds "$OUT/embeds.pt" || return 1
}

finish () {
  { echo "stage=$1"; echo "status=$2"; echo "finished_at=$(date -u +%FT%TZ)"; echo "elapsed_s=$SECONDS"; } \
    > "$OUT/DONE_$1"
  echo "[serve] ===== $1 $2 in ${SECONDS}s. STOP THE POD. =====" | tee -a "$LOG"
}

stage=${1:-all}
case "$stage" in
  smoke) if smoke; then finish smoke ok; else finish smoke fail; fi ;;
  full) if full; then finish full ok; else finish full fail; fi ;;
  all)
    if smoke; then
      if full; then finish all ok; else finish all fail_full; fi
    else
      finish all fail_smoke
    fi ;;
  *) echo "usage: $0 smoke|full|all" >&2; exit 2 ;;
esac
