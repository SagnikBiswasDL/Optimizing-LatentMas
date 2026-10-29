#!/usr/bin/env bash
# The five-long-request overflow (8064-token prompts, 32768 forced tokens each),
# stock vLLM preemption vs tail-only preemption, with prefix caching off and on.
#
#   bash scripts/run_jiayi_tail.sh            # all four arms, tail-off first
#   ARMS="tail_off stock_off" bash scripts/run_jiayi_tail.sh
#
# Env: OUT= results root (default /root/jiayi_tail). Writes OUT/DONE when finished.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 2
PY=${PY:-/root/venv311/bin/python}
OUT=${OUT:-/root/jiayi_tail}
ARMS=${ARMS:-"tail_off stock_off tail_on stock_on"}
mkdir -p "$OUT"
LOG=$OUT/run.log
echo "[jiayi] $(git log --oneline -1) arms=$ARMS $(date)" | tee -a "$LOG"
status=ok
for arm in $ARMS; do
  sched=${arm%_*}
  case ${arm#*_} in
    off) cache=--no-prefix-caching ;;
    on) cache=--prefix-caching ;;
    *) echo "[jiayi] bad arm $arm" | tee -a "$LOG"; exit 2 ;;
  esac
  if [[ -f $OUT/$arm/pressure.json ]]; then
    echo "[jiayi] $arm already done" | tee -a "$LOG"; continue
  fi
  echo "[jiayi] start $arm $(date)" | tee -a "$LOG"
  "$PY" -u scripts/jiayi_capacity.py --phase pressure --arm full --concurrency 5 \
    --scheduler "$sched" $cache --out "$OUT/$arm" > "$OUT/$arm.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    status=fail
    echo "[jiayi] $arm FAILED rc=$rc" | tee -a "$LOG"
    tail -20 "$OUT/$arm.log" | tee -a "$LOG"
    continue
  fi
  "$PY" - "$OUT/$arm/pressure.json" <<'EOF' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[jiayi] {sys.argv[1].split('/')[-2]}: e2e={d['e2e_s']:.0f}s preemptions={d['preemptions']} "
      f"all_32768={d['all_reached_32768']} events={d['events']}")
EOF
done
{ echo "status=$status"; echo "finished_at=$(date -u +%FT%TZ)"; echo "elapsed_s=$SECONDS"; } > "$OUT/DONE"
echo "[jiayi] DONE $status in ${SECONDS}s" | tee -a "$LOG"
