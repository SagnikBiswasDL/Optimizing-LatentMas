#!/usr/bin/env bash
# Stop this pod when its GPU has been idle for IDLE_MIN consecutive minutes.
#
#   setsid nohup bash scripts/pod/watchdog.sh > /root/logs/watchdog.log 2>&1 < /dev/null &
#
# Covers a finished queue, a crashed job and an absent operator alike. The pod has
# no persistent volume: copy /root/out off the pod before the idle limit runs out.
set -u
IDLE_MIN=${IDLE_MIN:-40}
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null 2>&1
idle=0
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  if [ "${util:-0}" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
  echo "$(date -u +%FT%TZ) util=$util idle_min=$idle"
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    echo "$(date -u +%FT%TZ) idle for $idle minutes; stopping pod $RUNPOD_POD_ID"
    runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
    sleep 300
  fi
  sleep 60
done
