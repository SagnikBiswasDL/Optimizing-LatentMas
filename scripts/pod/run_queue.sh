#!/usr/bin/env bash
# Run the commands in a jobs file one after another on the pod.
#
#   setsid nohup bash scripts/pod/run_queue.sh /root/jobs.txt > /root/logs/queue.log 2>&1 < /dev/null &
#
# Each non-empty, non-comment line is "<name> <command...>"; $M is the model
# snapshot. A failing job does not stop the queue; start, end and exit status of
# each job go to /root/out/queue_status.txt, and /root/out/QUEUE_DONE is written
# at the end. Stopping an idle pod is the job of scripts/pod/watchdog.sh.
set -u
JOBS=${1:?jobs file}
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export M=${M:-/root/hf/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1}
source /root/venv/bin/activate
cd /root/work/repo
mkdir -p /root/out /root/logs
rm -f /root/out/QUEUE_DONE
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  name=${line%% *}; cmd=${line#* }
  echo "=== $(date -u +%FT%TZ) start $name" | tee -a /root/out/queue_status.txt
  bash -c "$cmd" > "/root/logs/$name.log" 2>&1
  echo "=== $(date -u +%FT%TZ) end $name rc=$?" | tee -a /root/out/queue_status.txt
done < "$JOBS"
date -u +%FT%TZ > /root/out/QUEUE_DONE
