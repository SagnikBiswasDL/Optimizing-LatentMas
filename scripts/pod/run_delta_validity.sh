#!/usr/bin/env bash
# Detached validity screen. Safe to start twice: the python script resumes
# from the jsonl files under /workspace/delta_exit_validity.
set -eu
if [ -d /workspace ]; then
  mkdir -p /workspace/hf /workspace/delta_exit_validity
  export HF_HOME=/workspace/hf
else
  mkdir -p /root/hf /root/delta_exit_validity
  export HF_HOME=/root/hf
fi
export HF_HUB_ENABLE_HF_TRANSFER=0
export OUT=/workspace/delta_exit_validity
export PYTHONUNBUFFERED=1
if [ ! -d /root/venv ]; then
  python3 -m venv --system-site-packages /root/venv
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate
pip install -q --no-cache-dir transformers==4.57.1 datasets math-verify==0.9.0
cd /root/work/repo
exec python scripts/delta_exit_validity.py
