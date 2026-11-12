#!/usr/bin/env bash
# Environment for the looped-serving runs on a fresh runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 pod.
# Pins: configs/looped_serving/pins.json. Run once; safe to rerun.
set -ex
# A network volume at /workspace outlives the pod. Results, logs and the model
# snapshot go there; /root/out and /root/logs are links, so the queue scripts
# keep their paths. Without a volume everything stays on the container disk.
if [ -d /workspace ]; then
  mkdir -p /workspace/hf /workspace/out /workspace/logs
  ln -sfn /workspace/out /root/out
  ln -sfn /workspace/logs /root/logs
  export HF_HOME=/workspace/hf
else
  mkdir -p /root/out /root/logs
  export HF_HOME=/root/hf
fi
export HF_HUB_ENABLE_HF_TRANSFER=0
[ -d /root/venv ] || python3 -m venv --system-site-packages /root/venv   # the image's system Python refuses pip installs (PEP 668)
source /root/venv/bin/activate
pip install -q --no-cache-dir transformers==4.57.1 datasets math-verify==0.9.0 pytest 2>&1 | tail -3
mkdir -p /root/work
cd /root/work
[ -d vllm-rlt ] || git clone -q https://github.com/InferMatrixLab/vllm-rlt
cd vllm-rlt
git checkout -q e0a5515ec970d634546d2e1d38b3ba3602072884
pip install -q --no-cache-dir -e '.[dev]' 2>&1 | tail -5
python - <<'PY'
import importlib.metadata as md
import torch
from huggingface_hub import snapshot_download
print("torch", torch.__version__, "transformers", md.version("transformers"), "vllm-rlt", md.version("vllm-rlt"),
      "math-verify", md.version("math-verify"), "gpu", torch.cuda.get_device_name(0))
print(snapshot_download("ByteDance/Ouro-1.4B", revision="574fa66cb8bf5abdc979642d01cf2b79b16bfab1"))
PY
echo SETUP_DONE
