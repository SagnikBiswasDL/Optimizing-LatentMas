#!/usr/bin/env bash
# Bring a fresh RunPod box to a state where an experiment can start. Idempotent:
# safe to re-run on a pod that is already set up.
#
# Run it through the proxy, which takes no exec-form commands and no SCP:
#   RP_HOST=<id>-<user>@ssh.runpod.io scripts/rp.sh < scripts/pod_bootstrap.sh
#
# Encodes the things that cost time on 2026-09-23/24:
#   * HF_HOME must point at the volume, or the 28 GB model re-downloads into the
#     container and the quota blows.
#   * /workspace has a ~50 GB quota that `df` does not show (df reports the whole
#     MooseFS cluster). Exceeding it fails `git fetch` with "Disk quota exceeded".
#   * Large torch.save writes corrupt on the network mount, so tapes go to
#     container-local disk; container-local disk is wiped on stop, so results must
#     be harvested back.
#   * flash_attn is absent and is not worth installing for batch-1 decode.
set -u

REPO_DIR=${REPO_DIR:-/workspace/latentmas-baseline}
REPO_URL=${REPO_URL:-https://github.com/SagnikBiswasDL/Optimizing-LatentMas.git}
BRANCH=${BRANCH:-feature/seal-token-efficiency}
QUOTA_GB=${QUOTA_GB:-50}

say () { echo "[bootstrap] $*"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || {
  echo "[bootstrap] FATAL: no GPU visible" >&2; exit 1; }

# --- volume headroom -------------------------------------------------------
# du over the whole volume is slow but it is the only honest number; the quota is
# per-volume and invisible to df.
say "volume usage (this takes a moment; df would lie here)"
used_gb=$(du -sBG /workspace 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')
say "/workspace using ${used_gb}G of ~${QUOTA_GB}G quota"
if [ "${used_gb:-0}" -ge $(( QUOTA_GB - 4 )) ]; then
  say "within 4G of the quota — reclaiming the pip cache (fully regenerable)"
  rm -rf /workspace/.cache/pip
  used_gb=$(du -sBG /workspace 2>/dev/null | awk '{gsub("G","",$1); print $1+0}')
  say "now ${used_gb}G"
fi

# --- environment ----------------------------------------------------------
cat > /workspace/env_aime.sh <<'SH'
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/root/tmp
mkdir -p /root/tmp /root/logs
# Whichever venv actually has torch; the layout has moved between /root and
# /workspace across pod migrations.
for c in /workspace/venv/bin/python /root/venv/bin/python; do
  if [ -x "$c" ] && "$c" -c 'import torch' >/dev/null 2>&1; then export PY=$c; break; fi
done
export PY=${PY:-python3}
cd REPO_DIR_PLACEHOLDER
SH
sed -i "s#REPO_DIR_PLACEHOLDER#${REPO_DIR}#" /workspace/env_aime.sh
say "wrote /workspace/env_aime.sh"

# --- repo ------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  say "repo present; fetching"
  cd "$REPO_DIR"
  git remote set-url origin "$REPO_URL"
  # Stash rather than discard: a previous pod's uncommitted work has been found
  # here before, and it is cheap to keep.
  if ! git diff --quiet 2>/dev/null; then
    git stash push -u -m "pod-local before bootstrap $(date -u +%FT%TZ)" >/dev/null 2>&1 \
      && say "stashed pod-local edits"
  fi
  GIT_TERMINAL_PROMPT=0 git fetch origin "$BRANCH" 2>&1 | tail -2
  git checkout "$BRANCH" >/dev/null 2>&1
  git merge --ff-only "origin/$BRANCH" 2>&1 | tail -1
else
  say "cloning $REPO_URL"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR" 2>&1 | tail -2
  cd "$REPO_DIR"
fi
say "HEAD: $(git log --oneline -1)"

# --- interpreter ----------------------------------------------------------
source /workspace/env_aime.sh
say "PY=$PY"
"$PY" -c "import torch,transformers,sys;print('[bootstrap] py',sys.version.split()[0],
'torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)" \
  || { echo "[bootstrap] FATAL: interpreter lacks torch/transformers" >&2; exit 1; }
"$PY" -c "import pytest" >/dev/null 2>&1 || {
  say "installing pytest (--no-cache-dir so it does not rebuild a 10G pip cache)"
  "$PY" -m pip install --no-cache-dir -q pytest 2>&1 | tail -2; }

# --- model is cached, and resolves without a download ---------------------
say "checking Qwen3-14B resolves from the volume cache"
HF_HUB_OFFLINE=1 "$PY" -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen3-14B')
print(f'[bootstrap] config ok: {c.num_hidden_layers} layers, hidden {c.hidden_size}')
" || say "WARNING: not cached offline — the first run will download ~28 GB"

# --- the steering vector the experiments need ----------------------------
V=artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt
[ -f "$V" ] && say "steering vector present: $V" || say "WARNING: missing $V"

say "test suite"
"$PY" -m pytest tests/ -q 2>&1 | tail -3

say "READY. Next:"
say "  source /workspace/env_aime.sh"
say "  nohup bash scripts/run_aime_localize.sh signflip > /root/logs/signflip.log 2>&1 &"
say "Remember: results live on the volume, but STOP THE POD yourself — it cannot."
