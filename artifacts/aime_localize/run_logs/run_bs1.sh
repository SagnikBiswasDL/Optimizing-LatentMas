set -u
cd /root/latentmas-baseline
export HF_HOME=/workspace/.cache/huggingface
PY=/workspace/venv/bin/python
ROOT=/root/latentmas-baseline/artifacts/aime_localize
TAPES=/root/aime_localize_tapes
PERSIST=/workspace/aime_localize_results
VEC=/root/latentmas-baseline/artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt
IDX=0,1,2,4,10,18
# Unbatched: batching failed parity, so every row here must be bs=1. Arms in
# value order with a wall-clock budget, so a cut run still yields the best prefix.
$PY -u scripts/exp_aime_localize.py --mode views --task aime2024 \
  --out_dir $ROOT --tape_dir $TAPES --persist_dir $PERSIST \
  --judger_budget 8192 --view_arms real_seal40,none,real_seal60 \
  --view_indices $IDX --decode_bs 1 --time_budget_s 2220 \
  --seal_vector $VEC --seal_layer 28
$PY -u scripts/exp_aime_localize.py --mode report --out_dir $ROOT --tape_dir $TAPES --persist_dir $PERSIST
$PY -u scripts/exp_aime_localize.py --mode budget --task aime2024 --out_dir $ROOT --tape_dir $TAPES --persist_dir $PERSIST --budget_arms real,real_seal40,none
echo ALL_DONE
