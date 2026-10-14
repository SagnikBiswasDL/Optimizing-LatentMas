#!/usr/bin/env bash
# AIME localization: which silent-agent writes the Judger actually uses,
# then role-budgeted eviction. Restart-safe. GPU from `collect` onward.
#
#   bash scripts/run_aime_localize.sh smoke
#   bash scripts/run_aime_localize.sh efficiency # ~2 GPU-h: the coefficient decision run
#   bash scripts/run_aime_localize.sh confirm24  # ~2.5 GPU-h: held-out 24 items, gated
#   bash scripts/run_aime_localize.sh insight   # ~45 min: budget-limit + cache-content probes
#   bash scripts/run_aime_localize.sh blitz     # fixed GPU window, value-ordered
#   bash scripts/run_aime_localize.sh parity    # ~6 min: certify grouped decode. RUN FIRST.
#   bash scripts/run_aime_localize.sh sweep     # ~1.5h: n=30, real/none/seal40/seal60
#   bash scripts/run_aime_localize.sh localize30   # ~1.7h: n=30 localization arms
#   bash scripts/run_aime_localize.sh aime25_sweep # ~1.5h: held-out AIME 2025
#   bash scripts/run_aime_localize.sh quick     # ~20 min: Real arm + token-budget curve
#   bash scripts/run_aime_localize.sh seal      # ~25 min: Judger SEAL coef sweep on AIME
#   bash scripts/run_aime_localize.sh focus     # items 0,1,2,4,10,18, all arms (~3h)
#   bash scripts/run_aime_localize.sh full      # n=30
#   bash scripts/run_aime_localize.sh qwen      # Real decode at Qwen thinking sampler
#   bash scripts/run_aime_localize.sh aime25    # same table on AIME 2025
#
# Env: N= items per stage (default 30).  DECODE_BS= Judger decodes per batch
#        (default 6; set 1 when the run's purpose is per-item latency).
#      SWEEP_ARMS= / LOC_ARMS= override the arm lists.
#      CACHE= path to math1k/cache.pt for isolated_frozen (Jiayi line 4).
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
N=${N:-30}
# Judger decodes per generate() call. Grouping multiplies throughput ~4x because
# the per-step cost is dominated by streaming the weights. KV is ~160KiB/token,
# so batch 6 at budget 8192 is ~9GB on top of the weights — room to raise this on
# an H200. Keep it at 1 for any run whose purpose is per-item latency.
DECODE_BS=${DECODE_BS:-6}
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

# Pull previously-persisted results back in so a fresh pod does not re-decode
# work we already paid for. Never clobbers newer local files; tapes are not
# restored (they regenerate in seconds).
restore () {
  [[ -n $PERSIST_DIR && -d $PERSIST_DIR ]] || return 0
  mkdir -p "$ROOT_DIR/texts" 2>/dev/null || true
  for f in rows.jsonl agent_times.jsonl; do
    [[ -f "$PERSIST_DIR/$f" && ! -f "$ROOT_DIR/$f" ]] && cp "$PERSIST_DIR/$f" "$ROOT_DIR/$f"
  done
  if [[ -d $PERSIST_DIR/texts ]]; then
    cp -n "$PERSIST_DIR"/texts/*.txt "$ROOT_DIR/texts/" 2>/dev/null || true
  fi
  local n=0
  [[ -f "$ROOT_DIR/rows.jsonl" ]] && n=$(wc -l < "$ROOT_DIR/rows.jsonl")
  echo "[localize] restored $n prior rows from $PERSIST_DIR" | tee -a "$LOG"
}

# Kernel-path flags applied to every invocation in a stage. The provenance guard
# refuses to compare rows decoded on different paths, so the path has to be a
# property of the whole stage rather than of individual calls.
DECODE_FLAGS=${DECODE_FLAGS:-}

run_py () {
  echo "[localize] $* ${DECODE_FLAGS} $(date)" | tee -a "$LOG"
  # shellcheck disable=SC2086  # DECODE_FLAGS is deliberately word-split
  if "$PY" -u scripts/exp_aime_localize.py --out_dir "$ROOT_DIR" --tape_dir "$TAPE_DIR" \
      --persist_dir "$PERSIST_DIR" --seed "$SEED" --k "$K" "$@" ${DECODE_FLAGS}; then
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

# Fail before the 40s model load rather than after it: a missing steering vector
# is the one setup error that only surfaces at attach time.
need_seal_vector () {
  case "$1" in
    *seal*) ;;
    *) return 0 ;;
  esac
  if [[ ! -f "$SEAL_VECTOR" ]]; then
    echo "[localize] arms '$1' need a SEAL vector; none at $SEAL_VECTOR" | tee -a "$LOG"
    return 1
  fi
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

# SEAL at the Judger on AIME. This is the only lever aimed at the 99.4% of wall
# clock that the Judger decode owns, and it is already validated on GSM8K
# (coef 40 => -17% tokens at 95.0% vs 93.3% control). Coefficients ride in the
# arm name, so the whole sweep runs in a single model load.
seal () {
  local idx=${INDICES:-0,1,2,4,10,18}
  local arms=${SEAL_ARMS:-real,real_seal40,real_seal60}
  need_seal_vector "$arms" || return 1
  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192 --view_arms "$arms" \
    --seal_vector "$SEAL_VECTOR" --seal_layer "${SEAL_LAYER:-28}"
  run_py --mode report
  run_py --mode loops --task aime2024 --budget_arms "$arms"
}

# Certify grouped decoding against the unbatched baseline before any batched
# sweep is trusted. Re-decodes the six items we already have exact numbers for,
# as one batch, under the label real__bs<N>, then diffs the text byte-for-byte.
# Cheap (~6 min) and it gates everything below.
# ---------------------------------------------------------------- time budget
# A GPU-hour budget is only real if something enforces it. Each `views` call gets
# a fresh in-process clock, so the deadline has to live out here and be handed
# down per call; run_views stops cleanly between items rather than mid-decode.
DEADLINE_S=${DEADLINE_S:-0}
start_deadline () {
  DEADLINE_S=$(( SECONDS + $1 ))
  echo "[localize] deadline set: ${1}s from now" | tee -a "$LOG"
}
remaining () {
  local r=$(( DEADLINE_S - SECONDS ))
  (( r < 0 )) && r=0
  echo "$r"
}
# Refuse to start a sub-stage that cannot plausibly finish, instead of starting it
# and leaving a half-populated arm that later looks like a real measurement.
have_time () {
  local need=$1 label=$2 left
  left=$(remaining)
  if (( DEADLINE_S == 0 )); then return 0; fi
  if (( left < need )); then
    echo "[localize] SKIP $label: needs ~${need}s, ${left}s left in budget" | tee -a "$LOG"
    return 1
  fi
  echo "[localize] $label: ~${need}s needed, ${left}s left" | tee -a "$LOG"
  return 0
}

parity_sentinel () { echo "${PERSIST_DIR:-$ROOT_DIR}/PARITY_OK_bs${DECODE_BS}"; }

# Any stage that decodes with DECODE_BS>1 must call this first. Grouped decoding
# has already been measured to change greedy outputs and halve accuracy, and
# `--mode compare` only *prints* that; without a hard gate the sweep proceeds and
# writes corrupted rows that look ordinary. Parity is per batch size, so the
# sentinel is too.
require_parity () {
  [[ ${DECODE_BS:-1} -le 1 ]] && return 0
  if [[ -f "$(parity_sentinel)" ]]; then
    echo "[localize] parity certified for bs=$DECODE_BS: $(parity_sentinel)" | tee -a "$LOG"
    return 0
  fi
  if [[ ${ALLOW_UNCERTIFIED_BATCH:-0} == 1 ]]; then
    echo "[localize] WARNING: decoding at bs=$DECODE_BS with NO parity certificate." \
      "Rows from this stage are not comparable to unbatched rows." | tee -a "$LOG"
    return 0
  fi
  echo "[localize] REFUSING to decode at bs=$DECODE_BS: no parity certificate." >&2
  echo "[localize]   run: bash $0 parity      (certifies this batch size)" >&2
  echo "[localize]   or:  DECODE_BS=1 bash $0 ${stage:-<stage>}" >&2
  echo "[localize]   or:  ALLOW_UNCERTIFIED_BATCH=1 ... (rows will be flagged uncomparable)" >&2
  return 1
}

parity () {
  local idx=${INDICES:-0,1,2,4,10,18}
  if [[ ${DECODE_BS:-1} -le 1 ]]; then
    echo "[localize] parity is meaningless at DECODE_BS=1; set DECODE_BS>1" >&2
    return 2
  fi
  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192 --view_arms real \
    --decode_bs 1 --view_indices "$idx"
  ALLOW_UNCERTIFIED_BATCH=1 run_py --mode views --task aime2024 --judger_budget 8192 \
    --view_arms real --decode_bs "$DECODE_BS" --method_tag "bs${DECODE_BS}" \
    --view_indices "$idx"
  # Non-zero exit here is the gate. Do not add --allow_diverge.
  if run_py --mode compare --compare_arms "real,real__bs${DECODE_BS}"; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$(parity_sentinel)"
    echo "[localize] PARITY PASSED — certificate written to $(parity_sentinel)" | tee -a "$LOG"
  else
    rm -f "$(parity_sentinel)"
    echo "[localize] PARITY FAILED at bs=$DECODE_BS — batched sweeps stay blocked." \
      "See docs/READ_ME_FIRST_AGENT_BRIEFING.md on bf16 batch-invariance." | tee -a "$LOG"
    run_py --mode report || true
    return 1
  fi
  run_py --mode report
}

# The data run. n=30 on AIME 2024, grouped decode, accuracy-and-tokens arms:
# `real` (the method), `none` (does the cache matter at all), and the SEAL coefs
# (the one latency lever left). Roughly 20 min per arm at DECODE_BS=6 versus
# ~78 min unbatched, so this is ~4x more data per GPU-hour.
sweep () {
  local arms=${SWEEP_ARMS:-real,none,real_seal40,real_seal60}
  need_seal_vector "$arms" || return 1
  require_parity || return 1
  run_py --mode collect --task aime2024 --n "$N" --indices "0-$((N - 1))" \
    --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192 --view_arms "$arms" \
    --decode_bs "$DECODE_BS" --seal_vector "$SEAL_VECTOR" \
    --seal_layer "${SEAL_LAYER:-28}"
  run_py --mode report
  run_py --mode budget --task aime2024 --budget_arms "$arms"
  run_py --mode loops --task aime2024 --budget_arms "$arms"
}

# The localization arms at n=30. No longer a latency story (see
# docs/AIME_LATENCY_LOCALIZATION.md) — these answer which upstream writes the
# Judger actually reads, as an accuracy and KV-memory question.
localize30 () {
  local arms=${LOC_ARMS:-c1,c2,c3,c23,evict_seg}
  require_parity || return 1
  run_py --mode collect --task aime2024 --n "$N" --indices "0-$((N - 1))" \
    --judger_budget 8192
  run_py --mode views --task aime2024 --judger_budget 8192 --view_arms "$arms" \
    --decode_bs "$DECODE_BS"
  run_py --mode report
}

# Held-out generalization: the same table on AIME 2025, into its own out_dir.
# TAPE_DIR is deliberately left alone: exp_aime_localize.py now appends a
# task/model/k subdirectory to it, so 2025 cannot pick up 2024's tapes, and a
# stale tape is rejected on load rather than silently decoded against the wrong
# question. Before that, this stage swapped out_dir only and reused the 2024 tapes.
aime25_sweep () {
  local out="${ROOT_DIR}_aime25"
  local save=$ROOT_DIR save_log=$LOG
  mkdir -p "$out"
  ROOT_DIR=$out
  LOG=$out/localize.log
  local arms=${SWEEP_ARMS:-real,none,real_seal40,real_seal60}
  need_seal_vector "$arms" || { ROOT_DIR=$save; LOG=$save_log; return 1; }
  require_parity || { ROOT_DIR=$save; LOG=$save_log; return 1; }
  run_py --mode collect --task aime2025 --n "$N" --indices "0-$((N - 1))" \
    --judger_budget 8192
  run_py --mode views --task aime2025 --judger_budget 8192 --view_arms "$arms" \
    --decode_bs "$DECODE_BS" --seal_vector "$SEAL_VECTOR" \
    --seal_layer "${SEAL_LAYER:-28}"
  run_py --mode report
  ROOT_DIR=$save
  LOG=$save_log
}

# The insight run. `--mode answers` showed that across every arm, "graded correct"
# and "ever wrote the gold answer" never disagree, and the failures never emit a
# boxed answer at all. So nothing is being produced and then lost: every failure
# is a failure to *reach* an answer inside 8192 tokens. Two questions follow, and
# this stage asks both. Unbatched throughout, because batching failed parity.
insight () {
  local vec_ok=1
  need_seal_vector real_seal40 || vec_ok=0
  run_py --mode collect --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" \
    --judger_budget 8192

  # Q1. Budget-limited or capability-limited? Items 1 and 2 never reach an answer
  # in any arm. If 3x the budget solves them, then accuracy at a fixed budget is
  # measuring convergence *speed*, not reasoning ability, and every number in this
  # project has to be read that way. If it does not, the ceiling is the model's.
  run_py --mode views --task aime2024 --view_arms real --view_indices 1,2 \
    --judger_budget 24576 --method_tag b24k --decode_bs 1

  # Q1b. Was SEAL's damage speed or correctness? It pushed items 10 and 18 from
  # solved-with-EOS to never-finishing. At 2x budget, either they come back (SEAL
  # only slowed convergence) or they do not (SEAL broke the reasoning).
  if [[ $vec_ok -eq 1 ]]; then
    run_py --mode views --task aime2024 --view_arms real_seal40 --view_indices 10,18 \
      --judger_budget 16384 --method_tag b16k --decode_bs 1 \
      --seal_vector "$SEAL_VECTOR" --seal_layer "${SEAL_LAYER:-28}"
  fi

  # Q2. Does the cache's *content* matter, or only its presence? `none` halves
  # accuracy; `shuf` hands the Judger a different problem's cache at the same
  # size. Landing near `real` means the cache works as generic scaffolding;
  # landing near `none` means the upstream agents encode something specific.
  run_py --mode views --task aime2024 --view_arms shuf \
    --view_indices "${INDICES:-0,1,2,4,10,18}" --judger_budget 8192 --decode_bs 1

  run_py --mode report
  run_py --mode answers --task aime2024 \
    --budget_arms real,none,shuf,real_seal40,real__b24k,real_seal40__b16k
}

# ===========================================================================
# Stage A: is the GPU actually being used? Batch-1 decode of a 14B bf16 model is
# memory-bound, so the ceiling is weights/bandwidth: ~162 tok/s on an H200. We
# measured 42.5, i.e. 26% of roofline, with no attention kernel selected, no CUDA
# graphs and a DynamicCache. Recovering that is a latency win on every item at no
# accuracy cost, and it makes every stage below roughly twice as cheap — so it runs
# first for economic reasons even if the speedup is not itself the result.
# ===========================================================================
throughput () {
  local out=${THROUGHPUT_OUT:-$ROOT_DIR/throughput.json}
  echo "[localize] throughput probe -> $out" | tee -a "$LOG"
  "$PY" -u scripts/diag_throughput.py --model "${MODEL:-Qwen/Qwen3-14B}" \
    --prefix_len "${PREFIX_LEN:-700}" --new_tokens "${PROBE_TOKENS:-256}" \
    --seal_vector "$SEAL_VECTOR" --seal_layer "${SEAL_LAYER:-28}" \
    --out "$out" 2>&1 | tee -a "$LOG"
  persist
}

# Same dev-set decode on the fast path, to confirm the speedup does not move
# accuracy. Not bit-identical is expected and fine (different kernels, bf16); a
# different accuracy is not.
fastparity () {
  local idx=${INDICES:-0,1,2,4,10,18}
  # flash_attention_2 needs the flash_attn package, which is absent on some pods;
  # it also buys little at batch 1, where decode is bound by weight bandwidth
  # rather than attention. StaticCache is the lever that measured 1.70x, so that
  # is the default and compile is opt-in (it aborts inside generate(), see
  # docs/READ_ME_FIRST_AGENT_BRIEFING.md).
  local attn=${FAST_ATTN:-sdpa}
  local flags="--attn_impl $attn --static_cache"
  [[ ${FAST_COMPILE:-0} == 1 ]] && flags="$flags --compile_decode"
  echo "[localize] fast-path accuracy check: $flags" | tee -a "$LOG"
  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget 8192
  # shellcheck disable=SC2086
  run_py --mode views --task aime2024 --view_arms real --view_indices "$idx" \
    --judger_budget 8192 --method_tag fast --decode_bs 1 $flags
  # Byte parity will fail across kernels; what matters is that accuracy holds, so
  # record the comparison rather than gating on it.
  run_py --mode compare --compare_arms "real,real__fast" --allow_diverge
  run_py --mode report
}

# ===========================================================================
# The coefficient decision run. Question: can steering make the Judger finish
# sooner without dropping a correct answer?
#
# Design notes that cost money if ignored:
#
#  * Every arm decodes at the SAME cap (PROMOTE_CAP). Comparing a baseline given
#    24k tokens against a candidate given 8k would manufacture a token saving.
#  * Only *censored* rows are re-run. A greedy run that emitted EOS below the old
#    cap emits the same EOS at any larger cap, so those rows are cap-independent
#    and reusable — which is what makes this fit in two GPU-hours.
#  * --decode_bs 1 throughout. Grouped decoding is not bit-faithful in bf16 (see
#    docs/READ_ME_FIRST_AGENT_BRIEFING.md §1a); a throughput win here would be
#    paid for in exactly the quantity being measured.
#  * Screening runs on the two items coef 40 broke. A coefficient that cannot hold
#    those cannot pass the rule, so it is rejected for ~2 items of GPU instead of 6.
# ===========================================================================
# The same decision run with the coefficient sign flipped. Positive coefficients
# lengthen the Judger's output monotonically on item 10 (5848 -> 10604 -> 14704 ->
# capped; see docs/RESULTS_2026_09_24_COEFFICIENT_RUN.md), and the vector is
# mean(execution) - mean(reflection + transition), so steering the other way is
# the cheap test of whether the axis carries any brevity signal at all.
#
# BASELINE_INDICES defaults to the whole cohort here rather than the censored two,
# because a baseline reused from another pod carries no decode-path provenance and
# --mode promote refuses a mixed set.
signflip () {
  local flipped=${COEFS:-"-20,-40,-60"}
  COEFS=$flipped BASELINE_INDICES=${BASELINE_INDICES:-0,1,2,4,10,18} efficiency
}

# Small magnitudes, both signs. Everything tested through 2026-09-29 used
# |coef| >= 20 against a vector whose raw norm is 52.7, so coefficient 20 displaces
# the residual stream by ~38% of the direction's natural scale. Every one of those
# settings emitted *more* tokens in either direction, which looks like an
# off-distribution perturbation rather than a semantic effect. If that is right,
# the interesting regime is the one never sampled: near zero, where the
# perturbation is small enough that whatever semantics the direction carries might
# still dominate.
#
# The screen also demands *shorter*, not merely correct. On a token-efficiency
# question a correct-but-longer arm is already a failure, and letting it survive
# costs a full cohort to rediscover that.
# What the latent channel between agents actually carries.
#
# The Judger currently reads all three upstream roles' KV unconditionally. This
# runs the arms that decide whether that content matters: `none` (no cache at all)
# and `shuf` (another problem's cache, same size). Analysis and reading rules are
# pre-registered in scripts/channel_ablation.py.
#
# Reuses the 30 tapes already collected, so there is no upstream cost -- only
# Judger decodes. Item-major so a deadline truncates whole items and every item
# present has every arm.
# Jiayi's ask (see the `qwen` stage): the Qwen3 thinking sampler rather than
# greedy, which is the setting the paper actually decodes in. Every result in
# §6-§7 of the results doc is greedy, so the token effect is currently unproven
# in the regime it needs to hold in.
#
# The one thing this stage must not do is sample the UPSTREAM agents. run_collect
# copies --temperature into the namespace it builds the latent cache with, so
# passing the sampler to both steps would change the cache as well as the decode
# and stop being a clean comparison. Collect is therefore run at the default
# temperature 0 and only the Judger decode samples.
#
# Rows land as `real_t06__<tag>` / `none_t06__<tag>`, distinct from the greedy
# rows in the same file; channel_token_cost.py --sampler t06 selects them and
# refuses to pool the two regimes.
sampler () {
  local cap=${PROMOTE_CAP:-16384}
  local tag=${PROMOTE_TAG:-b16k}
  local arms=${SAMPLER_ARMS:-"real,none"}
  local temp=${SAMPLER_TEMP:-0.6}
  local top_p=${SAMPLER_TOP_P:-0.95}
  local top_k=${SAMPLER_TOP_K:-20}
  local n_items=${SAMPLER_N:-15}
  local items=${SAMPLER_ITEMS:-$(seq 0 $(( n_items - 1 )) | paste -sd, -)}
  items="${items%,}"
  n_items=$(( $(tr -cd ',' <<<"$items" | wc -c) + 1 ))
  local budget=${SAMPLER_S:-12600}
  start_deadline "$budget"

  # Items are the first n by index, a fixed arbitrary rule. Never pick the subset
  # by how the greedy run came out; see §3.7 on selecting items by outcome.
  echo "[localize] sampler: arms=$arms items=$n_items temp=$temp top_p=$top_p" \
    "top_k=$top_k cap=$cap tag=$tag budget=${budget}s" | tee -a "$LOG"

  # Cache built greedily on purpose: only the Judger samples.
  run_py --mode collect --task aime2024 --n "$n_items" --indices "$items" \
    --judger_budget "$cap"

  run_py --mode views --task aime2024 --view_arms "$arms" \
    --view_indices "$items" --judger_budget "$cap" --method_tag "$tag" \
    --temperature "$temp" --top_p "$top_p" --top_k "$top_k" \
    --decode_bs 1 --arm_order item --time_budget_s "$(remaining)"

  run_py --mode report
  "$PY" "$(dirname "$0")/channel_token_cost.py" --rows "$ROOT_DIR/rows.jsonl" \
    --tag "$tag" --sampler "t${temp//./}" \
    --out "$ROOT_DIR/token_cost_t${temp//./}.json" 2>&1 | tee -a "$LOG"
  echo "[localize] sampler used ${SECONDS}s of ${budget}s" | tee -a "$LOG"
}

channel () {
  local cap=${PROMOTE_CAP:-16384}
  local tag=${PROMOTE_TAG:-b16k}
  local arms=${CHANNEL_ARMS:-"none,shuf"}
  local budget=${CHANNEL_S:-25200}
  local items=${COHORT_ITEMS:-$(seq 0 29 | paste -sd, -)}
  items="${items%,}"
  local n_items; n_items=$(( $(tr -cd ',' <<<"$items" | wc -c) + 1 ))
  start_deadline "$budget"

  echo "[localize] channel: arms=$arms items=$n_items cap=$cap tag=$tag" \
    "budget=${budget}s" | tee -a "$LOG"

  # Tapes must already exist; collect is a no-op that verifies identity.
  run_py --mode collect --task aime2024 --n 30 --indices "$items" \
    --judger_budget "$cap"

  run_py --mode views --task aime2024 --view_arms "$arms" \
    --view_indices "$items" --judger_budget "$cap" --method_tag "$tag" \
    --decode_bs 1 --arm_order item --time_budget_s "$(remaining)"

  run_py --mode report
  "$PY" "$(dirname "$0")/channel_ablation.py" --rows "$ROOT_DIR/rows.jsonl" \
    --tag "$tag" --arms "$arms" \
    --out "$ROOT_DIR/channel_ablation.json" 2>&1 | tee -a "$LOG"
  echo "[localize] channel used ${SECONDS}s of ${budget}s" | tee -a "$LOG"
}

# The full paired cohort at one coefficient. This is the experiment the screen
# design could not do: §3.6 of the results doc showed token count swinging >150%
# of baseline between adjacent coefficients, so per-item trajectory chaos has to
# be averaged over many items rather than probed at two.
#
# Pair-major (--arm_order item) so that running out of wall clock costs whole
# pairs instead of leaving baselines without steered partners. The analysis and
# its decision rule were committed before any of this data existed; see
# scripts/paired_coef_test.py.
cohort () {
  local cap=${PROMOTE_CAP:-16384}
  local tag=${PROMOTE_TAG:-b16k}
  local coef=${COHORT_COEF:-5}
  local layer=${SEAL_LAYER:-28}
  local budget=${COHORT_S:-25200}
  # BSD seq leaves a trailing separator where GNU seq does not, and a trailing
  # comma would both miscount the cohort and reach the index parser.
  local items=${COHORT_ITEMS:-$(seq 0 29 | paste -sd, -)}
  items="${items%,}"
  local n_items; n_items=$(( $(tr -cd ',' <<<"$items" | wc -c) + 1 ))
  need_seal_vector "real_seal${coef}" || return 1
  start_deadline "$budget"
  local per_item=$(( cap * 100 / 4250 ))

  echo "[localize] cohort: coef=$coef items=$n_items cap=$cap tag=$tag" \
    "budget=${budget}s per-pair<=$(( per_item * 2 ))s" | tee -a "$LOG"

  # One collect for the whole cohort: tapes are reused across both arms and
  # re-collecting per item would reload the model 30 times.
  run_py --mode collect --task aime2024 --n 30 --indices "$items" \
    --judger_budget "$cap"

  run_py --mode views --task aime2024 --view_arms "real,real_seal${coef}" \
    --view_indices "$items" --judger_budget "$cap" --method_tag "$tag" \
    --decode_bs 1 --arm_order item \
    --seal_vector "$SEAL_VECTOR" --seal_layer "$layer" \
    --time_budget_s "$(remaining)"

  run_py --mode report
  "$PY" "$(dirname "$0")/paired_coef_test.py" --rows "$ROOT_DIR/rows.jsonl" \
    --tag "$tag" --coef "$coef" \
    --out "$ROOT_DIR/paired_coef${coef}.json" 2>&1 | tee -a "$LOG"
  echo "[localize] cohort used ${SECONDS}s of ${budget}s" | tee -a "$LOG"
}

smallsweep () {
  local coefs=${COEFS:-"2,5,10,-2,-5,-10"}
  COEFS=$coefs \
  SCREEN_REQUIRE_SHORTER=${SCREEN_REQUIRE_SHORTER:-1} \
  MAX_SURVIVORS=${MAX_SURVIVORS:-2} \
  BASELINE_INDICES=${BASELINE_INDICES:-0,1,2,4,10,18} \
    efficiency
}

# k samples per item at temperature, recording length, correctness, and layer-28
# activations. Two questions in one run: the noise floor (how far accuracy moves
# when only the seed changes), and whether correct solutions to the *same* problem
# differ in length at all — which is the precondition for brevity steering. If
# every correct solution to a problem is the same length there is no terse mode to
# steer toward, and that is worth one GPU-hour to learn before building a vector.
samples () {
  local idx=${INDICES:-0,1,2,4,10,18}
  local k=${K_SAMPLES:-4}
  local temp=${SAMPLE_TEMP:-0.6}
  local cap=${SAMPLE_CAP:-16384}
  local budget=${SAMPLES_S:-7200}
  start_deadline "$budget"
  local per_item=$(( cap * 100 / 4250 ))
  echo "[localize] samples: k=$k temp=$temp cap=$cap items=$idx" \
    "worst case $(( per_item * k * 6 / 60 ))min" | tee -a "$LOG"
  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget "$cap"
  run_py --mode samples --task aime2024 --view_arms "${SAMPLE_ARM:-real}" \
    --view_indices "$idx" --judger_budget "$cap" --k_samples "$k" \
    --temperature "$temp" --decode_bs 1 --act_window "${ACT_WINDOW:-128}" \
    --time_budget_s "$(remaining)"
  persist
}

efficiency () {
  local cap=${PROMOTE_CAP:-16384}
  local tag=${PROMOTE_TAG:-b16k}
  local idx=${INDICES:-0,1,2,4,10,18}
  local screen=${SCREEN_INDICES:-10,18}
  local coefs=${COEFS:-20,60,80}
  local budget=${EFFICIENCY_S:-7200}
  local layer=${SEAL_LAYER:-28}
  local max_survivors=${MAX_SURVIVORS:-2}
  need_seal_vector "real_seal40" || return 1
  start_deadline "$budget"
  # Per-item worst case in seconds: the cap, at the throughput this box measured.
  local per_item=$(( cap * 100 / 4250 ))

  echo "[localize] efficiency: cap=$cap tag=$tag coefs=$coefs screen=$screen" \
    "budget=${budget}s per_item<=${per_item}s" | tee -a "$LOG"

  run_py --mode collect --task aime2024 --n 6 --indices "$idx" --judger_budget "$cap"

  # --- Stage 1: de-censor the baseline. Items 1 and 2 never finished at 8192, so
  # the baseline's own token total is currently a lower bound, and the whole
  # comparison is against an unknown number.
  #
  # Only the censored items need re-running when the reusable rows carry the same
  # decode-path provenance as the new ones. Rows written before provenance existed
  # do not, and --mode promote refuses a mixed set, so set BASELINE_INDICES to the
  # full cohort when rebuilding a baseline on a different or newly-recorded path.
  local base_idx=${BASELINE_INDICES:-1,2}
  local n_base; n_base=$(( $(tr -cd ',' <<<"$base_idx" | wc -c) + 1 ))
  if have_time $(( per_item * n_base )) "stage1 de-censor baseline ($base_idx)"; then
    run_py --mode views --task aime2024 --view_arms real --view_indices "$base_idx" \
      --judger_budget "$cap" --method_tag "$tag" --decode_bs 1 \
      --time_budget_s "$(remaining)"
    # Did the longer run retrace the shorter one? If not, extended and original
    # token counts are not comparable and nothing downstream is valid.
    run_py --mode prefix --compare_arms "real,real__${tag}" --allow_diverge
  fi

  # --- Stage 2: screen coefficients on the items coef 40 broke.
  local survivors=""
  for c in ${coefs//,/ }; do
    have_time $(( per_item * 2 )) "stage2 screen coef $c" || break
    run_py --mode views --task aime2024 --view_arms "real_seal${c}" \
      --view_indices "$screen" --judger_budget "$cap" --method_tag "$tag" \
      --decode_bs 1 --seal_vector "$SEAL_VECTOR" --seal_layer "$layer" \
      --time_budget_s "$(remaining)"
    # Survives only if it solves BOTH screen items under the stopping policy.
    if "$PY" - "$ROOT_DIR/rows.jsonl" "real_seal${c}__${tag}" "$screen" \
         "real__${tag}" "${SCREEN_REQUIRE_SHORTER:-0}" <<'EOF'
import json, sys
path, arm, idxs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
base_arm, need_shorter = sys.argv[4], sys.argv[5] == "1"
rows = [json.loads(l) for l in open(path)]
by = {(r['method'], int(r['idx'])): r for r in rows}
reasons = []
ok = True
for i in idxs:
    r = by.get((arm, i))
    if r is None:
        ok = False; reasons.append(f"idx{i}: missing"); continue
    if not r.get('eos'):
        ok = False; reasons.append(f"idx{i}: never terminated"); continue
    if not r.get('correct'):
        ok = False; reasons.append(f"idx{i}: wrong"); continue
    # Correct-but-longer used to survive the screen and then consume a full
    # cohort. Every coefficient tested through 2026-09-29 emitted MORE tokens
    # than baseline in both directions, so on a token-efficiency question
    # "shorter" belongs in the screen rather than being discovered later.
    if need_shorter:
        b = by.get((base_arm, i))
        if b is None:
            ok = False; reasons.append(f"idx{i}: no baseline {base_arm}"); continue
        if int(r['tokens']) >= int(b['tokens']):
            ok = False
            reasons.append(f"idx{i}: {r['tokens']} >= baseline {b['tokens']}")
            continue
        reasons.append(f"idx{i}: {r['tokens']} < {b['tokens']}")
print(f"[screen] {arm} on {idxs}: " + ("SURVIVES" if ok else "eliminated")
      + (" (" + "; ".join(reasons) + ")" if reasons else ""))
sys.exit(0 if ok else 1)
EOF
    then
      survivors="${survivors}${survivors:+,}$c"
    fi
  done
  echo "[localize] survivors: ${survivors:-none}" | tee -a "$LOG"

  # --- Stage 3: complete the cohort for survivors, cheapest-first, capped in
  # number so a 3-way tie cannot silently blow the budget.
  local rest
  rest=$("$PY" - "$idx" "$screen" <<'EOF'
import sys
all_i = [x for x in sys.argv[1].split(',') if x]
scr = set(sys.argv[2].split(','))
print(','.join(i for i in all_i if i not in scr))
EOF
)
  local n=0 cands=""
  for c in ${survivors//,/ }; do
    (( n >= max_survivors )) && { echo "[localize] survivor cap reached, skipping coef $c" | tee -a "$LOG"; break; }
    have_time $(( per_item * 4 )) "stage3 cohort for coef $c" || break
    run_py --mode views --task aime2024 --view_arms "real_seal${c}" \
      --view_indices "$rest" --judger_budget "$cap" --method_tag "$tag" \
      --decode_bs 1 --seal_vector "$SEAL_VECTOR" --seal_layer "$layer" \
      --time_budget_s "$(remaining)"
    cands="${cands}${cands:+,}real_seal${c}"
    n=$(( n + 1 ))
  done

  # --- Stage 4: the decision.
  run_py --mode report
  run_py --mode latency --exclude_arms real__bs16
  if [[ -n $cands ]]; then
    run_py --mode promote --baseline_arm real --candidate_arms "$cands" \
      --promote_cap "$cap" --promote_tag "$tag" --promote_indices "$idx" \
      --min_saving "${MIN_SAVING:-0.10}"
  else
    echo "[localize] no candidate completed the cohort; nothing to score" | tee -a "$LOG"
  fi
  echo "[localize] efficiency used ${SECONDS}s of ${budget}s" | tee -a "$LOG"
}

# The whole program, in dependency order, unattended. Each phase writes a DONE_
# sentinel so a lost pod resumes rather than repeats, and `restore` pulls finished
# rows back from the durable volume at every stage start.
program () {
  echo "[localize] ===== phase A: throughput =====" | tee -a "$LOG"
  throughput
  echo "[localize] ===== phase A2: fast-path accuracy =====" | tee -a "$LOG"
  fastparity || echo "[localize] fast path unusable; later phases stay on the eager path" | tee -a "$LOG"
  finish throughput
  echo "[localize] ===== phase B: coefficient decision =====" | tee -a "$LOG"
  EFFICIENCY_S=${EFFICIENCY_S:-7200} efficiency
  finish efficiency
  echo "[localize] ===== phase C: held-out confirmation =====" | tee -a "$LOG"
  if confirm24; then finish confirm24; else
    echo "[localize] nothing promoted, so no confirmation run" | tee -a "$LOG"; fi
  run_py --mode latency --exclude_arms real__bs16 || true
  echo "[localize] program complete in ${SECONDS}s" | tee -a "$LOG"
}

# Held-out confirmation on the other 24 AIME-2024 items. Gated on a promotion,
# because at ~2.5 GPU-h for baseline plus one arm it is the most expensive thing
# here and is worthless without a candidate worth confirming.
confirm24 () {
  local cap=${PROMOTE_CAP:-16384}
  local tag=${PROMOTE_TAG:-b16k}
  local coef=${CONFIRM_COEF:-}
  local budget=${CONFIRM_S:-9000}
  if [[ -z $coef ]]; then
    coef=$("$PY" -c "
import json,sys
try: d=json.load(open('$ROOT_DIR/promote.json'))
except Exception: sys.exit(0)
w=d.get('promoted') or []
print(w[0].replace('real_seal','') if w else '')
" 2>/dev/null)
  fi
  if [[ -z $coef ]]; then
    echo "[localize] confirm24: nothing was promoted. Set CONFIRM_COEF=<n> to override." >&2
    return 1
  fi
  need_seal_vector "real_seal${coef}" || return 1
  start_deadline "$budget"
  echo "[localize] confirm24: coef=$coef cap=$cap on the 24 non-development items" | tee -a "$LOG"
  # The held-out set is everything in 0-29 that is NOT in the development cohort,
  # so the confirmation cannot be contaminated by the items used to choose the coef.
  local held
  held=$("$PY" - "${INDICES:-0,1,2,4,10,18}" <<'EOF'
import sys
dev = {int(x) for x in sys.argv[1].split(',') if x}
print(','.join(str(i) for i in range(30) if i not in dev))
EOF
)
  echo "[localize] held-out items: $held" | tee -a "$LOG"
  run_py --mode collect --task aime2024 --n 30 --indices "$held" --judger_budget "$cap"
  run_py --mode views --task aime2024 --view_arms real --view_indices "$held" \
    --judger_budget "$cap" --method_tag "$tag" --decode_bs 1 --time_budget_s "$(remaining)"
  run_py --mode views --task aime2024 --view_arms "real_seal${coef}" \
    --view_indices "$held" --judger_budget "$cap" --method_tag "$tag" --decode_bs 1 \
    --seal_vector "$SEAL_VECTOR" --seal_layer "${SEAL_LAYER:-28}" \
    --time_budget_s "$(remaining)"
  run_py --mode latency --exclude_arms real__bs16
  run_py --mode promote --baseline_arm real --candidate_arms "real_seal${coef}" \
    --promote_cap "$cap" --promote_tag "$tag" --promote_indices "$held" \
    --min_saving "${MIN_SAVING:-0.10}"
  run_py --mode report
}

# One command for a fixed GPU window. Arms run in value order and the sweep
# stops cleanly between batches when the budget expires, so whatever finished is
# complete and persisted rather than half-written. BLITZ_MIN sets the window.
blitz () {
  local arms=${BLITZ_ARMS:-real,none,real_seal40,real_seal60}
  local mins=${BLITZ_MIN:-80}
  need_seal_vector "$arms" || return 1
  autotune_bs
  echo "[localize] blitz n=$N bs=$DECODE_BS arms=$arms budget=${mins}min" | tee -a "$LOG"
  # Tapes first: cheap, and every decode below needs them.
  run_py --mode collect --task aime2024 --n "$N" --indices "0-$((N - 1))" \
    --judger_budget 8192
  # Certify grouped decode against the committed n=1 rows before trusting it.
  # These used to end in `|| true`, which made the certification decorative: a
  # failed parity check printed a warning and the data run below batched anyway.
  if [[ ${DECODE_BS:-1} -gt 1 ]]; then
    ALLOW_UNCERTIFIED_BATCH=1 run_py --mode views --task aime2024 \
      --judger_budget 8192 --view_arms real --decode_bs "$DECODE_BS" \
      --method_tag "bs${DECODE_BS}" --view_indices "${INDICES:-0,1,2,4,10,18}"
    if run_py --mode compare --compare_arms "real,real__bs${DECODE_BS}"; then
      date -u +%Y-%m-%dT%H:%M:%SZ > "$(parity_sentinel)"
    else
      rm -f "$(parity_sentinel)"
      echo "[localize] parity failed at bs=$DECODE_BS; falling back to DECODE_BS=1." \
        "This costs throughput and is the correct trade." | tee -a "$LOG"
      DECODE_BS=1
    fi
  fi
  require_parity || return 1
  # The data run, time-boxed.
  run_py --mode views --task aime2024 --judger_budget 8192 --view_arms "$arms" \
    --decode_bs "$DECODE_BS" --seal_vector "$SEAL_VECTOR" \
    --seal_layer "${SEAL_LAYER:-28}" --time_budget_s "$((mins * 60))"
  run_py --mode report
  run_py --mode budget --task aime2024 --budget_arms "$arms" || true
  run_py --mode loops --task aime2024 --budget_arms "$arms" || true
}

# Pick a decode batch size that fits. KV is ~160KiB/token for Qwen3-14B, so a
# sequence at budget 8192 plus its upstream tape costs ~1.4GB; leave the weights
# ~30GB and half the remainder as headroom for activations and fragmentation.
autotune_bs () {
  [[ -n ${DECODE_BS_FIXED:-} ]] && { DECODE_BS=$DECODE_BS_FIXED; return 0; }
  local total
  total=$("$PY" -c 'import torch;print(int(torch.cuda.get_device_properties(0).total_memory//2**20))' 2>/dev/null) || return 0
  [[ -z $total ]] && return 0
  local avail=$(( (total - 30000) / 2 ))
  local bs=$(( avail / 1400 ))
  (( bs < 1 )) && bs=1
  (( bs > N )) && bs=$N
  (( bs > 16 )) && bs=16   # past this the straggler dominates, not throughput
  DECODE_BS=$bs
  echo "[localize] autotune: ${total}MB GPU -> DECODE_BS=$DECODE_BS" | tee -a "$LOG"
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
[[ $stage != smoke ]] && restore
case "$stage" in
  smoke) smoke ;;
  quick) quick ;;
  seal) seal ;;
  insight) insight ;;
  throughput) throughput ;;
  fastparity) fastparity ;;
  program) program ;;
  efficiency) efficiency ;;
  signflip) signflip ;;
  smallsweep) smallsweep ;;
  cohort) cohort ;;
  channel) channel ;;
  sampler) sampler ;;
  samples) samples ;;
  confirm24) confirm24 ;;
  blitz) blitz ;;
  parity) parity ;;
  sweep) sweep ;;
  localize30) localize30 ;;
  aime25_sweep) aime25_sweep ;;
  compare) run_py --mode compare --compare_arms "${COMPARE_ARMS:?set COMPARE_ARMS=A,B}" ;;
  focus) focus ;;
  full) full ;;
  qwen) qwen ;;
  aime25) aime25 ;;
  report) run_py --mode report ;;
  budget) run_py --mode budget --task aime2024 --budget_arms "${BUDGET_ARMS:-real}" ;;
  collect) run_py --mode collect --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" ;;
  views) run_py --mode views ;;
  isolated) run_py --mode isolated --task aime2024 --n 6 --indices "${INDICES:-0,1,2,4,10,18}" ;;
  *) echo "usage: $0 program|throughput|fastparity|efficiency|signflip|smallsweep|cohort|channel|sampler|samples|confirm24|insight|blitz|parity|sweep|localize30|aime25_sweep|quick|seal|smoke|focus|full|qwen|aime25|compare|budget|report" >&2; exit 2 ;;
esac
echo "[localize] DONE $stage $(date)" | tee -a "$LOG"
persist
cat "${ROOT_DIR}/CHECKIN.md" 2>/dev/null || true
cat "${ROOT_DIR}/BUDGET.md" 2>/dev/null || true
finish "$stage"
