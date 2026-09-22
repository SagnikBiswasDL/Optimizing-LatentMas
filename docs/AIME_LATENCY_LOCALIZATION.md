# AIME latency localization — where the wall clock actually goes

Run date: 2026-09-21. Model: Qwen3-14B, bf16, single H200 SXM. Task: AIME 2024,
items `{0,1,2,4,10,18}` (n=6; the three "critical" items `{4,10,18}` are the ones
Real K=10 solves but Frozen/Mean-Replay does not, per `DELTABRIDGE_HANDOFF.md`).
`K=10` latent steps per silent agent, Judger budget 8192, greedy (`temperature=0`),
batch 1, `generate_bs=1`.

**Status: partial.** The pod was lost mid-sweep (see "How this run died"). The
`real` arm and the per-agent timing are complete and are the load-bearing
results. Arms `c1/c2/c3/c12/c23/latents/shuf/evict_*`, plus both `isolated`
wirings, did not finish — only `none` on item 0 landed. Everything below is
measured, not projected.

Harness: `scripts/exp_aime_localize.py`, `scripts/run_aime_localize.sh`,
`seal/segment_kv.py`, `seal/latent_eval.py::build_upstream_segments`.

---

## 1. Headline: the silent agents are not the latency

Per-agent wall clock, mean over the 6 items:

| Agent | seconds | % of end-to-end | KV positions written | peak MB |
|---|---:|---:|---:|---:|
| planner | 0.387 | 0.25% | 210 | 29624 |
| critic | 0.286 | 0.19% | 251 | 28657 |
| refiner | 0.277 | 0.18% | 237 | 28756 |
| **judger** | **151.794** | **99.38%** | 6591 (tokens emitted) | — |

All three silent agents together cost **0.95s of a 152.7s item**. Judger decode is
**99.4%**.

**This kills the latency premise of the whole cache-surgery program.** Segment
eviction, dropping Planner/Critic, Jiayi's alternate wirings, synthetic
scaffolds — every one of these can only ever recover some fraction of 0.95s, i.e.
**under 1% of AIME latency**, even if accuracy were free. The same is true of the
existing "zero upstream compute" synthetic-scaffold result: on AIME it buys
~0.6%, not a speedup anyone will notice. Those methods have to be justified by
accuracy or by KV memory, *not* by latency.

(Peak MB for the upstream roles is dominated by the resident model weights, so
read it as "no meaningful extra allocation", not as per-agent cost.)

## 2. Where the latency *is*: non-terminating Judger runs

Per-item `real` decode, sorted by cost:

| item | tokens | judger_s | EOS? | correct | pred |
|---:|---:|---:|---|---|---|
| 0 | 2600 | 59.7 | yes | yes | 204 |
| 10 | 5848 | 135.7 | yes | yes | 104 |
| 18 | 7062 | 164.1 | yes | yes | 23 |
| 4 | 7653 | 175.7 | yes | yes | 110 |
| 1 | **8192** | 187.7 | **no** | no | 30.25 |
| 2 | **8192** | 187.9 | **no** | no | 108 |

Measured decode speed **43.4 tok/s**; EOS rate **67%**.

The two items we get wrong are exactly the two that never stop — they run the
budget to the wall and each cost **3.1x** the solved item. We pay peak latency
precisely where we earn nothing. Mean latency is being set by failures, not by
successes.

This is the low-hanging fruit, and it lives entirely in the Judger's token
budget. **Both obvious interventions have now been measured and both are dead.**
See §2b. The arithmetic that predicted it:

**A hard token cap is nearly worthless here.** The boxed answer sits at the very
end of a healthy CoT, so any cap below an item's length destroys its answer. The
only *safe* cap on this n=6 is above 7653 (the longest solved item), which
truncates only items 1 and 2 by 539 tokens each — **1078 of 39547 total tokens,
i.e. 2.7%**. Push the cap lower and it starts eating solves: a 6000 cap saves 18%
but kills items 4 and 18, halving accuracy. `--mode budget` measures this curve
offline from saved generations (tokenizer only, no GPU) and is expected to come
back *negative*; it is worth running precisely to close the option honestly.

**Aborting degenerate tails is the real lever.** Items 1 and 2 never emit EOS and
are wrong anyway, so cutting them costs nothing. If their looping is detectable
by ~token 3000, aborting saves roughly **10400 of 39547 tokens (26% of all
decode) at zero accuracy cost**. `--mode loops` implements this: an n-gram
repetition detector over a sliding window, requiring two consecutive windows over
threshold so a restated equation does not trip it. It reports, per item, the
onset position, tokens saved, and a `would_lose_solve` flag that fires when the
abort point precedes a correct answer — that flag is the guard against a detector
tuned too aggressively.

## 2b. Both token-budget interventions came back negative (2026-09-22)

Second pod, `quick` stage, 18 minutes. **The §1 result replicated exactly** —
Judger 155.4s of a 156.4s item (99.36%), silent agents 1.00s, 42.4 tok/s, 4/6
correct with all of `{4,10,18}` kept, and identical per-item token counts. Greedy
decode is deterministic here, so the latency decomposition is confirmed on two
independent runs.

**The hard cap is bad, as predicted** (`BUDGET.md`):

| budget | acc | Δacc | mean tokens | est judger_s | speedup |
|---:|---:|---:|---:|---:|---:|
| 8192 | 0.667 | +0.000 | 6590 | 155.4 | 1.00x |
| 6144 | 0.500 | −0.167 | 5504 | 129.8 | 1.20x |
| 4096 | 0.167 | −0.500 | 3846 | 90.7 | 1.71x |
| 3072 | 0.167 | −0.500 | 2993 | 70.6 | 2.20x |
| 2048 | 0.000 | −0.667 | 2048 | 48.3 | 3.22x |

You buy 1.2x for a third of the solves. **Option closed.**

**The loop detector fired on nothing at all** — zero onsets across all six items
at 8-gram / 256-window / 0.80. The non-terminating items are *not* degenerate, so
there is no repetition to abort and the 26% saving does not exist. Reading the
generations shows what they are actually doing:

- **item 1** (gold 113) is cut mid-arithmetic, converting `625/36` to a common
  denominator. No answer in sight.
- **item 2** (gold 371, i.e. probability 115/256) had derived 109/256 and was
  checking whether 109 and 256 are coprime when the budget ran out. It was
  executing carefully on a **wrong subset count**, so more budget would not have
  saved it.

Both are honest failures that happen to be expensive, not detectable waste.
**Option closed.**

## 2c. What the generations say the waste actually is

Not repetition — **verbosity**. The text is dense with reflection and transition
moves (`But wait, there's a mistake here`, `let's check with another approach`,
`Alternatively, compute...`) and with fraction arithmetic spelled out digit by
digit. That is a *thought-type mix* problem, not a stopping problem, and it is
exactly what the existing Judger SEAL vector targets:
`mean(exec) − mean(reflection+transition)` at layer 28, which on GSM8K gives
−17% tokens at 95.0% (vs 93.3% control) at coef 40, and −30% at coef 60.

SEAL at the Judger is therefore the one proven lever pointed at the 99.4%, and it
has never been run on AIME. `bash scripts/run_aime_localize.sh seal` sweeps
coef 40 and 60 against the `real` baseline in a single model load (~25 min; the
`real` arm is restored from `PERSIST_DIR` rather than re-decoded).

Success criterion: token reduction at unchanged accuracy on `{0,4,10,18}`. The
risk is specific and worth stating up front — SEAL suppresses reflection, and
AIME may need reflection more than GSM8K does, in which case accuracy drops and
this closes too.

## 3. Accuracy (tracked in parallel, so we don't buy speed with solves)

`real` reproduces the locked ladder: **4/6 = 0.667**, correct `[0, 4, 10, 18]`,
critical items kept `[4, 10, 18]`. Good — the harness is measuring the right
thing.

One suggestive data point on the cache's value: item 0 solves with **no upstream
cache at all** (`none`: 3129 tokens, 72.7s, correct, pred 204) as well as with
the full Real tape (2600 tokens, 59.7s). Consistent with the established
content-insensitivity finding, but n=1 — do not over-read it.

## 4. Tape geometry (for whoever resumes this)

Growing Real K=10, positions written per role:

| item | planner | critic | refiner | total pos | KV MB | upstream_s |
|---:|---|---|---|---:|---:|---:|
| 0 | 0–234 | 234–509 | 509–770 | 770 | 120.3 | 1.54 |
| 1 | 0–208 | 208–457 | 457–692 | 692 | 108.1 | 0.84 |
| 2 | 0–177 | 177–395 | 395–599 | 599 | 93.6 | 0.83 |
| 4 | 0–166 | 166–373 | 373–566 | 566 | 88.4 | 0.84 |
| 10 | 0–178 | 178–397 | 397–602 | 602 | 94.1 | 0.83 |
| 18 | 0–298 | 298–637 | 637–962 | 962 | 150.3 | 0.83 |

Each role writes ~170–340 positions (prompt + K=10 latents), tape is ~90–150 MB.
Item 0's 1.54s upstream includes first-call CUDA warmup; steady state is 0.83s.

## 5. How this run died (operational, read before booting the next pod)

Three separate infra problems, all now fixed in the scripts:

1. **venv moved.** RunPod migration relocated the environment from `/root/venv`
   to `/workspace/venv`. `run_aime_localize.sh` now probes candidate
   interpreters for one that can `import torch` instead of hardcoding a path.
2. **`torch.save` corrupts large files on `/workspace`.** That mount is MooseFS
   (`mfs#us-nc-1.runpod.net:9421`); torch's zip writer seeks while writing and
   fails mid-file (`PytorchStreamWriter failed writing file data/0`,
   `unexpected pos 10880 vs 10774`). It killed the whole process group with no
   Python traceback, which made it look like an OOM. `save_tape` now serializes
   to a `BytesIO` buffer, writes once sequentially, and renames atomically; the
   126 MB tapes go to local disk via `--tape_dir` / `TAPE_DIR`.
3. **`/root` is ephemeral — this is what lost the results.** Putting artifacts on
   local disk dodged problem 2 but meant everything vanished when the pod
   stopped. Correct split: **big KV tapes on `/root`** (local, fast, expendable —
   they are regenerable in ~40s for all 6 items), **small JSON/text artifacts on
   `/workspace`** (`rows.jsonl`, `agent_times.jsonl`, `texts/`, `CHECKIN.md`), and
   rsync after every completed arm. Small writes to MooseFS work fine; it is only
   the large seeking writes that break.

Also: a dozen-plus duplicate `royal_white_vole-migration-migration-...` pod
records accumulated from repeated RunPod migrations. They bill storage while
stopped and are the likely cause of the credit burn. Delete the duplicates; the
`/workspace` network volume is a separate resource and survives pod deletion.

## 6. What to run first on the next pod

The sweep is restart-safe (`rows.jsonl` keys on `(idx, method)`), tapes regenerate
in well under a minute, and `views` is arm-major so partial runs are readable.

### The throughput fix that makes everything else affordable

Every number above was produced one item at a time. `decode_batch` already
supports true batching — it left-pads ragged KV caches and builds the matching
mask — and `decode_one` was simply calling it with a batch of one. Per decode
step the cost is dominated by streaming 14B weights through memory, so a batch
amortizes that across items and buys roughly **4x more data per GPU-hour**.

The catch is that `generate()` runs until every sequence in the batch finishes, so
a group costs `max(tokens)` steps rather than `sum(tokens)`, and **per-item wall
clock stops being observable** — a 2600-token item sharing a batch with a
runaway 8192-token one looks like it took just as long. So grouped rows store
`batch_s`/`batch_size` and a **null `judger_s`**, and the per-item latency table
skips them rather than dividing by batch size and quietly inventing numbers. The
arm tables also carry `wall_s` (from `batch_s/batch_size`, which sums back to
true wall clock either way), so cost is still reported correctly for grouped runs.
**Any run whose purpose is latency must use `--decode_bs 1`.**

Greedy decoding *should* be batch-invariant, but padding the cache and the
different GEMM shapes a batch takes can perturb logits enough to flip a tie, and
once two runs diverge they stay diverged. `--mode compare` diffs two arms'
saved generations byte-for-byte to settle it, and the `parity` stage runs it
against the six items whose exact token counts we already have. **Run `parity`
before trusting any batched number.** If it is not fully identical, batched
results remain valid on their own terms but stop being drop-in comparable to the
n=1 baseline, and that has to be said wherever they are quoted.

### Order of operations

1. `bash scripts/run_aime_localize.sh parity` — ~6 min. Certifies grouped
   decoding and measures the actual speedup. Gates everything below.
2. `bash scripts/run_aime_localize.sh sweep` — ~1.5 h. n=30 AIME-2024 for
   `real`, `none`, `real_seal40`, `real_seal60`: 120 decodes that would have cost
   ~5 h unbatched. This is the main data yield. It answers three things at once —
   the real accuracy of the method at a defensible n, whether the cache matters
   at all (`none`), and whether SEAL buys tokens without costing accuracy.
3. `bash scripts/run_aime_localize.sh localize30` — ~1.7 h. The localization arms
   at n=30, now framed as accuracy/KV-memory rather than latency.
4. `bash scripts/run_aime_localize.sh aime25_sweep` — ~1.5 h. Held-out AIME 2025,
   into its own `out_dir`. Only worth it once AIME-2024 says something.

n=6 is the reason several earlier conclusions had to be walked back; at that size
one item is 17 accuracy points. Everything above is n=30 for that reason.
`restore` pulls completed rows back from `PERSIST_DIR` at stage start, so these
stages can be run across several pods without repaying for finished decodes.

Operational: `restore` pulls `rows.jsonl`, `agent_times.jsonl` and `texts/` back
from `PERSIST_DIR` at stage start and never clobbers newer local files, so
completed decodes carry across pods. `--mode budget` and `--mode loops` are
CPU-only and run off saved `texts/`, so you can stop the pod and iterate on
analysis locally for free. That is the whole point of saving generations.
