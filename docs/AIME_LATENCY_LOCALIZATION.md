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

## 3. The cache earns its keep; SEAL does not (2026-09-22, measured)

Same six items throughout, all `--decode_bs 1`, greedy, budget 8192. `!` marks a
run that never emitted EOS and burned the whole budget.

| item | gold | real | none | seal40 | seal60 |
|---:|---:|---|---|---|---|
| 0 | 204 | ✓ 2600 | ✓ 3129 | ✓ 1807 | ✓ 1599 |
| 1 | 113 | ✗ 8192! | ✗ 8192! | **✓ 8192!** | **✓ 8192!** |
| 2 | 371 | ✗ 8192! | ✗ 8192! | ✗ 8192! | — |
| 4 | 110 | ✓ 7653 | ✗ 8192! | ✓ 5896 | — |
| 10 | 104 | ✓ 5848 | ✗ 8192! | **✗ 8192!** | — |
| 18 | 023 | ✓ 7062 | ✓ 8192! | **✗ 8192!** | — |

| arm | n | acc | mean tokens | EOS rate |
|---|---:|---:|---:|---:|
| real | 6 | **0.667** | 6591 | **0.67** |
| none | 6 | **0.333** | 7348 | **0.17** |
| real_seal40 | 6 | 0.500 | 6745 | 0.33 |
| real_seal60 | 2 | (1.000) | 4896 | 0.50 |

**The cache result is confounded by item selection and must not be quoted as
evidence.** Deleting the cache takes accuracy 0.667 → 0.333 and EOS 0.67 → 0.17,
which looks like strong evidence that the upstream agents contribute something the
Judger cannot reconstruct. It is not, and the reason is in §0: `{4,10,18}` were
chosen *because* Real K=10 solves them and Frozen/Mean-Replay does not. The
`real`-minus-`none` difference is exactly items **4 and 10** — both from that
selected set. So the arm gap is close to circular: we picked items where the real
cache uniquely wins, then measured that the real cache wins.

What this run legitimately shows is narrower: the harness reproduces the known
ladder (`real` = 4/6 with all three critical items kept), and the *mechanism* of
failure without a cache is refusal to terminate rather than wrong answers — 4 of 6
`none` runs never box anything at all (§3b). The mechanism claim is not
selection-dependent in the same way, because it is about how the failures look
rather than how many there are.

**An unbiased estimate needs the other 24 AIME-2024 items.** Until then, treat
"the latent cache helps on AIME" as untested, not supported.

**SEAL at the Judger is a net loss on AIME, and the risk we flagged is exactly
what happened.** Accuracy falls 0.667 → 0.500 and mean tokens go *up* (6591 →
6745), because EOS rate halves. The per-item detail is more interesting than the
average:

- Where it keeps a solve it is a large token win: item 0 −31% (2600→1807), item 4
  −23% (7653→5896). That is the GSM8K result reproducing.
- It *gains* item 1, which no other arm solves. Suppressing reflection stopped it
  from talking itself out of the answer.
- But it **breaks termination on items 10 and 18**, both of which the baseline
  solved while emitting EOS. They now run the full budget and answer wrong.

So the vector trades reflection for verbosity in both directions: less hedging
helps a problem that was over-thinking, and hurts two that needed the reflection
to know they were done. Averaged over n=6 that is negative. coef 60 got through
only 2 items before the budget cut, and both are ones coef 40 also solved, so it
says nothing yet about the items that broke — it is not evidence that 60 is better.

**Caveat that limits all of the above: n=6, so one item is 17 accuracy points.**
The cache result (a 2-item gap, and a mechanism visible in the EOS rate) is worth
believing directionally. The SEAL result is one lost solve away from neutral and
needs n=30 before it is quotable.

## 3b. Every failure is a failure to *reach* an answer (2026-09-23, free)

`--mode answers` scans the saved generations for `\boxed{}` and asks whether the
gold answer appears anywhere, at any point, regardless of how the run was graded.
Across all 26 generations:

| arm | n | correct | ever wrote gold | never boxed anything |
|---|---:|---:|---:|---:|
| real | 6 | 4 | **4** | 2 |
| none | 6 | 2 | **2** | 4 |
| real_seal40 | 6 | 3 | **3** | 3 |
| real_seal60 | 2 | 2 | **2** | 0 |
| real__bs16 | 6 | 2 | **2** | 3 |

**"Correct" and "ever wrote the gold answer" never disagree — 0 of 26 cases.**
That rules out two whole classes of explanation at once. Nothing is being
produced and then thrown away, so the model is not talking itself out of answers,
and our extraction is not losing them either. And the failures do not merely pick
the wrong answer: in almost every failing run the count of boxed expressions is
**zero**. The model is still mid-computation when the budget ends.

This is the third and strongest version of the same correction. We thought the
waste was non-termination, then degenerate looping, then verbosity. It is none of
those: **the failing items never get to an answer inside 8192 tokens.** Which
means accuracy on this set may be measuring convergence *speed* rather than
reasoning ability — and if so, every accuracy number in this project needs reading
that way, including the cache result in §3, where `none`'s collapse shows up
precisely as 4 of 6 items never boxing anything.

It also reframes SEAL cleanly. SEAL did not make items 10 and 18 *wrong*; it made
them **not finish**, taking both from solved-with-EOS to zero boxed answers. And
`real__bs16` item 4 is the one exception worth noting in the other direction: it
terminated with a confident wrong answer (`134`), so the batching bug does not
merely truncate, it changes the reasoning.

**The one experiment that resolves this is a budget extension** — `insight` stage,
~45 min. Give items 1 and 2 three times the budget under `real`: if they converge,
the ceiling here is speed and not capability. Give items 10 and 18 twice the budget
under `seal40`: if they come back, SEAL only slowed convergence rather than
breaking it. The same stage runs `shuf` to separate the cache's *content* from its
mere presence, the control that is missing between `real` (0.667) and `none`
(0.333).

### Grouped decoding failed parity — do not use it for accuracy (2026-09-22)

Measured, not predicted. Six items, batch of 6, left-padded prompts and
left-padded caches, against the same six decoded one at a time:

| | unbatched | batch 6 |
|---|---:|---:|
| accuracy | **0.667** (4/6) | **0.333** (2/6) |
| identical text | — | **0/6** |
| same token count | — | 2/6 |
| wall clock | 933s | 533s (**1.75x**) |

Items 4 and 10 flipped from correct to wrong (`110`→`134`, `104`→`33`), and item
10 stopped terminating at all, running 5848 tokens unbatched to the full 8192
batched. So grouping bought 1.75x and cost **half the solves**. Not a tie-break
perturbation — systematic degradation.

### Why (revised 2026-09-23; the first explanation was wrong)

`exp_aime_localize.py` never set `tokenizer.padding_side = "left"`, which every
other batched script in this repo does. That is fixed — and it is **not** the
explanation: the right-padded attempt is `run_logs/blitz_aborted.txt` (21:37 UTC),
which was killed on the warning and produced **no rows**. All six `real__bs16` rows
come from `run_logs/blitz.txt` (21:41 UTC) with zero padding warnings.

This document previously blamed a shared `cache_position = arange(pmax, pmax + L)`
opening a per-item RoPE gap. **Retracted.** `scripts/diag_batch_parity.py` runs the
real `pad_caches_left`/`decode_batch` wiring against a tiny random-weight Qwen3 on
CPU and measures it directly:

- Calling `model()` **directly** with a shared `cache_position`, an item whose cache
  is 6 short of `pmax` does get positions `[14,15,16]` instead of `[5,6,7]`, and its
  first-step argmax flips. So the mechanism is real — but this is not the code path.
- Through **`generate()`**, which is what `decode_batch` calls, positions are correct:
  `generate()` re-derives per-sequence `position_ids` from the attention mask.
- **float32: 0 of 48** batched items diverged, with cache shifts up to 15 positions.
- **bfloat16: 7 of 48** diverged, at statistically indistinguishable rates for
  nonzero shift (5/35) and zero shift (2/13).

Divergence tracks dtype, not position. The cause is **batch-variance in bf16**:
batch size changes kernel reduction order, logits move at the 1e-3 level, and greedy
`argmax` flips on near-ties. The per-item divergence points (455–1571 characters in,
not 0) fit that and not a structural position error, which would bite at token 1.

The 1.75x also falls well short of the ~4x the arithmetic predicted, so per-step cost
grows faster with batch than weight-streaming alone explains.

**Consequences.** There is no fix that makes bf16 batch 6 bit-match batch 1, so
"repair batching" is not a task. Grouped decoding stays valid only under a **fixed
batch composition** — same items, same groups, same batch size for every arm — and
may never be compared against unbatched rows. `--mode compare` now **exits non-zero**
on divergence, and `blitz`/`sweep`/`localize30`/`aime25_sweep` require a parity
certificate for their batch size or fall back to `DECODE_BS=1`.

The probe rows are kept under `real__bs16` rather than deleted, because the failure
is a result. (That label records the *configured* cap; the realized batch was 6, since
the parity run was scoped to six tapes.)

**And the sharper reading:** `real__bs16` differs from `real` only in arithmetic
order, yet accuracy moved 0.667 → 0.333 with items 4 and 10 flipping. That is a free
noise estimate, and it says two of the baseline's four solves are unstable under a
perturbation that should not matter. **No effect smaller than ±2 items is measurable
on this set** — which covers every SEAL effect reported here. Greedy decoding is
reproducible but not stable, and it gave false confidence.

### Audit of the other batched results

`exp_frozen_seal.py`, `exp_one_role.py` and `exp_math_ladder.py` share this decoder,
and their default batch sizes are 1 for AIME but **4** and **20** elsewhere — so the
GSM8K/MATH numbers, including the **GSM8K SEAL result**, were collected batched in
bf16. All three do set `padding_side="left"`; `exp_aime_localize.py` was the only
offender.

The structural check that matters, they pass. `exp_frozen_seal.py` loops
`for batch: for arm in (frozen, frozen_seal)`, so both arms see the same items in the
same group at the same batch size and receive the same numerical treatment. That is a
valid paired comparison, and it is the pattern to preserve. Two caveats remain:

- The resume filter `need = [it for it in batch if method not in done]` can leave the
  two arms with **different** `B` on a resumed run, silently breaking the pairing.
- They record `"judger_s": t_j / B` — batch wall clock over batch size. That is
  amortized throughput, **not single-request latency**, and cannot support a latency
  claim. `exp_aime_localize.py` records `None` for grouped rows instead, on purpose.

### Order of operations

1. `bash scripts/run_aime_localize.sh parity` — ~6 min, and **expect it to fail** on
   the evidence above. It now writes a certificate on success and leaves batched
   stages blocked on failure. Run it anyway: it is how you learn the batch size you
   are allowed to use, and it costs six minutes.
2. **Establish the noise floor.** A perturbation that should not matter moved accuracy
   0.667 → 0.333, so the current design cannot resolve the effects being claimed.
   Either sample k≥4 per item at temperature or go to n=30, and report paired
   item-level uncertainty. Steps 3–5 are uninterpretable without this.
3. `bash scripts/run_aime_localize.sh sweep` — ~1.5 h at `DECODE_BS=1` (~5 h if the
   parity gate blocks batching, which it should). n=30 AIME-2024 for `real`, `none`,
   and the SEAL coefficients. Answers the real accuracy at a defensible n, whether
   the cache matters once the item selection is no longer curated, and whether SEAL
   buys tokens without costing termination.
4. `bash scripts/run_aime_localize.sh localize30` — ~1.7 h. The localization arms
   at n=30, now framed as accuracy/KV-memory rather than latency.
5. `bash scripts/run_aime_localize.sh aime25_sweep` — ~1.5 h. Held-out AIME 2025,
   into its own `out_dir` *and* its own tape subdirectory. Only worth it once
   AIME-2024 says something.

Report with `--mode latency --exclude_arms real__bs16`, which keeps tokens-to-EOS,
correctness under the stopping policy, and work-under-cap separate, pairs each arm
against the baseline's own completed-and-correct items, and marks censored aggregates
as lower bounds instead of averaging them into a meaningless mean.

n=6 is the reason several earlier conclusions had to be walked back; at that size
one item is 17 accuracy points. Everything above is n=30 for that reason.
`restore` pulls completed rows back from `PERSIST_DIR` at stage start, so these
stages can be run across several pods without repaying for finished decodes.

Operational: `restore` pulls `rows.jsonl`, `agent_times.jsonl` and `texts/` back
from `PERSIST_DIR` at stage start and never clobbers newer local files, so
completed decodes carry across pods. `--mode budget` and `--mode loops` are
CPU-only and run off saved `texts/`, so you can stop the pod and iterate on
analysis locally for free. That is the whole point of saving generations.
