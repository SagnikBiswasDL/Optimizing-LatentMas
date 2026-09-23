# Briefing for an incoming agent

You are being pointed at this repo to propose how to get better performance out of
LatentMAS. This file exists because the repo contains several results that are easy
to over-read, a few that are outright invalidated, and one bug that silently
produces plausible-looking numbers. Read this before `MASTER_BRIEF.md`.

Written 2026-09-23, after a session that produced three negative results and one
methodological correction.

---

## 0. The goal is a LATENCY result. Here is the whole latency model.

**The end deliverable is a latency win, not an accuracy win.** Accuracy matters only
as a constraint: a latency win that loses correctness is not a win (that is exactly
how token-capping died). Read this section as the specification.

Three measurements, in increasing order of usefulness:

**(i) The Judger is the only thing that costs anything.** On AIME with Qwen3-14B the
Judger's decode is **99.4%** of a 156s item; all three silent agents together are
**1.0s**. Replicated exactly across two runs under greedy decoding.

**(ii) Decode throughput is a hard constant, invariant to the cache.** Across all 20
unbatched rows in `artifacts/aime_localize/rows.jsonl`:

```
tok/s = 42.51 +/- 0.31   (min 41.82, max 42.85, spread 1.025x)
```

and that holds while `cache_pos` ranges **0 to 962** and cache size ranges **0 to
150 MB**. The zero-cache `none` arm decodes at 42.76 tok/s; the 150 MB `real` arm at
41.91. Attending over the upstream latent cache is free at these scales, because a
few hundred cache positions are noise next to thousands of generated tokens.

**(iii) Therefore latency is exactly token count:**

```
latency_seconds = tokens_generated / 42.5
```

This single equation should drive the whole plan, and it has three consequences:

- **Cache surgery cannot produce a latency win.** Eviction, compression, role
  dropping, `c1`/`c2`/`c3`, medoids, cheaper scaffolds — all of it targets either the
  1.0s of agent time or a cache whose size provably does not affect tok/s. These are
  defensible as **KV-memory** results. They are not latency results. If a plan
  proposes them for latency, it has not read this section.
- **There are exactly two real levers: emit fewer tokens, or raise the 42.5 tok/s
  floor.** The second is serving-stack work (fix batching per §1a, vLLM, speculative
  decoding, quantization) and is orthogonal to everything else in the repo. The first
  is the research contribution.
- **You can report tokens instead of seconds.** Because tok/s is constant to 2.5%,
  token count is a faithful and hardware-independent latency proxy. This makes the
  experiments cheaper and the result far more robust than wall-clock timing.

### ⚠️ SEAL is currently a net latency LOSS. Retracted claim, read this.

An earlier version of this file reported "1.31x–1.63x lower latency at preserved
correctness" from the items where both arms terminated and were correct. **That was
a selected subset and the number is not a speedup.** Run `--mode latency`:

| arm | anchor items | baseline tok | arm tok | ratio | kept | lost | speedup on kept |
|---|---|---:|---|---|---|---|---|
| `none` | 0,4,10,18 | 23163 | ≥27705 | **≥1.196** | — | 4,10,18 | 0.83x |
| `real_seal40` | 0,4,10,18 | 23163 | ≥24087 | **≥1.040** | 0,4 | 10,18 | 1.33x |
| `real_seal60` | 0 only | 2600 | 1599 | 0.615 | 0 | — | 1.63x |

Paired over the four items the **baseline** completes and gets right, SEAL40 spends
**at least 4% more tokens than baseline**, because it converts items 10 and 18 from
solved-and-terminating into capped runs. A capped run books the full 8192 and would
book more if uncapped, so **de-censoring can only make this worse.** The "1.33x" is
real but is conditional on the two items SEAL kept, and quoting it without the two it
lost is the same selection error that produced the retracted cache result.

`real_seal60` looks best in the table on one item and has **n=1** completed-correct
observation. It is not evidence.

**So there is currently no latency result, in either direction.** What exists is a
mechanism worth chasing: on problems it does not break, steering cuts tokens 23–31%.
The open question is whether a coefficient exists that captures that without
destroying termination. Nothing in the data says one does.

### The blocker: 60% of the data is right-censored

**12 of 20 unbatched runs hit the 8192-token budget exactly**, so their latency is a
censored observation (">= 192s"), not a measurement. Consequences:

- **No mean latency or mean token count is computable from the current data.** Any
  such average is really an average of the cap.
- The reported "SEAL made mean tokens go *up*" is a **censoring artifact**: SEAL
  pushed two items from terminating into the cap, and a capped run books 8192.
- **Fixing this is mostly free: raise the budget** until runs terminate, then measure.
  A budget-extension probe is already written (§2) and never run.

For terminating runs, time-to-first-correct-boxed-answer equals time-to-EOS (the
answer lands at ~100% of the generation), so there is no hidden post-answer waste to
reclaim and the token count is already the right metric. Post-answer rambling only
occurs in capped runs. That check has been done; do not redo it.

## 1. The three landmines

**(a) Grouped decoding changes greedy outputs, and it is NOT a bug you can fix.**
Batching the Judger fails a byte-level parity check: batch 6 gave 1.75x throughput
and **halved the solves** (0.667 → 0.333), with 0 of 6 generations identical to
their unbatched counterparts.

An earlier version of this file blamed a shared `cache_position` opening a
per-sequence RoPE gap. **That explanation is wrong and has been retracted.**
`scripts/diag_batch_parity.py` settles it on CPU in about 5 seconds:

- Through a direct `model()` call, a shared `cache_position` *does* corrupt
  positions (item with a 6-short cache gets positions `[14,15,16]` instead of
  `[5,6,7]`). But that is not the production path.
- Through `generate()` — which is what `decode_batch` calls — positions are
  **correct**, because `generate()` re-derives per-sequence `position_ids` from the
  attention mask and ignores the shared `cache_position` for that purpose.
- In **float32, 0 of 48** batched items diverged from their unbatched runs, with
  cache shifts up to 15 positions.
- In **bfloat16, 7 of 48** diverged — and at the same rate whether the cache shift
  was nonzero (5/35) or exactly zero (2/13).

Divergence tracks **dtype, not position**. The cause is that batch size changes
reduction order in the attention and GEMM kernels, perturbing logits at the
1e-3 level; greedy `argmax` occasionally flips on a near-tie, and the trajectories
separate from there. In the real run, divergence starts 455–1571 characters in, not
at character 0 — exactly the signature of "identical until the first close call."

**Consequences, which matter more than the diagnosis:**
- There is **no fix** that makes batch 6 bit-match batch 1 in bf16. Do not plan one.
  Batch-invariant kernels or fp32 decode would do it, and both cost more than the
  1.75x they buy.
- Batching is still **scientifically usable under a fixed batch composition**: if
  every arm decodes the same items in the same groups at the same batch size, all
  arms get the same numerical treatment and the paired comparison is valid. What you
  may never do is compare a batched arm to an unbatched one.
- The safeguard is now enforced: `--mode compare` **exits non-zero** on divergence,
  and the driver's `blitz`/`sweep`/`localize30`/`aime25_sweep` stages require a
  parity certificate for the batch size they are about to use, falling back to
  `DECODE_BS=1` rather than proceeding. `ALLOW_UNCERTIFIED_BATCH=1` overrides, loudly.

**(a′) The same evidence says the n=6 measurement is dominated by noise.**
`real__bs16` is not corrupt data. It is the same model, prompts, cache and greedy
policy, differing only by a semantically-neutral change in arithmetic order — and
accuracy moved **0.667 → 0.333**, with items 4 and 10 flipping correct→wrong. Two of
the baseline's four solves are that fragile. **Any effect smaller than ±2 items on
this set is unmeasurable**, which includes every SEAL effect reported so far. Greedy
decoding gave false confidence here: it is reproducible run-to-run, but it is not
stable against perturbations that ought to be irrelevant. Plan for multiple samples
per item, or a much larger n, before believing any arm difference.

**(a″) Audit of the other batched results.** Every experiment sharing this decoder
sets `padding_side="left"` correctly; `exp_aime_localize.py` was the only one that
did not, and that is fixed. Default batch sizes are `1` for AIME but **4**
(`exp_one_role.py`) and **20** (`exp_math_ladder.py`) elsewhere, so the GSM8K/MATH
numbers — including the **GSM8K SEAL result** — were collected batched, in bf16.

The structural check they need, they pass: `exp_frozen_seal.py` loops
`for batch: for arm in (frozen, frozen_seal)`, so both arms see the same items in the
same group at the same batch size. That makes them paired and internally valid. Two
caveats survive:

- The resume filter (`need = [it for it in batch if method not in done]`) can leave
  the two arms with **different** `B` on a resumed run, silently breaking the pairing.
- Those scripts record `"judger_s": t_j / B` — batch wall clock divided by batch size.
  That is amortized throughput, **not single-request latency**, and must never be
  quoted as a latency result. `exp_aime_localize.py` deliberately records `None`
  instead for grouped rows.

**(b) `rows.jsonl` mixes trusted and deliberately-untrusted arms.**
Naive aggregation over all rows gives 0.5, which is meaningless. Specifically:
`real__bs16` is the **known-bad batched probe**, kept on purpose as evidence of
the bug — exclude it from every analysis. `real_seal60` has **n=2**, and both
items are ones `seal40` also solved, so it is *not* evidence that coef 60 is
better. Six rows have `judger_s = null` by design (grouped rows forfeit per-item
timing); code that treats null as 0 will silently corrupt latency means.

**(c) The n=6 focus set is selected, not random — and one conclusion was circular.**
Everything recent is n=6 on AIME-2024 items `{0,1,2,4,10,18}`, where one item is
17 accuracy points. Worse, `{4,10,18}` were chosen *because* Real K=10 solves them
and Frozen/Mean-Replay does not. So when `real` (0.667) beats `none` (0.333), the
entire gap is items 4 and 10 — both from that selected set. **That result is
circular and has been retracted in the docs.** Any claim of the form "the cache
helps" needs the other 24 items. The same bias applies to anything else you
measure on this set.

## 2. The most important open question (and it is a latency question)

`--mode answers` scans every saved generation for `\boxed{}` and asks whether the
gold answer appears *anywhere*, however the run was graded. Across all 26
generations, **"graded correct" and "ever wrote the gold answer" never disagree
(0/26)**, and failing runs emit **zero** boxed expressions — the model is still
mid-computation when the budget ends.

That rules out a lot at once: nothing is produced and then discarded, extraction
is not losing anything, and the failures are not wrong answers. It also means
**accuracy on this set may be measuring convergence speed, not reasoning
ability.** If so, "improve accuracy" and "reduce tokens" are the same axis, and
several plausible-sounding interventions are incoherent.

**This is unresolved and cheap to resolve. Run it first:**

```bash
bash scripts/run_aime_localize.sh insight   # ~45 min, unbatched
```

It gives items 1 and 2 three times the budget (do they converge? → speed-bound,
not capability-bound), gives items 10 and 18 twice the budget under SEAL (did SEAL
break the reasoning or merely slow it?), and runs the `shuf` control that
separates the cache's *content* from its mere presence — the arm missing between
`real` and `none`. The stage is written and dry-run tested but **has never been
executed.** Do not plan accuracy work before you know this answer.

## 3. What is already closed — do not re-propose these

Three successive hypotheses about Judger waste were each measured and killed. The
docs are layered chronologically, so a top-down read will surface dead ones as if
live:

| Idea | Verdict |
|---|---|
| Hard token cap / early exit | Dead. 1.20x for **−0.167 accuracy**; 1.71x for −0.500. |
| Degenerate-loop detection | Dead. Detector fires on **nothing**; the long runs are doing real non-repetitive work. |
| SEAL at the Judger (verbosity) | Negative at n=6. Accuracy 0.667 → 0.500 and mean tokens go **up**, because EOS rate halves. |
| Cache surgery for latency | Dead as a *latency* story (§0). Still open on memory/accuracy. |

SEAL's detail is more interesting than its verdict, and is the one live thread in
the efficiency direction: on the items it keeps it is a **−31% / −23% token win**,
and it uniquely solves item 1. It failed by pushing two items from
solved-with-EOS to never-finishing. So token reduction on already-solvable
problems is real; it is the accuracy side that breaks.

## 4. Infrastructure that will waste your time or money

- `/workspace` on the pod is **MooseFS**. `torch.save` corrupts there mid-file.
  Big KV tapes must go to container-local disk via `TAPE_DIR`; that is what
  `save_tape`'s buffer-then-atomic-rename and the `tape_dir` flag exist for.
- `/root` is **wiped when the pod stops**. A finished sweep was lost this way
  once. `PERSIST_DIR` mirrors small artifacts to the network volume after **every
  batch**, and `restore` pulls them back at stage start so a new pod does not
  repay for finished decodes.
- **Export `HF_HOME=/workspace/.cache/huggingface`** or you will re-download 28GB
  of Qwen3-14B.
- The pod **cannot stop itself** (`runpodctl` is installed but has no API key).
  An idle H200 burned ~6.5 hours of credit in this session. Stages write a
  `DONE_<stage>` sentinel to `PERSIST_DIR` and `--time_budget_s` stops cleanly
  between batches; use them, and tell the human to click Stop.
- `list_tapes` loads **every** tape (3.2GB at n=30) before `--view_indices`
  filters, so a 6-item run pays the full load. Easy, worthwhile fix.

## 5. Code facts worth knowing before you edit

- Arm naming is parsed, not free-form: `real_seal40` → base arm `real` with
  steering coef 40 (`parse_arm`), so a coefficient sweep runs in **one model load**
  by retuning the hook. `--method_tag X` appends `__X` to the row label **without**
  changing the cache view, which is how the same arm gets re-decoded and compared.
- `--indices` selects what to **collect** and defaults to the curated six.
  `--view_indices` scopes a **decode**. They are deliberately separate: filtering
  views on `--indices` would silently shrink every n=30 sweep to six items.
- `shuf` picks its donor as `others[(idx+1) % len(others)]` — not a clean
  permutation, and two items can draw the same donor. Fine for a presence-vs-content
  control, not fine if you need a rigorous permutation test.
- Greedy (`temperature=0`) throughout, so runs are deterministic and single-seed is
  not a variance problem — but it also means **n=6 carries no variance estimate at
  all**. If you want error bars without 30 items, sampling at temperature with
  several draws per item is the cheaper route to a distribution.
- CPU tests exist and are fast (69 total, ~4s): `tests/test_decode_group.py`
  (batch/row alignment, timing attribution, pricing), `tests/test_segment_kv.py`
  (KV slicing, role-aware eviction), and `tests/test_safeguards.py` (tape identity,
  the parity gate, censoring-aware latency). `--mode budget`, `loops`, `answers`,
  `compare`, `latency` are all CPU-only and run off saved generations, so iterate
  there for free. `scripts/diag_batch_parity.py` reproduces the batching failure on
  CPU with a random-weight model in seconds — no GPU, no checkpoint.
- **Tape reuse is now validated.** `tape_root` appends a `task__model__k` subdir, and
  a tape is rejected on load unless its task, k, model and question hash match the
  current config. Previously `aime25_sweep` swapped `out_dir` but kept `TAPE_DIR`, so
  AIME-2025 item 0 would have silently decoded against AIME-2024's cache. Tapes
  written before this carry no `identity` block and are reported as unverifiable
  rather than assumed good.

## 6. Unexplored, ordered for a latency deliverable

1. **Establish the noise floor before measuring anything** (§1a′). A
   semantically-neutral perturbation moved accuracy 0.667 → 0.333, so the current
   design cannot resolve the effects being claimed. Either sample k≥4 per item at
   temperature, or go to n=30, and report paired item-level uncertainty. Everything
   below is uninterpretable without this, and it is the cheapest thing to fix.
2. **Raise the budget to de-censor** (§0). 12 of 20 runs are censored; no mean is
   computable. Note the artifacts store no token IDs, so extended runs **cannot be
   resumed** — they re-decode from scratch, and you should verify the new run
   reproduces the old prefix before trusting the extension.
3. **Screen coefficients on the regression items first.** SEAL40 broke items 10 and
   18; any candidate that also breaks them is dead, so screen there for ~2 items of
   cost before spending the full cohort. `parse_arm` runs the whole sweep in **one
   model load** (`real_seal20`, `real_seal80`, ...). Expect non-monotonicity; pick one
   fixed coefficient by aggregate token cost subject to preserved correctness, with
   **no per-item oracle selection.**
4. **Then, and only then, n=30 with the frozen coefficient**, plus AIME-2025 as a
   holdout with its own tape directory. Report every item, including failures.
5. **Serving-stack throughput** (vLLM, speculative decoding, batch-invariant
   kernels). This is the *other* lever and it is entirely untouched. It raises the
   42.5 tok/s floor, composes with any token reduction, and unlike steering it cannot
   cost accuracy. For a latency deliverable this may well be the better bet.
   Caveat: batching improves throughput, not single-request latency — do not multiply
   the two (§1a″).
6. **Per-role localization** (`c1`/`c2`/`c3`/`c23`), never run: harness, segment-aware
   KV slicing, and eviction are written and unit-tested. Frame as **KV memory**, not
   latency (§0), and note that a memory win only becomes a throughput win if you show
   the freed memory buys useful concurrency. Same for K-sweeps.

## 7. One meta-warning

This program's dominant failure mode has been believing an appealing mechanism
before measuring it. Non-termination, then looping, then verbosity, then "the
cache helps" — each was plausible, each was written up, each turned out to be
wrong or confounded. The fastest way to be useful here is to propose the cheapest
experiment that could **falsify** your hypothesis, and to prefer the CPU-only
analyses over anything that needs a GPU. Be suspicious of any number in these docs
that does not state its `n`.
