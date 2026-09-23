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

### The latency result already half-exists

Restricting to same-item pairs where **both arms terminated and both were correct**
(the only honest latency comparison available):

| item | `real` | SEAL coef 40 | SEAL coef 60 |
|---|---|---|---|
| 0 | 61.1s / 2600 tok | 42.9s / 1807 tok (**1.42x**) | 37.4s / 1599 tok (**1.63x**) |
| 4 | 180.1s / 7653 tok | 137.6s / 5896 tok (**1.31x**) | not run |

**1.31x to 1.63x latency reduction at preserved correctness.** That is the paper, in
embryo, at n=3 observations over 2 items. Extending and de-censoring this is the
highest-value research task in the repo. Note also that coef 60 beat coef 40 on the
one item both ran, and 60 is the largest coefficient ever tried — the coefficient
sweep is unfinished and is the cheapest possible win.

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

**(a) Batched decoding is broken. Use `--decode_bs 1`.**
Batching the Judger through `decode_batch` fails a byte-level parity check: batch
6 gave 1.75x throughput and **halved the solves** (0.667 → 0.333), with 0 of 6
generations identical to their unbatched counterparts. One cause is fixed (a
missing `padding_side="left"`); the remaining cause is position handling — a single
`cache_position` is shared across a batch whose KV caches and prompts are both
left-padded to different real lengths, so each sequence gets a different-sized
artificial RoPE gap between cache and prompt. This is the most dangerous thing in
the repo because the failure is silent and looks like a normal accuracy number.
If you want the throughput back, **fixing this properly is the single highest-value
engineering task available** — it is what makes n=30 affordable. Verify any fix
with `--mode compare`, which diffs two arms' saved generations byte-for-byte.
This also puts a question mark on earlier batched results from
`exp_frozen_seal.py` and `exp_one_role.py`, which is where the **GSM8K SEAL
numbers** came from. They are less exposed (a frozen cache is uniform across the
batch, so only prompt padding bites) but they are not cleared. Check before citing.

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
- CPU tests exist and are fast: `tests/test_decode_group.py` (batch/row alignment,
  timing attribution, pricing) and `tests/test_segment_kv.py` (KV slicing and
  role-aware eviction). `--mode budget`, `loops`, `answers`, `compare` are all
  CPU-only and run off saved generations, so iterate there for free.

## 6. Unexplored, ordered for a latency deliverable

1. **Raise the budget to de-censor the data** (§0). Without this there is no
   denominator and no mean. Cheapest high-value action in the repo.
2. **Finish the SEAL coefficient sweep.** Only 40 and 60 have been tried; 60 was
   better than 40 and is the largest ever run. `parse_arm` means a whole sweep runs
   in **one model load** (`real_seal20`, `real_seal80`, ...). This directly extends
   the 1.31–1.63x result and is where the headline number will come from.
3. **Fix batched decoding** (§1a). A separate, multiplicative ~1.75x on the tok/s
   side, and it is what makes n=30 affordable. Verify with `--mode compare`.
4. **n=30 unbiased tokens-to-solution for `real` vs `real_seal*`** — the actual
   experiment behind the paper, and it de-confounds §1c for free.
5. **Serving-stack throughput** (vLLM, speculative decoding). Raises the 42.5 tok/s
   floor and is orthogonal to the steering work, so it composes with it. Untouched.
6. **Per-role localization** (`c1`/`c2`/`c3`/`c23`), never run: harness, segment-aware
   KV slicing, and eviction are written and unit-tested. Frame as **KV memory**, not
   latency (§0). Same for AIME-2025 holdout and K-sweeps — worth doing, but they are
   not the latency result.

## 7. One meta-warning

This program's dominant failure mode has been believing an appealing mechanism
before measuring it. Non-termination, then looping, then verbosity, then "the
cache helps" — each was plausible, each was written up, each turned out to be
wrong or confounded. The fastest way to be useful here is to propose the cheapest
experiment that could **falsify** your hypothesis, and to prefer the CPU-only
analyses over anything that needs a GPU. Be suspicious of any number in these docs
that does not state its `n`.
