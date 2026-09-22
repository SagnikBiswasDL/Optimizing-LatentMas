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
budget. But the two obvious interventions are **not** equally good, and the
arithmetic matters:

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

The detector's machinery is tested; its **threshold is uncalibrated** because it
has never seen a real degenerate AIME tail. Calibrating it is the first thing the
next GPU session should produce. Success criterion: fires on items 1 and 2, fires
on neither of the four solves.

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

Run `bash scripts/run_aime_localize.sh quick` — one ~20 minute stage that does all
of the below and writes a `DONE_quick` sentinel to `PERSIST_DIR` so you know when
it is safe to stop the pod.

1. `collect` (~40s for 6 items) then `views --view_arms real` (~15 min) to confirm
   parity with the numbers above and, critically, to save the generations.
2. `--mode loops` — calibrate the degenerate-tail detector. **Highest value per
   GPU second: it is the only thing targeting the 99.4%, and the 26% saving is
   real if the threshold separates items {1,2} from {0,4,10,18}.**
3. `--mode budget` — the cap curve, to close that option with a number rather
   than an argument.
4. Only then the localization arms (`c3`, `c23`, `isolated`, `evict_seg`), at
   n=30 if they are worth it at all. Reframe them as accuracy/memory questions;
   they are no longer a latency story.

Operational: both analysis modes are CPU-only and run off saved `texts/`, so once
step 1 has persisted you can stop the pod and iterate on the detector locally for
free. That is the whole point of saving generations.
