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
budget. Two directions, in priority order:

1. **Cap / early-exit.** If the 4 solved items stay solved under a ~4k cap, that
   is a ~2x mean-latency cut for free. `--mode budget` was written to answer this
   offline from saved generations (replays a hard cap, no GPU) — it needs one
   completed `views` sweep to run against. Note all 4 solves used ≤7653 tokens
   and 2 of 4 used ≤5848, so a cap strictly between 5848 and 8192 is already
   guaranteed to lose nothing on this n=6 and would cut both 188s runs.
2. **Detect non-termination early.** The failures are distinguishable by
   behaviour (no EOS, looping), not by answer quality. A cheap loop//repetition
   detector that aborts is pure latency win with zero accuracy cost on these 6.

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

1. `collect` (~40s for 6 items) then `views --view_arms real` to confirm parity.
2. `--mode budget --budget_arms real` — the cap curve. **Highest value per GPU
   second, and it is the only arm that targets the 99.4%.**
3. Only then the localization arms (`c3`, `c23`, `isolated`, `evict_seg`). Reframe
   these as accuracy/memory questions; they are no longer a latency story.
