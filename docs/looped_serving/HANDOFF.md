# Looped serving: handoff

For a research agent picking this work up. Branch `research/looped-serving`
(remote `git@github.com:SagnikBiswasDL/Optimizing-LatentMas.git`), based on
`641d891`. Last updated 2026-11-12, after the held-out run. The GPU is stopped.

`docs/looped_serving/EVIDENCE.md` is the ledger of what the runs support.
This file says where things stand and how to run them.
`docs/looped_serving/NEXT_AGENT.md` is the brief for an agent deciding the
next experiment. That agent reads, recommends, and does not start a GPU.

## The question

Model: `ByteDance/Ouro-1.4B`, a looped language model (the same layer stack is
applied 1 to 4 times per token; "depth"). Engine: `InferMatrixLab/vllm-rlt` at
`e0a5515` (v0.2.0), unmodified, driven in process. Task: MATH-Train questions,
answers in `\boxed{}`, graded by `math-verify` 0.9.0.

The project asks whether choosing answer mode, sample count, recurrent depth and
launch timing together delivers more correct answers within a latency target
than a strong fixed or adaptive-sampling policy on the same engine. The goal is
a paper claim. No claim is established yet.

## Where it stands

### Established

1. **Engine fidelity.** In fp32 the engine reproduces the Hugging Face
   reference exactly at every depth when either (a) the last prompt token is
   withheld and the engine emits it (`--prompt_mode boundary_shift`), or (b) the
   engine itself is depth-limited (`--total_ut_steps r --prompt_mode stock`).
   In bf16 the tokens depend on batch composition, so two bf16 runs with the
   same seeds are not the same candidate pool.
   `artifacts/looped_serving/fidelity_gate/`.
2. **Memory does not shrink with request depth.** The engine reserves all four
   KV planes for a request at any depth. Only a depth-limited engine reserves
   fewer.
3. **The first answer token picks an answer mode.** `<think>` (token id 3)
   opens a long free-form reasoning block; anything else (96% of the time the
   token `To`) starts a short worked solution. In natural sampling at depth 3
   about 40% of candidates open `<think>`; they use about 70% of all decode
   tokens, most hit the 2,048-token cap without an answer, and those that do
   finish are 98–100% correct. The think share rises with depth (33% / 42% /
   52% at depths 2 / 3 / 4) and with question level.
   `artifacts/looped_serving/mode_audit/modes.md`.
4. **Token savings are not latency savings.** On the smoke split, "two samples,
   then two more on disagreement" used 17% fewer loop-tokens than a fixed
   four-sample vote but had 56% higher mean latency and more than double the
   p95 (the follow-up waits for a slot and starts late). Cancelling on a strict
   vote certificate cut mean latency 15% at equal accuracy within noise.
   `artifacts/looped_serving/profile_smoke/table.md`.

### Held-out result (this is the current reading)

Depth-limited depth-3 engine, bf16, stock prompt mode, CUDA graphs and static
buffers, temperature 0.6, top-p 0.95, 32 slots, one seed. Validation split, 200
questions the calibration comparison never saw
(`artifacts/looped_serving/pod_out/modes_validation/`). Graded with
`math-verify` 0.9.0. Paired intervals are 10,000-draw bootstraps, seed 0.

| Arm | Accuracy (95% CI) | Capped | Charged loop-tokens/q | Latency mean / p50 / p95 | Questions/min |
|---|---|---|---|---|---|
| Natural SC@4, cap 2,048 | 78.0% (72.0, 83.5) | 27.1% | 14,115 | 49.8 / 55.0 / 81.8 s | 11.5 |
| Direct SC@4, cap 2,048 | 79.5% (74.0, 85.0) | 2.9% | 9,201 | 25.5 / 18.8 / 65.2 s | 20.4 |
| Natural SC@4, cap 4,096 | 84.0% (79.0, 89.0) | 18.4% | 19,709 | 92.3 / 66.0 / 209.0 s | 6.5 |
| Quorum 2-of-4, direct, live | 82.0% (76.5, 87.5) | 2.4% | 8,745 | 21.1 / 14.8 / 63.1 s | 22.0 |

Paired, same 200 questions:

- Direct minus natural at cap 2,048: **+1.5 points**, interval −2.0 to +5.0,
  8 against 5, exact McNemar p = 0.58. The practice-set accuracy gain did not
  replicate.
- Natural at 4,096 minus natural at 2,048: **+6.0 points**, interval +2.0 to
  +10.0, 15 against 3, p = 0.0075. This is the comparison that replicates.
- Natural at 4,096 minus direct at 2,048: **+4.5 points**, interval +1.0 to
  +8.0, 11 against 2, p = 0.022. The longer monologue is more accurate and
  about three times slower (92 s against 26 s).
- The live quorum run is a separate bf16 sample, not the direct pool with early
  stopping. Its +2.5 against direct SC@4 is not a stopping-rule effect. On the
  calibration direct pool, replayed on the token clock, stopping at two
  agreeing answers kept correctness on all 400 questions and saved 5.5% of
  loop-tokens.

"Direct" means the answer starts with the token `To`. It does not turn
reasoning off. One closed-loop run per arm, not interleaved, so the latency
columns are unreplicated.

### Practice set, where the withdrawn claim came from

Calibration, 400 questions, same engine, cap 2,048, 64 slots
(`artifacts/looped_serving/pod_out/modes_calibration/`, code `a95b3e1`):

| Arm | Accuracy (95% CI) | Capped | Charged loop-tokens/q | Latency mean / p50 / p95 | Questions/min |
|---|---|---|---|---|---|
| Natural SC@4 | 77.5% (73.2, 81.5) | 29.9% | 14,628 | 101.6 / 118.7 / 162.4 s | 11.5 |
| Direct SC@4 | 83.8% (80.0, 87.2) | 2.2% | 9,118 | 49.3 / 36.0 / 129.2 s | 21.7 |

Paired: direct +6.25 points (interval +3.0 to +9.5); 36 only direct, 11 only
natural; p = 0.00035. The gain sat on the 92 questions where all four natural
samples opened `<think>` (natural 32%, direct 61%). Do not cite this as the
result. The 64-slot latencies are not comparable to the 32-slot validation run:
throughput is about 700–900 tokens/s regardless of slots, so closed-loop latency
follows tokens in flight.

Controls on the same 400 questions, from the second GPU (32 slots):

- Natural cap 4,096: 82.5% (78.8, 86.0), 19.5% capped, 20,530 loop-tokens/q.
  Paired against natural 2,048: +5.0 points (interval +2.0 to +8.2, 31 against
  11, p = 0.0029). Paired against direct 2,048: −1.25 (interval −4.5 to +2.0,
  p = 0.53). Once the monologue can finish, it ties the short answer here.
- Direct depth 4: 83.8%, identical to depth-3 direct (15 against 15, p = 1),
  at 12,059 loop-tokens/q against 9,118.
- Direct depth 2: 75.0%, −8.75 points against depth-3 direct (interval −12.2
  to −5.2, 9 against 44, p = 1.2e-6).
- Think, two samples, cap 4,096, on the 75 questions whose direct votes never
  reached a 3-of-4 quorum: 28.0% (18.7, 38.7), 82.7% capped, mean latency
  198.7 s. Direct SC@4 on those same 75 is 30.7%. A monologue fallback does
  not rescue them.

The `nothink` control (the model's own first token with `<think>` banned, up
to 64 redraws) aborted on both splits: 57 fresh-set candidates and 162
practice-set candidates were still `<think>` after 64 draws. No candidate file
was written. First-token mass on `<think>` is extreme. That control has not
separated "block the monologue" from "start with the word To."

Smoke split, 100 questions (`pod_out/modes_smoke/`): direct SC@4 82.0% against
75.0% for an earlier natural SC@4 on a four-plane engine at 32 slots (accuracy
only; 9 against 2, p = 0.065). Forced think, two samples, cap 4,096: 75.0%.
Of 18 questions the direct vote missed, a think sample finished on 3 and was
right on 2. Pilot only.

### Failed or weakened hypotheses (keep these)

- Depth routing as the main control: depth 3 and depth 4 four-sample votes are
  indistinguishable on 300 saved questions (+0.33 points, interval about
  [−3.3, +4.0]); depth 2 is about 10 points worse.
- A deeper fallback: "two at depth 3, then two at depth 4" selects the same
  answers as the always-run 3+3+4+4 vote.
- Two-then-two as a serving policy: slower than the fixed vote (see 4 above),
  even on an ideal token clock.
- Repetition or compression detectors for doomed traces: AUROC 0.4–0.5 until
  late. Capped traces are long reasoning, not loops.
- The earlier cap-extension result (75% → 79% after rerunning 114 capped
  candidates at 4,096) is a selective rerun with new trajectories and a
  different slot count: none of the 114 reproduced its old prefix. It is not a
  continuation and should not be cited as the effect of a longer cap. The
  fresh full-pool natural arms at cap 4,096 are the cap comparison.
- Forcing direct answers as an accuracy improvement that replicates on fresh
  questions at a matched 2,048 cap. It was +6.25 points on the 400-question
  practice set and +1.5 points on the 200-question fresh set (p = 0.58).
  Direct remains about twice as fast at that cap (25.5 s against 49.8 s).
- A think fallback on the questions direct votes cannot settle: 28% against
  30.7% direct on those 75 practice questions, with 83% of think samples capped.

## Review of 2026-11-11 and what was done about it

An external review of `a95b3e1` found four defects. All are fixed in `3a894da`
with regression tests; none affects the fixed-width arms above.

| Finding | Fix |
|---|---|
| `FastThenThink` waited forever when all direct votes had finished in a tie | Escalates (or falls back) once no direct candidate is running; the driver now fails if a question is left without work or without a final decision, and the grader refuses a run with a missing question |
| Follow-up launches still queued when their question was answered ran anyway | The driver registers them as `queued`, shows them to the policy, and drops them when the question is answered |
| A box cut off by the cap (`\boxed{12`) could vote | `answer_identity.answer_from_text`: only a complete box votes or is graded; the grader recomputes answers from saved text. Incidence in everything saved so far: 3 candidates (2 in smoke `d3_w8`, 1 in `d4_w4`), no accuracy change, none in the forced-mode arms |
| A cap extension of a forced-mode arm silently ran in natural mode | The extension keeps the source mode and refuses changed generation settings |

Also from the review: manifests now record a content hash of the executed
source tree and the installed model snapshot and engine commit
(`extra.verified`), because the `code_sha` in the first session's manifests
(`225d4a7`) predates the script that produced them. Arms run before `3a894da`
in this session carry `a95b3e1`, which is the code they ran.

Open review points: the fresh-question repeat is done, and the direct-mode
accuracy claim did not survive it. The confirmation reserve is still locked.
The serving claim needs fixed arrival traces and deadlines, not only the closed
loop. Related work to position against includes depth-portfolio methods (DPA)
and adaptive-thinking methods ("When to Continue Thinking", "Learning When to
Think"); hold the runtime fixed, and acknowledge ExactLoop. The rerun manifests
record code `e7963b3` with a dirty working tree, because `scripts/pod/setup.sh`
and `scripts/pod/run_queue.sh` were edited after that commit and have not been
committed.

## Layout

| Path | What |
|---|---|
| `looped/modes.py` | Answer modes: `natural`, `direct`, `think`, `nothink` |
| `looped/allocation.py` | Policies: `FixedWidth`, `TwoThenTwo`, `CertificateCancel`, `QuorumCancel`, `FastThenThink`; `parse_policy` specs |
| `looped/answer_identity.py` | Extraction, vote identity (`structural-v2`), pinned grader |
| `looped/experiment_manifest.py` | Run manifests, stable candidate ids and seeds, split checks, source hash |
| `serving/looped/rlt_backend.py` | The engine adapter (prompt modes, depth-limited engines) |
| `serving/looped/driver.py` | Closed-loop executor for a policy on a backend |
| `serving/looped/replay_backend.py` | Saved candidates on a token clock (a length proxy, never a latency) |
| `scripts/looped_fidelity_gate.py` | Engine against the reference |
| `scripts/looped_profile.py` | Fixed arms `<depth>x<width>[:<mode>][@<cap>]`; `--extend_from`; `--only_questions` |
| `scripts/eval_looped_policies.py` | Live policies with real launches and cancellations |
| `scripts/grade_looped_profile.py` | Offline grading; writes `table.json` and `table.md` per run directory |
| `scripts/replay_looped_policies.py`, `scripts/analyze_looped_modes.py` | Offline policy replay and the mode audit |
| `scripts/select_looped_questions.py` | Questions whose votes reach no quorum (gold-free) |
| `scripts/pod/` | Pod setup, sequential job queue, idle watchdog |
| `configs/looped_serving/` | Pins and the frozen splits |
| `artifacts/looped_serving/` | Fidelity gate, smoke profile, mode audit, split exports, this session's outputs (`pod_out/`) |

Splits (`configs/looped_serving/splits.json`): smoke 100 (profiling only),
calibration 400 (fit), validation 200 (choose), confirmation reserve 300
(locked until the policy is frozen). Final benchmarks: GSM8K test, MATH-500.
Do not tune on the reserve or the benchmarks.

## How to run

Local (CPU, Python 3.9 in `.venv`; grading needs Python ≥ 3.10 and `math-verify`):

```bash
.venv/bin/python -m pytest tests/test_looped_splits.py tests/test_looped_history_audit.py \
  tests/test_answer_identity.py tests/test_ensemble_state.py tests/test_experiment_manifest.py \
  tests/test_looped_cache.py tests/test_looped_backends.py tests/test_looped_allocation.py -q   # 160 pass

uv run --no-project --python 3.12 --with math-verify==0.9.0 --with numpy \
  python scripts/grade_looped_profile.py --runs <run dir> \
  --gold artifacts/looped_serving/splits/<split>.gold.jsonl
```

GPU (one RTX PRO 6000 Blackwell 96 GB, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`): copy `looped scripts tests
configs serving/looped data.py utils.py` and the gold-free `*.items.jsonl` to
`/root/work/repo`, write the commit id to `CODE_SHA`, run `scripts/pod/setup.sh`,
then:

```bash
M=/root/hf/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1
X='{"cuda_graphs": true, "static_buffers": true}'
# fixed arms on a depth-3 engine (forced modes need stock prompt mode and engine depth == arm depth)
python scripts/looped_profile.py --model $M --items artifacts/looped_serving/splits/math_train_calibration.items.jsonl \
  --out /root/out/modes_calibration --total_ut_steps 3 --prompt_mode stock --arms 3x4:direct \
  --cap 2048 --slots 64 --kv_gb 80 --execution "$X"
# live policies
python scripts/eval_looped_policies.py --model $M --items <items> --out <dir> --total_ut_steps 3 \
  --prompt_mode stock --policies fixed:3x4:direct,quorum:3x4/2:direct,ftt:3x4/3+3x2@4096 \
  --slots 32 --kv_gb 80 --execution "$X"
```

The exact job lists of this session are in
`artifacts/looped_serving/logs/modes/` (`queue_status.txt` and one log per job).

Operational notes:

- Throughput is about 700–900 generated tokens/s on this GPU regardless of
  the slot count (one Python thread drives the engine), so in the closed loop
  latency follows tokens in flight.
- KV per candidate is reserved for the full cap: about 413 blocks per 2,048
  tokens per depth plane at depth 3. With `--kv_gb 80`: 64 slots at cap 2,048,
  32 at cap 4,096 (depth 3); 48 slots at cap 2,048 at depth 4.
- Write outputs to a network volume mounted at `/workspace`, and pull them
  before stopping. Container disk is wiped on stop. Pod `n89idixoif18c0`
  (CA-MTL-3, no volume) lost the unpulled arms of the first validation attempt
  that way; it is EXITED and was left unterminated. Pod `d44xrnlfc8qhy5`
  (EU-RO-1) is EXITED. Its volume `s7ctgyb8b7` (`looped-serving-results`,
  20 GB, about $1.40/month) still holds a copy. Do not delete either pod or
  that volume unless asked, and do not touch the pre-existing volume
  `hib98og0wy`.
- `scripts/pod/watchdog.sh` cannot stop a pod. `runpodctl` inside the pod is
  unauthorized. Stop from outside. In the first session an unattended pod
  idled for about 9 hours.
- A queue line must not be launched in parallel with setup. `bash` parses
  `A && B & C` as `(A && B) & C`. Capture a job's exit code before `$(date)`,
  which resets `$?`.

## Next steps

The accuracy comparison is done. Do not rerun it, and do not touch the
confirmation reserve until a policy is frozen.

1. Serving measurement, which is the paper-shaped claim and has not been run.
   Fix an arrival trace and a set of deadlines. The two policies that survived
   are direct at cap 2,048 (about 26 s, 79.5% on the fresh set) and natural at
   cap 4,096 (about 92 s, 84%). Report correct answers delivered by each
   deadline, queue-inclusive p95, and charged work. One RTX PRO 6000 Blackwell
   is $2.49/h. Do not start one without an explicit go.
2. If the question is "block `<think>`" rather than "start with To," the
   `nothink` arm needs a sampler that does not abort the run when 64 redraws
   are still the think token (a logit penalty, or skip only the stubborn
   candidates). Then run it on the validation split only, matched to the arms
   above.
3. An agent deciding between those, or deciding that neither is worth a GPU,
   should follow `docs/looped_serving/NEXT_AGENT.md` and write the
   recommendation before spending anything. The graded tables are in
   `artifacts/looped_serving/pod_out/`.
