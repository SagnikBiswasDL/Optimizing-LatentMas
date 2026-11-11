# Looped serving: handoff

For a research agent picking this work up. Branch `research/looped-serving`
(remote `git@github.com:SagnikBiswasDL/Optimizing-LatentMas.git`), based on
`641d891`. Last updated 2026-11-11, during the second GPU session.

`docs/looped_serving/EVIDENCE.md` is the ledger of what the runs support.
This file says where things stand, how to run them, and what to do next.

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

### Measured in the current session, not yet replicated

Depth-limited depth-3 engine, bf16, stock prompt mode, CUDA graphs and static
buffers, temperature 0.6, top-p 0.95. Accuracy intervals are question
bootstraps.

Calibration split, 400 questions, depth 3, four samples, cap 2,048, 64 slots
(`artifacts/looped_serving/pod_out/modes_calibration/`, code `a95b3e1`):

| Arm | Accuracy (95% CI) | Capped | Charged loop-tokens/q | Latency mean / p50 / p95 | Questions/min |
|---|---|---|---|---|---|
| Natural SC@4 | 77.5% (73.2, 81.5) | 29.9% | 14,628 | 101.6 / 118.7 / 162.4 s | 11.5 |
| Direct SC@4 | 83.8% (80.0, 87.2) | 2.2% | 9,118 | 49.3 / 36.0 / 129.2 s | 21.7 |

Paired: direct +6.25 points (bootstrap 95% interval +3.0 to +9.5); 36 questions
only direct gets right, 11 only natural; exact McNemar p = 0.00035. The gain
sits on the 92 questions where all four natural samples opened `<think>`
(natural 32%, direct 61%); on the other 308 the two arms are within a point.
Latency and throughput are from one closed-loop run each on the same engine
configuration, not repeated.

Smoke split, 100 questions, same engine (`pod_out/modes_smoke/`): direct SC@4
82.0% against 75.0% for the first session's natural SC@4 (which ran on the
four-plane engine with 32 slots, so only the accuracy is comparable; 9 against
2 discordant, p = 0.065). Forced think, two samples, cap 4,096: 75.0%, 29.5%
capped, mean latency 136.8 s.

Reading, with its limits:

- Forcing the direct mode is more accurate than the natural four-sample vote at
  the 2,048-token cap and costs fewer loop-tokens. "Direct" here means the
  answer is started with the token `To`; the answer can still contain a worked
  derivation. The `nothink` arm (the model's own first token with `<think>`
  excluded) tests whether the effect is the mode or the word.
- This says nothing about reasoning given room. A natural arm at cap 4,096 is
  queued; it is also the truncation control (each trajectory against its own
  first 2,048 tokens).
- On the smoke split, forced reasoning at cap 4,096 finished almost only on
  questions the direct vote already got right: of the 18 questions direct SC@4
  missed, a reasoning sample finished on 3 and was right on 2. If that holds on
  calibration, "direct first, reasoning on disagreement" adds little. The
  queued think arm runs exactly on the calibration questions whose direct votes
  reach no 3-of-4 quorum (75 of 400), which is the population that fallback
  would act on.

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
  continuation and should not be cited as the effect of a longer cap.

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

Open review points: the comparison must be repeated on fresh questions
(validation, then the locked confirmation set); the serving claim needs fixed
arrival traces and deadlines, not only the closed loop; related work to position
against includes depth-portfolio methods (DPA) and adaptive-thinking methods
("When to Continue Thinking", "Learning When to Think").

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
- The pod has no persistent volume. Copy `/root/out` off it while it runs.
- `scripts/pod/watchdog.sh` stops the pod after 40 idle GPU minutes; its stop
  call has not been observed working (a read with the pod's own key was
  refused). In the first session an unattended pod idled for about 9 hours.

## Next steps

1. Finish the calibration queue (running): `nothink`, direct at depths 4 and 2,
   think on the no-quorum questions, natural at cap 4,096, greedy direct.
2. From calibration only: pick depth, width, quorum and whether a reasoning
   fallback earns its latency. Predeclare the primary comparison.
3. Validation (200 questions), all policies live on one engine configuration
   with the same slot count, arm order randomised, at least two run seeds:
   natural SC@4 at caps 2,048 and 4,096, natural with certificate cancellation,
   direct SC@4, a direct quorum policy, and direct-first with a reasoning
   fallback if step 2 keeps it. Report paired question-level differences.
4. Freeze, then one run on the confirmation reserve.
5. Open-loop serving: fixed arrival traces and deadlines; correct answers
   delivered by the deadline per second, queue-inclusive p95, charged work;
   best static policy against a load-blind and a load-aware controller.
