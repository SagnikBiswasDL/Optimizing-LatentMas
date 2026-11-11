# Looped serving: evidence ledger

Branch `research/looped-serving`, based on `641d891` (`perf/vllm-fork`). Last updated 2026-11-11.

This file records what the saved runs support, what was recomputed, and what has not been
run. The project question is whether choosing answer mode, recurrent depth, sampling width
and launch timing jointly improves correct answers delivered within a latency target, beyond
a strong adaptive-sampling policy on a strong looped-model engine. `HANDOFF.md` in this
directory is the current state for someone continuing the work.

## Status

Done on 2026-11-10, on CPU, with no model inference:

- The plan's replay of the saved Stage-0b candidates is reproduced. Every point estimate
  matches.
- Vote identity is repaired (`structural-v2`), with the historical key kept for old numbers.
- Question splits are frozen with text-derived IDs; every question the Ouro runs have seen is
  listed as exposed.
- Run manifests refuse incompatible resumes.
- The backend, models, datasets and grader are pinned.

The supplied `looped_history_audit.py` was not on this machine, so the replay
was rebuilt from the plan's description; agreement with the plan's numbers is the check.

Since then, on GPU: the fidelity gate, the smoke profile and the first calibration arms.
See the session reports below. Not run: any deadline or arrival-trace measurement, anything
on the validation split, and anything on the locked confirmation reserve.

## Pins

Full detail in `configs/looped_serving/pins.json`.

| What | Pin |
|---|---|
| Base commit | `641d891fc4d879b98d15f5c46af9be2f2ed40109` |
| Reference backend | `looped/ouro_adapter.py` (Hugging Face; depth via `total_ut_steps`) |
| Serving engine | `InferMatrixLab/vllm-rlt` at `e0a5515ec970d634546d2e1d38b3ba3602072884` (v0.2.0); installed unmodified on 2026-11-11, its own tests 916 passed / 633 skipped |
| Model | `ByteDance/Ouro-1.4B` at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| MATH-Train | `EleutherAI/hendrycks_math` at `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| GSM8K | `gsm8k` (main) at `740312add88f781978c0658806c59bc2815b9866` |
| MATH-500 | `HuggingFaceH4/MATH-500` at `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| Grader | `math-verify` 0.9.0 |
| Splits | `configs/looped_serving/splits.json`, hash `e5213c5b831e62ed` |

## What the saved runs support

Source for the MATH rows: `artifacts/loopdelta_run/artifacts/loopdelta/stage0b/`. 300
MATH-Train questions (60 per level), depths 1 to 4, one greedy and four sampled answers per
depth, temperature 0.6, top-p 0.95, cap 2,048 new tokens, `Ouro-1.4B`, RTX PRO 6000
Blackwell. Recomputed by `scripts/audit_looped_history.py`; output in
`artifacts/looped_serving/history_audit/audit.json`; pinned by
`tests/test_looped_history_audit.py`.

| Policy (historical vote key) | Accuracy | Mean loop-tokens |
|---|---|---|
| Depth 3, greedy | 66.33% | 3,524.93 |
| Depth 4, greedy | 68.33% | 5,139.88 |
| Depth 3, four samples | 77.33% | 13,677.85 |
| Depth 4, four samples | 77.00% | 19,860.12 |
| Two depth-3 samples; stop if they agree, else two more at depth 3 | 77.61% | 10,667.78 |
| Same start; on disagreement two depth-4 samples | 77.06% | 12,118.63 |
| Two depth-3 plus two depth-4 samples, always | 77.06% | 16,768.99 |

Accuracy intervals are question bootstraps over this one pool, about ±5 points for each row.
Costs are logical loop-tokens, not time. The adaptive rows average over every choice of
initial samples from the four saved ones. None of this is a fresh evaluation.

Other recomputed figures:

| Claim | Value | Note |
|---|---|---|
| Depth-3 minus depth-4, four samples | +0.33 points; 15 vs 14 discordant questions | 95% interval about [−3.3, +4.0]. The plan quotes [−3.0, +3.67]; endpoints sit on a 1/3-point grid and move one step between bootstrap seeds. Does not establish equivalence. |
| Two-then-two work vs four depth-3 samples | 22.01% less | |
| Depth-3 vs depth-4 four-sample work | 31.13% less | |
| Two depth-3 samples agree | 61.83% of pairs | 2.25% of agreeing pairs are wrong. Pairs within a question are dependent. |
| Strict vote-margin cancellation, depth 3 | 12.81% fewer completion rounds, 4.25% fewer decode tokens | Length proxy with synchronized starts, not a latency. Winner unchanged in all 173 questions where it fires. |
| Ideal reuse of three duplicate prompts | 7.35% of depth-3 four-sample loop-tokens | Mean prompt 111.68 tokens. Not a wall-time bound. |
| Sampled traces at the cap | 24.6% (depth 3), 28.3% (depth 4) | |
| Within-question AUROC, depth 3 | 0.518 (negative final delta) vs 0.659 (mean log-prob) | 276 correct/incorrect pairs from 82 questions |
| Within-question AUROC, depth 4 | 0.591 vs 0.839 | 254 pairs from 80 questions |
| Sampled pass, depths 2 to 4 | 4.66 GPU-hours | Proportional guess for 600 questions × 8 samples: 18.65 GPU-hours on the old path |

GSM8K pilots, recomputed from `per_question.jsonl` (200 questions each):

| Run | Single | Team | Four samples | Self-revision |
|---|---|---|---|---|
| Thinking model, `artifacts/looped/mas_gate_r4/` | 89.5% | 83.5% | 91.0% | 89.0% |
| Plain model, `artifacts/looped_base2/mas_gate/` | 84.5% | 72.0% | 89.0% | 88.0% |

`artifacts/looped_base/` is the invalid pre-mask-fix pilot. It is not read by any new code.

Read from saved summaries, not recomputed here: adding the aggregate KV delta to length and
subject changes out-of-fold difficulty correlation from 0.4160 to 0.4159
(`artifacts/loopdelta_run/artifacts/loopdelta/analysis/summary.json`).

## Findings from this audit

1. **Capped traces mostly cast no vote.** At depths 2, 3 and 4, 23.5%, 22.4% and 26.1% of
   sampled traces contain no boxed answer, and 95%, 91% and 92% of capped traces are in that
   group. So the 2,048-token cap removes roughly a quarter of the votes, and depth changes
   output length. The depth comparison cannot be trusted until the longer-cap check is run.
2. **The deeper fallback is the mixed portfolio with early stopping.** With the first-listed
   tie rule, "two at depth 3, then two at depth 4 on disagreement" selects exactly the answer
   the always-run 3+3+4+4 vote selects. Their accuracies are equal by construction; only the
   work differs (12,118.63 vs 16,768.99). It still does not beat the same-depth fallback.
3. **The vote-key defect is real but did not bite in this pool.** The historical key deletes
   braces, so `\frac{1}{23}` and `\frac{12}{3}` both become `\frac123`. No pair of sampled
   answers within a question is merged or separated differently by the repaired key. The only
   effect is two truncated answers (`\dfrac{`) that voted under the old key and are parse
   failures under the new one. Replay accuracies move by at most 0.17 points
   (for example the deeper fallback, 77.06% to 77.18%).
4. **Split votes among correct answers are rare.** Of 3,323 pairs of sampled answers that are
   both graded correct, 4 have different keys, all from one question whose answer is an
   unordered list. Neither key handles reordered lists.
5. **The grader version is now known.** `math-verify` 0.9.0 reproduces all 4,282 stored
   verdicts. String matching disagrees with it on 67 of them, and `grade_math` falls back to
   string matching silently when the package is missing, as it is in the local Python 3.9
   environment. New code fails instead of falling back.
6. **The engine's prompt depth differs from the reference.** The pinned `vllm-rlt` snapshot
   documents, and its `stage_engine.py` states, that prompt prefill always runs full depth
   and `max_loops` only limits later decode. The saved depth-r candidates applied r loops to
   the prompt too. `min_loops = max_loops = r` there would be a different model.
7. **One MATH-Train question appears twice**, differing only by a double space (loader rows
   924 and 958). The second copy is dropped from the fresh pool.
8. **`data.load_math("train")` can fall back silently** to other datasets, which would change
   every `idx`. The split builder reproduces the four historical subset fingerprints before it
   does anything, and stops if it cannot.

## Evaluation issues from the plan (section 8)

| Issue | Status |
|---|---|
| Vote normalisation removes braces | Fixed for new runs: `looped/answer_identity.py`, `structural-v2`. Legacy key kept verbatim for old numbers. |
| Grading and vote identity are one operation | Separated. Grader pinned and fails loudly when missing. Tie rules are named (`first_listed` historical, `lowest_candidate` order-independent). Parse failures carry a status and cast no vote. |
| Resume checks only row count and rewrites metadata | Fixed for new runs: `looped/experiment_manifest.py`. A directory resumes only under an identical manifest hash; the stored manifest is never rewritten; rows without a manifest are not adopted. Not yet wired into a generator, because the new generator does not exist yet. |
| Split discipline | Fixed: separate calibration, validation and locked confirmation sets, by question, in `splits.json`. The old even/odd frontier in `loopdelta_analyze.py` is exploratory only. |
| `straggler_ratio_mean` is an output-length ratio | Reported in the audit as a length-ratio proxy. Real latency needs the engine. |
| Quantile endpoints at zero or one | Open. Belongs with the threshold policies (P3), with tests for zero-mass bins, ties and out-of-range inputs. |
| Policies charged a full-depth planner regardless | Open. Belongs with cost accounting in P3. |
| Teacher-forced rescoring pass is uncharged | Open. `CandidateRecord` has a `probe_work` field; generation-time confidence is P2. |
| Global mutable `total_ut_steps` | Open. Stays in the sequential reference only; per-request depth is P2. |
| High cap rates | Measured (finding 1). The longer-cap subset needs a GPU. |

Not covered yet: GSM8K answers still use `looped/tasks.py` (last number in the box); the
structural key has not been applied to that task. Symbolic equivalence (for example
`32\sqrt{3}i` vs `32i\sqrt{3}`) is not attempted by either key.

## Splits

`configs/looped_serving/splits.json`, built by `scripts/build_looped_splits.py`. A question's
ID is a hash of dataset, revision, split and text. Overlap is checked on the text hash alone,
so it holds across datasets and revisions.

| Split | Questions | Use |
|---|---|---|
| `math_train_exposed` | 1,000 (300 with Stage-0b generations) | History only |
| `math_train_smoke` | 100 | Correctness and runtime profiling; not a quality claim |
| `math_train_calibration` | 400 | Fit value tables, cost models, stopping rules |
| `math_train_validation` | 200 | Choose policies, prices, thresholds |
| `math_train_confirmation_reserve` | 300 | Locked until the policy is frozen |
| `gsm8k_test` | 1,319 (200 exposed to Ouro pilots) | Final evaluation |
| `math500_test` | 500 | Final evaluation |

Fresh MATH-Train splits are balanced, 20% per level. No MATH-Train question text appears in
MATH-500. Level 1 has only 164 fresh questions left unassigned, which limits how far a
level-balanced calibration set can grow. Both test sets were evaluated in full by the
June–October Qwen experiments; they are not historically unseen, only unseen by Ouro apart
from the tagged GSM8K items. Exact-text matching does not catch reworded variants.

## Reproduce

```bash
python -m pytest tests/test_answer_identity.py tests/test_ensemble_state.py \
    tests/test_experiment_manifest.py tests/test_looped_history_audit.py tests/test_looped_splits.py -q
python scripts/audit_looped_history.py            # writes artifacts/looped_serving/history_audit/audit.json
python scripts/build_looped_splits.py --check     # rebuilds splits.json and compares byte for byte
```

All three run on CPU with Python 3.9 or later, NumPy, and (for the split builder and one
parity test) `datasets` and `torch`. The audit takes about two seconds.

## Session report, 2026-11-10

- **Hypothesis tested.** None of the research hypotheses. This session checked that the
  historical evidence is reproducible and repaired the evaluation code.
- **Outcome.** Every replay number in the plan reproduces from the saved rows. The
  vote-identity repair changes replay accuracy by at most 0.17 points.
- **Costs.** CPU only. No GPU time.
- **Known confounds.** One candidate pool, one decoding seed per depth, 2,048-token cap with
  about a quarter of traces capped and unanswered, loop-token costs instead of time, depth-4
  samples drawn with a different sampling seed than depth-3.
- **Next gate.** Backend fidelity (P2): on the pod, install the pinned engine, run its tests,
  and compare first-token logits and greedy continuations against `OuroRunner` at each depth.
  That decides between adding a per-request prefill depth and regenerating the candidate
  evidence under full-depth prefill. Then the 100-question smoke profile.

## Session report, 2026-11-11

Two pods, both `NVIDIA RTX PRO 6000 Blackwell Server Edition`, Secure Cloud, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, bf16, temperature 0.6, top-p 0.95.
The first (`ipm1mt8fwpq22x`) ran the fidelity gate and the smoke profile and was deleted
after verification (about $29.5, of which about $22 was unattended idle). The second
(`n89idixoif18c0`, $2.49/h, no persistent volume) is running the calibration queue; its
raw output so far is in `artifacts/looped_serving/pod_out/`.

**Fidelity.** In fp32 the engine matches `OuroRunner` at every depth in two configurations:
withholding the last prompt token so the engine emits it (`boundary_shift`), and running
the engine itself at depth r (`--total_ut_steps r`, stock prompt mode). Stock mode on the
full engine differs from uniform depth only in the first generated token. bf16 tokens
depend on the batch, so equal seeds do not reproduce a pool. The `last_exited` KV layout
reserves all four planes whatever the request depth; only a depth-limited engine reserves
fewer. Report: `artifacts/looped_serving/fidelity_gate/fidelity.json`.

**The first answer token is an answer mode.** `<think>` opens a long reasoning block that
uses most of the decode budget and usually hits the 2,048 cap without an answer; any other
first token starts a short solution. Think share rises with depth (33% / 42% / 52% at
depths 2 / 3 / 4 on the saved Stage-0b pool). Finished think answers are 98–100% correct,
but they finish on the questions the short answers already get right.
`artifacts/looped_serving/mode_audit/modes.md`.

**Smoke profile, 100 questions** (`artifacts/looped_serving/profile_smoke/table.md`),
four-plane engine, 32 slots, cap 2,048. Natural four-sample vote: depth 2 72%, depth 3
75%, depth 4 78%, intervals about ±9 points. Certificate cancellation at depth 3: 77%,
mean latency 49.0 s against 57.4 s for the fixed vote. Two-then-two: 74%, 17% less charged
work, mean latency 89.7 s and p95 183 s against 88 s. Token savings and latency move in
opposite directions. The 75% → 79% figure from rerunning 114 capped candidates at 4,096 is
not a continuation: 0 of 114 reproduced its prefix and the slot count changed. It is not
cited as a cap effect.

**Calibration, 400 questions, depth-3 engine, 64 slots, cap 2,048**
(`artifacts/looped_serving/pod_out/modes_calibration/`). Forcing the direct mode (the
answer starts with `To`) against natural sampling, same engine and settings:

| Arm | Accuracy (95% CI) | Capped | Charged loop-tokens/q | Mean / p50 / p95 latency |
|---|---|---|---|---|
| Natural, four samples | 77.5% (73.2, 81.5) | 29.9% | 14,628 | 101.6 / 118.7 / 162.4 s |
| Direct, four samples | 83.8% (80.0, 87.2) | 2.2% | 9,118 | 49.3 / 36.0 / 129.2 s |

Paired difference +6.25 points (bootstrap interval +3.0 to +9.5); 36 questions only direct
gets right, 11 only natural; exact McNemar p = 0.00035. The gain is concentrated on the 92
questions where every natural sample opened `<think>` (32% against 61%). One closed-loop
run each, so the latency columns are unreplicated. "Direct" forces one lexical prefix; the
queued `nothink` arm (the model's own first token with `<think>` excluded) is the control
that separates the mode from the word.

Replay of the direct pool on the token clock, which is a length proxy and not a latency:
stopping at the first two agreeing answers keeps the full vote's accuracy exactly (0 of
400 questions change correctness; the chosen answer differs on 1) and saves 5.5% of
loop-tokens, because direct answers are already short. Early stopping is not where the
gain is. `replay_direct.json` in the same directory.

**Known confounds.** bf16 batch dependence; one run per arm; the natural and direct arms
ran in that engine but not interleaved, so a slow drift in machine state is not averaged
out; the smoke natural baseline used a different engine configuration and is only
comparable on accuracy; no validation or confirmation number exists yet.

**Next gate.** Finish the calibration controls (nothink, direct at depths 2 and 4, think
on the 75 questions whose direct votes reach no quorum of 3, natural at cap 4,096), fit on
calibration only, then one predeclared comparison live on the 200-question validation
split. The confirmation reserve stays locked.
