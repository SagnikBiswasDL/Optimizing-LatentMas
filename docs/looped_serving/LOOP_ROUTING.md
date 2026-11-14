# Query-aware loop selection: protocol

Written 2026-11-13, before any pilot result was read. Results go in the last
section and do not change anything above it.

## Question

Ouro-1.4B runs each token through the same 24 layers 1 to 4 times (the depth, or
number of loops). Can we choose the depth for each question **before its answer is
written**, so that we get more right answers per unit of GPU work than any fixed
depth? And does that also give more right-and-on-time answers when many questions
share one GPU?

The open risk is whether the benefit of an extra loop is a property of the question
at all, or just which samples happened to land. The pilot below tests that first.

## Positioning

- Continuous Depth Batching (arXiv 2608.09444) serves Ouro with per-token early exit
  and a scheduler that re-forms batches between loop steps. It does not change exit
  decisions.
- Here the depth is chosen once per question and every token of the answer runs at
  that depth, so ordinary batching applies.
- CDB files two papers under sequence-level depth (all tokens exit at one depth).
  LoopFormer (Jeddi et al., ICLR 2026) trains a looped model to work at any budget;
  the user picks the budget and step schedule at inference, not a predictor of the
  question. Parcae (Prairie et al., COLM 2026) studies scaling laws for stable looped
  models. Neither predicts a depth from the question.
- Ouro's own exit gate is per token and is scored here as a baseline.

## Fixed setup

- Stock solver prompt. The model chooses its own answer style.
- Four sampled answers per question, temperature 0.6, top-p 0.95.
- Majority vote over boxed answers (structural answer keys). Ties go to the lowest
  sample index. An unparseable or cut-off answer casts no vote. No vote counts as
  wrong.
- Grader: math-verify 0.9.0, run locally. Gold answers never go to the pod.
- Cap of 8,192 tokens as a runaway guard. The share of answers hitting it is
  reported for every depth.
- Each depth runs on its own depth-limited vLLM-rlt engine (`--total_ut_steps d`,
  stock prompt), which computes exactly like a uniform-depth run. Depths 2 and 3 were
  shown exact earlier; depth 1 is checked as the pilot's first job; depth 4 is the
  full model.

## Definitions

For question q, depth d and run s:

- C[q, d, s] = 1 if the vote of the four answers is correct.
- cost[q, d, s] = loop-tokens for the prompt and all four answers.
- "q needs loop d+1 in run s" = C[q, d, s] = 0 and C[q, d+1, s] = 1.

## Pilot

200 practice questions (`artifacts/looped_serving/splits/router_pilot_questions.txt`,
40 per level, sha256 `300d480f…0bad`). Depths 1, 2, 3, 4. Two runs (seeds 0 and 1).
Prefill and probe features for the same 200 (`scripts/loop_probe_features.py`).

### Gate (both checks must pass)

1. **Repeatability.** For the steps 1→2 and 2→3, Cohen's kappa between the "needs the
   next loop" indicators of run 0 and run 1. The 95% bootstrap interval (2,000
   resamples of questions) must exclude 0. The step 3→4 is reported but not gated.
2. **A router could learn it.** Read the ideal routing off one run: each question
   goes to its cheapest depth with a correct vote, or to that run's most accurate
   fixed depth if none was correct. Score that routing on the other run and subtract
   the best question-independent mixture of depths at the same mean cost on that run.
   Averaged over both directions, the bootstrap interval must stay above 0.

Check 2 replaces the earlier "oracle gap between depths versus oracle gap between
replicate runs" form. That form scores only accuracy, so it reads zero whenever the
deeper setting is never worse, which is exactly when routing saves cost. The change
was made before any pilot data existed (caught by `tests/test_loop_select.py`).

If the gate fails, the finding is that per-question depth needs do not repeat, so
there is nothing stable for a router to learn, and the full experiment is not run.

### Selectors scored on the pilot (development only)

Out-of-fold, 5 folds stratified by level, per-depth logistic heads with
cross-validated L2 strength (PCA to 10 components inside each fold for the 144-number
Δ profiles):

- Text only (TF-IDF over the question, the semantic-router-style baseline).
- Question length. MATH level and subject (a label ceiling, not deployable).
- Prefill: Δ per loop, Δ of keys and values, per-loop entropy, KL between loops,
  top-1 agreement with the last loop, Ouro's gate, plus length.
- Probe: the same signals over the first 32 tokens of a greedy depth-4 answer, plus
  whether the answer opens with `<think>`, plus length.
- Threshold rules, kept as direct policies: prefill Δ (`r = min{t : Δ_t < ε}`), probe
  Δ, Ouro's gate (smallest depth whose cumulative exit probability passes q).

Decision rule for the heads: the smallest depth whose estimated vote accuracy is
within tau of the best depth's. Each selector is compared at mean cost no higher than
fixed 3 loops with: fixed depths, the best question-independent mixture at the same
cost, and 2,000 matched-random permutations of its own depths (a one-sided
permutation p-value). The operating tau is picked on the same out-of-fold
predictions, so these are development reads, not claims.

## Full experiment (only if the gate passes)

- New split file `configs/looped_serving/splits_router_v1.json` from MATH-train
  questions no split has used (offset 200 onward in the existing per-level order):
  router-train 400 (80 per level) and a sealed test set of 300 (60 per level). The
  old validation set (200) has informed decisions and is development data. The
  confirmation reserve is untouched.
- Labels at depths 1 to 4 for router-train, the rest of the practice set and the
  sealed set. Sealed results are written but not read until the selector, tau and
  serving settings are frozen.
- Serving: one engine with per-request depth in `boundary_shift` mode, with the
  withheld prompt token injected deterministically rather than sampled. Gate:
  identical tokens to the depth-limited reference at depths 1 to 4. Fallback: one
  engine per depth, with the memory budget measured including weights and graphs.
- Policies: fixed 2, 3, 4; matched-random; best mixture at equal cost; the frozen
  selector. Three seeds on the sealed set, the existing alternating arrival trace
  (4 and 12 per minute, 240 s deadline), identical trace and order per seed.
  Right-and-on-time out of all arrivals, wait, and GPU time; paired per-question
  comparisons pooled across seeds.

## Results

### Stage A, saved data (development)

`artifacts/looped_serving/loop_routing/stage_a_stage0b/selectors.md` and
`stage_a_practice/selectors.md`.

LoopDelta stage 0b, 300 questions, votes over the four sampled answers (greedy answer
excluded), model's own answer style, cap 2,048:

| | accuracy | loop-tokens per question |
|---|---|---|
| 1 loop | 30.7% | 4,224 |
| 2 loops | 67.7% | 8,656 |
| 3 loops | 77.3% | 13,678 |
| 4 loops | 77.0% | 19,860 |
| Oracle (cheapest correct depth) | 84.0% | 7,422 |

Best point of each selector at a mean cost no higher than fixed 3 loops:

| selector | accuracy | loop-tokens | best mixture, same cost | matched random (p) |
|---|---|---|---|---|
| length | 75.7% | 12,724 | 75.5% | 75.2% (0.38) |
| level + subject | 76.7% | 12,855 | 75.7% | 74.4% (0.02) |
| prefill signals | 74.0% | 11,125 | 72.4% | 71.5% (0.02) |
| prefill Δ profile | 73.7% | 13,227 | 76.5% | 73.4% (0.48) |
| text (TF-IDF) | 75.0% | 13,449 | 76.9% | 75.1% (0.60) |
| prefill Δ threshold | 77.0% | 13,521 | 77.0% | 76.8% (0.52) |
| Ouro gate threshold | 77.3% | 13,592 | 77.2% | 76.9% (0.27) |

No selector beats fixed 3 loops. The prefill signals (Δ with entropy, KL and the gate)
and the level/subject labels are the only ones that beat matched-random assignment of
their own depths (p about 0.02), by 1 to 2.5 points, at lower cost than 3 loops. Δ by
itself (threshold or profile) and question text carry no routing signal here.

Practice set, 400 questions, forced short answers at depths 2 to 4 (text, length and
labels only): no selector beats matched random (p 0.24 to 0.69) or the best mixture.

### Pilot (200 practice questions, two runs, cap 8,192, model's own answer style)

Both runs finished. Graded locally with math-verify 0.9.0.
`artifacts/looped_serving/loop_routing/pilot/gate.md`.

| loops | run 0 | run 1 | loop-tokens (run 0) | answers cut off (run 0) | opened a monologue (run 0) |
|---|---|---|---|---|---|
| 1 | 28.5% | 28.5% | 11,487 | 27.8% | 5.6% |
| 2 | 76.0% | 78.0% | 18,987 | 17.4% | 34.5% |
| 3 | 87.5% | 86.5% | 29,966 | 13.4% | 49.1% |
| 4 | 89.5% | 88.0% | 35,406 | 8.8% | 53.6% |

The two runs agree closely on accuracy. The 8,192 cap is not a rare runaway guard: 9% to 28% of answers hit it, more at fewer loops.

**Check 1 passed for the steps that were gated.** "Needs the next loop" repeats across runs for 1→2 (97 and 101 questions, 66 in both, kappa 0.34, interval 0.20 to 0.47) and for 2→3 (30 and 23, 15 in both, kappa 0.50, interval 0.32 to 0.66). It does not repeat for 3→4 (10 and 7, only 1 in both, kappa 0.08, interval −0.05 to 0.32).

**Check 2 failed, so the gate fails.** The ideal routing read off one run, scored on the other, is 4.1 points *worse* than the best question-blind mix of loop counts at the same cost (interval −7.6 to −0.6, entirely below zero). It sends 36 questions to 1 loop and 36 to 4 loops because they happened to come out that way in the first run. Those picks do not survive the second run. A same-run oracle looks like 93.5% at 17,410 loop-tokens; that number is sampling luck.

Selectors fit out of fold on run 0 (`pilot_selectors_seed0/selectors.md`) do not beat fixed 3 loops. The closest, Ouro's own exit gate used as a threshold, is 88.0% against 87.5%, one question each way. Question text beats a random assignment of the same depths (87.0% vs 84.6%, p = 0.045) but not fixed 3 loops (4 won, 5 lost).

**Decision.** The full label collection and the serving experiment are not run. There is a real, repeated accuracy gap between loop counts, and the questions that gain from the 2nd and 3rd loop are partly the same ones both times. What does not survive is using one run's per-question outcomes to choose a loop count for the next run: the wins a router would chase are mostly luck, and chasing them loses to ignoring the question.
