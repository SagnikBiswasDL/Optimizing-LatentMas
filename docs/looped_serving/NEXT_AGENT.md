# Next agent: decide what to do

You are picking this up after the held-out run. Your job is to analyze the
evidence and recommend the next experiment. Do not start a GPU, do not spend
money, and do not commit unless the user asks. The confirmation reserve stays
locked.

Read, in this order:

1. This file.
2. `docs/looped_serving/HANDOFF.md` (where the work stands).
3. The 2026-11-12 session report in `docs/looped_serving/EVIDENCE.md`.
4. `artifacts/looped_serving/pod_out/modes_validation/table.md` and
   `artifacts/looped_serving/pod_out/modes_calibration/table.md`.

The graded tables and the candidate files for the held-out run are on this
branch, under `artifacts/looped_serving/pod_out/`. A slide deck of the same
numbers exists only as a local Cursor canvas (`looped-serving-talk.canvas.tsx`)
and is not in the repo.

## What you must not overturn

These were computed from the saved runs (question bootstrap, plus paired
10,000-draw bootstrap, seed 0, and exact McNemar). Do not recompute them into
a different headline, and do not lead with the practice-set number.

Fresh set, 200 MATH questions, depth-3 engine, stock prompt, bf16, temperature
0.6, top-p 0.95, 32 slots, one seed, self-consistency of 4:

| Arm | Accuracy | Mean latency |
|---|---|---|
| Natural, cap 2,048 | 78.0% (72.0–83.5) | 49.8 s |
| Direct, cap 2,048 | 79.5% (74.0–85.0) | 25.5 s |
| Natural, cap 4,096 | 84.0% (79.0–89.0) | 92.3 s |

- Direct minus natural at cap 2,048: +1.5 points, interval −2.0 to +5.0,
  p = 0.58. Withdrawn as an accuracy claim.
- Natural 4,096 minus natural 2,048: +6.0 points, interval +2.0 to +10.0,
  p = 0.0075. This replicated on the 400-question practice set at +5.0
  (p = 0.0029).
- Natural 4,096 minus direct 2,048: +4.5 points, interval +1.0 to +8.0,
  p = 0.022. About three times slower.
- Practice set, the number not to cite as the result: direct 83.8% against
  natural 77.5% at cap 2,048 (+6.25, p = 0.00035). That gap was the cap
  cutting `<think>` traces off. At cap 4,096 the monologue ties direct there
  (−1.25, p = 0.53).
- Direct depth 4 ties depth 3 (both 83.8%, p = 1). Direct depth 2 is −8.75
  (p = 1.2e-6). Think on the 75 unsettled practice questions: 28% against
  30.7% for direct.
- "Direct" is the token `To` prefixed onto the answer. It does not disable
  reasoning. The `nothink` control, which would separate that prefix from
  banning `<think>`, aborted: 57 fresh-set candidates and 162 practice-set
  candidates were still `<think>` after 64 redraws. No candidate file.

Latency is one closed-loop run per arm, not interleaved. The token-clock
replay is a length proxy and is not a latency. The live quorum arm on the
fresh set is a separate bf16 sample, so its accuracy is not a stopping-rule
effect. The stopping-rule result is the calibration direct-pool replay: same
answers, 5.5% fewer loop-tokens.

## Constraints that still bind

- Calibration is for fitting, validation for choosing, then freeze and confirm.
  The confirmation reserve (`math_train_confirmation_reserve`, 300) and the
  final benchmarks (GSM8K test, MATH-500) are untouched. Do not tune on them.
- Record a failed hypothesis before changing policy. The withdrawn ones are
  listed in `HANDOFF.md`.
- Predeclare the comparison before running it. Match engine settings across
  the arms you compare.
- One RTX PRO 6000 Blackwell Server Edition, Secure Cloud, is $2.49/h. State
  the hourly price before creating anything billable, and do not create it
  without an explicit go from the user. Write to a network volume. Container
  disk is wiped on stop. Pull before stopping. `runpodctl` inside a pod cannot
  stop the pod.
- Pod `d44xrnlfc8qhy5` is EXITED. Volume `s7ctgyb8b7` (20 GB, EU-RO-1, about
  $1.40/month) still holds a copy of the results. Pod `n89idixoif18c0` is
  EXITED with no volume; its disk was wiped. Do not terminate either pod or
  delete that volume unless the user asks. Do not touch volume `hib98og0wy`.

## The decision to write

Produce a short recommendation, not a new run. It should contain:

1. One sentence a paper could defend from the runs already saved, and one
   sentence it cannot. The goal is a claim, not another table. "Think less"
   is not that claim: related work already holds the runtime fixed (DPA,
   "When to Continue Thinking", "Learning When to Think"; acknowledge
   ExactLoop).
2. The single next measurement, if any. Name the policies, the split, the
   predeclared comparison, the success criterion, and a GPU-hour estimate at
   $2.49/h. The measurement the project was built for, and has not run, is
   correct answers delivered by a deadline under a fixed arrival trace. The
   two policies that survived are direct at cap 2,048 and natural at cap
   4,096. Say whether that is actually the right next measurement, or whether
   the saved accuracy-latency tradeoff is already the result and a deadline
   sweep would not change the claim.
3. What you considered and rejected. Include another accuracy rerun of direct
   against natural at cap 2,048, a think fallback, depth 4, opening the
   confirmation reserve, and the `nothink` sampler. Keep `nothink` only if you
   can say what claim it would change.
4. What you would leave alone: the stopped pods, the volume, and the reserve.

Do not propose a GPU run in the same breath as starting one. Hand the
recommendation to the user.
