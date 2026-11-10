# LoopDelta: execution plan (2026-11-04)

Source: Jiayi's proposal `docs/jiayi/loopdelta-proposal.md` (email of 2026-11-03).
Backbone: `ByteDance/Ouro-1.4B` (plain, revision `574fa66`), the model behind the
84.5% single / 89% vote / 88% self-revision pilot numbers. Everything is
training-free inference on one GPU.

## The idea in one paragraph

A planner prefills the question at full depth and measures how much the KV cache
changes from loop to loop (Δ). Large Δ is read as "hard, needs more loops". It
picks a depth r; four solvers sample answers in parallel at depth r, reusing the
planner's first r loop caches for the prompt (exact, already verified); votes are
weighted by each solver's own late-loop Δ (settled loops count more).

## What has to be true, and how stage 0 tests it

| Claim | Test | Baseline it must beat |
|---|---|---|
| Prefill Δ tracks difficulty | Spearman of Δ with MATH level 1–5, raw and with prompt length + subject partialled out; out-of-fold prediction of level from length+subject with vs without Δ | Prompt length, subject; Ouro's exit gate; logit-space signals (entropy, KL between loops) |
| Δ predicts the depth a question needs | Min depth at which the 4-sample vote is right, vs planner Δ | Prompt length |
| Adaptive depth beats fixed depth at equal compute | Depth policies fitted on even idx, scored on odd idx, planner cost included | Fixed r, random assignment at the same mix, length-based policy; oracle as ceiling |
| Solver Δ picks correct samples | Within-question pairwise AUROC (the quantity vote weighting uses) and pooled AUROC; Δ-weighted vs unweighted vote | Mean token log-prob, mean entropy |

Direction is fixed in advance: larger Δ = harder = deeper. The reversed direction
is computed only as a diagnostic.

## Known constraints going in

- Ouro is trained with 4 loops: only Δ_2..Δ_4 exist, so r ∈ {2, 3, 4} in practice.
  Stage 0a also prefills at 8 loops (`--extrapolate`) to see whether Δ keeps
  falling or the model drifts; anything beyond 4 is out of distribution.
- Lower depth costs accuracy on Ouro (Thinking sweep on GSM8K: r=4 89.5%,
  r=3 88%, r=2 84%, r=1 37.5%) and shallower runs write longer answers. Stage 0b
  redoes this on the plain model and on MATH.
- Vote latency: 4 solvers batch on one GPU; the cost is waiting for the slowest.
  From the GSM8K pilot: slowest-of-4 is 1.24× a single answer (median), 1.52×
  (mean). Stage 0b records the same ratio on MATH.

## Stages and gates

**Stage 0a: prefill Δ vs difficulty (minutes on an H200).** 200 MATH-Train
questions per level (1,000), solver prompt, prefill at depth 4 and at depth 8.
Gate: Δ's partial Spearman with level (given length and subject) has a 95% CI
excluding 0, and adding Δ improves out-of-fold level prediction over
length+subject.

**Stage 0b: needed depth and solver Δ (≈1–2 GPU hours).** 60 questions per level
(300, a prefix of 0a's sample), depths 1–4, 1 greedy + 4 samples (T=0.6,
top_p=0.95, 2,048 new tokens). Gates: planner-Δ policy beats both fixed depth and
the length policy on the held-out half at matched loop-tokens; solver Δ's
within-question AUROC is above 0.5 and not below log-prob's.

If a gate fails, stop and report; the proposal's own fallback (a short probe
decode at R_max instead of prefill-only Δ) is the first thing to try.

**Stage 1 (only if 0 passes):** B3 vs fixed-r single on MATH-500 and GSM8K test,
ε frozen from MATH-Train. **Stage 2:** B4, then B5 (vote, then Δ weighting),
with measured wall-clock latency. **Stage 3:** Huginn-0125 for the 4–64 range.

## Run

```bash
bash scripts/run_loopdelta_stage0.sh          # on the pod, repo root
python scripts/loopdelta_analyze.py --stage0a artifacts/loopdelta/stage0a_r4 \
    --stage0b artifacts/loopdelta/stage0b --out artifacts/loopdelta/analysis
```

Code: `looped/loopdelta.py` (Δ, MATH sampling, grading), `scripts/loopdelta_stage0a.py`,
`scripts/loopdelta_stage0b.py`, `scripts/loopdelta_analyze.py`.

Implementation notes:
- Δ_t = ||X_t − X_{t−1}||² / ||X_{t−1}||² per layer (X = K or V), summed over heads,
  head dims and selected positions, then averaged over layers. Loop t of an
  R_max prefill equals what a depth-t run holds (tested), so one prefill gives
  every shallower trajectory.
- The solver's Δ is measured by re-prefilling prompt + answer at depth r and
  reading Δ over the answer positions; the same pass gives token log-probs.
- On Apple MPS, sdpa turns fully padded rows into NaN, which leaks into the next
  loop; `kv_deltas` raises if that happens. Use `--attn eager` there. CUDA batched
  runs matched unbatched runs in the pilots.
- Grading uses `math-verify` when installed (Python ≥ 3.10), else normalised
  string match; both verdicts are stored per sample.
