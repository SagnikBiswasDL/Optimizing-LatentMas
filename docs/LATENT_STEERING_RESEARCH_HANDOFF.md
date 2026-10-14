# Latent Steering in LatentMAS — Research Handoff

Updated: 2026-07-19

**Audience:** a research agent taking over to work on the *fundamental* question below.
This doc is self-contained: it records everything we tried, the exact numbers, the
mechanism that explains the (mostly negative) results, the reusable infrastructure, and
a set of grounded directions for *designing a latent steering mechanism* rather than
brute-forcing one. Read it fully before running anything — most of the obvious
experiments have already been done and are catalogued here so you don't repeat them.

---

## 0. The fundamental question

> **LatentMAS agents (Planner → Critic → Refiner) communicate through a non-text latent
> channel — a shared, growing KV cache — that only the Judger decodes. Can we design a
> steering mechanism that makes this latent channel a *usable control surface* for
> improving reasoning (accuracy), and if so, what is the right mechanism?**

Every steering method we've tried lands the same way: **leverage for both accuracy and
efficiency is concentrated at the text-emitting Judger; the upstream latent channel is
either inert (weak interventions) or destructive (strong interventions), never
usefully improving.** The open problem is whether that is a fundamental property of
latent MAS, or an artifact of the *mechanisms* we used (static linear vectors, applied
naively). We believe the mechanism design space is under-explored — see §6.

---

## 1. System under study (facts, not assumptions)

- **Pipeline:** Planner → Critic → Refiner → Judger (`methods/__init__.py`,
  `methods/latent_mas.py`). The first three reason in latent space: each runs a fixed
  `K = latent_steps` forward passes, feeding the last-layer hidden state back as the
  next input embedding (via realignment `W_a`), emitting **no text**. They communicate
  via a shared, growing `past_key_values`. Only the **Judger** decodes tokens.
- **Compute:** total latent forwards ≈ `3·(K+1)`; `K` is global/per-upstream-agent, not
  one shared K-sequence.
- **Backbones used:** Qwen3-14B (hidden 5120, 40 layers) for most prior work; Qwen3-4B
  (hidden 2560, 36 layers) for the learned-CES study. **Steering vectors do NOT transfer
  across sizes** (different hidden dim).
- **Env (RunPod H200):** torch 2.8+cu128, transformers==4.57.1 (5.x breaks KV handling),
  vllm 0.11. `/workspace` persists; `/root` venv is ephemeral (rebuild via
  `/workspace/recover_env.sh`, ~5 min). Proxy SSH is interactive-only (drive via forced
  PTY + heredoc). See `docs/HANDOFF_LATENT_STEERING.md` §7 for gotchas.

---

## 2. Everything tried, and how it landed

| # | Intervention | Where | Objective | Outcome |
|---|---|---|---|---|
| 1 | **SEAL residual** (diff-of-means, training-free) | **Judger** decode | shorten CoT | **POSITIVE**: −17 to −39% tokens, acc held/improved, ~2× faster; transfers GSM8K→ARC (`RESULTS.md` §3–4) |
| 2 | SEAL residual, per-agent (isolation) | Planner/Critic/Refiner | shorten/quality | Only Judger cuts tokens; a *critic-native* vector gave a small isolated acc bump (`RESULTS.md` §5,§7) |
| 3 | **Native correctness-contrastive** residual (in-pipeline) | upstream {P,C,R}, layer 28 | improve Judger | **NEGATIVE**: within noise on GSM8K/ARC, hurts MedQA, no token savings; native-Judger worse than generic SEAL (`RESULTS.md` §9) |
| 4 | **KV-cache steering** of the handoff | latent handoff columns | improve/shorten | **FALSIFIED** for content columns: `handoff_last` too weak (~1/130 attn), `handoff_all` saturates → collapse (94.7%→72% at cv=1). `judger_token` (aggregation pos) works: −25 to −33% tokens (`KV_STEERING_FINDINGS.md`) |
| 5 | **K-budget CES** (learned `v`, recover small-K) | upstream latent steps | recover acc at smaller K w/ compute savings | **STOPPED at Gate 1**: no operating point saves wall-clock (Judger CoT length dominates; more latent steps *shorten* Judger output). `GATE1_STATUS.md` |
| 6 | **Learned CES "boost"** (this work) | upstream {P,C,R} latent steps | push acc **above** best-K | **NEGATIVE (destructive-only)**: see §3. `CES_LATENT_STEERING_FINDINGS.md` |
| 7 | **DeltaBridge** (Frozen Mean-Replay + one-token Real−Frozen residual) | Judger first generate step, L32 r=16 | recover AIME’s extra K=10 solves without 33 forwards | **MATH go / AIME hard no.** Oracle rank-16 MATH 85% vs Real 74%; AIME 6/30 vs Frozen-class 19/30, `oracle_full` 8/30. Do not train `g_φ`. `DELTABRIDGE_HANDOFF.md` |
| 8 | **One live K=10 role** on Frozen MATH-1k | one of {P,C,R} at K=10, then Judger | keep AIME recurrent depth, 33→11 forwards | **DONE, AIME no.** MATH n=20: Planner/Refiner 0.80, Critic 0.85 vs Frozen 0.60. AIME n=6/12 same 2/6 and 7/12 as Frozen (swap idx4/5). Extra AIME Real solves are not one-role. `artifacts/one_role/RESULTS.md` |
| 9 | **Frozen + Judger SEAL** (lock-in) | Frozen MATH-1k prefix + GSM8K SEAL L28 coef 40 on Judger | latency / tokens, iso-acc | **THIS GPU.** `scripts/run_frozen_seal.sh`. Kill if GSM8K/MATH acc drops or AIME n=6 loses net solves. Do not hunt AIME accuracy. `docs/LOCK_IN.md` |

**One-line synthesis:** three independent methods (residual SEAL, KV-cache, learned CES)
all conclude the upstream latent channel is not a usable lever; the Judger is the locus.

---

## 3. The learned-CES "boost" study (most recent, most controlled)

Qwen3-4B, MedQA, K=10 (best unsteered point), layer 20, CES rank loss
`softplus(E(y+)−E(y−))` trained end-to-end through the differentiable multi-agent KV;
steering upstream latent steps only. Controls: α=0 sanity, α<0 negate, random-v; paired
bootstrap CIs; dev split (test never consumed). Full detail:
`docs/CES_LATENT_STEERING_FINDINGS.md`.

- **Last-token injection (`apply_to=last`)**: inert. Coef sweep α∈{4,8,16,−8}: acc stays
  0.60–0.65 (unsteered 0.65), all CIs straddle 0, no polarity. Even α=16 (Δ‖ ‖≈80) does
  nothing. Training was ill-conditioned (grad norms 1e4–1e5).
- **All-tokens injection (`apply_to=all`)**: real leverage, **destructive-only**.
  Training well-conditioned (grad norms ~15). Coef sweep (dev n=40, unsteered 0.65):

  | α | acc | Δ | mean toks |
  |---|---|---|---|
  | +1 | 0.600 | −0.05 | 962 |
  | +2 | 0.675 | +0.025 | 1018 |
  | +4 | 0.625 | −0.025 | 969 |
  | +8 | 0.600 | −0.05 | 1005 |
  | **−8** | **0.000** | **−0.65** | **1946** |

  Positive α within noise; α=−8 collapses to 0 with output length ≈ the 2048 cap — the
  Judger **degenerates/rambles** (disruption, not "steer to a wrong answer").

**Interpretation:** all-token perturbation *can* causally break the pipeline, proving the
channel matters, but there is **no direction/strength that improves** accuracy. CES finds
destructive directions, not an accuracy one.

---

## 4. The central mechanistic clue (why it keeps failing)

Three independent observations converge on one story:

1. **Correctness is decided late, at the Judger** (`RESULTS.md` §9.1 probe). Per-agent
   5-fold correctness probes on real in-pipeline layer-28 activations:

   | Agent (AUC) | GSM8K | ARC-C | MedQA |
   |---|---|---|---|
   | Planner | 0.54 | 0.57 | 0.57 |
   | Critic | 0.62 | 0.51 | 0.63 |
   | Refiner | 0.59 | 0.46 | 0.64 |
   | **Judger** | **0.69** | **0.73** | **0.80** |

   Final correctness is **barely linearly encoded** in upstream latents (near chance);
   it becomes decodable only at the Judger. So a **linear** steering direction toward
   "correct" essentially doesn't exist upstream — which is exactly why linear
   diff-of-means and linear CES vectors can't find one.

2. **Two functionally different KV position types** (`KV_STEERING_FINDINGS.md`): dense
   *content* positions (latent thoughts) that steering **corrupts**, vs
   *aggregation/routing* positions (e.g., the Judger's last prompt token) that steering
   **rides**. Steering all content columns shifts the Judger's attention output by the
   full `c·S` per layer (attention weights sum to 1) → saturates into oversteering almost
   immediately (the α=−8 collapse and the `handoff_all cv=1` collapse are the same
   phenomenon). A single content column is ~1/130 of attention → too weak (the
   `handoff_last` and `apply_to=last` nulls).

3. **The Judger barely needs the latent channel** (`GATE1_STATUS.md`): unsteered
   accuracy is best at K=10 and flat/declining to K=40; K=0→10 is the only real gain.
   If the Judger's answer is largely insensitive to the fine detail of the latent KV,
   no steering of that KV can move the answer much (except by breaking it).

**Together:** the upstream latent channel is (a) not linearly organized around task
correctness, (b) only globally perturbable in a way that saturates/destroys, and (c) not
something the Judger strongly depends on. Any *useful* latent steering mechanism must
overcome all three.

---

## 5. Reusable infrastructure (build on this; it works)

Differentiable path + steerers + losses are implemented and unit-tested
(`tests/test_ces_steering.py`; Gate-2 grad smoke passed, `artifacts/ces/gate2_smoke.json`).

- **Steerers** (`seal/`):
  - `TrainableSteerer` (`seal/ces_steerer.py`): `v` as `nn.Parameter`, residual add at a
    layer, `apply_to ∈ {last, all}`, phase gating (`latent_only` / `prefill_and_latent`),
    per-role. Clone-before-edit (no in-place view mutation).
  - `SealSteerer` (`seal/hooks.py`): detached diff-of-means residual add.
  - `KVCacheSteerer` (`seal/kv_steer.py`): one-shot `K/V += c·S` on handoff columns.
  - `ActivationRecorder` (`seal/capture.py`): in-pipeline activation capture for native
    contrastive vectors.
- **Losses** (`seal/ces.py`): `length_normalized_nll_from_logits`, `ces_rank_loss`
  (softplus), `kl_tokenwise`, `hinge_kl_penalty`, `combined_ces_objective`.
- **Differentiable pipeline** (`models.py`): `generate_latent_batch_grad`
  (grad-enabled latent rollout through the KV), `teacher_force_nll` (energy `E_v(y|x)`),
  `attach_ces`/`_ces_prepare`. Gradients reach `v` only; host frozen.
- **Scripts** (`scripts/`): `mine_budget_pairs.py` (`--single_k` correctness mining),
  `train_ces_claim_b.py` (`--apply_to`, `--grad_clip`, `--max_v_norm`),
  `eval_ces_heldout.py` (`--mode boost --control {negate,random,zero,all}`, `--split`),
  `coef_sweep.py`, `build_native_vectors.py`, `native_eval_sweep.py`,
  `kv_eval_sweep.py`, `ablation_sweep.py`. Drivers: `run_boost_experiment.sh`,
  `run_all_tokens.sh`, `run_offline_chain.sh`.
- **CLI flags** (`run.py`): `--ces*`, `--seal*`, `--kvsteer*`, `--capture_acts`,
  `--latent_steps` / `--planner_steps` / `--critic_steps` / `--refiner_steps`,
  `--agents`, `--ces_steer_phase`.

**Key capability for new mechanisms:** you can backprop from the Judger's answer NLL,
through the differentiable shared KV, into any parameter you insert during the upstream
latent steps. The current work only inserted a *static vector*; the path supports
arbitrary learned modules (see §6).

---

## 6. Fundamental directions for *designing* a latent steering mechanism

These attack the three obstacles in §4. Ordered roughly diagnosis → mechanism. Each is a
hypothesis with a cheap first test; prefer the diagnostics before building.

### 6A. Diagnose the channel before steering it (do this first, cheap)
- **What is decodable from upstream latents, beyond final correctness?** Probe P/C/R
  latents for *intermediate* task features (e.g., "plan mentions the right operation",
  "critique caught the error", numeric sub-results), not just final right/wrong. If
  *nothing* task-relevant is linearly/nonlinearly decodable, latent steering is hopeless
  and that itself is a publishable structural claim. If intermediate features *are*
  decodable, steer **those**, not final correctness. (Reuse `ActivationRecorder`.)
- **Causal dependence of the Judger on the latent KV.** Quantify how much the Judger's
  output changes under controlled ablations/perturbations of specific latent positions
  and layers (an activation-patching study). This maps *where* leverage exists before we
  try to use it.

### 6B. Non-linear / state-conditioned steering (attacks obstacle 4-a)
- A static linear `v` is the wrong tool if correctness isn't linearly encoded. Replace
  `v` with a small **learned steering function** `g(h)` (low-rank or tiny MLP, or a
  learned per-layer gate) conditioned on the current latent state, trained through the
  same differentiable KV path. The infra already backprops to arbitrary inserted params.
  First test: does a rank-`r` conditioned steerer beat the static-`v` null on dev?

### 6C. Manifold-preserving / anti-saturation steering (attacks obstacle 4-b)
- The destructive collapse is off-manifold saturation. Constrain the intervention to stay
  on the latent manifold: steer in a **whitened / PCA-projected** subspace of the latent
  activations; or add a **trust region on the latent distribution itself** (KL between
  steered and unsteered latent states), not just on the Judger output; or steer with a
  norm-preserving (rotation-like) map instead of additive translation. Goal: allow
  movement that changes *content* without breaking the Judger's attention aggregation.
- Explicitly separate the two KV position types (§4.2): steer only positions the Judger
  *routes through*, leave dense content untouched — or vice versa, learn which columns
  are safe to move.

### 6D. Steer the *read*, not the *write* (attacks obstacle 4-c / late binding)
- Since correctness crystallizes at the Judger, act where the signal is: steer the
  **Judger's early-layer integration of the latent cache** (a cross-over between "latent"
  and "Judger" steering), or learn to modulate the Judger's *attention over* the latent
  handoff, rather than editing the upstream latent loops. This is the probe's implied
  target (`RESULTS.md` §11).

### 6E. Make the channel matter first (changes the system, not just the steer)
- If the Judger barely depends on the latent KV (§4.3), no steering of it can help.
  Precondition experiments: force reliance on the channel (truncate the Judger's own
  re-derivation so it *must* use the handoff; or bottleneck the handoff so it carries a
  compressed decision). Then re-test steering. This reframes the question as
  co-designing the channel and the steer.

### 6F. Richer supervision (the current signal is impoverished)
- CES rank with `y+`=gold letter / `y-`=wrong letter on 4-way MC is ~1 token of signal.
  Try full-CoT/logit **distillation from a stronger teacher config**, process-level or
  step-level contrasts on *reasoning trajectories*, or preference data over
  plans/critiques — anything that gives the latent steerer a dense gradient about *what
  good upstream thinking looks like*, not just the final letter.

---

## 7. Concrete, cheap next experiments (in priority order)

1. **Intermediate-feature probes** (§6A) — no training; reuse capture. Decides whether
   the whole direction is viable. **Do this first.**
2. **Activation-patching leverage map** (§6A) — where/what in the latent KV causally
   moves the Judger.
3. If (1)/(2) find steerable signal: **state-conditioned steerer** (§6B) and/or
   **manifold-constrained steer** (§6C) vs the static-`v` null, on the same 4B/MedQA
   dev harness (fast iteration; `coef_sweep.py` pattern).
4. Only if a dev signal emerges: 14B confirm + 2nd task + held-out test.

If (1) and (2) come back empty (no steerable upstream signal, Judger insensitive), the
honest, publishable conclusion is: *the LatentMAS latent inter-agent channel is not a
controllable surface for reasoning quality; steering value lives at the text boundary
(Judger).* That, plus the SEAL-Judger efficiency positive, is a coherent paper.

---

## 8. Artifact & doc index

- **Results docs:** `RESULTS.md` (SEAL Judger + per-agent + native + probe),
  `KV_STEERING_FINDINGS.md`, `GATE1_STATUS.md`, `CES_LATENT_STEERING_FINDINGS.md`,
  `K_BUDGET_CES.md`, `PROGRESS.md`, `HANDOFF_LATENT_STEERING.md` (env/gotchas),
  `EXPERIMENT_OVERVIEW.md`, `RESEARCH_PLAN.md`, **`DELTABRIDGE_HANDOFF.md`** (2026-09
  Mean-Replay + oracle residual; read before any new GPU).
- **On-pod artifacts (`/workspace/latentmas-baseline/artifacts/`):**
  `ces/boost_4b/` (last-token: pairs, train, dev report),
  `ces/boost_4b_tuned/` (grad-clipped train + dev),
  `ces/boost_4b_all/` (all-tokens train + `coef_sweep/report.json`),
  `ces/gate2_smoke.json`, `gate1/*/summary.json`, `sweeps/kv_*`, `kv_steer_vectors/`,
  `plots/`. (Local checkout lacks `seal_vectors/`, `capture/`, native CSVs — they live
  on the pod volume.)
- **Numbers cheat-sheet:**
  - Gate-1 MedQA (14B, test n=100): K0 0.78 / K5 0.77 / **K10 0.83** / K20 0.79 / K40 0.79.
  - SEAL Judger GSM8K (14B): 621 tok/93.3% → coef80 379 tok/94.2% (−38.9%).
  - Learned-CES 4B/MedQA dev: unsteered 0.65; steered ≤ +noise for all positive α;
    all-tokens α=−8 → 0.00.

---

## 9. Guardrails

- Baseline LatentMAS fork is verified byte-for-byte; layer new work as additions, don't
  change upstream semantics. Only commit when the user asks.
- The pod's working copy has **uncommitted** infra edits (`models.py`, `seal/hooks.py`,
  `methods/*`) on top of `origin/feature/seal-token-efficiency`; don't `git reset`/`pull`
  destructively. Sync new files without clobbering (base64/scp), don't overwrite
  `models.py`/`seal/*` blindly.
- Report numbers with n/seed/split caveats; a clean negative is a valid result.
