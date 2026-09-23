# LatentMAS Steering — Master Research Brief (single self-contained doc)

Updated: 2026-09-16. This is the one document to hand a research agent. It consolidates
everything: the system, the full experiment log with exact numbers, the mechanism, the
one positive method result, reusable infra, environment/repro, and prioritized open
directions. Source docs it subsumes: `EXPERIMENTS_UNIFIED_RESULTS.md`,
`DIAG_READOUT_FINDINGS.md`, `SCAFFOLD_RESULTS.md`, `RESULTS.md`, `KV_STEERING_FINDINGS.md`,
`GATE1_STATUS.md`, `CES_LATENT_STEERING_FINDINGS.md`, `LATENT_STEERING_RESEARCH_HANDOFF.md`,
`PAPER_OUTLINE.md`, `DELTABRIDGE_HANDOFF.md`.

---

## 0. TL;DR (read this first)

- **Goal evolution:** started as "SEAL + LatentMAS", pivoted to "can we steer the latent
  agents to improve reasoning?", and after that came back negative, pivoted to "what does
  the latent inter-agent cache actually do, and can we replace it?"
- **Main negative (well-established):** steering the upstream latent agents does NOT improve
  accuracy — shown with 3 independent methods (diff-of-means SEAL, KV-cache, learned CES),
  cross-model and cross-task, with controls.
- **Main mechanism (well-established):** LatentMAS's downstream benefit is **insensitive to
  the semantic content** of the inter-agent KV cache. A wrong-question cache helps the Judger
  as much as the right one; the answer is barely decodable from upstream latents. The cache
  acts as a **content-independent conciseness scaffold** (~40% fewer Judger tokens), and its
  accuracy edge over judger-only is largely a token-budget effect.
- **Main positive (the method result):** on Qwen3-14B, the 3 upstream agents (33 latent
  forwards) can be replaced by **one fixed, question-independent synthetic KV cache**
  (Gaussian matched to averaged real-cache per-channel statistics, full length) that
  reproduces full-LatentMAS accuracy and conciseness with **zero upstream compute**.
- **Latency reality check (2026-09-21, measured):** on AIME with Qwen3-14B, the 3 silent
  agents cost **0.95s of a 152.7s item (0.6%)**; the Judger decode is **99.4%**. So the whole
  cache-surgery family (eviction, dropping roles, alternate wirings, synthetic scaffolds)
  **cannot be sold as a latency win on AIME** — justify it by accuracy or KV memory instead.
  The real latency target is Judger tokens: the items we get **wrong** are exactly the ones
  that never emit EOS and burn the full 8192 budget (188s vs 60s for a solved item).
  Full numbers + next steps: `AIME_LATENCY_LOCALIZATION.md`.
- **Every AIME failure we have is a failure to *reach* an answer (2026-09-23):** across
  all 26 generations, "graded correct" and "ever wrote the gold answer anywhere" **never
  disagree (0/26)**, and failing runs emit **zero** boxed expressions — still
  mid-computation at the budget. So accuracy here may be measuring convergence *speed*,
  not reasoning ability, and SEAL's damage was making items **not finish** rather than
  answer wrong. Next run (`insight`) extends the budget to settle it.
- **The latent cache earns its keep on AIME (2026-09-22, measured, n=6):** deleting the
  upstream cache halves accuracy (**0.667 → 0.333**) and collapses termination
  (**EOS 0.67 → 0.17**). Shuffling the cache was survivable; removing it is not. The
  cache's contribution looks like knowing when the problem is *done*.
- **SEAL at the Judger is a net loss on AIME (same run):** acc **0.667 → 0.500** and
  mean tokens go *up* (6591 → 6745) because EOS rate halves. It is a −31%/−23% token
  win on the items it keeps, and it uniquely solves item 1, but it breaks termination
  on two items the baseline solved. The third latency lever is closed at n=6; needs
  n=30 before it is quotable either way.
- **Grouped (batched) Judger decoding fails parity — do not use it for accuracy:**
  batch 6 gave 1.75x throughput for **half the solves** (0.667 → 0.333), 0/6 texts
  identical. Root cause is padding/position handling with a prefix cache. This also
  puts a question mark on earlier batched numbers from `exp_frozen_seal.py` /
  `exp_one_role.py`; check them with `--mode compare` before publishing.
- **Two of the three Judger levers are now dead (2026-09-22, measured):** a hard token cap
  buys 1.20x for **−0.167 accuracy** (and 1.71x for −0.500), and the degenerate-tail
  detector **fires on nothing** — the non-terminating AIME items are doing real
  non-repetitive arithmetic on a wrong intermediate count, not looping. The waste is
  **verbosity** (reflection/transition moves), which is precisely what the Judger SEAL
  vector targets. `run_aime_localize.sh seal` is the go/no-go.
- **What didn't work:** shrinking the scaffold to a few slots (learned m=64 → 0.50 vs fixed
  full-length 0.54–0.60), so we save compute but not KV memory yet. Mean-Replay of MATH-1k
  matches Real on GSM8K/MATH but **not AIME**. DeltaBridge (one-token Real−Frozen residual
  at L32) **beats Real on MATH** (85 vs 74) and **destroys AIME** (even uncompressed
  `oracle_full`). Full write-up: `DELTABRIDGE_HANDOFF.md`.

---

## 1. System under study (facts)

- **LatentMAS** (Zou et al., arXiv:2511.20639), verified byte-for-byte fork of
  `Gen-Verse/LatentMAS@9a9e4d3` (reproduces paper accuracy). Pipeline: **Planner → Critic →
  Refiner → Judger** (`methods/__init__.py`, `methods/latent_mas.py`).
- Upstream 3 agents reason in **latent space**: each runs `K = latent_steps` forward passes,
  feeding last-layer hidden state back as next input embedding (via realign `W_a`), emitting
  **no text**. They communicate through a shared, growing **KV cache** (`past_key_values`).
  Only the **Judger** decodes text.
- Total upstream latent forwards ≈ `3·(K+1)` (K=10 default ⇒ 33). Judger cache ≈ ~1000
  positions (agent prefills contain the question 3× + 30 latent tokens).
- Backbones: **Qwen3-14B** (hidden 5120, 40 layers, 8 KV heads, head_dim 128) primary;
  **Qwen3-4B** (hidden 2560, 36 layers) for cheap iteration. Vectors/scaffolds do NOT
  transfer across sizes (different dims).

---

## 2. Complete experiment log (exact numbers)

Unless noted: temp 0.6/top-p 0.95 seed 42 for §2.1–2.6; greedy (temp 0) for §2.7–2.10.

### 2.1 Gate 1 — unsteered K-curve (14B MedQA n=100, 4096 tok)
K0 0.78 / K5 0.77 / **K10 0.83** / K20 0.79 / K40 0.79. Best at K=10; cutting K does NOT save
wall-clock (Judger CoT dominates and *shrinks* as K rises). AIME24 n=30: K10 0.67 > K0 0.53 >
K40 0.50. → K-budget/latency framing dead; Judger insensitive to K beyond ~10.

### 2.2 SEAL residual @ Judger — POSITIVE (14B GSM8K n=120)
Vector `mean(exec) − mean(reflection+transition)`, layer 28. Control 621 tok/93.3%:
coef40 515/95.0 (−17%), coef60 437/94.2 (−30%), coef80 379/94.2 (−39%), ~2× faster.

### 2.3 SEAL transfer
ARC-C −25% tokens @ iso-acc (95.8%); MedQA task-sensitive (coef60: −12% tok, −5 pts acc).

### 2.4 Per-agent SEAL (GSM8K n=120)
Only the Judger cuts tokens (515/95.0 @ coef40). Planner/Critic/Refiner ~unchanged length.
All-agents ≈ Judger-only. Thought-type mix ~identical (~88% exec) → Judger's role is
**structural** (sole text-emitter).

### 2.5 Native correctness-contrastive upstream steering — NEGATIVE (RESULTS.md §9)
`v = mean(latent|correct) − mean(latent|incorrect)` per agent, in-pipeline. Within noise on
GSM8K/ARC, hurts MedQA, no token savings; does NOT beat Judger-only SEAL.

### 2.6 Correctness probe (mechanism, 14B, layer 28 latent)
Per-agent 5-fold AUC vs final correctness: Planner 0.54 / Critic 0.62 / Refiner 0.59 /
**Judger 0.69–0.80**. Correctness decided late; barely encoded upstream.

### 2.7 KV-cache steering (14B GSM8K n=150)
Handoff steering **falsified** (handoff_last too weak ≈1/130 attention; handoff_all cv=1
collapses 0.947→0.72). `judger_token` (aggregation position) works: −25 to −33% tokens at
unchanged acc. Mechanism: content positions corrupt; aggregation positions ride.

### 2.8 Learned CES "boost" — NEGATIVE (4B MedQA, K=10, layer 20, dev)
Trained residual `v` via CES rank loss through the differentiable KV. Last-token: inert
(coef sweep α=4/8/16 flat, α−8 flat; all within ±0.05 of unsteered 0.65). All-tokens:
**destructive-only** (α+ within noise; **α−8 → 0.00** via output degeneration). Gate-2 grad
smoke passed (grads reach v only). Matches the probe: no linear "toward-correct" direction upstream.

### 2.9 Role ablation
Judger-only ≈ full pipeline (GSM8K 0.91 vs 0.93; MedQA 0.737 vs 0.717) — low upstream causal value.

### 2.10 Readout diagnostics — cache-usage + answer probes (the pivot)
Decode the Judger with its cache replaced by variants, across Judger token budgets. Greedy.
Scripts `scripts/diag_cache_usage.py`, `diag_answer_probes.py`.

14B MedQA, n=40 (real / shuffled / none):
| budget | real | shuffled | none | tokens real/shuf/none |
|---|---|---|---|---|
| 1024 | 0.675 | 0.625 | 0.350 | 722 / 712 / 916 |
| 2048 | 0.775 | 0.775 | 0.700 | 882 / 867 / 1329 |
| 4096 | 0.800 | 0.775 | 0.750 | 892 / 878 / 1603 |

4B MedQA n=60 b1024: real 0.417 / shuffled 0.467 / zero 0.217 / none 0.167 (zero ≈ none).
GSM8K 14B: b1024 real 0.933 / shuffled ~0.93 / none 0.800.
Answer probes (gold 4-way, chance 0.25, 4B+14B): best-layer MLP 0.28–0.33, AUC ~0.50 — near chance.

**Reads:** (a) **real ≈ shuffled** → Judger ignores cache CONTENT; (b) real−none collapses
as budget grows (token-budget artifact; judger re-solves when unconstrained); (c)
content-independent **conciseness** (~40% fewer tokens); (d) answer not decodable upstream.

### 2.11 Synthetic latent scaffold — the POSITIVE (`SCAFFOLD_RESULTS.md`)
Mechanism sweep (which variant reproduces the effect?):
- 4B MedQA b1024 n=40 (real 0.45, none 0.15): only real/shuffled work (0.45); matched-noise,
  single-vector, trunc≤256, pca≤64 ALL fail (4B brittle to off-manifold caches).
- **14B MedQA b1024 n=40 — FLIP** (real 0.675, none 0.350): matched (0.600) and pca32 (0.650)
  now reproduce; repeat_last (0.225), trunc64 (0.400) fail. → scaffold = coarse stats at full length.

Fixed universal synthetic scaffold `synthglobal` (one fixed cache from averaged stats of 8
train caches, question-independent, 0 upstream forwards):
| set | real | synthglobal | none |
|---|---|---|---|
| 14B MedQA b1024 n=40 | 0.675 | **0.600** (701 tok) | 0.350 |
| 14B MedQA b1024 n=80 | 0.600 | **0.537** (~76% recovery) | 0.338 |
| 14B GSM8K b512 n=60 | 0.567 | **0.567** | 0.467 |
| 14B GSM8K b1024 n=60 | 0.933 | **0.883** | 0.800 |
Robustness: 3 synth seeds → 0.55/0.60/0.65 (not a lucky cache).

Shrinking (14B MedQA b1024 n=40; real 0.675, none 0.350):
- length: synth l32/l64 ≈ 0.575 (work), l256 0.30 (single-seed noise); rank r64 0.55 → r8/16 0.45.
- **learned m=64 slots** (distilled to full-LatentMAS behavior): **0.500** — trains (loss 0.08,
  grad flows) but below the fixed full-length synth. m=32/16 not run.

### 2.12 Mean-Replay + DeltaBridge (14B greedy, 2026-09)

Frozen 422-pos MATH-1k Mean-Replay (`artifacts/math_ladder/math1k/cache.pt`) vs Real K=10:

| Task | Real K=10 | Frozen Mean-Replay | None |
|---|---|---|---|
| GSM8K n=100 | 92% | **92%** | 84% |
| MATH n=100 | 74% | **76%** | 73% |
| AIME24 n=30 | **66.7% (20/30)** | **56.7% (17/30)** | 53.3% |

`frozen_k2`/`k5` on AIME stayed 56.7%. AIME’s extra 3 solves need instance-specific recurrent
compute, not a shorter frozen prefix.

**DeltaBridge oracle** (newline bridge token; residual at **L32 r=16**, first generate step
only). Bank from MATH-train only. Full report: `docs/DELTABRIDGE_HANDOFF.md`.

| Arm | MATH n=100 | GSM8K n=100 | AIME24 n=30 |
|---|---|---|---|
| zero (Frozen + dummy bridge) | 76% | 86% (dummy tax vs 92%) | **19/30 (63%)** |
| shuffled projected | 80% | 90% | 10/30 |
| oracle full \(d\) | 82% | 91% | **8/30** |
| oracle rank-16 | **85%** (+11 vs Real) | **92%** | **6/30** (recovered 0, lost 13) |

MATH: **go** — compressed residual beats Real and halves Judger tokens. AIME: **hard no**.
`oracle_full` also collapsed, so this is not “rank too small.” Do **not** train `g_φ` for
AIME. GPU stopped. Next bets in `DELTABRIDGE_HANDOFF.md` §3: scale/layer sweep of existing
`eval_states.pt`, or MATH-only predicted residual as a separate claim.

---

## 3. Mechanism synthesis (what's true)

1. **Leverage is at the text-emitting Judger**, not the latent agents (agreed by SEAL,
   KV-steer, CES, role-ablation, and the diagnostics).
2. **The latent channel is content-insensitive**: a wrong-question cache = right one; answer
   not decodable upstream (probes ≈ chance); steering its content can't improve accuracy.
3. **What the cache does**: a content-independent **conciseness scaffold** — any full-length,
   on-manifold model-generated (or, on 14B, statistically-matched synthetic) KV prefix puts
   the Judger into a shorter/more-accurate reasoning mode. Its accuracy edge over judger-only
   is largely a token-budget effect (vanishes when the Judger has room to re-solve).
4. **Model-size effect**: 4B is brittle (only real caches work); 14B is robust (matched-stat
   and rank-32 synthetic caches work) → the effect is coarse-statistical on capable models.

---

## 4. Infrastructure to reuse (all validated, in-repo)

- **Pipeline**: `methods/latent_mas.py` `LatentMASMethod.run_batch` (HF; @no_grad). `run.py`
  CLI (flags: `--latent_steps/--planner_steps/...`, `--agents`, `--seal*`, `--kvsteer*`,
  `--ces*`, `--capture_acts`).
- **Model wrapper** (`models.py`): `generate_latent_batch` (no_grad) / `generate_latent_batch_grad`
  (grad-enabled latent rollout through the differentiable KV) / `teacher_force_nll` (energy
  E(y|x); differentiable) / `generate_text_batch` (Judger decode; builds mask from
  `_past_length(past)` so ANY cache works as `past_key_values`) / `attach_ces`.
- **Steerers/losses** (`seal/`): `SealSteerer` (detached diff-of-means residual),
  `TrainableSteerer` (nn.Parameter residual; `apply_to last|all`; phase gating),
  `ces.py` (`ces_rank_loss`, `kl_tokenwise`, `combined_ces_objective`, NLL),
  `kv_steer.py` (KVCacheSteerer), `capture.py` (ActivationRecorder).
- **Diagnostic/experiment scripts** (`scripts/`):
  - `diag_cache_usage.py` — cache real/shuffled/zero/none × token budget (idx-aligned paired CIs).
  - `diag_answer_probes.py` — per-agent/layer capture + torch linear/MLP probes; saves features.npz.
  - `diag_scaffold_sweep.py` — 12 cache variants incl. `matched`, `repeat_last`, `trunc{m}`,
    `pca{r}`, and `synth*` (fixed synthetic scaffold at parsed length/rank/seed, e.g. `synth_l64_r16`).
  - `train_synth_scaffold.py` — learn m KV slots (frozen model) via distillation to full-LatentMAS
    behavior through `teacher_force_nll`; evals vs none.
  - `train_ces_claim_b.py`, `eval_ces_heldout.py` (`--mode boost --control`), `coef_sweep.py`.
  - drivers: `run_diagnostics.sh`, `run_boost_experiment.sh`, `run_all_tokens.sh`,
    `run_scaffold_chain.sh`, `run_learn_chain.sh`.
- **Key capability**: the differentiable KV path can train *any* inserted object (a residual
  vector, or learnable KV slots as the Judger's `past`) with the host frozen.

---

## 5. Environment & reproduction (RunPod)

- **Pod** (may differ after restart): direct TCP `ssh root@103.196.86.103 -p 21444 -i ~/.ssh/id_ed25519`
  (supports exec + scp; preferred). Proxy `dhdjbpkrk690q2-64410b2b@ssh.runpod.io` is
  interactive-only and was intermittently unreachable.
- **Env**: torch 2.8.0+cu128, **transformers==4.57.1** (5.x breaks KV handling), vllm 0.11,
  1× H200 (140 GB). `/workspace` persists; `/root` venv is ephemeral → rebuild with
  `/workspace/recover_env.sh` (~5 min). `source env_native.sh` sets HF_HOME + activates venv.
- **Durability**: run long jobs in **tmux** on the pod (survives disconnects). CAUTION: a
  `while tmux has-session` completion check is unreliable (sessions can linger) — use a
  file-sentinel or `pgrep` on the python instead. `runpodctl` is installed but has **no API
  key** → cannot auto-stop the pod from inside; stop it from the RunPod console to avoid idle billing.
- **Data**: MedQA local (`data/medqa.json`; splits test[0,100), train[100,220), dev[220:]);
  GSM8K/ARC/AIME via HF. Grading via `extract_gsm8k_answer` + `normalize_answer`.
- **Artifacts** (`/workspace/latentmas-baseline/artifacts/`): `diag/{4b,14b}/*`,
  `ces/synth_scaffold_m64/`, `ces/boost_4b*`, `gate1/*`, `sweeps/*`, `kv_steer_vectors/`, `plots/`.

---

## 6. Open questions & prioritized next directions

**Lock-in (this week): Frozen + Judger SEAL (`scripts/run_frozen_seal.sh`).**
Zero upstream + GSM8K SEAL coef 40 on the MATH-1k prefix. GSM8K n=40 → MATH n=40 →
AIME n=6. Kill if accuracy drops. This is a latency result, not an AIME accuracy hunt.
AIME one-role and DeltaBridge residual for AIME accuracy are **stopped**.

**DeltaBridge (read `DELTABRIDGE_HANDOFF.md`).** Oracle residual at L32 destroyed AIME.
Do not train the predictor. ε-scale / L28 of existing `eval_states.pt` is optional
background, not the next GPU.

**A. Harden the positive for a paper (low risk).** More seeds + n on the synthglobal≈real
claim; a hard task (AIME) for the strongest budget-crossover; **measure the actual
compute/latency/latent-forward savings**; accuracy–token Pareto (real vs synthglobal vs none).

**B. Shrink the scaffold — the higher-upside systems win.** pca32 works but learned m=64
(0.50) < fixed full-length (0.54–0.60). Try: low-rank *stored* scaffold (rank-32 factors, not
full KV) to cut memory; better-optimized learned slots (longer training, KL-to-real-cache
target instead of behavior-clone, larger m like 128/256); find min length that holds
(multi-seed length sweep — the l256 dip needs repeats). Goal: save compute AND KV memory.

**C. Mechanistic depth.** Why does a statistical scaffold induce concise mode? Probe the
Judger's attention over the scaffold (attention-sink / effective-context-length?); which
layers' stats matter (per-layer synth vs real ablation); is it norm, covariance, or rank?

**D. Generality.** Where's the 4B→14B robustness threshold (8B?); other families (Llama);
other tasks (AIME/ARC); hierarchical LatentMAS.

**Paper framing (`PAPER_OUTLINE.md`):** *"Synthetic Latent Scaffolds — LatentMAS's inter-agent
cache is a content-independent conciseness scaffold, replaceable by a fixed synthetic KV
object with zero upstream compute."* Contributions: (1) content-insensitivity mechanism +
causal shuffled-cache evidence; (2) the fixed synthetic scaffold method; (3) compute savings;
(4) honest limits (shrinking, gap to real). Realistic venue: workshop/Findings now; stronger
with B (memory savings) + A (hardening) + D (generality).

---

## 7. Honest status line

One paper-worthy positive (synthetic scaffold replaces upstream agents on 14B) + a clean,
well-controlled mechanistic story. Mean-Replay extends that to a *type-level* math cache
(Frozen=Real on GSM8K/MATH; Frozen misses Real on AIME). DeltaBridge’s MATH oracle beating
Real is a second method-shaped result; the AIME one-token residual is an honest negative.
Not a hero SOTA; the "tiny learned scaffold matches real" outcome did not materialize. All
numbers above are single-seed at n=40–150 unless stated — hardening (seeds/n/tasks + compute
measurements) is the main gap before write-up.
