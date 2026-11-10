# LoopDelta: MSE-Triggered Adaptive Depth and Convergence-Weighted Ensembling for Looped Transformers

**Working title.** **One line:** one measured signal — the loop-to-loop KV delta — picks per-query depth *and* weights the vote.

## 1. Summary

We propose a training-free adaptive-computation system on a single frozen looped (recurrent-depth) transformer, aimed at high accuracy *and* low latency. A planner runs *R_max* loops over the prompt and measures the KV-cache MSE delta across loops, Δ_t; it allocates depth *r* = the first loop with Δ_t < ε (ε calibrated once on a held-out set). Harder queries churn longer → larger *r*. *N* solvers then run exactly *r* loops in parallel at the same depth — diversity comes from temperature sampling, not depth — each installing the planner's first-*r* loop KV for the shared prompt as prefix: bit-exact, zero compute, so the selector roughly pays for its own overhead. Aggregation is convergence-weighted majority vote: each solver's vote gets weight *w_i ∝ 1/(Δ_i + c)* from its late-loop delta, a looped-native confidence signal a standard transformer cannot emit. Everything runs on one engine: no manufactured sender/receiver boundary, co-location as the honest baseline.

## 2. Motivation

Test-time compute scaling for reasoning is converging on a simple empirical fact from our own pilots (Sagnik, Nov 2026, Ouro-1.4B, GSM8K n=200, matched compute): a planner→critic→solver agent team scores **72%** vs **84.5%** for the single model (base backbone; −12.5pp — wider than the −6pp on the thinking backbone), while **majority vote over 4 samples reaches 89%** and self-revision 88%. The critic approved 197/200 plans and caught 30% of planted errors vs 80% for plain re-solving. The lesson is structural, not a bug: verification fails, generation works — one mind iterating beats two minds handing off. We therefore build on the patterns with positive evidence (sampling-based ensembling) and supply what they lack: per-query depth adaptivity and a principled aggregation score. The goal is a high-accuracy, low-latency looped Transformer.

On deployment honesty we keep our earlier stance: CacheBack manufactures its problem by splitting GPUs into sender/receiver pools — a setup with no production counterpart — and OBF is explicit that its chain runs "in a single process on one device," where relay bytes are computed quantities and full relay is fastest end-to-end. We design for boundaries that actually exist: planner and solvers share one engine, one weight copy, and prefix reuse is an in-process pointer handoff, not a transfer.

## 3. Related Work

**Self-consistency** (Wang et al., 2022): sample multiple reasoning paths, unweighted majority vote — the canonical aggregation we build on (ranked-voting variants add +3–5% on 3B models, arXiv:2505.10772, confirming small models benefit). Our delta trigger differs from learned halting approaches throughout: no trained halting head (which often fails to realize savings in practice, Wu et al., 2026), no per-token decisions — a training-free, per-query halt based on the measured KV delta. **LatentMAS** (Zou et al., ICML 2026): motivated this program; its sequential team structure is what our pilots refute (§2). Verified: it never used thinking mode, so its +14.6% was measured where the single baseline could not reason internally. **RecursiveMAS** (Yang, Zou et al., 2026): MAS-as-recursion with *trained* RecursiveLinks across heterogeneous models; closest prior art — we differ throughout: single frozen backbone, training-free, exact reuse instead of learned maps. **CacheBack / OBF**: position selection at the handoff at full feature dimension — an orthogonal axis; neither touches the recurrence axis. **Ouro** (Zhu et al., ByteDance, 2026): our pilot backbone (1.4B/2.6B, 7.7T tokens); code-verified that the forward unconditionally runs all `total_ut_steps` — the learned gate only selects the LM-head input and saves no forward compute, making it the built-in-adaptivity baseline. Their §5.4.2 studies loop-axis KV reuse (first-step-only collapses decoding, 18.7 vs 78.9 GSM8K; last-step/averaged matches at 4× less memory) — within-model decode sharing, not our cross-pass use. **Huginn-0125** (Geiping et al.): full-range backbone, *R* ∈ 1–64 via `num_steps` (*R* ≥ 4 floor, 4K context, Apache-2.0).
**ResidualQuant** (Kim et al., 2026): KV-cache quantization for looped transformers, premised on the same inter-loop KV similarity our signal exploits — their Eq. (3.1) bounds ‖X⁽ʳ⁾−X⁽ʳ⁻¹⁾‖₂, the unnormalized form of our Δ_t, measured on our exact backbones (Ouro-1.4B, Huginn-3.5B). Orthogonal use: they compress the residual (INT2, last-loop anchor); we read the residual's magnitude as a difficulty/confidence signal. Closest existing precedent for *using* the quantity: their loop-wise mixed precision already allocates bits by residual size (larger residual → INT4) — we allocate loops (depth) and vote weight by delta size instead: same "read the residual" principle, different resource. Their finding that late-loop KVs are stable enough to anchor on independently supports our convergence-weighting premise. (Their §1 also points to MoR/PLT/MELT on KV sharing across recurrences — adjacent, not overlapping.)
**Adaptive depth granularity.** Nearly all looped-transformer adaptivity is *per-token* (Ouro's gate, Huginn's zero-shot exit, Mixture-of-Recursions). *Per-query* depth exists for standard transformers via trained heads or user budgets, with Jeddi et al. (2026) reporting fragility of learned strategies on looped models. Training-free, per-query, measured-delta depth allocation appears unexplored (hedge: search not exhaustive).

## 4. Method

**Backbone.** One frozen looped transformer serves as planner and solvers (Ouro-1.4B pilots, *r* ≤ 4; Huginn-0125 for the full 4–64 range). Same weights throughout — one engine, one weight copy.

**MSE-triggered loop selector.** The planner prefills the prompt at *R_max* loops and records per-loop KV caches. Difficulty signal: Δ_t = mean normalized MSE(KV_t, KV_{t−1}) over K and V, all layers, prompt positions. Allocate *r* = min{t : Δ_t < ε}; ε is calibrated once on MATH-Train to hit a target mean budget, then frozen before any test-set evaluation (§6). Rationale: easy queries settle fast (small Δ early); hard queries churn (large Δ persists) — measured directly from the cache, training-free, per-query. This replaces LM-head integer decoding entirely: no regex parsing, no fallback defaults, nothing the model was never trained to emit. Fallback under test: if prefill Δ doesn't track difficulty (reasoning lives at decode), use a short probe decode of *k* tokens at *R_max* instead — same signal, slightly higher cost.

**Adaptive ensemble.** *N* solvers (*N*=4) run exactly *r* loops — the *same* depth. Depth diversity is deliberately excluded for now: underpowered solvers would cast wrong votes on hard queries and confound the adaptivity claim (a follow-up experiment, not this one). Diversity comes from temperature sampling (*T*≈0.6; at *T*=0 the ensemble collapses to one path). Each solver installs the planner's first-*r* loop KV over the shared prompt as prefix: bit-exact (same model, same input, aligned loop indices), zero compute. Saving *N·r·n₁* prefill loop·tokens ≈ the planner's own *R_max·n₁* — the selector pays for itself.

**Convergence-weighted voting.** After decoding, each solver reports its late-loop delta Δ_i (MSE between loops *r*−1 and *r* over its generated positions; optionally answer-stability across the last loops via cheap LM-head reads). Vote weight *w_i ∝ 1/(Δ_i + c)*; final answer = argmax_a Σ_i *w_i*·1[a_i = a]. Converged paths outvote churning ones. Unweighted majority is the ablation (B4). The signal is looped-native: a standard transformer has no loop axis on which to measure convergence.

**Figure 1.** MSE-triggered selector → same-depth parallel solvers with first-*r* prefix reuse → convergence-weighted vote.

![LoopDelta architecture](loopdelta-ensemble.png)

## 5. Scientific Claims & Evaluation

**Claims to measure:** (C1) prefill Δ_t tracks query difficulty → planner-*r* beats fixed-*r* at matched compute; (C2) late-loop Δ_i predicts solver correctness → weighted vote beats unweighted; (C3) the full system dominates the accuracy-vs-compute Pareto at lower latency than a fixed-*R_max* ensemble.

**Benchmarks:** GSM8K, MATH, GPQA first (difficulty-stratified); rest of LatentMAS's suite as compute allows. Same frozen backbone for all baselines:

| ID | Baseline | Isolates |
| --- | --- | --- |
| B1 | Single looped, fixed *R_max*, greedy | "Why not just run the model?" |
| B2 | Single + self-consistency, unweighted vote, *N*=4, fixed *R_max* | The 89% to beat |
| B3 | Planner-*r*, single solver | Adaptivity alone (load-bearing) |
| B4 | Planner-*r* + *N* solvers, unweighted vote | Weighting alone |
| B5 | Full: planner-*r* + *N* solvers + Δ-weighted vote | Everything proposed |
| B6 | Ouro native gate (readout selection) | Learned per-token adaptivity |

**Metrics:** accuracy, total loop·tokens, measured wall-clock latency, *r* histogram by difficulty. **Key figures:** (1) accuracy-vs-compute Pareto — B5 must dominate; (2) *r* distribution shifting with difficulty; (3) planner-Δ vs MATH difficulty levels 1–5 (validates the signal); (4) solver-Δ vs correctness (validates the weighting); (5) weighted vs unweighted vote ablation; (6) latency: adaptive vs fixed-*R_max* ensemble (expect ≈ planner + *r*·T, straggler-bound, not *N*×).

## 6. Validation Plan & Timeline (target ACL 2027, early Jan)

**Stage 0 (days — kill fast):** Sagnik first uses **MATH-Train as the calibration set**: prefill each question at *R_max* loops (prefill-only, no decode — cheap), record the Δ_t trajectory, and check whether it tracks the annotated difficulty levels 1–5 (e.g., area under the Δ curve, or the loop index where Δ first drops below candidate ε values, vs level). A stratified sample (~200 per level) suffices — no need for all 7500. In the same pass, compute solver late-loop Δ_i on questions with known answers → correlate with correctness. If either correlation is ~zero, pivot before building anything. If they hold, **ε is calibrated on MATH-Train to hit a target mean budget, then frozen** — test sets (MATH-500, GSM8K) are never touched during calibration.
**Stage 1:** B3 vs fixed-*r* single at matched compute — the adaptivity gate.
**Stage 2:** B4, then B5 — ensemble first, then weighting.
**Stage 3:** Huginn-0125 full-range Pareto; MATH/GPQA; ablations; writing. Training-free throughout: inference sweeps only.

## 7. Contributions

1. **MSE-triggered per-query loop-depth allocation**: training-free, measured-not-predicted — replacing fragile LM-head integer decoding with a halting signal that comes free from the KV cache.
2. **Convergence-weighted voting**: the loop-delta as a training-free confidence signal for aggregation — looped-native, unavailable to standard transformers.
3. **First-*r* prefix reuse, planner→solver**: bit-exact, zero-compute; the selector pays for itself; single-engine, zero communication.
4. **One signal, two uses**: the loop KV-delta drives both depth allocation and vote weighting.
5. **Data-driven pivot**: published pilot evidence that agent teams fail where sampling ensembles succeed (a verifier deficit, not a scale story), grounding the design in a negative result.

*Supersedes: the looped-LatentMAS v3 multi-agent design (Oct 29, 2026) — the Nov 1–3 pilots refuted the team structure; the AB-projection email draft (Oct 23–25, likely never sent).*
