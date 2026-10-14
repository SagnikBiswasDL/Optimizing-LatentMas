# DeltaBridge — Handoff to lead research agent

Updated: 2026-09-16. Qwen3-14B, greedy, seed 42. Pod `hij9pvt2b23caz` is **EXITED**.
Artifacts live on the network volume:
`/workspace/latentmas-baseline/artifacts/delta_bridge/`.

Code: `seal/delta_bridge.py`, `scripts/exp_delta_bridge.py`, `scripts/run_delta_bridge.sh`.
Driver: `bash scripts/run_delta_bridge.sh`. Do **not** train `g_φ` from this run.

---

## 0. What we tested

Hypothesis (your DeltaBridge note): sequential K=10 decomposes into a **task-level prior**
(the frozen 422-pos MATH-1k Mean-Replay cache) plus a **low-rank, problem-conditioned
residual** injectable at one designated bridge token.

\[
C(x)=C_{\text{Mean-Replay}}+\Delta(x),\qquad
\Delta \approx U\alpha,\ \ \alpha\ \text{from Real}{-}\text{Frozen bridge states}.
\]

Protocol: append one existing-vocab token (newline) after Frozen+Judger-prefill; add the
residual at **one layer, last token, first generate step only**; then Judger-decode.
No vocab resize. Bank fit on **MATH-train n=1000 only**. No AIME in the bank.

Oracle is a cheat: Real K=10 is run at eval **only to read** \(d=h_{\text{Real}}-h_{\text{Frozen}}\).
If that cannot recover AIME’s extra solves, a predictor cannot either.

Locked baselines (same cache, same model):

| Task | Real K=10 | Frozen Mean-Replay | None |
|---|---|---|---|
| GSM8K n=100 | 92% | **92%** | 84% |
| MATH n=100 | 74% | **76%** | 73% |
| AIME24 n=30 | **66.7% (20/30)** | **56.7% (17/30)** | 53.3% |

`frozen_k2`/`frozen_k5` on AIME stayed at 56.7% (already dead).

---

## 1. Results

PCA pick from MATH-train residuals: **layer 32, rank 16**
(explained 0.697, residual energy **440**). Layer 28 was close (explained 0.716, energy 240).
We injected only the picked layer.

### MATH n=100, T=2048 (protocol check — **passed**)

| Arm | Acc | Tokens |
|---|---|---|
| zero (Frozen + dummy bridge) | **76%** = locked Frozen | 1239 |
| shuffled projected residual | 80% | 730 |
| oracle full \(d\) | 82% | 665 |
| **oracle rank-16** | **85%** | 633 |
| LatentMAS Real K=10 (gate, B=20) | 74% | 1145 |

Verdict on MATH: **go**. Recovered 12 Frozen misses, lost 3. Shuffled recovered a
different set. Rank-16 **beat Real by +11 pts** and halved Judger length.
This is the surprising positive: a compressed Real−Frozen handoff on top of Mean-Replay
is a *better Judger context than Real’s full ~1000-pos cache* on MATH.

### GSM8K n=100, T=1024

| Arm | Acc |
|---|---|
| zero | **86%** (dummy bridge ding vs locked Frozen 92%) |
| shuffled | 90% |
| oracle full | 91% |
| oracle rank-16 | **92%** = locked Real/Frozen |

Dummy newline is not free on GSM8K. The residual puts accuracy back on the locked number.

### AIME24 n=30, T=8192 (the actual claim) — **hard no**

| Arm | Acc | n/30 | Tokens |
|---|---|---|---|
| zero | **63.3%** | 19/30 | 6070 |
| shuffled | 33.3% | 10/30 | 3293 |
| oracle full \(d\) (item-specific, unprojected) | 26.7% | 8/30 | 3643 |
| oracle rank-16 | **20.0%** | 6/30 | 2850 |

- `n_recovered` vs zero = **0**. Lost **13** items `[5,6,8,10,11,12,14,18,19,23,25,26,27]`.
- Gate recommend: `retry_4token`. That was the pre-registered rule when 1-token fails.
- **Do not follow it blindly.** `oracle_full` also collapsed. The failure is not “rank 16
  was too small.” A single-token patch of Real−Frozen at L32, full scale, is *destructive*
  on AIME, including the item’s own uncompressed \(d\).

Zero 19/30 is a bit above locked Frozen 17/30 (dummy bridge / B=1 vs old B=1 gate).
Treat as “Frozen-class,” not as a new SOTA. The residual arms are unambiguously worse.

Traces got shorter *and* worse. Same signature as over-strong SEAL / CES: the intervention
pushes a concise MATH-like mode that AIME cannot afford.

---

## 2. What this falsifies vs what it does not

**Falsified (at this exact intervention):**
> AIME’s 3 extra K=10 solves live in one last-token residual at L32 that you can add
> at coefficient 1 onto Mean-Replay.

**Not falsified:**
- Mean-Replay as a GSM8K/MATH prior (still holds).
- AIME needing *instance-specific recurrent* compute (still the leading account).
- A *scaled* or *earlier-layer* residual helping AIME (never swept).
- A MATH-only method: Frozen + predicted residual beating Real on MATH/GSM8K.

**Important mechanistic read of AIME:** shuffled 33% and oracle 20–27% are *both* far
below zero 63%. Any residual from this family hurts. This is not “wrong α for this
item.” It is “this patch is off-manifold for contest decode.”

**Important mechanistic read of MATH:** oracle > Real. Full Real KV is not the
information-optimal Judger prefix when a type-level cache already exists. That is
compatible with the older “cache is a conciseness scaffold” story, plus a *small*
instance residual that MATH can use and AIME cannot.

Residual energy at L32 (440) vs L16 (52) is huge. Full-scale add at L32 is a very
strong intervention. MATH happened to like it. AIME did not.

---

## 3. Where to go (priority)

GPU is stopped. Do not train the MLP, SEAL combo, eviction, or fallback until an
AIME oracle is actually non-destructive.

### P0 — Cheap ablations that reuse `eval_states.pt` (no recapture)

AIME (and MATH) `eval_states.pt` already store per-item `h_real` / `h_frozen` / `q`
at layers **16, 24, 28, 32**. We can re-decode with different inject recipes.

1. **Scale sweep** of the *same* AIME \(d\): \(\varepsilon \in \{0.05, 0.1, 0.25, 0.5\}\)
   at L32. Hypothesis: coeff 1.0 is a hammer; a SEAL-sized step might not collapse
   contest decode. If ε-AIME stays at 19/30, stop. If it climbs toward 20–22/30, the
   original hypothesis is still alive and we under-specified the dose.
2. **Layer 28** at ε=1 and at the best ε, using the same states. L28 has higher explained
   variance and ~half the energy.
3. **No new MATH-train collect** for these.

Cost: AIME decode is still ~1.2 h/arm in the worst case (8192 cap). Three ε values at
one layer ≈ one more night, restart-safe. **This is more informative than 4-token
recollect**, because `oracle_full` already failed.

### P1 — Do not spend on (unless P0 moves AIME)

- Coefficient MLP / teacher KL / answer NLL.
- SEAL on the bridge “for AIME accuracy.”
- SnapKV eviction of the 422 cache.
- Real-fallback controller.
- Four bridge tokens (re-collect 1000 tapes). Only if P0 shows that *some* residual
  injection is non-destructive but capacity-limited. Right now it looks destructive,
  not capacity-limited.

### P2 — MATH-only productization (separate claim)

If the paper can live without AIME accuracy:

> On tasks where Mean-Replay already matches Real, a rank-16 Real−Frozen residual at
> the Judger boundary beats full LatentMAS (MATH 85 vs 74) and cuts tokens ~2×.

Then it *is* worth training \(g_\phi(q)\to\hat\alpha\) on MATH-train features (already
in `residuals.pt`) and evaluating **predicted** residual on MATH/GSM8K held-out.
Never tune on AIME. AIME stays the honest negative: type-level prior, Judger must
re-solve.

This is a stronger write-up than “sometimes use K=2,” which already failed.

### P3 — Mechanism, not a new method

- Why does Frozen+residual beat Real’s cache on MATH? Attention to the 422 vs the
  ~1000 Real positions; last-token residual energy vs KV content.
- Dummy-bridge tax: GSM8K −6 pts, MATH 0, AIME slightly *up*. Fix the protocol for any
  follow-up: patch the last Judger prompt token instead of inserting a newline, so
  zero ≡ Frozen exactly.
- Confirm zero vs locked Frozen pairing on AIME item-level (19 vs 17). Pull
  `eval_aime24/latency_rows.jsonl` and the old gate jsonl if it still exists on the volume.

### P4 — If you insist on AIME accuracy next

The interventions that are *not* “another SEAL vector” and are still untried:

- **Scale + layer** (P0). First.
- **Do not inject during all of Judger decode** — we already inject once; good.
- **Prefill the problem into the frozen cache before the bridge** is already what we do.
- Instance-specific **small-K from scratch** (not on the frozen prefix) already failed
  (`k2` 53%, `k5` 50%).
- Subject-conditional medoid already 56.7%.
- Remaining long-shot: let Real run **only Planner** (11 forwards) then Judger; or
  adaptive K with an OOD detector. That is not DeltaBridge; it is a compute scheduler.
  Scientifically weaker than a mechanism, but it is how you actually save AIME
  forwards if P0 dies.

---

## 4. Paper sentence (current evidence)

**Keep:** Mean-Replay is a reusable math-type prior. Frozen = Real on GSM8K/MATH;
Frozen misses Real on AIME; residual TTC on the prefix does not close AIME.

**Add:** On MATH, the instance residual of K=10 is low-rank and, as an oracle patch,
*improves* on both Frozen and Real. On AIME, the same one-token L32 patch is
destructive even with the uncompressed residual. So K=10’s AIME gain is not a
compressible last-token handoff at this site.

**Do not write:** DeltaBridge recovers AIME. It doesn’t.

---

## 5. Files the next agent should read first

| Path | Why |
|---|---|
| `artifacts/delta_bridge/CHECKIN.md` | gate |
| `artifacts/delta_bridge/eval_math/report.json` | MATH go + per-item flags |
| `artifacts/delta_bridge/eval_gsm8k/report.json` | dummy-bridge tax |
| `artifacts/delta_bridge/eval_aime24/report.json` + `latency_rows.jsonl` | AIME collapse, pairing |
| `artifacts/delta_bridge/fit_report.json` | layer energies |
| `artifacts/delta_bridge/eval_aime24/eval_states.pt` | reuse for P0 |
| `artifacts/math_ladder/gate_{math,gsm8k,aime24}/report.json` | locked Real/Frozen |
| `docs/MASTER_BRIEF.md` | prior negatives (CES, native, KV steer) |

Local checkout may lack the `.pt` / jsonl; they are on `/workspace/latentmas-baseline/`.
Copy them down before another pod if you want offline analysis.

---

## 6. One-line ask for the lead

Is the remaining bet **(i)** dose/layer of the same AIME states, **(ii)** MATH-only
predicted-residual method, or **(iii)** stop DeltaBridge and return to scaffold
hardening / Judger SEAL? Recommend **(i) then (ii)**; skip 4-token until (i) says the
patch is non-destructive.
