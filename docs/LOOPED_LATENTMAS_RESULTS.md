# Looped-LatentMAS: results of the validation gates (2026-10-31, updated 2026-11-02)

**Verdict: stop on GSM8K, for both Ouro checkpoints.** The planner → critic → final-solver
team, run on one frozen Ouro backbone, is *less* accurate than the same model answering
alone, and it uses more compute.

| | Thinking model (`Ouro-1.4B-Thinking`) | Plain model (`Ouro-1.4B`) |
|---|---|---|
| Single model alone | 89.5% | 84.5% |
| Agent team | 83.5% | 72.0% |
| **Team score minus single score** | **−6.0 points** (team worse) | **−12.5 points** (team worse) |

The plain-model rerun tested Jiayi's diagnosis that the Thinking backbone "already reasons
internally" and leaves the team nothing to add. Her test was that the team should beat the
single model on the plain model. It did not, and the gap doubled, so the diagnosis is not
supported. Adaptive depth and latent cache reuse were not built (stopping rule of the
spec). The loop-cache mechanism itself is sound and is documented below for future use.

Agents hand off **text** in every experiment here. The latent version of the handoff was
deliberately not built, because the text version of the team already loses.

The proposal document was not available to this run; the specification used is the
task description that requested it.

## How to read the numbers

- **Accuracy** is the share of the 200 GSM8K questions answered correctly. Every setup
  gets the same questions, prompts, answer extraction, and grading.
- **"A − B" (e.g. "team − single")** is A's accuracy minus B's accuracy, in percentage
  points. *Negative means A scored lower than B; positive means A scored higher.* Example:
  team 72.0% − single 84.5% = −12.5 points, so the team got 25 fewer of the 200 questions
  right.
- **"A-only / B-only"** compares the two setups question by question. It counts the
  questions A got right and B got wrong, then the reverse. Questions both got right or
  both got wrong are left out. "9 / 34" means the team won 9 head-to-head questions and the
  single model won 34. The difference, 9 − 34 = −25 out of 200, is the −12.5 points.
- **CI** is the 95% confidence interval for the gap, from a paired bootstrap. If the whole
  range is below zero (e.g. [−19, −6.5]), A is reliably worse than B, not just unlucky on
  this sample. If the range crosses zero, the gap could be noise.
- **p** is an exact McNemar test on the A-only / B-only counts. It is the chance of
  seeing a split this lopsided if the two setups were actually equally good. Below 0.05
  is the usual bar.
- **Loop-tokens** measure compute. Ouro reuses the same 24 layers for each loop, so a
  token processed at depth 4 counts as 4 loop-tokens. We count every prompt, handoff, and
  generated token for every agent. The fair rival for the team is **self-consistency
  (SC@m)**: the single model answers m times and takes a majority vote, with m picked so
  the cost roughly matches the team's.
- **Self-revision** means the single model answers, then is shown its own answer and
  asked to check and revise it once.

## Backbone

| | |
|---|---|
| Model | `ByteDance/Ouro-1.4B-Thinking`, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, Apache-2.0, not gated |
| Code | remote `modeling_ouro.py` (trust_remote_code); transformers 4.57.1 on the pod, 4.57.6 locally (card asks <4.56; the merged cache fix makes 4.57 work, verified by the tests) |
| Hardware | 1x H200, bf16, sdpa, frozen weights, greedy unless stated |
| Depths | 1–4 (`total_ut_steps=4`); 24 shared layers per loop |

**How depth is controlled.** `OuroModel.forward` runs `for current_ut in range(self.total_ut_steps)`.
`OuroRunner.depth(r)` sets that attribute, so exactly `r` loops execute. Upstream
`exit_at_step` / `exit_threshold` only choose which loop's hidden state is read out *after
all 4 loops ran* — a test asserts this trap. The adapter reads logits from
`lm_head(last_hidden_state)` of loop `r`, bypassing the early-exit gate. Executed loops are
counted by a hook on layer 0, not inferred.

**Cache structure.** `UniversalTransformerCache` is a flat list of 96 K/V tensors indexed
`loop * 24 + layer`, each `[batch, 16 heads, seq, 128]`. One RoPE table is shared by all
loops. Loop `i` of token `t` reads only loop-`i` keys of tokens `<= t`, so "the first `r`
loop-cache sets" is entries `[0, 24r)`. KV costs ~0.79 MB/token at depth 4 (4x a
non-looped model of this size) — this caused the first run's OOM.

**Mapping LatentMAS onto Ouro.** Not a drop-in swap. Between loops the hidden state passes
through the final norm, so the latent vector is the normed loop-`r` output; it is fed back
as `inputs_embeds` after the repo's default realignment (identity, rescaled to the mean
input-embedding norm). Ouro's embeddings are untied. Qwen prompt builders hard-assert a
Qwen model, so roles live in `looped/prompts.py` (question first, role text after).

## Stage 2 — depth sweep (n=200, same questions/prompt/scorer)

| r | acc (95% CI) | mean output tok | capped at 3072 | mean loop-tokens | batch-1 ms/token |
|---|---|---|---|---|---|
| 1 | 0.375 (0.31–0.44) | 1591 | 80 | 1,694 | 13.2 |
| 2 | 0.840 (0.78–0.88) | 886 | 19 | 1,978 | 25.9 |
| 3 | 0.880 (0.83–0.92) | 779 | 8 | 2,648 | 38.7 |
| 4 | 0.895 (0.84–0.93) | 745 | 6 | 3,394 | 51.0 |

Per-token latency is linear in depth (r=1 is 3.9x faster per token), so lower depth really
cuts executed compute. But shallower runs write longer answers, so per-question work falls
less: r=3 saves 22% of loop-tokens for −1.5 pp, r=2 saves 42% for −5.5 pp, r=1 collapses.
Batched wall times use depth-scaled batch sizes (32/42/64/100) and are throughput, not
latency. Peak memory 52–78 GB. The r=4 arm reproduced the gate's `single` arm exactly
(same executed loop-tokens, 683,892).

## Stage 3 — role conditioning (n=50, full depth)

- **Planner** gave a numbered plan without the answer in 41/50 (0 boxed answers); 8/50 hit
  the 3,072 cap still thinking. The solver on the same questions boxed an answer 50/50 and
  scored 0.90. Plans often pad to the 6-step limit with "(Optional: verify…)" lines.
- **Critic** on 50 deliberately corrupted solutions (one altered step, known first error):
  flagged INCORRECT 98%, named the exact first wrong step 82%, corrected answer 90%,
  copied the wrong answer 6%. On 50 correct solutions it wrongly rejected 12%.
- **Generic solver control** (same candidate, no critic role): implied-verdict accuracy
  100% on corrupted, final answer 94% on corrupted (critic − solver −4 pp, CI [−12, +4],
  McNemar p=0.63), 96% on correct.

The roles do their jobs (format, withholding, error localisation), but the critic role adds
no answer accuracy over simply re-solving.

## Stage 4 — main gate: MAS vs single model (n=200, r=4, text handoffs)

| arm | acc (95% CI) | mean loop-tokens | batched wall (s) |
|---|---|---|---|
| single (greedy) | 0.895 (0.84–0.93) | 3,394 | 602 |
| self-revision (2 passes) | 0.890 (0.84–0.93) | 7,224 | 1,237 |
| **planner → critic → solver** | **0.835 (0.78–0.88)** | **14,902** | **2,180** |
| self-consistency@2 (T=1, p=0.7) | 0.895 | 7,203 | 1,815 |
| self-consistency@3 | 0.910 | 10,808 | 2,683 |
| self-consistency@4 (compute-matched) | 0.910 (0.86–0.94) | 14,465 | 3,566 |

Head-to-head on the same 200 questions. Negative means the first setup scored lower; see
"How to read the numbers".

| comparison | gap (first − second) | 95% CI | first-only / second-only | p | plain reading |
|---|---|---|---|---|---|
| team − single | −6.0 pts | [−10.5, −2.0] | 4 / 16 | 0.012 | team reliably worse |
| team − self-revision | −5.5 pts | [−10.0, −1.0] | 5 / 16 | 0.027 | team reliably worse |
| team − SC@4 (same compute) | −7.5 pts | [−11.0, −4.0] | 0 / 15 | 0.0001 | team reliably worse |
| self-revision − single | −0.5 pts | [−2.5, +1.5] | 2 / 3 | 1.0 | no real difference |

Attention-weighted work (Σ kv-length × depth) makes MAS look worse still: 12.2M vs 9.4M
for SC@4, because handoffs lengthen every downstream prompt.

**Why MAS loses (16 questions single solved and MAS did not):** 9 are the final solver
hitting the 3,072 cap; 5 had a planner that never closed its thinking, so the raw reasoning
tail was handed off; critics also "found" errors in correct plans and the final solver
deliberated over them (e.g. idx 1161, 1070). Across all 200, 45 planners and 12 critics
never closed thinking.

**Could a revised protocol rescue it?** Not on this benchmark. Even if every one of the 16
losses were recovered, MAS would win at most 4 extra items over single (≈0.915) — no better
than SC@4 at the same compute, which MAS never beat on a single item.

## Rerun on the plain model `ByteDance/Ouro-1.4B` (2026-11-02)

**Why.** Jiayi asked for both pilots (role check and team vs single) to be rerun on the
non-thinking checkpoint. LatentMAS never enables thinking, and its single-model GSM8K
baseline is ~80%, which leaves room for a team to help. Her test: if the team beats the
single model here, the Thinking backbone explains the first failure.

**Setup.** Identical to the stages above except for the model: `ByteDance/Ouro-1.4B`,
revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, Apache-2.0. Same 200 questions
(fingerprint `b8488b7fbc045f03`) and 50 role-check questions (`09f20eefc2b22c80`), depth
4, sdpa, batch 24, SC with T=1.0 and top_p=0.7. This checkpoint's chat template has no
thinking mode, so no `<think>` block is ever opened.

**Bug fixed before the valid run.** The plain checkpoint's `modeling_ouro.py` lacks
`UniversalTransformerCache.get_mask_sizes` (the Thinking checkpoint has it). Without it,
left-padded batches give the longest row a correct answer and garble every shorter row.
`looped/ouro_adapter.py` patches the method onto whichever cache class is loaded. After the
patch, batched output matches one-question-at-a-time output token for token. An earlier
unpatched run (`artifacts/looped_base/`) is **invalid; do not cite it**. The valid results
are in `artifacts/looped_base2/`.

### Role check (n=50, depth 4)

| measure | Thinking | Plain |
|---|---|---|
| Planner gives a numbered plan without the answer | 82% | 100% |
| Plain solver accuracy (reference) | 90% | 92% |
| Critic calls a planted wrong solution wrong | 98% | **30%** |
| Critic ends with the right answer on a planted wrong solution | 90% | **34%** |
| Critic copies the planted wrong answer | 6% | **64%** |
| Plain solver, shown the same wrong solution, ends with the right answer | 94% | 80% |
| Critic wrongly rejects a correct solution | 12% | 0% |

On the plain model the critic is much worse than simply re-solving. On planted wrong
solutions, critic answer accuracy minus solver answer accuracy is −46 points, CI
[−64, −28]: the critic was right where the solver was wrong 4 times, and the reverse
27 times (p < 0.0001). The plain-model critic almost never says "incorrect". It also
accepts every correct solution, which is why it shows 0% wrongful rejections. That is not
good judgment.

### Team vs single (n=200, depth 4, text handoffs)

| setup | accuracy (95% CI) | mean loop-tokens | batched wall (s) |
|---|---|---|---|
| single (greedy) | 84.5% (78.8–88.9) | 2,435 | 446 |
| self-revision (2 passes) | 88.0% (82.8–91.8) | 7,504 | 1,783 |
| **planner → critic → solver** | **72.0% (65.4–77.8)** | **8,756** | **1,672** |
| SC@1 | 86.0% | 2,474 | 592 |
| SC@2 | 86.0% | 4,857 | 1,228 |
| SC@3 | 88.0% | 7,374 | 1,901 |
| SC@4 (closest compute to team) | 89.0% (83.9–92.6) | 9,716 | 2,435 |

Head-to-head on the same 200 questions. Negative means the first setup scored lower.

| comparison | gap (first − second) | 95% CI | first-only / second-only | p | plain reading |
|---|---|---|---|---|---|
| team − single | −12.5 pts | [−19.0, −6.5] | 9 / 34 | 0.0002 | team reliably worse |
| team − self-revision | −16.0 pts | [−21.5, −10.5] | 3 / 35 | < 0.0001 | team reliably worse |
| team − SC@4 (similar compute) | −17.0 pts | [−22.5, −12.0] | 1 / 35 | < 0.0001 | team reliably worse |
| self-revision − single | +3.5 pts | [+0.5, +7.0] | 9 / 2 | 0.065 | small gain, not significant |

**Why the team loses on the plain model.** Of the 34 questions the single model solved and
the team missed:

- The critic approved the plan in 32 of them ("The plan is sound. The steps are correct…").
  Across all 200 questions it called the plan correct 197 times. Planner mistakes pass
  straight through. Example: idx 192, gold 91, the plan reaches 56, the critic approves,
  and the final answer is 56.
- The final solver hit the 3,072-token cap in 20 of the 34, often while debating the
  plan's interpretation against the question (e.g. idx 1161).

The single model at 84.5% is close to Jiayi's ~80% figure, so headroom existed. The team
lost ground instead of using it. Her ~80% comes from the LatentMAS paper's models, not
Ouro, so the two numbers are not directly comparable.

### What changed between the two models

The team loses on both, for different reasons. On the Thinking model the critic catches
errors well (98%). The losses there come from planners that never finish thinking and hand
off half-written reasoning, final solvers running out of length, and critics flagging
correct plans. On the plain model the critic rubber-stamps, so wrong plans go unchallenged.

## Stage 5a — cache correctness on the real checkpoint (done; 5b–6 not built)

History = planner prompt + 96 planner-generated tokens (full depth), continuation = the
next agent's turn; 3 questions, bf16 on H200 (1 more on MPS):

- **Truncate vs reuse at depth r** (first r sets of the full-depth cache vs a stored depth-r
  cache): K/V max |diff| **0.0** at every loop, logits max |diff| **0.0**, greedy
  continuations identical in 9/9 cases.
- **Single-pass recompute vs split (stored cache + continuation)**: logits differ by
  0.20–0.38 in bf16 and greedy diverges within 32 tokens in 4/9 cases. This is bf16
  chunking, not truncation: in fp32 the unit tests agree to 1e-5.
- **Latent replay** (saved full-depth latent vectors replayed at depth r) equals truncation
  exactly (0.0). **Latent regeneration** at depth r is a different input: cosine to the
  full-depth vectors is −0.29 to −0.24 at r=1, 0.18–0.30 at r=2, 0.47–0.74 at r=3, and
  continuation logits move by 0.6–4.3. So "keep the first r sets" is only exact when the
  inputs were fixed at full depth; it does not equal running the latent agent at depth r.

## Recommendations by gate

| gate | call |
|---|---|
| Backbone / depth control | **Continue** — mechanism verified; r=3 is a mild accuracy/compute trade |
| Role conditioning | **Revise** — Thinking: roles comply but add no accuracy. Plain: the critic fails (catches 30% of planted errors) |
| MAS vs single (main gate) | **Stop** on GSM8K — Thinking −6.0 pts at 4.4x compute; plain −12.5 pts at 3.6x compute |
| Jiayi's "Thinking backbone" diagnosis | **Not supported** — the team loses by more on the plain model |
| Cache reuse / adaptive depth | **Not started**, per the stopping rule |

If the idea is revisited, test the main gate first on a task where the single model is far
from ceiling (MATH-500 levels 4–5 or AIME), with planner/critic thinking disabled or capped
so handoffs are always closed text, and keep SC@m as the compute-matched baseline.

## Reproduce

```bash
pip install torch==2.8.0 transformers==4.57.1 datasets matplotlib pytest
python -m pytest tests/test_looped_cache.py -q                      # 16 CPU tests, ~3 s
python scripts/looped_cache_check.py --device cuda --dtype bfloat16 --n_questions 3
python scripts/looped_role_pilot.py --n 50 --bs 32 --out artifacts/looped/role_pilot
python scripts/looped_mas_gate.py --n 200 --depth 4 --bs 32 --sc_k 4 --out artifacts/looped/mas_gate_r4
python scripts/looped_depth_sweep.py --n 200 --bs 32 --latency_n 10 --latency_max_new 256 --out artifacts/looped/depth_sweep
python scripts/looped_report.py --root artifacts/looped

# plain-model rerun
python scripts/looped_role_pilot.py --model ByteDance/Ouro-1.4B --attn sdpa --n 50 --bs 24 --out artifacts/looped_base2/role_pilot
python scripts/looped_mas_gate.py --model ByteDance/Ouro-1.4B --attn sdpa --n 200 --depth 4 --bs 24 --sc_k 4 --out artifacts/looped_base2/mas_gate
```

GSM8K subset: test split, seeded permutation (seed 0), first n; indices and a fingerprint
are in each `env.json`. Raw outputs: `rows*.jsonl` / `per_question.jsonl` /
`*_rows.jsonl` under `artifacts/looped/`; environment and model revision in `env.json`.
Code: `looped/` (adapter, prompts, tasks, stats, harness), `scripts/looped_*.py`.
