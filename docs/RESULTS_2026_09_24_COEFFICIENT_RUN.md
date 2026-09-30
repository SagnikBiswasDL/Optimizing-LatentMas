# AIME coefficient decision run + throughput probes — 2026-09-24

Transcribed from the run log by hand because the raw `rows.jsonl` did not survive
the pod. `ROOT_DIR` was on `/workspace`, so `PERSIST_DIR` was correctly `none`
(mirroring to the same volume is pointless), but nothing was ever copied back to
the repo, and the container is now gone. **Harvest small artifacts to git at the
end of a pod session**; `scripts/rp.sh -c 'cat <path>' > local` is enough, since
the RunPod proxy has no SCP.

Hardware: 1x NVIDIA H200 (143 GB), pod `t79vcyoczutmkp`. Model Qwen3-14B, bf16,
`sdpa`, dynamic cache, `--decode_bs 1`, greedy. Every row below is
`decode_path=sdpa+dynamic`. Stage elapsed 3277s (~55 GPU-min).

## 1. The de-censored baseline (`real__b16k`, cap 16384)

The previous cohort ran at cap 8192 and 2 of 6 items were censored, so the
baseline's own token total was a lower bound. At 16384:

| idx | tokens | judger_s | tok/s | eos | correct | pred |
|---:|---:|---:|---:|:--|:--|:--|
| 0 | 2,600 | 62.1 | 41.9 | yes | **yes** | 204 |
| 1 | 11,036 | 263.2 | 41.9 | yes | **yes** | 113 |
| 2 | 11,092 | 265.7 | 41.7 | yes | no | 365 |
| 4 | 7,653 | 182.6 | 41.9 | yes | **yes** | 110 |
| 10 | 5,848 | 139.9 | 41.8 | yes | **yes** | 104 |
| 18 | 7,062 | 169.0 | 41.8 | yes | **yes** | 23 |

**Solved 5/6; 34,199 tokens across solved items.**

Two things this settled:

* **Item 1 was never wrong, only unfinished.** At 8192 it was censored and
  scored incorrect; given room it answers correctly at 11,036 tokens. Baseline
  accuracy moves 4/6 -> 5/6. Item 2 finishes and is genuinely wrong (11,092).
* **The extension is faithful.** `--mode prefix` reported **6/6 prefix
  preserved, verdict "reproduced"**: each longer run retraces its shorter one, so
  extended and original token counts are comparable. Items 0/4/10/18 reproduced
  their cap-8192 token counts *exactly* (2600, 7653, 5848, 7062), which also
  makes the decode deterministic on a fixed path.

Throughput held at 41.7-41.9 tok/s across 2.6k-11.1k token runs, consistent with
the documented 42.51 +/- 0.31 and with `latency = tokens / 42.5`.

## 2. The coefficient screen: all candidates eliminated

Vector `artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt` at layer 28, Judger
only, `apply_to=last`. Screened on the two items coef 40 had broken.

| arm | idx 10 | idx 18 |
|---|---|---|
| baseline | 5,848 ✓ | 7,062 ✓ |
| `real_seal20__b16k` | 10,604 ✓ (+81%) | 16,384 ✗ censored, pred 3 |
| `real_seal60__b16k` | 14,704 ✗ pred `\frac{21` | 16,384 ✗ censored, pred 2 |
| `real_seal80__b16k` | 16,384 ✗ censored, pred 107 | 6,050 ✓ (-14%) |

**Survivors: none.** No promotion, so `confirm24` correctly never ran.

Reading:

* **Steering makes the Judger more verbose, not less.** On item 10 the effect is
  cleanly monotonic in coefficient: 5,848 -> 10,604 -> 14,704 -> cap.
* Exactly 1 of 6 cells improved (coef 80 on item 18, -14%), and that same
  coefficient ran item 10 into the cap. Item 18 is *not* monotonic, so the
  direction is not uniformly signed across problems.
* At coef 60 the output degrades structurally, not just in length — it emits
  `\frac{21...` instead of an integer answer.
* With coef 40 already measured at +4% tokens paired, coefficients
  **20/40/60/80 all fail**. There is no working operating point on this axis.

**This is SEAL's own token-efficiency instrument, not a correctness vector.**
`gsm8k_layer28.pt` is `mean(execution) - mean(reflection + transition)` over 50
GSM8K traces (`extract_seal_vector.py`), so a positive coefficient steers toward
execution and away from second-guessing, which should *shorten* output. It
lengthens it. The correctness vector is a different artifact
(`native_gsm8k/judger_native_layer28.pt`) and is nearly orthogonal to this one,
cosine **+0.22**. `gsm8k_layer28_n200` is cosine **+0.946** with the vector used
here, so re-running with it is not a new hypothesis.

Untested and now cheap: **the opposite sign.** `_SEAL_ARM` accepted `\d+` only,
so a negative coefficient was unspellable until this was fixed.

## 3. Serving throughput: a retracted 1.70x

Roofline for batch-1 bf16 decode is weights over bandwidth,
29.6 GB / 4.8 TB/s = 6.17 ms/token = **162 tok/s**. Measured **43.8 tok/s, 27%
of roofline** — a 3.7x headroom.

`StaticCache` appeared to close most of it and does not. Reproduced twice
(44.5/74.0 then 43.8/74.3), then falsified:

| configuration | tok/s | vs dynamic |
|---|---:|---:|
| `generate` + dynamic | 43.7 | 1.00x |
| `generate` + static, auto len 956 | 75.4 | **1.72x** |
| `generate` + static, `max_cache_len` 9216 | 14.9 | 0.34x |
| `generate` + static, `max_cache_len` 16384 | 12.3 | 0.28x |

StaticCache attends over every *allocated* slot each step, masked, so its cost
tracks the allocation rather than the true length. A probe generating 256 tokens
from a 700-token prefix allocates ~956 slots and sees 1.72x; the Judger allocates
`prefix + 8192` ~= 9000 and loses 3x. Confirmed in-pipeline before it was caught:
item 0 ran at **23.7 tok/s** and item 1 at **28.1** on `sdpa+static`, against
42.6 and 42.5 on dynamic.

`diag_static_cache_len.py` isolates it (prefix 770, 64 new tokens, hand-rolled
eager loop):

| cache | tok/s | vs dynamic |
|---|---:|---:|
| dynamic | 45.2 | 1.00x |
| static 1024 | 40.1 | 0.89x |
| static 2048 | 40.2 | 0.89x |
| static 4096 | 38.3 | 0.85x |
| static 9216 | 25.8 | 0.57x |
| static 16384 | 17.7 | 0.39x |

Note static is *slower than dynamic at every length* in the eager loop. So the
1.72x under `generate` is not an attention or Python-overhead effect: it is
**CUDA graphs**. `generate` auto-compiles when handed a compileable cache
(`generation/utils.py:2759` -> `_valid_auto_compile_criteria`, which requires
`past_key_values.is_compileable`; `StaticCache.is_compileable` is `True`). That
makes tight-but-growing allocation a real lever — see briefing §6.

Also: do not call `torch.compile(model.forward)` yourself on top of this. It
collides with generate's compiled call and took the interpreter down with no
Python traceback, which is why `diag_throughput.py` now flushes results per
config instead of only at the end.

## 3.5 The sign flip (2026-09-29): steering costs tokens in *both* directions

Coefficients **-20/-40/-60**, same cap, same screen items, same rule. The baseline
rows were recovered off the pod volume, so no baseline re-decode was needed.

| coef | idx 0 | idx 1 | idx 2 | idx 4 | idx 10 | idx 18 |
|---:|---|---|---|---|---|---|
| **0** | 2,600 ✓ | 11,036 ✓ | 11,092 ✗ | 7,653 ✓ | **5,848 ✓** | **7,062 ✓** |
| +20 | | | | | 10,604 ✓ | 16,384! ✗ |
| +60 | | | | | 14,704 ✗ | 16,384! ✗ |
| +80 | | | | | 16,384! ✗ | 6,050 ✓ |
| -20 | | | | | 16,384! ✗ | 16,384! ✗ |
| -40 | 2,703 ✓ | 16,384! ✗ | 16,384! ✗ | 7,444 ✗ | 9,850 ✓ | 7,214 ✓ |
| -60 | 3,310 ✓ | 11,017 ✓ | 16,384! ✗ | 16,384! ✓ | 15,021 ✓ | 7,996 ✓ |

`!` = hit the cap, so the token count is a lower bound and the run never
terminated.

Both survived the *screen* — the first candidates ever to do so, since the screen
only asks for correctness on items 10 and 18 — and both were rejected on the
cohort:

| candidate | solves kept | solves lost | tokens on baseline's solves | vs baseline |
|---|---|---|---:|---:|
| `real_seal-40` | 0, 10, 18 | 1, 4 | 43,595 | **+27.5%** |
| `real_seal-60` | 0, 1, 10, 18 | 4 (never terminated) | 53,728 | **+57.1%** |

`-60` is the best-behaved arm yet on correctness: it keeps four of five baseline
solves outright and *does* produce the right answer on item 4, but only by running
into the cap, so under a stopping policy it has not delivered an answer. It pays
+57% tokens for that.

**The finding is that the sign does not matter.** Counting every steered setting
against its own baseline:

* item 10: **6 of 6** emit more tokens than the unsteered 5,848
* item 18: **5 of 6** emit more than the unsteered 7,062

The single exception is +80 on item 18 (6,050, -14%), and that same coefficient
runs item 10 into the cap. The curve through zero is not monotonic — it is a
valley whose minimum is at **zero** (see `coef_sweep.png`, from
`scripts/plot_coef_sweep.py`).

That shape says the token increase is a **magnitude effect, not a direction
effect**: displacing the residual stream at layer 28 by a fixed-norm vector makes
the model ramble regardless of which way the vector points. It is not the
execution-vs-reflection semantics the vector is supposed to carry. Non-monotonicity
supports this too — -20 runs both screen items into the cap while the *larger*
-40 and -60 do not, which is not how a semantic axis with a dose-response should
behave.

**Consequence for the brevity-vector plan.** The prediction is that *any*
diff-of-means direction injected at this site with comparable norm will also
lengthen output, so a length-supervised vector is not obviously exempt. Before
spending GPU on extraction, it is worth running `--mode samples` to check whether
correct solutions to the same problem even differ in length, and worth treating
"does this vector reduce tokens at small coefficient" as a gate rather than an
expectation. A cheaper variant also becomes interesting: sweep *small* magnitudes
(coef 2, 5, 10) to find where the perturbation cost begins, since everything
tested so far is >= 20 and may simply be far off-distribution.

> **Superseded in part by §3.6.** That sweep ran, and the claim above that the
> minimum is "at zero" is wrong for item 18: coefficients 2, 5 and 10 all come in
> below its baseline, with a minimum of -34% at coefficient 5. The "valley with
> its minimum at zero" reading was an artifact of only ever sampling |coef| >= 20.
> The magnitude-effect conclusion survives for item 10, which never improves at
> any coefficient, but §3.6 shows the sweep resolves neither claim: adjacent
> coefficients differ by up to 168% of baseline.

## 3.6 The small-magnitude sweep (2026-09-29): the sweep cannot resolve the effect

§3.5 proposed sweeping small coefficients on the theory that everything tested so
far (|coef| >= 20 against a vector of raw norm 52.7) was simply far
off-distribution. That ran: coefficients 2, 5, 10, -2, -5, -10 on the two screen
items, 54 minutes, `DECODE_FLAGS="--decode_path sdpa+dynamic"`. **Zero survivors.**

The headline finding is not about any coefficient. It is that **token count is not
a smooth function of the coefficient**, so this experimental design cannot measure
what it was built to measure.

Item 18, every setting ever run (baseline 7,062):

| coef | -60 | -40 | -20 | -10 | -5 | -2 | 0 | +2 | +5 | +10 | +20 | +60 | +80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens | 7,996 | 7,214 | cap | 7,298 | 7,017 | 13,451 | 7,062 | 5,392 | 4,684 | 6,194 | cap | cap | 6,050 |
| vs base | +13% | +2% | — | +3% | -1% | **+90%** | — | -24% | **-34%** | -12% | — | — | -14% |

Item 10, same settings (baseline 5,848): **0 of 8** terminating settings came in
shorter, across the whole range -60 to +80. Four of twelve hit the cap.

Read in isolation, item 18's positive side looks like exactly the hoped-for
result: a monotone dip to -34% at coefficient 5, then a turn back up as
perturbation cost takes over, with correctness preserved throughout. That reading
does not survive the negative side. Coefficient -5 sits at baseline (-0.6%) while
its neighbour -2 is +90%. On item 10, -2 and -10 both terminate while -5 between
them runs into the cap.

**This is not measurement noise.** Decoding is greedy on a fixed decode path, and
§1 established that token counts reproduce exactly. Every number above is exact.
What varies is the trajectory: a small change in the steering coefficient tips the
Judger onto a different reasoning path, and the length of that path is close to
arbitrary.

The quantity that settles it, printed by `scripts/plot_coef_sweep.py`:

* largest jump between **adjacent** coefficients, item 10: **168% of baseline** (-5 -> -2)
* largest jump between adjacent coefficients, item 18: **146% of baseline** (+60 -> +80)

The effect under investigation is on the order of 30%. When neighbouring settings
of the independent variable differ by 150%, a single run per (item, coefficient)
resolves nothing, and the apparent dip at +2/+5/+10 on item 18 is as easily three
draws from a wide distribution as it is a dose-response.

**What this costs the plan.** The small-magnitude hypothesis is not confirmed and
not refuted; it is unmeasurable at n=2 items with one run per cell. Two honest
routes remain, and they are not cheap:

1. **Pay for the statistics.** Many more items per coefficient — 20-30, so
   per-item trajectory chaos averages out and coefficients can be compared as
   distributions. At ~385 s/item worst case that is ~3 GPU-hours per coefficient,
   so ~10 GPU-hours for a three-point curve. This is the only route that can
   produce a defensible token-efficiency claim about this vector.
2. **Stop measuring this vector.** The one durable result across 24 settings is
   that 0/8 terminating settings helped item 10 at any coefficient of either
   sign. If a second or third item behaves like item 10, the direction is not a
   brevity control at this injection site and the remaining GPU is better spent
   elsewhere.

Route 1 is also the correct precondition for the brevity vector: without a
per-item noise band there is no way to tell a real 30% saving from trajectory
luck. The `--mode samples` run (k=4 at temperature) measures a *different* band
(within-item, under sampling) and is still worth having, but it does not
substitute for item count, because the variation documented here is across
coefficients at fixed item, not across runs at fixed coefficient.

## 3.7 The 30-item paired cohort (2026-09-29): a powered null

§3.6 said the only route to a defensible claim was to pay for statistics. That
ran: the full AIME-2024 pool, all 30 items, each contributing an unsteered and a
coefficient-5 run over the same upstream tape, greedy on `sdpa+dynamic`,
pair-major so a deadline could only cost whole pairs. 2.7 hours, 30/30 pairs
complete, 0 unpaired. Analysis and decision rule were committed in
`scripts/paired_coef_test.py` **before** the data existed.

**Result: REJECT.** Coefficient 5 does not reduce Judger tokens.

| quantity | value |
|---|---|
| usable pairs (both arms finished and correct) | 21 of 30 |
| median token change | **-6.0%** |
| mean token change, 95% CI | **-1.0%  [-12.4%, +11.8%]** |
| shorter on | 11 of 21 items (sign test p = 1.00) |
| paired permutation p | 0.87 |
| accuracy | 22/30 -> 22/30 (fixed 1, broke 1, McNemar p = 1.00) |
| censored (hit cap) | baseline 3, steered 4 |

Three of the four pre-registered clauses fail: the saving is under 10%, it is not
significant, and censoring is slightly worse. Only "accuracy not worse" passes.

**This is a real null, not an underpowered one.** Per-item spread is large (SD of
the log ratio 0.286, a 1.33x factor; range -41.7% to +109.2%), but with n=21 the
cohort still had 80% power to detect a 16% saving, and the confidence interval
excludes everything at or beyond -13%. A saving big enough to matter would have
shown up. What it cannot rule out is something small: detecting a 10% saving would
need 58 items and a 5% saving 244, against a benchmark that contains 30. **For
this vector at this site, AIME-2024 is at its resolution limit, and the answer
within that limit is no.**

### The methodological error this exposes

Item 10 ranks **21st of 21** in the cohort (+109.2%, the largest increase) and
item 18 ranks **2nd of 21** (-33.7%, the second largest decrease). The two items
every screen since 2026-09-22 was built on are the two extremes of the
distribution.

That is not bad luck. Those two items were chosen, back in §2, as *"the items coef
40 broke"* — selected on the outcome being measured. Every subsequent
two-item screen inherited that selection, which is why small samples kept
producing dramatic, contradictory, non-reproducing results: item 18's -34% at
coefficient 5 and item 10's +109% at the same coefficient are both real, exact,
deterministic numbers, and they are both unrepresentative. Averaged over the pool
they cancel to -1%.

Concretely, for future screens: **do not select screen items on the dependent
variable.** If a cheap screen is wanted, draw items at random or stratify by
baseline token count, and treat any two-item result as a smoke test that a run
completed, never as evidence about an effect.

### Status of the brevity-vector plan

The premise was that a *better-targeted* direction (length-supervised rather than
SEAL's execution-vs-reflection heuristic) would cut tokens where this one does
not. Nothing here refutes that, but the cost of testing it is now known and it is
not small: any candidate vector needs a 30-item paired cohort (~2.7 GPU-hours) to
be evaluated at all, and a positive result under 15% cannot be established on this
benchmark at any cohort size. A brevity vector is worth building only if the
expected effect is large, or if the evaluation moves to a benchmark with enough
items to resolve smaller ones.

## 4. What this run did not establish

* **Nothing about accuracy at n=6.** The noise floor is +/-2 items (a
  semantically neutral bf16 batching change moved accuracy 0.667 -> 0.333), so
  only the token effects above are large enough to read. +81% and cap hits are;
  a -14% on one item is not.
* The screen covers 2 items by design (a cheap kill criterion), not a cohort.
* `flash_attn` is not installed on the pod, so both flash configs failed to load
  and are unmeasured. At batch 1 this is unlikely to matter — decode is bound by
  weight bandwidth, not attention.

## 5. The latent channel ablation (2026-09-29), and what the paper actually claims

Arms run on all 30 AIME-2024 items, Qwen3-14B, cap 16384, `sdpa+dynamic`, paired
per item, analysis pre-registered in `scripts/channel_ablation.py`.

| arm | KV to Judger | correct | vs `real` | 95% CI | censored |
|---|---|---|---|---|---|
| `real` (planner+critic+refiner) | 108.3 MB | 22/30 | — | — | 3 |
| `none` (no cache) | 0 MB | 21/30 | -1 item | [-9.9, +3.2] pp | 8 |
| `shuf` (another problem's cache) | 108.3 MB | 19/30 | -3 items | [-24.2, +4.2] pp | 4 |

### This reproduces the paper, it does not contradict it

LatentMAS (arXiv 2511.20639), Table 2, AIME24, Qwen3-14B, **Sequential** setting:

| | Single | TextMAS | LatentMAS | reported |
|---|---|---|---|---|
| accuracy | 63.3% (19/30) | 63.3% (19/30) | 66.7% (20/30) | **+3.4** |

**The paper's own AIME24 gain is +3.4 pp, which on a 30-problem benchmark is one
problem.** Our `real` - `none` gap is +1 item (+3.3 pp). The effect size matches
almost exactly; the headline "up to 14.6% higher accuracy" comes from other tasks
and settings, not this one. In the Hierarchical setting the paper reports 63.3 ->
73.3 (19 -> 22 items) against Single, and our `real` is 22/30, so absolute levels
line up too, ours running ~2 items above theirs (plausibly the token budget: 20000
in the paper against 16384 here, plus protocol differences).

So the finding is not "LatentMAS does not replicate". It is that **on AIME24 the
published gain rests on a one-problem difference**, and our interval on that same
difference, [-9.9, +3.2] pp, contains both their +3.4 and zero.

### A confound in our own `none` arm

`none` reuses the Judger prompt, which reads *"You are provided with latent
information for reference"* while supplying none (`prompts.py:53`). The arm is
therefore handicapped by an incoherent prompt, which is the likely source of its
censoring jumping 3 -> 8: the Judger looks for context that does not exist and
rambles. Note the direction — this makes `none` a *pessimistic* baseline, so the
bound on the channel's contribution survives, and arguably tightens.

`shuf` being numerically worse than `none` (19 vs 21) says the Judger does read the
channel rather than ignoring it: a wrong problem's latents actively mislead. Both
differences are inside noise at n=30 and are quoted as directions only.

### The experiment this sets up

`prompts.py:694` already contains `build_agent_messages_single_agent`, the paper's
own Single baseline. Running it as a `single` arm on the same 30 items, same budget
and decode path, gives an internally consistent decomposition with nothing
confounded by prompt text:

```
single (plain CoT)  ->  none (judger prompt, no latents)  ->  real (full channel)
```

That separates two things the paper reports as one: the benefit of **multi-agent
prompt scaffolding** from the benefit of **latent KV transfer**, which is the
actual contribution. If `single` lands near 19/30 as the paper reports, then
scaffolding is worth ~2 items and the latent channel ~1, and the claim becomes that
most of the AIME24 gain is prompting rather than latent collaboration. Cost is one
arm, 30 items, ~1.7 GPU-hours.

### 5.1 Pre-registered reading of the `single` arm

*Written 2026-09-30 while the arm was decoding, before any of its rows existed.*

The decomposition is cleaner than assumed when it was proposed. Diffing the two
user prompts, they are identical except for the two sentences announcing latent
information (plus incidental `**bold**` markers around "provided Target
Question"); same system message, same `\boxed{}` instruction, same
step-by-step wording. So the two contrasts really do isolate one thing each:

| contrast | what varies | what it measures |
| --- | --- | --- |
| `single` vs `none` | prompt text only (neither has a cache) | the multi-agent framing |
| `none` vs `real` | cache only (prompt held fixed) | the latent KV transfer |
| `single` vs `real` | both | the paper's headline comparison |

`none` vs `real` is already the prompt-matched measurement of the contribution,
and it is +1 item, CI [-9.9, +3.2] pp. The `single` arm cannot change that. What it
decides is how the paper's +3.4 pp divides, and whether our harness reproduces
their Single baseline at all. Three outcomes, all of which are informative:

1. **`single` ≈ 19/30.** We reproduce the paper's baseline, and its +3 items
   decompose as ~2 from prompt framing and ~1 from latents. The contribution the
   paper names is the smaller half of its own effect.
2. **`single` ≈ `none` ≈ 21/30.** The prompt text is inert; the paper's gap is our
   `none`→`real` gap, one item, and §5's reading stands unchanged.
3. **`single` > `real`.** The Judger prompt *hurts* when latents are absent, our
   `none` floor was too pessimistic in a way that flatters LatentMAS, and in this
   harness the whole pipeline buys nothing over one agent with a plain prompt.

Outcome 3 is the one that would matter most and the one I am least able to
dismiss, so it is named before the data rather than after.

**A falsifiable prediction of my own confound explanation.** §5 above attributes
`none`'s censoring (8/30 against `real`'s 3/30) to a prompt that promises context
it does not supply. If that is right, `single` — which makes no such promise —
should censor closer to 3 than to 8. If `single` censors ~8 as well, the
explanation is wrong and the censoring is just what an unaided 14B model does on
AIME at a 16384-token cap.

**Power, stated in advance.** n=30 with ~22 correct: this run can detect a shift of
roughly 4-5 items, not 1. Every number below is reported with its interval, and no
one-item difference in either direction will be called a finding.

---

## 6. The channel does something, and it is not communication (2026-09-30)

*This section is the reason the project has a claim at all. §3 and §4 closed two
dead ends; §5 returned a null on the metric the paper reports. This is a positive,
well-powered result on a metric the paper does not report.*

`scripts/channel_token_cost.py`, run on the 178 rows already collected.

### 6.1 Why accuracy was never going to work

Accuracy on 30 AIME items is a 30-way binomial. §3.7 measured what that buys:
80% power for a 16% effect, and 58 items needed for 10%. The paper's own AIME24
gain is +3.4 pp, which is one problem. No amount of care makes 30 items resolve
one problem.

The Judger's **token count** is a different instrument. It is continuous, one
measurement per item rather than one bit, and under greedy decoding on a fixed
decode path it reproduces exactly (§3.6). The same 30 problems support a test with
real power.

### 6.2 The result

Paired on the same items, `real` against `none` with the Judger prompt held fixed:

| step | what changes | sign test (all 30 items) | median | 95% CI |
| --- | --- | --- | --- | --- |
| `none` → `shuf` | attach *any* cache, from a **different problem** | 23 shorter / 5 longer, **p=0.0009** | **-21%** | [-34.2, -15.3] |
| `shuf` → `real` | make that cache the **right** problem's | 19 shorter / 9 longer, p=0.087 | -6% | [-16.1, -3.2] |
| `none` → `real` | the published configuration | 23 shorter / 4 longer, **p=0.0003** | **-33%** | [-41.4, -22.9] |

The channel is worth a third of the answering agent's tokens. **Most of that is
reproduced by a cache belonging to a different problem.** Upgrading the placebo to
the genuine article buys a further 6%, which does not reach significance at n=30.

The saving survives full cost accounting. The upstream roles emit only `k=10`
latent steps each and no text, so they cost **0.9 s against a 124 s Judger**.
End-to-end, `none` → `real` is -37% wall clock, 20/22 faster, p=0.0001.

So the two headline metrics point opposite ways, and both are defensible:

- **Accuracy: null.** Deleting the channel costs one item of thirty, CI [-9.9, +3.2] pp.
- **Tokens: -33%, p=0.0003.** And largely not attributable to the channel's contents.

### 6.3 Two methodological commitments, encoded in the script rather than promised

**The primary test discards nothing.** Keeping only pairs where both arms
terminated conditions on the outcome — precisely the error §3.7 records us making
with items 10 and 18. But the *sign* of a paired difference is identified under
censoring: if one arm stops at 5000 tokens while the other is still running at the
16384 cap, the second is longer, and no assumption is needed. So the primary test
is an exact sign test over all 30 items, dropping only pairs where **both** hit the
cap. Magnitudes require completed runs and are labelled as conditional; the
correctness-filtered version is printed beside the unfiltered one to show the
filter moves nothing (-21.1% either way on `none` → `shuf`).

**`shuf`'s *time* is confounded, and the script prints the confound.** Item *i*
borrows item *i+1*'s cache, so prefix length is matched only in the mean: 98.8 MB
against 100.1 MB on average, but differing by 18 MB per item and up to 41. Token
counts do not depend on prefix length; prefill and per-step attention do. The
`shuf` time number is therefore not quoted as evidence, and `TIME_CONFOUNDED`
marks it in the output.

The placebo itself is verified rather than assumed: all 30 `shuf` rows record a
`donor_idx`, and every donor's cache size matches `real[donor_idx]` to the byte.

### 6.4 What the mechanism is not, yet

The honest reading is *"largely not problem-specific information"*, which is
weaker than *"not information"*. A donor cache is still a latent trace of a model
reasoning about a competition maths problem. It could plausibly carry transferable
mathematical procedure while carrying nothing about *this* problem.

One mechanism is cheap to test and was tested for free: if what the Judger gains
is simply a longer prefix — behaving as though it has already been reasoning, and
so moving to conclude — then the saving should scale with prefix length. Within
this data it does not. Regressing the per-item saving on the length of the cache
actually attached, over 515-962 positions:

```
shuf:  spearman +0.12, permutation p=0.61   (n=20)
real:  spearman +0.26, permutation p=0.24   (n=22)
```

No gradient, and the sign is if anything backwards. That is consistent with a
**threshold** effect — any sufficient prefix, then flat — but the range is only
1.9x, so it cannot exclude a gradient that acts far below 515 positions.

### 6.5 The experiment that decides it, and what it is worth if it lands

`evict_uniform` at `--evict_budget 64`: keep 64 of ~650 positions, a **10x smaller
cache**, chosen by key-norm importance. Combined with `real` (~650) and `none` (0),
one arm yields a three-point dose-response curve rather than a single contrast.

- If 64 positions retain the saving, the effect is threshold-like, and LatentMAS's
  measurable benefit is available at a tenth of the KV bytes. That is a
  compression result with a real metric — **KV bytes at fixed accuracy** — and it
  follows from a mechanism rather than from tuning.
- If 64 positions lose it, the effect is graded in length after all, §6.4's null
  was range restriction, and the next question is where the knee sits.

Queued behind the `single` arm on the same pod (`/workspace/chain_evict64.sh`),
~2.5 GPU-hours.

### 6.6 Where this leaves the write-up

The claim is no longer "we made LatentMAS faster", which was the fair criticism of
§4. It is a mechanism claim plus an efficiency claim, both falsifiable and both
measured on the authors' own benchmark and model:

> On AIME24 with Qwen3-14B, LatentMAS's latent channel does not measurably change
> accuracy (+1 item of 30, CI [-9.9, +3.2] pp), but it does cut the answering
> agent's tokens by a third (p=0.0003) at negligible upstream cost. A cache from an
> unrelated problem reproduces most of that saving, so the benefit is largely not
> the transfer of problem-specific information.

Both halves are needed. The first alone reads as a failed replication of a claim
the benchmark cannot support either way. The second explains what the method is
actually doing, and makes a prediction that §6.5 tests.
