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

## 4. What this run did not establish

* **Nothing about accuracy at n=6.** The noise floor is +/-2 items (a
  semantically neutral bf16 batching change moved accuracy 0.667 -> 0.333), so
  only the token effects above are large enough to read. +81% and cap hits are;
  a -14% on one item is not.
* The screen covers 2 items by design (a cheap kill criterion), not a cohort.
* `flash_attn` is not installed on the pod, so both flash configs failed to load
  and are unmeasured. At batch 1 this is unlikely to matter — decode is bound by
  weight bandwidth, not attention.
