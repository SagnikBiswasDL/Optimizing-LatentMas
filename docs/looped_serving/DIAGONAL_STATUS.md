# Diagonal schedule: status for the next agent

Date: 2026-11-24. Branch `research/looped-serving`. Pod `nj8lgo50xa5ah8` (`delta-exit-validity`) is stopped. Raw rows are on network volume `s7ctgyb8b7`, mounted at `/workspace/delta_exit_validity/` (`lossless.jsonl`, `math.jsonl`, `gsm8k.jsonl`, `summary.txt`, `DONE`).

## What was implemented

Jiayi's 2026-11-14 proposal, the two decode schedules only. Code: `looped/delta_exit.py`. Tests that do not load Ouro: `tests/test_wavefront.py`, `tests/test_delta_exit.py`. Suite: `scripts/delta_exit_validity.py`.

* **Wavefront (`wfd`).** One step advances every in-flight word by one repeat. The newest word is one repeat behind the word before it. A draft made at repeat 1 is kept only when the previous word, at full depth, predicts that same id. A miss deletes the in-flight suffix and restarts from the full-depth id.
* **Convergence diagonal (`diagonal`).** The next word's first repeat runs beside the parent's next repeat. That repeat is kept only when the parent's cross-loop KV delta falls below epsilon. Otherwise it is discarded and recomputed against the parent's newer state. If the parent never settles, the next word starts over from the parent's final repeat and then runs one word at a time.

Prompt words still run every repeat. Epsilon was set to 0.05, 0.1, and 0.2. It was not calibrated on a held-out set. Model: non-thinking Ouro-1.4B, greedy, bf16, solver prompt with a boxed answer. An answer with no boxed value counts as wrong.

## Data

* Token match: 20 MATH-train questions, 4 from each difficulty level, seed 0, cap 192 new tokens. These 20 are inside the 100 below.
* MATH accuracy: 100 MATH-train questions, 20 per level, seed 0, cap 1,024. Source: `EleutherAI/hendrycks_math`.
* GSM8K accuracy: 100 test questions, seed-0 shuffle, cap 512. Source: `gsm8k` main.

## Result

The diagonal fails the check the proposal set for it, and it loses the answers.

Token match against ordinary depth 4:

| Schedule | Identical sequences |
|---|---|
| Wavefront | 7/20 |
| Diagonal, epsilon 0 | 9/20 |

The first disagreement is around word 25. A short non-math generation (38 tokens) did match. The MATH traces do not.

Correct answers. The second number is the mean repeats actually run.

| Schedule | MATH, 100 | GSM8K, 100 |
|---|---|---|
| Always 4 repeats | 51, 4.00 | 62, 4.00 |
| Always 3 repeats | 51, 3.00 | 76, 3.00 |
| Always 2 repeats | 57, 2.00 | — |
| Always 1 repeat | 26, 1.00 | — |
| Ouro gate, threshold 0.5 | 52, 4.00 | 67, 4.00 |
| KV-delta, epsilon 0.05, one word at a time | 51, 3.78 | — |
| KV-delta, epsilon 0.1 | 47, 3.36 | 64, 3.31 |
| KV-delta, epsilon 0.2 | 42, 3.14 | — |
| Diagonal, epsilon 0.05 | 27, 3.77 | — |
| Diagonal, epsilon 0.1 | 25, 3.34 | 34, 3.30 |
| Diagonal, epsilon 0.2 | 17, 3.11 | — |
| Second-order KV rule, epsilon 0.1 | 51, 4.00 | — |
| Random halt, mean depth about 3.3 | 46, 3.31 | — |
| KV-delta, epsilon 0.1, no copy-forward | 46, 4.00 | — |

The gate still runs all 4 repeats. The plain KV-delta stays close to depth 4 and drops about half a repeat. The diagonal stops after about the same number of repeats as that plain rule and gets about half as many answers. Deeper runs hit the length cap more often (MATH depth 4 finished 55/100; depth 2 finished 68/100), so the depth-2 accuracy is partly the cap.

## What is still open

The raw jsonl files are on the volume, not in this commit. The wavefront mismatch is unexplained: a schedule that is supposed to be exact diverges by word 25. Whether that is a bug in `generate_wavefront` or bf16 disagreement from a different reduction order has not been separated. Prefill words were not early-exited. Epsilon was not fit on a calibration split. MATH-500, the MATH test split, and GPQA were not run.
