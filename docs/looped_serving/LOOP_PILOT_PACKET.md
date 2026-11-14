# Loop-count pilot packet

For a follow-up strategy. Protocol and decision are in
`docs/looped_serving/LOOP_ROUTING.md`. This file is the numbers and the files.

## What was asked

Does a MATH question have a stable number of loops on Ouro-1.4B, or does that
depend on the random draw? 200 practice questions, answered at depths 1, 2, 3,
and 4, then the whole grid repeated (run seeds 0 and 1).

Fixed: stock solver prompt, model picks its own answer style, temperature 0.6,
top-p 0.95, four samples, majority vote, ties to the lowest sample index,
unparseable or cut-off answers cast no vote, cap 8192. Each depth ran on its
own depth-limited vLLM-rlt engine (`total_ut_steps = d`, stock prompt, bf16).
Graded locally with math-verify 0.9.0. Gold never went to the pod.

Question list: `artifacts/looped_serving/splits/router_pilot_questions.txt`
(sha256 `300d480f99cf1e73108027575c0cf07180079bdf2671ee45ea5dc0f1b50a0bad`),
40 questions from each MATH level 1–5, taken from the practice split
`math_train_calibration`.

## Verdict

The average at each depth repeats. A per-question loop count read off one run
does not. Using run 0's ideal per-question depth on run 1 scores 79.0%, which
is 4.1 points worse than the best question-blind mix at the same cost
(95% interval −7.6 to −0.6). Gate: fail. The larger routing experiment was not run.

3 loops is where the reliable gain sits (about 87%). The 4th loop adds about
1–2 points and the questions it saves do not repeat (1 question in both runs).

## Counts (200 questions)

| Loops | Run 0 right | Run 0 wrong | Run 1 right | Run 1 wrong | Votes that differ |
|---|---:|---:|---:|---:|---:|
| 1 | 57 | 143 | 57 | 143 | 48 |
| 2 | 152 | 48 | 156 | 44 | 22 |
| 3 | 175 | 25 | 173 | 27 | 14 |
| 4 | 179 | 21 | 176 | 24 | 15 |

Same outcome both runs: depth 1, 33 right in both and 119 wrong in both.
Depth 2, 143 right in both and 35 wrong in both. Depth 3, 167 right in both
and 19 wrong in both. Depth 4, 170 right in both and 15 wrong in both.

"Needs the next loop" means wrong at d and right at d+1.

| Step | Run 0 | Run 1 | Both | kappa (95% interval) |
|---|---:|---:|---:|---|
| 1 → 2 | 97 | 101 | 66 | 0.34 (0.20 to 0.47) |
| 2 → 3 | 30 | 23 | 15 | 0.50 (0.32 to 0.66) |
| 3 → 4 | 10 | 7 | 1 | 0.08 (−0.05 to 0.32) |

## Run 0, graded summary

| Loops | Accuracy (95% CI) | Think % | Answered % | Capped % | Loop-tokens / q | Mean wait s |
|---|---|---:|---:|---:|---:|---:|
| 1 | 28.5% (22.0–35.0) | 5.6 | 49.8 | 27.8 | 11487 | 150.9 |
| 2 | 76.0% (70.0–82.0) | 34.5 | 83.1 | 17.4 | 18987 | 157.0 |
| 3 | 87.5% (83.0–92.0) | 49.1 | 87.1 | 13.4 | 29966 | 227.6 |
| 4 | 89.5% (85.0–93.5) | 53.6 | 91.8 | 8.8 | 35406 | 277.6 |

## Run 1, graded summary

| Loops | Accuracy (95% CI) | Think % | Answered % | Capped % | Loop-tokens / q | Mean wait s |
|---|---|---:|---:|---:|---:|---:|
| 1 | 28.5% (22.5–34.5) | 6.6 | 53.4 | 27.4 | 11423 | 150.8 |
| 2 | 78.0% (72.5–83.5) | 35.1 | 86.4 | 14.1 | 17507 | 333.9 |
| 3 | 86.5% (81.5–91.0) | 46.8 | 86.8 | 13.4 | 28653 | 214.6 |
| 4 | 88.0% (83.0–92.5) | 54.2 | 90.1 | 10.2 | 35827 | 285.2 |

Run 1 at depth 2 used 96 slots; every other arm used 32. Its wait is not
comparable. Loop-tokens are the cost measure. The 8192 cap is hit often
(9% to 28%), so it is not a rare runaway guard.

## Selectors tried on run 0 only (development, not a claim)

Out-of-fold logistic heads, operating point = best accuracy at a mean cost no
higher than fixed 3 loops. None beat fixed 3 loops. Full table:
`artifacts/looped_serving/loop_routing/pilot_selectors_seed0/selectors.md`.

Same-run oracle (cheapest correct depth, using run 0's own labels) is 93.5%
at 17410 loop-tokens, depths [49, 96, 32, 23]. That number uses the labels it
is scored on. The cross-run transfer above is the one that counts.

Earlier, on saved LoopDelta data (300 questions, four sampled answers, cap
2048) and on the 400-question practice set (forced short answers, depths 2–4,
cap 2048), the same selectors also failed to beat a fixed depth. Tables:
`artifacts/looped_serving/loop_routing/stage_a_stage0b/selectors.md` and
`stage_a_practice/selectors.md`.

## Files

Root: `artifacts/looped_serving/loop_routing/pilot/`.

Eight arms, 800 candidate rows each (200 questions × 4 samples):

```
pilot/seed0/d{1,2,3,4}_w4_cap8192/
pilot/seed1/d{1,2,3,4}_w4_cap8192/
```

Each directory has:

- `candidates.jsonl` — one row per sample. Fields include `question_id`,
  `candidate_id` (`{qid}:d{depth}:s{0-3}:{mode}`), `depth`, `sample`, `text`,
  `pred`, `answer_key`, `parse_status`, `finish_reason` (`eos` or `length_cap`),
  `think`, `gen_tokens`, `logical_loop_tokens`, `prompt_tokens`.
- `graded.jsonl` — `question_id`, `candidate_id`, `pred`, `answer_key`,
  `parse_status`, `finish_reason`, `correct`.
- `questions.jsonl` — per-question latency and `all_candidate_ids`.
- `summary.json`, `manifest.json`.

`pilot/seed0/table.md` and `pilot/seed1/table.md` are the graded summaries.
`pilot/gate.json` is the repeatability gate, including per-run fixed accuracies.
`pilot/features/features.jsonl` is 200 rows of prefill and probe features
(probe = first 64 tokens of one greedy depth-4 answer). Keys: `prefill` and
`probe` (`"16"`, `"32"`, `"64"`), each with `delta`, `delta_k`, `delta_v`,
`entropy`, `kl_consec`, `top1_agree_last`, `gate_lambda`, and layerwise
`delta_layers_k` / `delta_layers_v`.

Code: `looped/loop_select.py` (vote, gate, transfer), `scripts/loop_pilot_gate.py`,
`scripts/eval_loop_selectors.py`, `scripts/loop_probe_features.py`.
Job lists: `scripts/pod/jobs_router_pilot.txt` and the `_a` / `_b` / `_rest` splits.
Pods are stopped. GPU logs were on the pod volume, not in this repo. The result
files above are the record.
