# Deadline feasibility study

Decided 2026-11-12. Implemented, not run. The constants are
`looped/deadline.py`. `configs/looped_serving/deadline_study.json` is the same
document; the runner refuses the file if the two differ.

## Question

Can adapting the reasoning budget deliver more correct answers before a
deadline than a strong fixed allocation, on this Ouro-1.4B workload?

A pass is a development screen. It is not a new algorithm, and it is not a
reason to open the confirmation reserve. Certaindex/Dynasor, LASER and "Late
but Valuable" already study reasoning progress, load and the value of a late
correct answer. This measurement asks whether the saved direct-versus-long
tradeoff survives a queue.

## What is held fixed

Depth-3 engine, stock prompt, bf16, temperature 0.6, top-p 0.95, 32 slots,
80 GB of KV, CUDA graphs and static buffers. Direct is four samples at cap
2,048 starting with the token `To`. Long is four natural samples at cap
4,096. The confirmation reserve, GSM8K test and MATH-500 are refused.

## Workload

Questions stay in split order. Block of 20 at 4 per minute, then 20 at 12 per
minute, repeating. Question 0 arrives at time 0. The gap in front of a
question is one interarrival of its own block. The deadline is 240 seconds
from arrival. These are synthetic stress rates, not a production trace.

The calibration trace's last arrival is about 66 minutes (the run ends by
about 70, because work past a deadline is cancelled). The validation trace
ends by about 37 minutes.

## Arms

| Arm | Rule |
|---|---|
| always-direct | Every question is direct. |
| always-long | Every question is long. |
| mixture | One of `p25`, `p50`, `p75`, `disagree`, chosen on calibration and then frozen. The fractions hash the question id, so repetitions share the assignment. `disagree` launches direct and starts the long vote only when three direct samples can no longer agree. |
| queue-aware | Long only when no other question is in service and at least 120 seconds of the deadline remain. Otherwise direct. 120 is half the deadline. It is not fit on either split. |

The mixture winner is the calibration run with the most correct-in-time
answers, then the fewest questions that actually started a long answer, then
the candidate name. Seed 0 only. One run, so the choice is noisy; that is the
declared procedure.

## Score and gate

The score is the fraction of **all arrivals** whose selected answer grades
correct and whose decision time is at or before the deadline. Expired, late
and wrong answers stay in the denominator. Gold is applied by
`scripts/grade_deadline_study.py`, not by the driver.

The gate passes when the queue-aware arm beats the best of the three static
arms by at least 2 points on average across seeds 0, 1 and 2, and the
difference is positive on every seed. A failure, or a pass, stops there. It
does not open the confirmation reserve.

## How to run

One RTX PRO 6000 Blackwell is $2.49/h. Four calibration runs and twelve
validation runs occupy the GPU for about the length of the traces: roughly
13–15 hours, about $32–37, before a slow model load on every job. Do not
start a pod without an explicit go. Write to a network volume. Pull before
stopping. `runpodctl` inside the pod cannot stop it.

Gold stays off the pod. Run the calibration job list, pull the outputs, freeze
the mixture locally, copy `selection.json` up, then run the validation list.

```bash
# on the pod, after setup.sh. $M is the pinned snapshot.
bash scripts/pod/run_queue.sh scripts/pod/jobs_deadline_calibration.txt
```

```bash
# locally, after the pull
python scripts/grade_deadline_study.py --role select \
  --runs artifacts/looped_serving/deadline_study/calibration \
  --gold artifacts/looped_serving/splits/math_train_calibration.gold.jsonl
```

```bash
# on the pod. The mixture jobs read /root/work/repo/selection.json.
bash scripts/pod/run_queue.sh scripts/pod/jobs_deadline_validation.txt
```

```bash
python scripts/grade_deadline_study.py --role gate \
  --runs artifacts/looped_serving/deadline_study/validation \
  --gold artifacts/looped_serving/splits/math_train_validation.gold.jsonl \
  --selection artifacts/looped_serving/deadline_study/calibration/selection.json
```

A failed job does not stop `run_queue.sh`. Do not start the validation list
until `selection.json` exists. `--smoke N` runs a prefix and is labelled so
the grader ignores it. It is not evidence.

The closed-loop latencies already measured cannot be replayed into this
score. Service time depends on what else is in the batch, and the batch here
is the arrival trace.
