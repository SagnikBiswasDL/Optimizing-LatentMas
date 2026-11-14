# Loop-routing pilot gate (200 questions, two runs)

| depth | run 0 accuracy | run 1 accuracy | loop-tokens (run 0) | cap hit (run 0) | think share (run 0) | votes that differ between runs |
|---|---|---|---|---|---|---|
| 1 | 28.5% | 28.5% | 11487 | 27.8% | 5.6% | 48 |
| 2 | 76.0% | 78.0% | 18987 | 17.4% | 34.5% | 22 |
| 3 | 87.5% | 86.5% | 29966 | 13.4% | 49.1% | 14 |
| 4 | 89.5% | 88.0% | 35406 | 8.8% | 53.6% | 15 |

**Check 1: does "needs the next loop" repeat across runs?**

| step | run 0 | run 1 | both | kappa (95% interval) | gated |
|---|---|---|---|---|---|
| 1->2 | 97 | 101 | 66 | 0.34 (0.20 to 0.47) | pass |
| 2->3 | 30 | 23 | 15 | 0.50 (0.32 to 0.66) | pass |
| 3->4 | 10 | 7 | 1 | 0.08 (-0.05 to 0.32) | reported |

**Check 2: does the ideal routing from one run still win on the other?**

Transferred oracle minus best mixture at the same cost: -4.1 points (95% interval -7.6 to -0.6); run 0 to run 1: 79.0% at 24764 loop-tokens, depths [36, 96, 32, 36]. fail

**Gate: FAIL**
