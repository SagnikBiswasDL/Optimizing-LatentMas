# Parity: `real` vs `real__bs16`

n=6 · identical text **0/6** · same tokens 2/6 · same correct 4/6
acc 0.667 -> 0.333
wall clock 933s -> 533s (**1.75x**)

**verdict: diverged**

| idx | identical | tokens A | tokens B | correct A | correct B | diverge @char |
|---:|---|---:|---:|---|---|---:|
| 0 | NO | 2600 | 2598 | 1 | 1 | 460 |
| 1 | NO | 8192 | 8192 | 0 | 0 | 479 |
| 2 | NO | 8192 | 8192 | 0 | 0 | 455 |
| 4 | NO | 7653 | 6901 | 1 | 0 | 944 |
| 10 | NO | 5848 | 8192 | 1 | 0 | 907 |
| 18 | NO | 7062 | 6823 | 1 | 1 | 1571 |

Not bit-identical, so batched rows are self-consistent but not drop-in comparable to unbatched ones. Keep each comparison within a single batch size, and say so when quoting the numbers.
