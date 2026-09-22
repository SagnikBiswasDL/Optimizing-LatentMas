# Judger token-budget curve

Hard cap replayed over saved generations; no answer inside the cap = wrong, so this is the conservative bound on an early-exit policy (a real policy could force an answer at the cap and do better).

Caveat: lengths come from re-tokenizing the saved text, which can differ by a few tokens from the generation-time count. `est_judger_s` prices each budget at the measured tok/s for that arm.

## `real` (n=6, 42.4 tok/s measured)

| budget | acc | Δacc | mean tokens | est judger_s | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.000 | -0.667 | 512 | 12.1 | 12.87x |
| 1024 | 0.000 | -0.667 | 1024 | 24.1 | 6.44x |
| 1536 | 0.000 | -0.667 | 1536 | 36.2 | 4.29x |
| 2048 | 0.000 | -0.667 | 2048 | 48.3 | 3.22x |
| 3072 | 0.167 | -0.500 | 2993 | 70.6 | 2.20x |
| 4096 | 0.167 | -0.500 | 3846 | 90.7 | 1.71x |
| 6144 | 0.500 | -0.167 | 5504 | 129.8 | 1.20x |
| 8192 | 0.667 | +0.000 | 6590 | 155.4 | 1.00x |

