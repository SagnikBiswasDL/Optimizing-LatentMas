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

## `real_seal40` (n=6, 42.5 tok/s measured)

| budget | acc | Δacc | mean tokens | est judger_s | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.000 | -0.500 | 512 | 12.0 | 13.17x |
| 1024 | 0.000 | -0.500 | 1024 | 24.1 | 6.59x |
| 1536 | 0.000 | -0.500 | 1536 | 36.1 | 4.39x |
| 2048 | 0.167 | -0.333 | 2008 | 47.2 | 3.36x |
| 3072 | 0.167 | -0.333 | 2861 | 67.3 | 2.36x |
| 4096 | 0.167 | -0.333 | 3714 | 87.4 | 1.82x |
| 6144 | 0.333 | -0.167 | 5380 | 126.6 | 1.25x |
| 8192 | 0.500 | +0.000 | 6745 | 158.7 | 1.00x |

## `none` (n=6, 42.6 tok/s measured)

| budget | acc | Δacc | mean tokens | est judger_s | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.000 | -0.333 | 512 | 12.0 | 14.35x |
| 1024 | 0.000 | -0.333 | 1024 | 24.1 | 7.18x |
| 1536 | 0.000 | -0.333 | 1536 | 36.1 | 4.78x |
| 2048 | 0.000 | -0.333 | 2048 | 48.1 | 3.59x |
| 3072 | 0.167 | -0.167 | 3072 | 72.2 | 2.39x |
| 4096 | 0.167 | -0.167 | 3935 | 92.5 | 1.87x |
| 6144 | 0.167 | -0.167 | 5641 | 132.6 | 1.30x |
| 8192 | 0.333 | +0.000 | 7348 | 172.7 | 1.00x |

