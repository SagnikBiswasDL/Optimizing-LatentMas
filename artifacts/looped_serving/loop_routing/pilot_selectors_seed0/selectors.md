# Loop selectors, source `pilot` (200 questions)

Fixed depths: 1 loops 28.5% at 11487 loop-tokens, 2 loops 76.0% at 18987 loop-tokens, 3 loops 87.5% at 29966 loop-tokens, 4 loops 89.5% at 35406 loop-tokens
Oracle (cheapest correct depth): 93.5% at 17410, depths [49, 96, 32, 23]

Operating point: best accuracy at mean cost <= fixed 3 loops (29966); picked on the same out-of-fold predictions, so this is a development read.

| selector | accuracy | loop-tokens | best mixture, same cost | matched random | p (random) | vs 3 loops (only sel / only fixed, p) | depths |
|---|---|---|---|---|---|---|---|
| length | 86.5% | 29317 | 86.8% | 87.0% | 0.913 | 0 / 2, 0.5 | [0, 8, 191, 1] |
| level+subject | 86.0% | 28259 | 85.7% | 84.8% | 0.237 | 3 / 6, 0.508 | [0, 54, 113, 33] |
| prefill | 86.0% | 29900 | 87.4% | 84.5% | 0.188 | 3 / 6, 0.508 | [0, 57, 114, 29] |
| prefill_profile | 85.5% | 29689 | 87.2% | 85.0% | 0.425 | 2 / 6, 0.289 | [0, 45, 143, 12] |
| probe | 86.5% | 29915 | 87.4% | 87.1% | 0.794 | 1 / 3, 0.625 | [0, 13, 147, 40] |
| probe_profile | 85.0% | 28396 | 85.9% | 86.1% | 0.872 | 2 / 7, 0.18 | [0, 30, 139, 31] |
| prefill+probe | 86.0% | 27901 | 85.3% | 84.3% | 0.141 | 3 / 6, 0.508 | [0, 59, 122, 19] |
| text | 87.0% | 27729 | 85.2% | 84.6% | 0.045 | 4 / 5, 1 | [0, 53, 137, 10] |
| prefill_delta_threshold | 87.5% | 29807 | 87.3% | 87.0% | 0.358 | 1 / 1, 1 | [0, 11, 182, 7] |
| gate_threshold | 88.0% | 29929 | 87.5% | 86.9% | 0.090 | 1 / 0, 1 | [0, 10, 188, 2] |
| probe_delta_threshold | 87.5% | 29732 | 87.3% | 87.1% | 0.448 | 0 / 0, 1 | [0, 7, 190, 3] |
