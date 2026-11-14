# Loop selectors, source `stage0b` (300 questions)

Fixed depths: 1 loops 30.7% at 4224 loop-tokens, 2 loops 67.7% at 8656 loop-tokens, 3 loops 77.3% at 13678 loop-tokens, 4 loops 77.0% at 19860 loop-tokens
Oracle (cheapest correct depth): 84.0% at 7422, depths [127, 116, 41, 16]

Operating point: best accuracy at mean cost <= fixed 3 loops (13678); picked on the same out-of-fold predictions, so this is a development read.

| selector | accuracy | loop-tokens | best mixture, same cost | matched random | p (random) | vs 3 loops (only sel / only fixed, p) | depths |
|---|---|---|---|---|---|---|---|
| length | 75.7% | 12724 | 75.5% | 75.2% | 0.376 | 0 / 5, 0.0625 | [0, 66, 230, 4] |
| level+subject | 76.7% | 12855 | 75.7% | 74.4% | 0.022 | 0 / 2, 0.5 | [0, 89, 192, 19] |
| prefill | 74.0% | 11125 | 72.4% | 71.5% | 0.021 | 5 / 15, 0.0414 | [6, 154, 136, 4] |
| prefill_profile | 73.7% | 13227 | 76.5% | 73.4% | 0.477 | 1 / 12, 0.00342 | [6, 91, 154, 49] |
| text | 75.0% | 13449 | 76.9% | 75.1% | 0.599 | 5 / 12, 0.143 | [0, 69, 187, 44] |
| prefill_delta_threshold | 77.0% | 13521 | 77.0% | 76.8% | 0.522 | 2 / 3, 1 | [0, 14, 278, 8] |
| gate_threshold | 77.3% | 13592 | 77.2% | 76.9% | 0.274 | 1 / 1, 1 | [0, 14, 284, 2] |
