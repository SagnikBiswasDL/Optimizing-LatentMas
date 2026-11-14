# Loop selectors, source `practice` (400 questions)

Fixed depths: 2 loops 75.0% at 6265 loop-tokens, 3 loops 83.8% at 9118 loop-tokens, 4 loops 83.8% at 12059 loop-tokens
Oracle (cheapest correct depth): 89.0% at 6970, depths [336, 51, 13, 0]

Operating point: best accuracy at mean cost <= fixed 3 loops (9118); picked on the same out-of-fold predictions, so this is a development read.

| selector | accuracy | loop-tokens | best mixture, same cost | matched random | p (random) | vs 3 loops (only sel / only fixed, p) | depths |
|---|---|---|---|---|---|---|---|
| length | 83.5% | 9057 | 83.6% | 83.5% | 0.694 | 0 / 1, 1 | [11, 389, 0, 0] |
| level+subject | 81.0% | 8396 | 81.5% | 80.2% | 0.237 | 2 / 13, 0.00739 | [159, 226, 15, 0] |
| text | 81.5% | 8757 | 82.6% | 80.9% | 0.314 | 1 / 10, 0.0117 | [129, 228, 43, 0] |
