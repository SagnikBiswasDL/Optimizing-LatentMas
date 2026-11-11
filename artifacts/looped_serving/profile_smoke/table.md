| depth | policy | accuracy % (95% CI) | candidates/q | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | greedy | 54.0 (44.0, 63.0) | 1.00 | 73.0 | 28.0 | 2448 | 26.6 | 16.8 | 62.4 | 791 | 48.4 | 52.3 |
| 2 | SC@4 | 72.0 (63.0, 81.0) | 4.00 | 75.5 | 26.5 | 9959 | 42.3 | 43.6 | 67.6 | 869 | 13.0 | 53.4 |
| 3 | greedy | 62.0 (52.0, 71.0) | 1.00 | 74.0 | 28.0 | 3680 | 40.6 | 27.5 | 91.7 | 584 | 32.9 | 51.9 |
| 3 | SC@4 | 75.0 (66.0, 83.0) | 4.00 | 74.5 | 28.5 | 14746 | 57.4 | 65.7 | 88.1 | 674 | 9.5 | 53.4 |
| 3 | SC@4 + certificate cancel (live) | 77.0 (69.0, 85.0) | 4.00 | 71.6 | 27.0 | 14132 | 49.0 | 39.9 | 86.4 | 691 | 10.2 | 52.0 |
| 3 | two-then-two (live) | 74.0 (65.0, 82.0) | 2.90 | 64.8 | 38.6 | 12196 | 89.7 | 68.7 | 183.2 | 613 | 10.4 | 51.2 |
| 3 | two-then-two (live), fallback depth 4 | 75.0 (66.0, 83.0) | 2.90 | 65.9 | 36.6 | 13681 | 100.2 | 63.7 | 205.6 | 561 | 9.4 | 50.6 |
| 3 | SC@8 | 74.0 (65.0, 82.0) | 8.00 | 73.0 | 29.0 | 29388 | 59.9 | 75.6 | 86.9 | 692 | 4.9 | 56.7 |
| 4 | greedy | 59.0 (49.0, 69.0) | 1.00 | 69.0 | 35.0 | 5377 | 56.6 | 47.2 | 112.7 | 493 | 24.2 | 51.9 |
| 4 | SC@4 | 78.0 (69.0, 86.0) | 4.00 | 72.2 | 31.5 | 20327 | 79.5 | 98.1 | 104.4 | 585 | 7.6 | 53.4 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 2 | two-then-two (replay, first 4 samples) | d2_w4_cap2048 | 72.0 (63.0, 81.0) | 7342 | 54.0 | 18.3 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap2048 | 75.0 (66.0, 83.0) | 11212 | 61.0 | 21.4 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w8_cap2048 | 72.0 (63.0, 81.0) | 11093 | 62.0 | 23.0 | +0.0 |
| 4 | two-then-two (replay, first 4 samples) | d4_w4_cap2048 | 78.0 (69.0, 86.0) | 16067 | 53.0 | 21.0 | +0.0 |

| arm | cap | capped candidates rerun | old tokens reproduced | finish with EOS | answered before -> after | correct before -> after | policy accuracy before -> after % |
|---|---|---|---|---|---|---|---|
| d3_w4_cap4096_extend | 2048 -> 4096 | 114 | 0 | 47 | 12 -> 54 | 12 -> 51 | 75.0 -> 79.0 |
