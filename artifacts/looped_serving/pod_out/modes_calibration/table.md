| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | SC@4, direct | 2048 | 75.0 (70.5, 79.2) | 4.00 | 0.0 | 95.7 | 4.3 | 6265 | 32.8 | 23.8 | 82.8 | 1533 | 34.4 | 54.8 |
| 3 | SC@2, think | 4096 | 28.0 (18.7, 38.7) | 2.00 | 100.0 | 23.3 | 82.7 | 23826 | 198.7 | 213.6 | 226.6 | 558 | 4.4 | 78.2 |
| 3 | SC@4 | 2048 | 77.5 (73.2, 81.5) | 4.00 | 47.7 | 73.2 | 29.9 | 14628 | 101.6 | 118.7 | 162.4 | 844 | 11.5 | 79.9 |
| 3 | SC@4 | 4096 | 82.5 (78.8, 86.0) | 4.00 | 47.7 | 81.7 | 19.5 | 20530 | 96.8 | 75.4 | 203.7 | 678 | 6.4 | 77.5 |
| 3 | SC@4, direct | 2048 | 83.8 (80.0, 87.2) | 4.00 | 0.0 | 97.8 | 2.2 | 9118 | 49.3 | 36.0 | 129.2 | 932 | 21.7 | 79.9 |
| 4 | SC@4, direct | 2048 | 83.8 (80.0, 87.2) | 4.00 | 0.0 | 97.8 | 2.2 | 12059 | 41.4 | 31.3 | 109.1 | 841 | 19.7 | 80.0 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 2 | two-then-two (replay, first 4 samples) | d2_w4_direct_cap2048 | 75.0 (70.5, 79.2) | 4759 | 64.0 | 24.0 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap2048 | 77.5 (73.2, 81.5) | 11321 | 62.5 | 22.6 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap4096 | 82.5 (78.8, 86.0) | 15699 | 70.5 | 23.5 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_direct_cap2048 | 83.8 (80.0, 87.2) | 6128 | 78.8 | 32.8 | +0.0 |
| 4 | two-then-two (replay, first 4 samples) | d4_w4_direct_cap2048 | 83.8 (80.0, 87.2) | 8293 | 77.0 | 31.2 | +0.0 |
