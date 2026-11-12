| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 4 launched, stop at 2 agreeing (live), direct | 2048 | 82.0 (76.5, 87.5) | 4.00 | 0.0 | 95.9 | 2.4 | 8745 | 21.1 | 14.8 | 63.1 | 892 | 22.0 | 40.7 |
| 3 | SC@4 | 2048 | 78.0 (72.0, 83.5) | 4.00 | 43.4 | 75.4 | 27.1 | 14115 | 49.8 | 55.0 | 81.8 | 808 | 11.5 | 42.4 |
| 3 | SC@4 | 4096 | 84.0 (79.0, 89.0) | 4.00 | 43.4 | 82.6 | 18.4 | 19709 | 92.3 | 66.0 | 209.0 | 664 | 6.5 | 78.3 |
| 3 | SC@4, direct | 2048 | 79.5 (74.0, 85.0) | 4.00 | 0.0 | 97.0 | 2.9 | 9201 | 25.5 | 18.8 | 65.2 | 880 | 20.4 | 42.3 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap2048 | 78.0 (72.0, 83.5) | 11087 | 62.0 | 21.5 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap4096 | 84.0 (79.0, 89.0) | 15110 | 70.0 | 23.3 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_direct_cap2048 | 79.5 (74.0, 85.0) | 6370 | 75.5 | 30.8 | +0.0 |
