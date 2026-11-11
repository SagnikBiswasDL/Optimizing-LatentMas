| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | SC@4 | 2048 | 77.5 (73.2, 81.5) | 4.00 | 47.7 | 73.2 | 29.9 | 14628 | 101.6 | 118.7 | 162.4 | 844 | 11.5 | 79.9 |
| 3 | SC@4, direct | 2048 | 83.8 (80.0, 87.2) | 4.00 | 0.0 | 97.8 | 2.2 | 9118 | 49.3 | 36.0 | 129.2 | 932 | 21.7 | 79.9 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap2048 | 77.5 (73.2, 81.5) | 11321 | 62.5 | 22.6 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_direct_cap2048 | 83.8 (80.0, 87.2) | 6128 | 78.8 | 32.8 | +0.0 |
