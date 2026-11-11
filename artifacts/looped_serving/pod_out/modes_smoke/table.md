| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | SC@2, think | 4096 | 75.0 (66.0, 83.0) | 2.00 | 100.0 | 71.5 | 29.5 | 15319 | 136.8 | 129.2 | 229.2 | 539 | 6.7 | 75.1 |
| 3 | SC@4, direct | 2048 | 82.0 (74.0, 89.0) | 4.00 | 0.0 | 97.0 | 3.0 | 9082 | 49.8 | 41.7 | 126.6 | 740 | 17.5 | 77.6 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 3 | two-then-two (replay, first 4 samples) | d3_w4_direct_cap2048 | 82.0 (74.0, 89.0) | 5843 | 79.0 | 35.7 | +0.0 |
