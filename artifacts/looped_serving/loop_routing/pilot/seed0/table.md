| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SC@4 | 8192 | 28.5 (22.0, 35.0) | 4.00 | 5.6 | 49.8 | 27.8 | 11487 | 150.9 | 222.1 | 240.0 | 1056 | 5.8 | 50.2 |
| 2 | SC@4 | 8192 | 76.0 (70.0, 82.0) | 4.00 | 34.5 | 83.1 | 17.4 | 18987 | 157.0 | 63.9 | 418.2 | 605 | 4.0 | 80.0 |
| 3 | SC@4 | 8192 | 87.5 (83.0, 92.0) | 4.00 | 49.1 | 87.1 | 13.4 | 29966 | 227.6 | 155.0 | 592.3 | 394 | 2.5 | 79.9 |
| 4 | SC@4 | 8192 | 89.5 (85.0, 93.5) | 4.00 | 53.6 | 91.8 | 8.8 | 35406 | 277.6 | 210.8 | 751.0 | 273 | 2.0 | 80.0 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 1 | two-then-two (replay, first 4 samples) | d1_w4_cap8192 | 28.5 (22.0, 35.0) | 11305 | 13.0 | 1.6 | +0.0 |
| 2 | two-then-two (replay, first 4 samples) | d2_w4_cap8192 | 76.0 (70.0, 82.0) | 16087 | 62.0 | 15.3 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap8192 | 87.5 (83.0, 92.0) | 23284 | 74.5 | 22.3 | +0.0 |
| 4 | two-then-two (replay, first 4 samples) | d4_w4_cap8192 | 89.5 (85.0, 93.5) | 25601 | 76.5 | 27.7 | +0.0 |
