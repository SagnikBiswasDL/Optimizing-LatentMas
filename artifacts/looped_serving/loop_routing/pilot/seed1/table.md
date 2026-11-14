| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SC@4 | 8192 | 28.5 (22.5, 34.5) | 4.00 | 6.6 | 53.4 | 27.4 | 11423 | 150.8 | 220.6 | 238.8 | 1051 | 5.8 | 49.9 |
| 2 | SC@4 | 8192 | 78.0 (72.5, 83.5) | 4.00 | 35.1 | 86.4 | 14.1 | 17507 | 333.9 | 250.5 | 911.6 | 603 | 4.4 | 80.0 |
| 3 | SC@4 | 8192 | 86.5 (81.5, 91.0) | 4.00 | 46.8 | 86.8 | 13.4 | 28653 | 214.6 | 145.5 | 550.4 | 394 | 2.6 | 80.0 |
| 4 | SC@4 | 8192 | 88.0 (83.0, 92.5) | 4.00 | 54.2 | 90.1 | 10.2 | 35827 | 285.2 | 197.1 | 755.3 | 268 | 1.9 | 80.0 |

| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | work saved vs SC@4 % | paired diff vs SC@4 (pts) |
|---|---|---|---|---|---|---|---|
| 1 | two-then-two (replay, first 4 samples) | d1_w4_cap8192 | 28.5 (22.5, 34.5) | 11154 | 13.5 | 2.4 | +0.0 |
| 2 | two-then-two (replay, first 4 samples) | d2_w4_cap8192 | 78.0 (72.5, 83.5) | 14456 | 61.5 | 17.4 | +0.0 |
| 3 | two-then-two (replay, first 4 samples) | d3_w4_cap8192 | 86.5 (81.5, 91.0) | 21773 | 75.0 | 24.0 | +0.0 |
| 4 | two-then-two (replay, first 4 samples) | d4_w4_cap8192 | 88.0 (83.0, 92.5) | 26840 | 77.5 | 25.1 | +0.0 |
