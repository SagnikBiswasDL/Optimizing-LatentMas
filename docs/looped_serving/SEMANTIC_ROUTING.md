# Semantic loop routing on the saved pilot

Cross-validated development experiment on the 200-question router pilot. It does not retrain Ouro, generate answers, or change serving. The confirmation reserve and the final benchmarks were not read. One frozen encoder was used. No second encoder was tried.

## Setup

Encoder `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, frozen, 384-d, L2-normalized. 12 questions needed more than one 256-wordpiece window. Device: cpu (Apple M5).

Each question is one group. Both generation runs stay in that group. Five outer folds are stratified by MATH level. Logistic `C`, ridge `alpha`, and the utility penalty `lambda` are chosen on four inner folds of the outer-training questions. The reported accuracy uses the outer-test questions only.

The action list is depths 1, 2, 3, and 4. The router maximizes predicted vote accuracy minus `lambda` times predicted loop-tokens / 10000. The primary budget is 0.9 times the outer-training mean cost of fixed 3 loops. Budgets 0.75 and 1.0 are reported and were not used to pick the primary.

Length features are log(1 + characters) and log(1 + whitespace words). TF-IDF uses word unigrams and bigrams, `min_df=2`, sublinear tf, and L2 normalization, fit on training questions only. Semantic and TF-IDF both include the length features.

Repository HEAD at the run: `890702b8dfcb27031356a7e350bdf02f6e493bb3`. That is the inspected commit. Later deadline-study files in the worktree were not inputs.

Primary vote counts reproduced before fitting: run 0 [57, 152, 175, 179], run 1 [57, 156, 173, 176]. Strict cap exclusion, rescored and not refit: run 0 [57, 152, 175, 177], run 1 [57, 155, 172, 175].

Seed 1 at depths 1, 3, and 4 records a different engine source-tree hash from the other arms. Seed 1 at depth 2 used 96 slots; every other arm used 32. Wait times are not comparable. Prompt, model revision, temperature, top-p, cap, width, and natural mode match.

## Primary budget, 0.9

| method | accuracy | loop-tokens | share of fixed-3 cost | equal-cost surplus | minus training mixture | strict-cap accuracy | depth counts |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_1 | 28.5% | 11455 | 0.391 | 0.0 | None | 28.5% | [200.0, 0.0, 0.0, 0.0] |
| fixed_2 | 77.0% | 18247 | 0.623 | 0.0 | None | 76.8% | [0.0, 200.0, 0.0, 0.0] |
| fixed_3 | 87.0% | 29309 | 1.000 | 0.0 | None | 86.8% | [0.0, 0.0, 200.0, 0.0] |
| fixed_4 | 88.8% | 35617 | 1.215 | 0.0 | None | 88.0% | [0.0, 0.0, 0.0, 200.0] |
| mixture@0.9 | 84.3% | 26354 | 0.899 | 0.0002 | None | 84.1% | [0.0, 53.07886412394285, 146.92113587605715, 0.0] |
| length@0.9 | 77.2% | 19832 | 0.677 | -0.0118 | -0.0709 | 77.0% | [2.0, 156.0, 41.0, 1.0] |
| tfidf@0.9 | 81.2% | 22489 | 0.767 | 0.0042 | -0.0309 | 81.0% | [1.0, 104.0, 67.0, 28.0] |
| semantic@0.9 | 82.5% | 24052 | 0.821 | 0.0025 | -0.0184 | 82.5% | [3.0, 104.0, 74.0, 19.0] |
| tfidf_quality_length_cost@0.9 | 78.0% | 22515 | 0.768 | -0.0286 | -0.0634 | 78.0% | [3.0, 127.0, 50.0, 20.0] |
| semantic_quality_length_cost@0.9 | 80.2% | 24138 | 0.824 | -0.0207 | -0.0409 | 80.2% | [2.0, 107.0, 73.0, 18.0] |

Equal-cost surplus is accuracy minus the retrospective question-blind frontier at the same realized cost. That frontier is rebuilt from held-out outcomes and was not used to choose the router. The training mixture is the deployable blind baseline: its weights were fit on outer-training questions only.

Budget miss is realized test cost divided by the budget fixed from that fold's training questions:
- mixture@0.9: realized/budget 1.005, folds over the training budget 4 of 5.
- length@0.9: realized/budget 0.754, folds over the training budget 0 of 5.
- tfidf@0.9: realized/budget 0.856, folds over the training budget 0 of 5.
- semantic@0.9: realized/budget 0.918, folds over the training budget 2 of 5.
- tfidf_quality_length_cost@0.9: realized/budget 0.858, folds over the training budget 1 of 5.
- semantic_quality_length_cost@0.9: realized/budget 0.921, folds over the training budget 2 of 5.

Bootstrap intervals resample questions, carry both runs, and use the same draws for every method. They describe these frozen predictions. They do not refit the heads.

- Semantic accuracy interval: 0.7750 to 0.8725.
- Semantic equal-cost surplus interval: -0.0233 to 0.0281.
- TF-IDF equal-cost surplus interval: -0.0183 to 0.0267.
- Length equal-cost surplus interval: -0.0272 to 0.0043.
- Semantic minus TF-IDF equal-cost surplus interval: -0.0346 to 0.0313.

## Other budgets

| method | accuracy | loop-tokens | equal-cost surplus |
|---|---:|---:|---:|
| mixture@0.75 | 80.4% | 21921 | 0.0004 |
| length@0.75 | 76.8% | 18257 | -0.0026 |
| tfidf@0.75 | 79.0% | 19102 | 0.0123 |
| semantic@0.75 | 80.8% | 20899 | 0.0135 |
| mixture@1.0 | 87.0% | 29309 | 0.0 |
| length@1.0 | 77.2% | 19832 | -0.0118 |
| tfidf@1.0 | 81.5% | 23071 | 0.0014 |
| semantic@1.0 | 82.2% | 24901 | -0.0077 |

## Per run, primary budget

- semantic: run 0: 82.0% at 24759; run 1: 83.0% at 23344
- tfidf: run 0: 80.5% at 23463; run 1: 82.0% at 21515

## Prediction quality

Log loss and Brier score are held-out scores of the refit heads. A better score here does not by itself mean a better depth assignment.

| method | depth | log loss | Brier | cost MAE | constant-cost MAE | predicted mean | observed mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| length | 1 | 0.554 | 0.187 | 5677 | 6218 | 11430 | 11455 |
| length | 2 | 0.504 | 0.164 | 12417 | 15250 | 18310 | 18247 |
| length | 3 | 0.363 | 0.110 | 20481 | 23258 | 29312 | 29309 |
| length | 4 | 0.313 | 0.092 | 23515 | 25736 | 35669 | 35617 |
| tfidf | 1 | 0.556 | 0.187 | 5305 | 6218 | 11337 | 11455 |
| tfidf | 2 | 0.495 | 0.161 | 12702 | 15250 | 18769 | 18247 |
| tfidf | 3 | 0.361 | 0.109 | 19707 | 23258 | 29717 | 29309 |
| tfidf | 4 | 0.293 | 0.084 | 22587 | 25736 | 36639 | 35617 |
| semantic | 1 | 0.550 | 0.184 | 5236 | 6218 | 11458 | 11455 |
| semantic | 2 | 0.500 | 0.162 | 11783 | 15250 | 18375 | 18247 |
| semantic | 3 | 0.346 | 0.106 | 20174 | 23258 | 29081 | 29309 |
| semantic | 4 | 0.293 | 0.085 | 23572 | 25736 | 35390 | 35617 |

## Shuffled depths

Within each outer test fold, the learned depths were permuted across questions. This is descriptive. It was not a selection criterion.

- length: shuffled accuracy 78.4% (interval 76.8% to 80.2%), mean loop-tokens 20589.
- tfidf: shuffled accuracy 81.6% (interval 79.7% to 83.5%), mean loop-tokens 24116.
- semantic: shuffled accuracy 81.0% (interval 78.8% to 83.3%), mean loop-tokens 23957.
- tfidf_quality_length_cost: shuffled accuracy 79.7% (interval 77.5% to 81.8%), mean loop-tokens 22768.
- semantic_quality_length_cost: shuffled accuracy 81.1% (interval 79.2% to 83.8%), mean loop-tokens 23828.

## Encoder overhead

Hardware Apple M5, device cpu. Model load 1762 ms. First batch-1 encode after load 3.2 ms. Warm batch-1, tokenization included, p50 3.0 ms, p95 3.1 ms. These milliseconds are not loop-tokens and are not a serving claim.

## Conclusion

The frozen MiniLM router is inconclusive. At the precommitted 0.9 budget it scores 82.5% at 24052 loop-tokens. TF-IDF scores 81.2% at 22489, length scores 77.2% at 19832, and the training-only blind mixture scores 84.3% at 26354. Fixed 3 loops scores 87.0% at 29309. Semantic equal-cost surplus over the retrospective blind frontier has interval -0.0233 to 0.0281. Semantic surplus minus TF-IDF surplus has interval -0.0346 to 0.0313. The gap is small or the interval covers zero, so this pilot does not show that question text allocates loops better than the controls. The single next experiment is not a search over encoders on these same questions. Keep the confirmation reserve sealed.

Artifacts: `artifacts/looped_serving/semantic_router/pilot_v1/`.
