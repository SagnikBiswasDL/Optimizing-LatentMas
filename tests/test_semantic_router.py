"""Routing rules that do not download a sentence-transformer model."""
import numpy as np

from looped.routing_evaluation import (
    bootstrap_intervals,
    evaluate_routers,
    fit_representation,
    frontier_accuracy,
    mixture_weights,
    select_lambda,
    stratified_group_folds,
)
from looped.semantic_router import (
    TfidfBuilder,
    cache_fingerprint,
    choose_depth,
    chunk_content_ids,
    combine_window_vectors,
    embed_from_content_ids,
    fit_cost_head,
    fit_quality_head,
    loop_tokens_from_scaled,
)


DEPTHS = [1, 2, 3, 4]


def test_groups_stay_inside_one_fold_and_levels_are_balanced():
    levels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    groups = ["a", "a", "b", "c", "d", "d", "e", "f"]
    folds = stratified_group_folds(levels, groups, 2, seed=7)
    seen = []
    for train, test in folds:
        assert set(np.array(groups)[train]).isdisjoint(set(np.array(groups)[test]))
        seen.extend(test.tolist())
        assert abs(np.sum(levels[test] == 1) - np.sum(levels[test] == 2)) <= 1
    assert sorted(seen) == list(range(8))
    again = stratified_group_folds(levels, groups, 2, seed=7)
    assert all(np.array_equal(a, b) and np.array_equal(c, d) for (a, c), (b, d) in zip(folds, again))


def test_choose_depth_can_return_every_action_and_breaks_ties_deterministically():
    eye = np.eye(4)
    costs = np.ones((4, 4))
    assert choose_depth(eye, costs, 0.0, DEPTHS).tolist() == [0, 1, 2, 3]
    tied = np.array([[0.5, 0.5, 0.2, 0.2]])
    cheaper_second = np.array([[100.0, 40.0, 100.0, 100.0]])
    assert int(choose_depth(tied, cheaper_second, 0.0, DEPTHS)[0]) == 1
    flat = np.ones((1, 4))
    assert int(choose_depth(flat, np.full((1, 4), 10.0), 0.0, DEPTHS)[0]) == 0


def test_one_class_constant_cost_and_negative_cost_are_finite():
    model, info = fit_quality_head(np.ones((4, 2)), np.ones((4, 2)), C=1.0)
    assert info["constant"] is True
    assert abs(model.probability - (5 / 6)) < 1e-12
    zeros, zero_info = fit_quality_head(np.ones((4, 2)), np.zeros((4, 2)), C=1.0)
    assert zero_info["constant"] is True and abs(zeros.probability - (1 / 6)) < 1e-12
    cost, cost_info = fit_cost_head(np.random.default_rng(0).normal(size=(5, 3)), np.full(5, 0.5), 1.0)
    assert cost_info["constant"] is True
    assert np.allclose(cost.predict(np.zeros((2, 3))), 0.5)
    tokens, hit = loop_tokens_from_scaled(np.array([-0.5, 0.2]), 10000)
    assert tokens[0] == 1.0 and bool(hit[0]) and tokens[1] == 2000.0


def test_tfidf_ignores_a_validation_only_word_and_does_not_refit_scaling():
    texts = ["alpha beta", "alpha beta", "alpha beta extra", "alpha beta extra", "alpha gammaunique"]
    builder = TfidfBuilder(texts, min_df=2)
    builder.fit(np.array([0, 1, 2, 3]))
    assert "gammaunique" not in builder.vectorizer.vocabulary_
    assert "alpha" in builder.vectorizer.vocabulary_
    mean = builder.scaler.mean_.copy()
    builder.transform(np.array([4]))
    assert np.allclose(builder.scaler.mean_, mean)


def test_chunking_keeps_every_token_and_short_inputs_use_the_normal_vector():
    ids = list(range(20))
    chunks = chunk_content_ids(ids, 6)
    assert [token for chunk in chunks for token in chunk] == ids
    assert max(len(chunk) for chunk in chunks) <= 6
    assert chunk_content_ids(ids[:4], 6) == [ids[:4]]
    pooled = np.array([[3.0, 0.0], [0.0, 4.0]])
    assert np.allclose(combine_window_vectors(pooled, np.array([1.0, 3.0])), [0.25, 0.75])

    def encode_short():
        return np.array([3.0, 4.0])

    final, raw, n_windows, _chunks = embed_from_content_ids([1, 2], 10, encode_short, lambda _c: None)
    assert n_windows == 1
    assert np.allclose(raw, [3.0, 4.0])
    assert np.allclose(final, [0.6, 0.8])
    seen = {}

    def encode_windows(windows):
        seen["windows"] = windows
        return np.ones((len(windows), 2))

    final, _raw, n_windows, kept = embed_from_content_ids(list(range(5)), 2, encode_short, encode_windows)
    assert n_windows == 3 and [token for chunk in kept for token in chunk] == list(range(5))
    assert seen["windows"] == kept
    assert abs(np.linalg.norm(final) - 1.0) < 1e-12


def test_cache_fingerprint_changes_with_text_or_revision():
    base = dict(encoder="sentence-transformers/all-MiniLM-L6-v2", revision="abc", text_hash="1",
                max_seq_length=256, tokenizer="bert")
    assert cache_fingerprint(**base) != cache_fingerprint(**{**base, "text_hash": "2"})
    assert cache_fingerprint(**base) != cache_fingerprint(**{**base, "revision": "def"})


def test_blind_mixture_and_frontier_match_a_hand_example():
    accuracy = np.array([0.2, 0.5, 0.8, 0.8])
    cost = np.array([10.0, 20.0, 30.0, 40.0])
    assert mixture_weights(accuracy, cost, 5.0) is None
    assert frontier_accuracy(accuracy, cost, 5.0) is None
    assert np.allclose(mixture_weights(accuracy, cost, 10.0), [1, 0, 0, 0])
    assert np.allclose(mixture_weights(accuracy, cost, 15.0), [0.75, 0.0, 0.25, 0.0])
    assert abs(frontier_accuracy(accuracy, cost, 15.0) - 0.35) < 1e-12
    weights = mixture_weights(accuracy, cost, 25.0)
    assert np.allclose(weights, [0.25, 0, 0.75, 0])
    assert abs(frontier_accuracy(accuracy, cost, 25.0) - 0.65) < 1e-12
    assert np.allclose(mixture_weights(accuracy, cost, 100.0), [0, 0, 1, 0])


def _pilot(n_per_level=8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per_level * 2
    levels = np.array([1] * n_per_level + [2] * n_per_level)
    texts = [f"question {i} alpha beta shared" for i in range(n)]
    vote = np.zeros((n, 2, 4))
    cost = np.zeros((n, 2, 4))
    for d, base in enumerate((0.2, 0.5, 0.7, 0.75)):
        vote[:, :, d] = rng.random((n, 2)) < base
        cost[:, :, d] = (d + 1) * 1000 + rng.normal(0, 10, size=(n, 2))
    embeddings = rng.normal(size=(n, 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    class Pilot:
        pass

    pilot = Pilot()
    pilot.question_ids = [f"q{i}" for i in range(n)]
    pilot.group_ids = list(pilot.question_ids)
    pilot.texts = texts
    pilot.levels = levels
    pilot.subjects = ["Algebra"] * n
    pilot.text_hashes = [f"h{i}" for i in range(n)]
    pilot.depths = (1, 2, 3, 4)
    pilot.run_seeds = (0, 1)
    pilot.vote_correct = vote
    pilot.vote_correct_strict = vote.copy()
    pilot.loop_tokens = cost
    pilot.n = n
    return pilot, embeddings


def _config():
    return {
        "outer_folds": 2,
        "inner_folds": 2,
        "split_seed": 20261114,
        "logistic_C_grid": [1.0],
        "ridge_alpha_grid": [1.0],
        "cost_unit_loop_tokens": 10000,
        "lambda_grid": [0.0, 1.0],
        "budget_ratios": [1.0],
        "budget_reference_depth": 3,
        "logistic_max_iter": 200,
        "tfidf_min_df": 1,
    }


def test_held_out_labels_do_not_change_the_fit_or_the_action():
    pilot, embeddings = _pilot()
    folds = stratified_group_folds(pilot.levels, pilot.group_ids, 2, 20261114)
    train, test = folds[0]
    config = _config()
    first = fit_representation(pilot, config, train, test, "length", None)
    decision = select_lambda(first["oof_p"], first["oof_cost"], pilot.vote_correct[train],
                             pilot.loop_tokens[train], config["lambda_grid"], budget=1.0e12,
                             depths=pilot.depths, unit=10000)
    actions = choose_depth(first["test_p"], first["test_cost"], decision["lambda"], pilot.depths)
    pilot.vote_correct[test] = 1.0 - pilot.vote_correct[test]
    pilot.loop_tokens[test] += 5000
    second = fit_representation(pilot, config, train, test, "length", None)
    decision2 = select_lambda(second["oof_p"], second["oof_cost"], pilot.vote_correct[train],
                              pilot.loop_tokens[train], config["lambda_grid"], budget=1.0e12,
                              depths=pilot.depths, unit=10000)
    actions2 = choose_depth(second["test_p"], second["test_cost"], decision2["lambda"], pilot.depths)
    assert first["C"] == second["C"]
    assert first["alpha"] == second["alpha"]
    assert decision["lambda"] == decision2["lambda"]
    assert np.allclose(first["test_p"], second["test_p"])
    assert np.allclose(first["test_cost"], second["test_cost"])
    assert np.array_equal(actions, actions2)


def test_tiny_router_runs_offline_with_a_fake_embedding():
    pilot, embeddings = _pilot()
    result = evaluate_routers(pilot, _config(), embeddings)
    assert len(result["folds"]) == 2
    covered = sorted(i for _train, test in result["folds"] for i in test)
    assert covered == list(range(pilot.n))
    semantic = [row for row in result["predictions"] if row["method"] == "semantic"]
    assert len(semantic) == pilot.n
    assert len(semantic[0]["p_correct"]) == 4
    assert semantic[0]["depths"] == DEPTHS
    assert "vote_correct" not in semantic[0]


def test_bootstrap_reuses_question_resamples_across_methods():
    pilot, _embeddings = _pilot(n_per_level=4, seed=1)
    correct = pilot.vote_correct.mean(axis=1)
    spent = pilot.loop_tokens.mean(axis=1)
    summaries = {}
    for name, column in (("semantic", 3), ("tfidf", 1)):
        summaries[name] = {
            "defined": True,
            "accuracy": float(correct[:, column].mean()),
            "mean_loop_tokens": float(spent[:, column].mean()),
            "per_question_correct": pilot.vote_correct[:, :, column],
            "per_question_cost": pilot.loop_tokens[:, :, column],
        }
    first = bootstrap_intervals(pilot, summaries, n_boot=30, seed=20261115)
    second = bootstrap_intervals(pilot, summaries, n_boot=30, seed=20261115)
    assert first["methods"]["semantic"]["accuracy"] == second["methods"]["semantic"]["accuracy"]
    gap = first["semantic_minus_tfidf_surplus"]
    assert gap["n_defined"] == 30
    # A question resample moves both runs together: the per-question means are what the interval uses.
    assert summaries["semantic"]["per_question_correct"].shape == (pilot.n, 2)
