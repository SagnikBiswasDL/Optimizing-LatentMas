import numpy as np

from looped import loop_select as ls


def test_vote_ties_go_to_lowest_sample_and_none_casts_no_vote():
    assert ls.vote_key([(0, "a"), (1, "b"), (2, "b"), (3, "a")]) == "a"
    assert ls.vote_key([(0, None), (1, "b"), (2, None), (3, None)]) == "b"
    assert ls.vote_key([(0, None), (1, None)]) is None
    assert ls.vote_correct([(0, "a", False), (1, "b", True), (2, "b", True), (3, "a", False)]) is False
    assert ls.vote_correct([(0, None, False), (1, None, False)]) is False


def test_policy_from_probs_picks_smallest_depth_within_tau():
    P = np.array([[0.40, 0.85, 0.90, 0.90], [0.9, 0.9, 0.9, 0.9], [0.1, 0.2, 0.3, 0.9]])
    assert ls.policy_from_probs(P, 0.05).tolist() == [1, 0, 3]
    assert ls.policy_from_probs(P, 0.0).tolist() == [2, 0, 3]
    assert ls.policy_from_probs(P, 1.0).tolist() == [0, 0, 0]


def test_frontier_and_best_mixture():
    acc = [0.30, 0.70, 0.75, 0.74]
    cost = [100.0, 200.0, 300.0, 400.0]
    fr = ls.mixture_frontier(acc, cost)
    assert fr == [(100.0, 0.30), (200.0, 0.70), (300.0, 0.75)]
    assert abs(ls.best_mixture_acc(fr, 150.0) - 0.50) < 1e-12
    assert abs(ls.best_mixture_acc(fr, 250.0) - 0.725) < 1e-12
    assert ls.best_mixture_acc(fr, 1000.0) == 0.75
    assert ls.best_mixture_acc(fr, 50.0) == 0.30


def test_matched_random_beats_nothing_when_labels_are_independent_of_policy():
    rng = np.random.default_rng(0)
    C = rng.integers(0, 2, size=(400, 4))
    cost = np.tile([1.0, 2.0, 3.0, 4.0], (400, 1))
    pol = rng.integers(0, 4, size=400)
    mr = ls.matched_random(pol, C, cost, n_perm=500)
    assert 0.0 < mr["p_value"] <= 1.0
    assert abs(mr["cost"] - ls.evaluate(pol, C, cost)[1]) < 1e-9
    oracle = ls.oracle_policy(C, cost)
    assert ls.matched_random(oracle, C, cost, n_perm=200)["p_value"] < 0.01


def test_mcnemar_and_kappa():
    a = np.array([1] * 8 + [0] * 5 + [1] * 50)
    b = np.array([0] * 8 + [1] * 5 + [1] * 50)
    m = ls.mcnemar_exact(a, b)
    assert (m["only_a"], m["only_b"]) == (8, 5) and abs(m["p"] - 0.5811) < 1e-3
    x = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    assert ls.cohen_kappa(x, x) == 1.0
    assert ls.cohen_kappa(x, 1 - x) < 0


def test_repeatability_gate_passes_on_stable_signal_and_fails_on_noise():
    rng = np.random.default_rng(1)
    n = 400
    hard = rng.random(n) < 0.3                      # these questions need the third loop
    base = np.ones((n, 4), int)
    base[:, 0] = 0
    base[hard, 1] = 0
    CA, CB = base.copy(), base.copy()
    flip = rng.random((n, 4)) < 0.03
    CB[flip] = 1 - CB[flip]
    cost = np.tile([1.0, 2.0, 3.0, 4.0], (n, 1))
    g = ls.repeatability_gate(CA, CB, cost, cost, steps=(1,), n_boot=300)
    assert g["pass"], g
    noiseA = (rng.random((n, 4)) < np.array([0.3, 0.7, 0.75, 0.75])).astype(int)
    noiseB = (rng.random((n, 4)) < np.array([0.3, 0.7, 0.75, 0.75])).astype(int)
    assert not ls.repeatability_gate(noiseA, noiseB, cost, cost, steps=(1,), n_boot=300)["pass"]
