"""Nested, question-grouped evaluation for the semantic loop router.

Hyperparameters and the utility penalty are chosen on inner folds of the
outer-training questions. Held-out questions are scored after those choices
are frozen. The retrospective mixture frontier is not used to choose anything.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from looped.semantic_router import (
    choose_depth,
    fit_cost_head,
    fit_quality_head,
    loop_tokens_from_scaled,
    make_builder,
    predict_quality,
    question_brier,
    question_log_loss,
)

METHODS = ("length", "tfidf", "semantic")
ATTRIBUTION = (
    ("tfidf_quality_length_cost", "tfidf", "length"),
    ("semantic_quality_length_cost", "semantic", "length"),
)


def stratified_group_folds(levels: np.ndarray, groups: Sequence[str], n_splits: int,
                           seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Level-stratified folds that keep every row of a group on one side.

    Within each level, groups are shuffled with ``seed`` and dealt round-robin.
    A group must have a single level. This is the grouped-stratification contract
    used instead of retuning a library split after the fact.
    """
    levels = np.asarray(levels)
    groups = [str(group) for group in groups]
    if len(levels) != len(groups):
        raise ValueError("levels and groups have different lengths")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    level_of: Dict[str, int] = {}
    for level, group in zip(levels.tolist(), groups):
        previous = level_of.get(group)
        if previous is not None and previous != int(level):
            raise ValueError(f"group {group} spans more than one level")
        level_of[group] = int(level)
    by_level: Dict[int, List[str]] = {}
    for group, level in level_of.items():
        by_level.setdefault(level, []).append(group)
    rng = np.random.default_rng(seed)
    fold_of: Dict[str, int] = {}
    for level in sorted(by_level):
        members = np.array(sorted(by_level[level]))
        members = members[rng.permutation(len(members))]
        if len(members) < n_splits:
            raise ValueError(f"level {level} has {len(members)} groups, fewer than {n_splits} folds")
        for position, group in enumerate(members.tolist()):
            fold_of[group] = position % n_splits
    folds = []
    group_arr = np.array(groups)
    for fold in range(n_splits):
        test = np.flatnonzero(np.array([fold_of[group] == fold for group in groups]))
        train = np.flatnonzero(np.array([fold_of[group] != fold for group in groups]))
        if len(test) == 0 or len(train) == 0:
            raise ValueError(f"fold {fold} is empty")
        if set(group_arr[test]) & set(group_arr[train]):
            raise RuntimeError("a group crossed a fold boundary")
        folds.append((np.sort(train), np.sort(test)))
    return folds


def mixture_weights(accuracy: np.ndarray, cost: np.ndarray, budget: float) -> Optional[np.ndarray]:
    """Best question-independent weights at training cost ``<= budget``.

    The optimum is at a vertex of the simplex cut by the budget, so it is enough
    to check feasible depths and the budget boundary on each edge. Returns None
    when even the cheapest depth exceeds the budget.
    """
    accuracy = np.asarray(accuracy, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n_actions = len(accuracy)
    if cost.min() > budget + 1e-6:
        return None
    candidates = []

    def consider(weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.min() < -1e-8 or abs(weights.sum() - 1.0) > 1e-6:
            return
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()
        realized = float(weights @ cost)
        if realized > budget + 1e-6:
            return
        candidates.append((float(weights @ accuracy), -realized, tuple(np.round(weights, 10)), weights))

    for action in range(n_actions):
        if cost[action] <= budget + 1e-6:
            one = np.zeros(n_actions)
            one[action] = 1.0
            consider(one)
    for left, right in itertools.combinations(range(n_actions), 2):
        gap = cost[left] - cost[right]
        if abs(gap) < 1e-9:
            continue
        if (cost[left] - budget) * (cost[right] - budget) >= 0:
            continue
        weight_left = (budget - cost[right]) / gap
        weights = np.zeros(n_actions)
        weights[left] = weight_left
        weights[right] = 1.0 - weight_left
        consider(weights)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def frontier_accuracy(accuracy: np.ndarray, cost: np.ndarray, budget: float) -> Optional[float]:
    """Retrospective best blind accuracy at cost ``<= budget``. Infeasible below the cheapest depth."""
    weights = mixture_weights(accuracy, cost, budget)
    if weights is None:
        return None
    return float(weights @ np.asarray(accuracy, dtype=np.float64))


def _gather_depth(values: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """``values`` is (n, runs, depths). ``actions`` is one depth index per question."""
    n, n_runs, _ = values.shape
    return values[np.arange(n)[:, None], np.arange(n_runs)[None, :], actions[:, None]]


def scored_outcomes(values: np.ndarray, depth_index: np.ndarray, weights: np.ndarray,
                    is_mixture: np.ndarray) -> np.ndarray:
    """Per-question, per-run outcome of a depth assignment or a blind mixture."""
    n, n_runs, n_depths = values.shape
    out = np.zeros((n, n_runs), dtype=np.float64)
    depth_rows = np.flatnonzero(~is_mixture)
    mix_rows = np.flatnonzero(is_mixture)
    if len(depth_rows):
        out[depth_rows] = _gather_depth(values[depth_rows], depth_index[depth_rows])
    if len(mix_rows):
        out[mix_rows] = (values[mix_rows] * weights[mix_rows][:, None, :]).sum(axis=2)
    return out


def _pool_question_scores(pieces: List[np.ndarray]) -> float:
    if not pieces:
        raise ValueError("no validation questions")
    return float(np.concatenate(pieces).mean())


def _fit_cached(builder, train_idx: np.ndarray, inner: Sequence[Tuple[np.ndarray, np.ndarray]]):
    cached = []
    for inner_train, inner_val in inner:
        global_train = train_idx[inner_train]
        global_val = train_idx[inner_val]
        builder.fit(global_train)
        cached.append((inner_train, inner_val, builder.transform(global_train), builder.transform(global_val)))
    return cached


def fit_representation(pilot, config: dict, train_idx: np.ndarray, test_idx: np.ndarray,
                       method: str, embeddings: Optional[np.ndarray]) -> dict:
    """Select C and alpha on inner folds, then refit and predict the outer test questions.

    ``test_idx`` is used only as the rows to transform after the refit. Test labels
    are not arguments to this function.
    """
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    y = pilot.vote_correct
    cost = pilot.loop_tokens
    depths = list(pilot.depths)
    unit = float(config["cost_unit_loop_tokens"])
    builder = make_builder(method, pilot.texts, embeddings, min_df=int(config.get("tfidf_min_df", 2)))
    inner = stratified_group_folds(
        pilot.levels[train_idx], [pilot.group_ids[int(i)] for i in train_idx],
        int(config["inner_folds"]), int(config["split_seed"]))
    cached = _fit_cached(builder, train_idx, inner)
    max_iter = int(config.get("logistic_max_iter", 2000))
    chosen_C = []
    chosen_alpha = []
    quality_notes = []
    cost_notes = []
    local_y = y[train_idx]
    local_cost = cost[train_idx]
    mean_scaled = local_cost.mean(axis=1) / unit

    for depth_index in range(len(depths)):
        best_C, best_loss = None, None
        for C in config["logistic_C_grid"]:
            pieces = []
            for _tr, _va, X_train, X_val in cached:
                model, _info = fit_quality_head(X_train, local_y[_tr, :, depth_index], float(C), max_iter)
                pieces.append(question_log_loss(local_y[_va, :, depth_index], predict_quality(model, X_val)))
            loss = _pool_question_scores(pieces)
            if best_loss is None or loss < best_loss - 1e-12 or (
                    abs(loss - best_loss) <= 1e-12 and float(C) < best_C):
                best_loss, best_C = loss, float(C)
        best_alpha, best_mse = None, None
        for alpha in config["ridge_alpha_grid"]:
            pieces = []
            for _tr, va, X_train, X_val in cached:
                model, _info = fit_cost_head(X_train, mean_scaled[_tr, depth_index], float(alpha))
                pred = np.asarray(model.predict(X_val), dtype=np.float64)
                pieces.append(np.square(pred - mean_scaled[va, depth_index]))
            mse = _pool_question_scores(pieces)
            if best_mse is None or mse < best_mse - 1e-12 or (
                    abs(mse - best_mse) <= 1e-12 and float(alpha) > best_alpha):
                best_mse, best_alpha = mse, float(alpha)
        chosen_C.append(best_C)
        chosen_alpha.append(best_alpha)
        quality_notes.append({"selected_log_loss": best_loss})
        cost_notes.append({"selected_mse": best_mse})

    oof_p = np.zeros((len(train_idx), len(depths)), dtype=np.float64)
    oof_scaled = np.zeros_like(oof_p)
    oof_info = []
    for inner_train, inner_val, X_train, X_val in cached:
        for depth_index in range(len(depths)):
            model, info = fit_quality_head(
                X_train, local_y[inner_train, :, depth_index], chosen_C[depth_index], max_iter)
            oof_p[inner_val, depth_index] = predict_quality(model, X_val)
            cost_model, cost_info = fit_cost_head(
                X_train, mean_scaled[inner_train, depth_index], chosen_alpha[depth_index])
            oof_scaled[inner_val, depth_index] = np.asarray(cost_model.predict(X_val), dtype=np.float64)
            oof_info.append({"depth_index": depth_index, "quality": info, "cost": cost_info})
    oof_cost, oof_floor = loop_tokens_from_scaled(oof_scaled, unit)

    builder.fit(train_idx)
    X_train = builder.transform(train_idx)
    X_test = builder.transform(test_idx)
    test_p = np.zeros((len(test_idx), len(depths)), dtype=np.float64)
    test_scaled = np.zeros_like(test_p)
    refit_info = []
    train_mean_scaled = np.zeros(len(depths), dtype=np.float64)
    for depth_index in range(len(depths)):
        model, info = fit_quality_head(X_train, local_y[:, :, depth_index], chosen_C[depth_index], max_iter)
        test_p[:, depth_index] = predict_quality(model, X_test)
        cost_model, cost_info = fit_cost_head(X_train, mean_scaled[:, depth_index], chosen_alpha[depth_index])
        test_scaled[:, depth_index] = np.asarray(cost_model.predict(X_test), dtype=np.float64)
        train_mean_scaled[depth_index] = float(mean_scaled[:, depth_index].mean())
        refit_info.append({"depth_index": depth_index, "quality": info, "cost": cost_info})
    test_cost, test_floor = loop_tokens_from_scaled(test_scaled, unit)
    return {
        "method": method,
        "C": chosen_C,
        "alpha": chosen_alpha,
        "oof_p": oof_p,
        "oof_cost": oof_cost,
        "test_p": test_p,
        "test_cost": test_cost,
        "train_mean_scaled_cost": train_mean_scaled,
        "floor_hits_oof": int(oof_floor.sum()),
        "floor_hits_test": int(test_floor.sum()),
        "n_test": int(len(test_idx)),
        "quality_selection": quality_notes,
        "cost_selection": cost_notes,
        "refit": refit_info,
        "oof_fit": oof_info,
        "empty_tfidf_vocabulary": bool(getattr(builder, "empty_vocabulary", False)),
    }


def select_lambda(oof_p: np.ndarray, oof_cost: np.ndarray, y_train: np.ndarray, c_train: np.ndarray,
                  lambdas: Sequence[float], budget: float, depths: Sequence[int], unit: float) -> dict:
    """Pick lambda on inner out-of-fold predictions. Training outcomes only."""
    curve = []
    for lam in lambdas:
        actions = choose_depth(oof_p, oof_cost, float(lam), depths, unit)
        chosen_y = _gather_depth(y_train, actions)
        chosen_c = _gather_depth(c_train, actions)
        curve.append({
            "lambda": float(lam),
            "accuracy": float(chosen_y.mean()),
            "mean_loop_tokens": float(chosen_c.mean()),
            "feasible": bool(chosen_c.mean() <= budget + 1e-6),
        })
    feasible = [row for row in curve if row["feasible"]]
    mean_cost = c_train.mean(axis=(0, 1))
    cheapest = float(mean_cost.min())
    if cheapest > budget + 1e-6:
        return {"status": "infeasible", "lambda": None, "curve": curve, "budget": float(budget),
                "cheapest_training_cost": cheapest}
    if not feasible:
        weights = mixture_weights(y_train.mean(axis=(0, 1)), mean_cost, budget)
        if weights is None:
            raise RuntimeError("the cheapest training depth is inside the budget but no mixture was found")
        return {"status": "mixture_fallback", "lambda": None, "weights": weights.tolist(),
                "curve": curve, "budget": float(budget), "cheapest_training_cost": cheapest}
    feasible.sort(key=lambda row: (-row["accuracy"], row["mean_loop_tokens"], -row["lambda"]))
    winner = feasible[0]
    return {"status": "lambda", "lambda": winner["lambda"], "curve": curve, "budget": float(budget),
            "inner_accuracy": winner["accuracy"], "inner_mean_loop_tokens": winner["mean_loop_tokens"],
            "cheapest_training_cost": cheapest}


def _empty_assignment(n: int, n_depths: int) -> dict:
    return {
        "depth_index": np.zeros(n, dtype=np.int64),
        "weights": np.zeros((n, n_depths), dtype=np.float64),
        "is_mixture": np.zeros(n, dtype=bool),
        "defined": np.ones(n, dtype=bool),
    }


def assign_fold(decision: dict, test_idx: np.ndarray, test_p: np.ndarray, test_cost: np.ndarray,
                assignment: dict, depths: Sequence[int], unit: float) -> None:
    if decision["status"] == "infeasible":
        assignment["defined"][test_idx] = False
        return
    if decision["status"] == "mixture_fallback":
        weights = np.asarray(decision["weights"], dtype=np.float64)
        assignment["is_mixture"][test_idx] = True
        assignment["weights"][test_idx] = weights
        return
    actions = choose_depth(test_p, test_cost, decision["lambda"], depths, unit)
    assignment["depth_index"][test_idx] = actions
    assignment["is_mixture"][test_idx] = False


def training_mixture_assignment(pilot, folds, config: dict) -> Dict[str, dict]:
    """Freeze a blind mixture on each outer-training split, for every budget ratio."""
    out = {}
    for rho in config["budget_ratios"]:
        assignment = _empty_assignment(pilot.n, len(pilot.depths))
        fold_info = []
        defined = True
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            y = pilot.vote_correct[train_idx]
            c = pilot.loop_tokens[train_idx]
            reference = list(pilot.depths).index(int(config["budget_reference_depth"]))
            budget = float(rho) * float(c[:, :, reference].mean())
            weights = mixture_weights(y.mean(axis=(0, 1)), c.mean(axis=(0, 1)), budget)
            if weights is None:
                defined = False
                assignment["defined"][test_idx] = False
                fold_info.append({"fold": fold_id, "budget": budget, "status": "infeasible", "weights": None})
                continue
            assignment["is_mixture"][test_idx] = True
            assignment["weights"][test_idx] = weights
            fold_info.append({"fold": fold_id, "budget": budget, "status": "mixture", "weights": weights.tolist(),
                              "training_accuracy": (y.mean(axis=(0, 1))).tolist(),
                              "training_cost": (c.mean(axis=(0, 1))).tolist()})
        assignment["defined"][:] = defined and assignment["defined"].all()
        if not defined:
            assignment["defined"][:] = False
        out[str(rho)] = {"assignment": assignment, "folds": fold_info, "defined": defined}
    return out


def evaluate_routers(pilot, config: dict, embeddings: Optional[np.ndarray] = None,
                     folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None) -> dict:
    """Fit every method under the frozen config and return predictions plus selection records.

    The returned assignments are one depth, or one training mixture, per held-out question.
    """
    if folds is None:
        folds = stratified_group_folds(pilot.levels, pilot.group_ids, int(config["outer_folds"]),
                                       int(config["split_seed"]))
    _assert_partition(folds, pilot.n, pilot.group_ids)
    unit = float(config["cost_unit_loop_tokens"])
    depths = list(pilot.depths)
    fitted = {method: [] for method in METHODS}
    for train_idx, test_idx in folds:
        for method in METHODS:
            fitted[method].append(fit_representation(pilot, config, train_idx, test_idx, method, embeddings))

    assignments = {}
    selection = {}
    for method in METHODS:
        for rho in config["budget_ratios"]:
            key = f"{method}@{rho}"
            assignment = _empty_assignment(pilot.n, len(depths))
            fold_rows = []
            defined = True
            for fold_id, ((train_idx, test_idx), fit) in enumerate(zip(folds, fitted[method])):
                y_train = pilot.vote_correct[train_idx]
                c_train = pilot.loop_tokens[train_idx]
                reference = depths.index(int(config["budget_reference_depth"]))
                budget = float(rho) * float(c_train[:, :, reference].mean())
                decision = select_lambda(fit["oof_p"], fit["oof_cost"], y_train, c_train,
                                         config["lambda_grid"], budget, depths, unit)
                if decision["status"] == "infeasible":
                    defined = False
                assign_fold(decision, test_idx, fit["test_p"], fit["test_cost"], assignment, depths, unit)
                fold_rows.append({
                    "fold": fold_id,
                    "budget": budget,
                    "decision": {k: v for k, v in decision.items() if k != "curve"},
                    "lambda_curve": decision["curve"],
                    "C": fit["C"],
                    "alpha": fit["alpha"],
                    "floor_hits_test": fit["floor_hits_test"],
                    "floor_hits_oof": fit["floor_hits_oof"],
                    "empty_tfidf_vocabulary": fit["empty_tfidf_vocabulary"],
                    "refit": fit["refit"],
                    "n_test": int(len(test_idx)),
                })
            if not defined:
                assignment["defined"][:] = False
            assignments[key] = assignment
            selection[key] = {"defined": defined, "folds": fold_rows}

    for name, quality_method, cost_method in ATTRIBUTION:
        for rho in config["budget_ratios"]:
            key = f"{name}@{rho}"
            assignment = _empty_assignment(pilot.n, len(depths))
            fold_rows = []
            defined = True
            for fold_id, (train_idx, test_idx) in enumerate(folds):
                quality = fitted[quality_method][fold_id]
                cost_fit = fitted[cost_method][fold_id]
                y_train = pilot.vote_correct[train_idx]
                c_train = pilot.loop_tokens[train_idx]
                reference = depths.index(int(config["budget_reference_depth"]))
                budget = float(rho) * float(c_train[:, :, reference].mean())
                decision = select_lambda(quality["oof_p"], cost_fit["oof_cost"], y_train, c_train,
                                         config["lambda_grid"], budget, depths, unit)
                if decision["status"] == "infeasible":
                    defined = False
                assign_fold(decision, test_idx, quality["test_p"], cost_fit["test_cost"],
                            assignment, depths, unit)
                fold_rows.append({
                    "fold": fold_id,
                    "budget": budget,
                    "decision": {k: v for k, v in decision.items() if k != "curve"},
                    "reused_quality": quality_method,
                    "reused_cost": cost_method,
                    "quality_C": quality["C"],
                    "cost_alpha": cost_fit["alpha"],
                })
            if not defined:
                assignment["defined"][:] = False
            assignments[key] = assignment
            selection[key] = {"defined": defined, "folds": fold_rows}

    mixtures = training_mixture_assignment(pilot, folds, config)
    for rho, payload in mixtures.items():
        assignments[f"mixture@{rho}"] = payload["assignment"]
        selection[f"mixture@{rho}"] = {"defined": payload["defined"], "folds": payload["folds"]}

    predictions = _prediction_rows(pilot, folds, fitted, assignments, config)
    return {
        "folds": [(train.tolist(), test.tolist()) for train, test in folds],
        "fitted_summary": {
            method: [{"C": fit["C"], "alpha": fit["alpha"],
                      "floor_hits_test": fit["floor_hits_test"],
                      "empty_tfidf_vocabulary": fit["empty_tfidf_vocabulary"]}
                     for fit in rows]
            for method, rows in fitted.items()
        },
        "selection": _jsonable_selection(selection),
        "assignments": assignments,
        "predictions": predictions,
        "test_predictions": _stacked_test_predictions(pilot, folds, fitted),
    }


def _assert_partition(folds, n: int, groups: Sequence[str]) -> None:
    seen = np.zeros(n, dtype=np.int64)
    for train, test in folds:
        if set(train) & set(test):
            raise RuntimeError("train and test overlap")
        seen[test] += 1
        train_groups = {groups[int(i)] for i in train}
        test_groups = {groups[int(i)] for i in test}
        if train_groups & test_groups:
            raise RuntimeError("a group appears on both sides of a fold")
    if not np.array_equal(seen, np.ones(n, dtype=np.int64)):
        raise RuntimeError("outer folds are not a partition of the questions")


def _stacked_test_predictions(pilot, folds, fitted) -> dict:
    """Held-out quality and cost predictions from the refit heads, one row per question."""
    stacked = {}
    for method, fits in fitted.items():
        p = np.zeros((pilot.n, len(pilot.depths)), dtype=np.float64)
        cost = np.zeros_like(p)
        for (_train, test), fit in zip(folds, fits):
            p[test] = fit["test_p"]
            cost[test] = fit["test_cost"]
        stacked[method] = {"p_correct": p, "expected_loop_tokens": cost}
    return stacked


def _prediction_rows(pilot, folds, fitted, assignments, config) -> List[dict]:
    fold_of = np.empty(pilot.n, dtype=np.int64)
    for fold_id, (_train, test) in enumerate(folds):
        fold_of[test] = fold_id
    rows = []
    depths = [int(depth) for depth in pilot.depths]
    for method, preds in _stacked_test_predictions(pilot, folds, fitted).items():
        for i, qid in enumerate(pilot.question_ids):
            choices = {}
            for rho in config["budget_ratios"]:
                key = f"{method}@{rho}"
                if key not in assignments:
                    continue
                assignment = assignments[key]
                if not assignment["defined"][i]:
                    choices[str(rho)] = {"status": "infeasible"}
                elif assignment["is_mixture"][i]:
                    choices[str(rho)] = {"status": "mixture_fallback",
                                         "weights": assignment["weights"][i].tolist()}
                else:
                    index = int(assignment["depth_index"][i])
                    choices[str(rho)] = {"status": "depth", "depth_index": index, "depth": depths[index]}
            rows.append({
                "question_id": qid,
                "fold": int(fold_of[i]),
                "method": method,
                "depths": depths,
                "p_correct": preds["p_correct"][i].tolist(),
                "expected_loop_tokens": preds["expected_loop_tokens"][i].tolist(),
                "choices": choices,
            })
    return rows


def _jsonable_selection(selection: dict) -> dict:
    def convert(value):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value
    return convert(selection)


def fixed_depth_assignment(n: int, n_depths: int, depth_index: int) -> dict:
    assignment = _empty_assignment(n, n_depths)
    assignment["depth_index"][:] = depth_index
    return assignment


def summarize_assignment(pilot, assignment: dict, *, label: str, vote: Optional[np.ndarray] = None) -> dict:
    """Accuracy and realized loop-tokens. Undefined when any question's fold was infeasible."""
    if vote is None:
        vote = pilot.vote_correct
    if not np.all(assignment["defined"]):
        return {"method": label, "defined": False}
    chosen_y = scored_outcomes(vote, assignment["depth_index"], assignment["weights"], assignment["is_mixture"])
    chosen_c = scored_outcomes(pilot.loop_tokens, assignment["depth_index"], assignment["weights"],
                               assignment["is_mixture"])
    counts = []
    for depth_index in range(len(pilot.depths)):
        hard = np.sum((~assignment["is_mixture"]) & (assignment["depth_index"] == depth_index))
        soft = float(assignment["weights"][:, depth_index][assignment["is_mixture"]].sum()) if assignment["is_mixture"].any() else 0.0
        counts.append(int(hard) + soft)
    per_run = {}
    for s, seed in enumerate(pilot.run_seeds):
        per_run[str(seed)] = {"accuracy": float(chosen_y[:, s].mean()), "mean_loop_tokens": float(chosen_c[:, s].mean())}
    return {
        "method": label,
        "defined": True,
        "accuracy": float(chosen_y.mean()),
        "mean_loop_tokens": float(chosen_c.mean()),
        "depth_counts": counts,
        "per_run": per_run,
        "per_question_correct": chosen_y,
        "per_question_cost": chosen_c,
    }


def calibration_table(pilot, test_predictions: dict) -> dict:
    """Held-out log loss, Brier score, and cost error. These do not choose the router."""
    tables = {}
    unit = 10000.0
    for method, preds in test_predictions.items():
        per_depth = []
        for d, depth in enumerate(pilot.depths):
            y = pilot.vote_correct[:, :, d]
            p = preds["p_correct"][:, d]
            observed = pilot.loop_tokens[:, :, d].mean(axis=1)
            predicted = preds["expected_loop_tokens"][:, d]
            mae = float(np.mean(np.abs(predicted - observed)))
            baseline = float(np.mean(np.abs(observed.mean() - observed)))
            per_depth.append({
                "depth": int(depth),
                "log_loss": float(question_log_loss(y, p).mean()),
                "brier": float(question_brier(y, p).mean()),
                "cost_mae": mae,
                "cost_mae_over_mean": float(mae / observed.mean()) if observed.mean() else None,
                "mean_predicted_loop_tokens": float(predicted.mean()),
                "mean_observed_loop_tokens": float(observed.mean()),
                "constant_cost_mae": baseline,
                "scaled_unit": unit,
            })
        tables[method] = per_depth
    return tables


def shuffled_assignment(assignment: dict, folds: Sequence[Tuple[np.ndarray, np.ndarray]],
                        repeats: int, seed: int) -> List[dict]:
    """Permute chosen depths inside each outer test fold. Mixture folds stay mixtures."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(repeats)):
        depth = assignment["depth_index"].copy()
        for _train, test in folds:
            hard = test[~assignment["is_mixture"][test]]
            if len(hard) > 1:
                depth[hard] = depth[hard][rng.permutation(len(hard))]
        shuffled = {
            "depth_index": depth,
            "weights": assignment["weights"],
            "is_mixture": assignment["is_mixture"],
            "defined": assignment["defined"],
        }
        draws.append(shuffled)
    return draws


def bootstrap_intervals(pilot, summaries: Dict[str, dict], n_boot: int, seed: int,
                        vote: Optional[np.ndarray] = None) -> dict:
    """Question-level intervals. The same resamples are used for every method.

    The retrospective frontier is rebuilt on each resample. Intervals describe the
    frozen predictions; they do not refit the heads.
    """
    if vote is None:
        vote = pilot.vote_correct
    rng = np.random.default_rng(seed)
    n = pilot.n
    draws = rng.integers(0, n, size=(int(n_boot), n))
    depth_acc = vote.mean(axis=1)
    depth_cost = pilot.loop_tokens.mean(axis=1)
    names = [name for name, summary in summaries.items() if summary.get("defined")]
    point = {}
    surplus = {name: np.empty(n_boot) for name in names}
    accuracy = {name: np.empty(n_boot) for name in names}
    cost = {name: np.empty(n_boot) for name in names}
    defined = {name: np.ones(n_boot, dtype=bool) for name in names}
    for name in names:
        correct = summaries[name]["per_question_correct"].mean(axis=1)
        spent = summaries[name]["per_question_cost"].mean(axis=1)
        for b in range(n_boot):
            idx = draws[b]
            accuracy[name][b] = float(correct[idx].mean())
            cost[name][b] = float(spent[idx].mean())
            front = frontier_accuracy(depth_acc[idx].mean(axis=0), depth_cost[idx].mean(axis=0), cost[name][b])
            if front is None:
                defined[name][b] = False
                surplus[name][b] = np.nan
            else:
                surplus[name][b] = accuracy[name][b] - front
        full_front = frontier_accuracy(depth_acc.mean(axis=0), depth_cost.mean(axis=0),
                                       summaries[name]["mean_loop_tokens"])
        point[name] = {
            "accuracy": summaries[name]["accuracy"],
            "mean_loop_tokens": summaries[name]["mean_loop_tokens"],
            "equal_cost_surplus": None if full_front is None else summaries[name]["accuracy"] - full_front,
            "frontier_accuracy": full_front,
        }
    gaps = None
    if "semantic" in surplus and "tfidf" in surplus:
        both = defined["semantic"] & defined["tfidf"]
        gap = surplus["semantic"] - surplus["tfidf"]
        gaps = _interval(gap[both]) if both.any() else None
        gaps = None if gaps is None else {**gaps, "n_defined": int(both.sum()), "n_boot": int(n_boot)}
    intervals = {}
    for name in names:
        mask = defined[name]
        intervals[name] = {
            "accuracy": _interval(accuracy[name]),
            "mean_loop_tokens": _interval(cost[name]),
            "equal_cost_surplus": _interval(surplus[name][mask]) if mask.any() else None,
            "n_defined_surplus": int(mask.sum()),
            "n_boot": int(n_boot),
            "point": point[name],
        }
    return {"methods": intervals, "semantic_minus_tfidf_surplus": gaps, "seed": int(seed), "n_boot": int(n_boot)}


def _interval(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "low": None, "high": None}
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def budget_miss(selection_folds: Sequence[dict], per_question_cost: np.ndarray, fold_tests: Sequence[np.ndarray]) -> dict:
    """Compare realized test cost with the budget that was fixed on that fold's training questions."""
    rows = []
    for fold, test_idx in zip(selection_folds, fold_tests):
        if "budget" not in fold:
            continue
        realized = float(per_question_cost[test_idx].mean())
        budget = float(fold["budget"])
        rows.append({
            "fold": fold.get("fold"),
            "budget": budget,
            "realized_mean_loop_tokens": realized,
            "realized_over_budget": realized / budget if budget else None,
            "over_budget": bool(realized > budget + 1e-6),
        })
    if not rows:
        return {"folds": []}
    return {
        "folds": rows,
        "mean_realized_over_budget": float(np.mean([row["realized_over_budget"] for row in rows])),
        "folds_over_budget": int(sum(row["over_budget"] for row in rows)),
    }


def reference_cost(pilot, config: dict) -> float:
    """Overall mean loop-tokens at the reference depth, for reporting cost ratios only."""
    index = list(pilot.depths).index(int(config["budget_reference_depth"]))
    return float(pilot.loop_tokens[:, :, index].mean())


def build_policy_table(pilot, config: dict, assignments: dict, selection: dict, folds) -> List[dict]:
    vote = pilot.vote_correct
    strict = pilot.vote_correct_strict
    ref_cost = reference_cost(pilot, config)
    depth_acc = vote.mean(axis=(0, 1))
    depth_cost = pilot.loop_tokens.mean(axis=(0, 1))
    rows = []
    labels = []
    for depth_index, depth in enumerate(pilot.depths):
        labels.append((f"fixed_{depth}", fixed_depth_assignment(pilot.n, len(pilot.depths), depth_index), None))
    for key, assignment in assignments.items():
        labels.append((key, assignment, selection.get(key)))
    fold_tests = [np.asarray(test, dtype=np.int64) for _train, test in folds]
    for label, assignment, meta in labels:
        summary = summarize_assignment(pilot, assignment, label=label, vote=vote)
        if not summary["defined"]:
            rows.append({"method": label, "defined": False})
            continue
        front = frontier_accuracy(depth_acc, depth_cost, summary["mean_loop_tokens"])
        strict_summary = summarize_assignment(pilot, assignment, label=label, vote=strict)
        rho = label.split("@")[-1] if "@" in label else None
        mixture_key = f"mixture@{rho}" if rho is not None else None
        mixture = None
        if mixture_key and mixture_key in assignments and label != mixture_key:
            mixture = summarize_assignment(pilot, assignments[mixture_key], label=mixture_key, vote=vote)
        miss = None
        if meta and meta.get("defined") and "folds" in meta:
            miss = budget_miss(meta["folds"], summary["per_question_cost"], fold_tests)
        row = {
            "method": label,
            "defined": True,
            "accuracy": summary["accuracy"],
            "mean_loop_tokens": summary["mean_loop_tokens"],
            "cost_over_fixed_reference": summary["mean_loop_tokens"] / ref_cost,
            "budget_miss": miss,
            "depth_counts": summary["depth_counts"],
            "per_run": summary["per_run"],
            "equal_cost_surplus": None if front is None else summary["accuracy"] - front,
            "retrospective_frontier_accuracy": front,
            "strict_cap_accuracy": strict_summary["accuracy"],
            "strict_cap_mean_loop_tokens": strict_summary["mean_loop_tokens"],
        }
        if mixture and mixture["defined"]:
            row["mixture_accuracy"] = mixture["accuracy"]
            row["mixture_mean_loop_tokens"] = mixture["mean_loop_tokens"]
            row["accuracy_minus_mixture"] = summary["accuracy"] - mixture["accuracy"]
        row["_per_question_correct"] = summary["per_question_correct"]
        row["_per_question_cost"] = summary["per_question_cost"]
        rows.append(row)
    return rows


def primary_summaries_for_bootstrap(table_rows: Sequence[dict]) -> Dict[str, dict]:
    wanted = {}
    for row in table_rows:
        method = row["method"]
        keep = method in {"length@0.9", "tfidf@0.9", "semantic@0.9",
                          "tfidf_quality_length_cost@0.9", "semantic_quality_length_cost@0.9",
                          "mixture@0.9", "fixed_1", "fixed_2", "fixed_3", "fixed_4"}
        if keep and row.get("defined") and "_per_question_correct" in row:
            wanted[method.split("@")[0] if method.endswith("@0.9") else method] = {
                "defined": True,
                "accuracy": row["accuracy"],
                "mean_loop_tokens": row["mean_loop_tokens"],
                "per_question_correct": row["_per_question_correct"],
                "per_question_cost": row["_per_question_cost"],
            }
    return wanted


def conclusion_label(bootstrap: dict) -> str:
    """Promising, inconclusive, or negative, from the primary-point intervals only."""
    methods = bootstrap.get("methods", {})
    semantic = methods.get("semantic", {}).get("point", {})
    surplus = methods.get("semantic", {}).get("equal_cost_surplus")
    gap = bootstrap.get("semantic_minus_tfidf_surplus")
    if not surplus or surplus.get("low") is None:
        return "inconclusive"
    loses_frontier = surplus["high"] < 0
    beats_frontier = surplus["low"] > 0
    beats_tfidf = bool(gap and gap.get("low") is not None and gap["low"] > 0)
    loses_tfidf = bool(gap and gap.get("high") is not None and gap["high"] < 0)
    if loses_frontier or loses_tfidf:
        return "negative"
    if beats_frontier and beats_tfidf and semantic.get("equal_cost_surplus", 0) > 0:
        length = methods.get("length", {}).get("point", {})
        if semantic.get("accuracy", 0) > length.get("accuracy", 1):
            return "promising"
    return "inconclusive"
