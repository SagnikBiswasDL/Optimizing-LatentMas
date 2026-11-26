#!/usr/bin/env python3
"""Offline semantic loop router on the saved Ouro pilot.

Stages are resumable only when the output directory was produced by the same
config, source hashes, and module hashes. A completed run is not overwritten
by a different stamp.

    python scripts/eval_semantic_router.py --config configs/looped_serving/semantic_router_pilot.json --stage all --out artifacts/looped_serving/semantic_router/pilot_v1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped.routing_dataset import (  # noqa: E402
    PilotDataset,
    file_sha256,
    load_pilot,
    save_dataset_arrays,
)
from looped.routing_evaluation import (  # noqa: E402
    ATTRIBUTION,
    bootstrap_intervals,
    build_policy_table,
    calibration_table,
    conclusion_label,
    evaluate_routers,
    frontier_accuracy,
    primary_summaries_for_bootstrap,
    shuffled_assignment,
    stratified_group_folds,
    summarize_assignment,
)
from looped.semantic_router import (  # noqa: E402
    CHUNKING_VERSION,
    cache_fingerprint,
    embedding_device,
    encode_texts,
    load_sentence_transformer,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_FILES = (
    "looped/routing_dataset.py",
    "looped/semantic_router.py",
    "looped/routing_evaluation.py",
    "scripts/eval_semantic_router.py",
)
PLACEHOLDER = "RESOLVE_TO_IMMUTABLE_COMMIT_BEFORE_EMBEDDING"


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")


def _convert(value):
    if isinstance(value, dict):
        return {str(k): _convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical(payload) -> str:
    return json.dumps(_convert(payload), sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_hashes() -> dict:
    return {path: file_sha256(os.path.join(ROOT, path)) for path in CODE_FILES}


def git_provenance() -> dict:
    def run(args):
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "inspected_commit": "890702b8dfcb27031356a7e350bdf02f6e493bb3",
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status_porcelain": run(["git", "status", "--porcelain"]),
    }


def load_config(path: str) -> dict:
    config = _read_json(path)
    config["_path"] = os.path.relpath(path, ROOT)
    return config


def _pilot_from_saved(out: str) -> PilotDataset:
    questions = _read_json(os.path.join(out, "questions.json"))
    arrays = np.load(os.path.join(out, "outcomes.npz"))
    audit = _read_json(os.path.join(out, "data_audit.json"))
    return PilotDataset(
        question_ids=questions["question_ids"],
        group_ids=questions["group_ids"],
        texts=questions["texts"],
        text_hashes=questions["text_hashes"],
        levels=arrays["levels"],
        subjects=questions["subjects"],
        depths=tuple(questions["depths"]),
        run_seeds=tuple(questions["run_seeds"]),
        vote_correct=arrays["vote_correct"],
        vote_correct_strict=arrays["vote_correct_strict"],
        loop_tokens=arrays["loop_tokens"],
        audit=audit,
    )


def _guard_completed(out: str, stamp: str) -> None:
    path = os.path.join(out, "results.json")
    if not os.path.exists(path):
        return
    previous = _read_json(path).get("stamp")
    if previous != stamp:
        raise SystemExit(
            f"{path} was produced with stamp {previous}, not {stamp}. "
            "Refusing to overwrite a completed experiment.")


def stage_validate(config: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    data = load_pilot(ROOT, depths=config["depths"], run_seeds=config["run_seeds"], reproduce=True)
    data.audit["repo"] = git_provenance()
    data.audit["code_hashes"] = code_hashes()
    data.audit["config_sha256"] = _sha(_canonical({k: v for k, v in config.items() if not k.startswith("_")}))
    save_dataset_arrays(data, out)
    print(f"validated {data.n} questions; primary counts reproduced", flush=True)


def resolve_revision(encoder: str) -> str:
    url = "https://huggingface.co/api/models/" + encoder
    with urllib.request.urlopen(url, timeout=60) as response:
        info = json.load(response)
    sha = info["sha"]
    if not isinstance(sha, str) or len(sha) < 40:
        raise RuntimeError(f"Hub did not return an immutable revision for {encoder}")
    return sha


def stage_embed(config: dict, out: str) -> dict:
    if not os.path.exists(os.path.join(out, "questions.json")):
        raise SystemExit("run --stage validate first")
    questions = _read_json(os.path.join(out, "questions.json"))
    revision = config["encoder_revision"]
    if revision == PLACEHOLDER:
        revision = resolve_revision(config["encoder"])
        config_path = os.path.join(ROOT, config["_path"])
        on_disk = _read_json(config_path)
        on_disk["encoder_revision"] = revision
        _write_json(config_path, on_disk)
        config["encoder_revision"] = revision
        print(f"pinned encoder revision {revision}", flush=True)
    device, hardware = embedding_device()
    t0 = time.perf_counter()
    model = load_sentence_transformer(config["encoder"], revision, device)
    load_ms = (time.perf_counter() - t0) * 1000.0
    texts = questions["texts"]
    encoded = encode_texts(model, texts)
    tokenizer_name = type(model.tokenizer).__name__
    fingerprints = [
        cache_fingerprint(encoder=config["encoder"], revision=revision, text_hash=text_hash,
                          max_seq_length=encoded["max_seq_length"], tokenizer=tokenizer_name)
        for text_hash in questions["text_hashes"]
    ]
    # Short questions must match ordinary normalized encoding.
    short_diffs = []
    for index, windows in enumerate(encoded["window_counts"]):
        if windows != 1:
            continue
        ordinary = model.encode([texts[index]], normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)[0]
        short_diffs.append(float(np.max(np.abs(ordinary - encoded["embeddings"][index]))))
        if len(short_diffs) >= 8:
            break
    if short_diffs and max(short_diffs) > 1e-4:
        raise RuntimeError(f"short-question encoding disagrees with model.encode by {max(short_diffs)}")
    meta = {
        "encoder": config["encoder"],
        "revision": revision,
        "chunking": CHUNKING_VERSION,
        "normalize": "l2",
        "precision": "float32",
        "max_seq_length": encoded["max_seq_length"],
        "window_size": encoded["window_size"],
        "tokenizer": tokenizer_name,
        "device": device,
        "hardware": hardware,
        "n_questions": len(texts),
        "n_long": encoded["n_long"],
        "dimension": int(encoded["embeddings"].shape[1]),
        "token_counts": encoded["token_counts"],
        "window_counts": encoded["window_counts"],
        "text_hashes": questions["text_hashes"],
        "fingerprints": fingerprints,
        "errors": encoded["errors"],
        "short_path_max_abs_diff": None if not short_diffs else max(short_diffs),
    }
    if meta["dimension"] != 384:
        raise RuntimeError(f"expected 384-d MiniLM embeddings, got {meta['dimension']}")
    np.savez_compressed(os.path.join(out, "embeddings.npz"),
                        embeddings=encoded["embeddings"], question_index=np.arange(len(texts)))
    overhead = _overhead(model, texts[0], load_ms, device, hardware)
    meta["overhead_ms"] = overhead
    _write_json(os.path.join(out, "embedding_metadata.json"), meta)
    _write_json(os.path.join(out, "resolved_config.json"),
                {k: v for k, v in config.items() if not k.startswith("_")})
    print(f"embedded {len(texts)} questions on {device}; {encoded['n_long']} needed more than one window", flush=True)
    return meta


def _overhead(model, text: str, load_ms: float, device: str, hardware: str) -> dict:
    t0 = time.perf_counter()
    model.encode([text], normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    for _ in range(5):
        model.encode([text], normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        model.encode([text], normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples = np.sort(np.asarray(samples))
    return {
        "hardware": hardware,
        "platform": platform.platform(),
        "device": device,
        "model_load_ms": load_ms,
        "cold_batch1_ms": cold_ms,
        "warm": True,
        "batch_size": 1,
        "tokenization_included": True,
        "batch1_p50_ms": float(np.quantile(samples, 0.50)),
        "batch1_p95_ms": float(np.quantile(samples, 0.95)),
        "note": "Milliseconds are encoder overhead only. They are not added to loop-tokens.",
    }


def _load_embeddings(out: str, config: dict) -> np.ndarray:
    meta_path = os.path.join(out, "embedding_metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit("run --stage embed first")
    meta = _read_json(meta_path)
    if meta["revision"] != config["encoder_revision"] or meta["encoder"] != config["encoder"]:
        raise SystemExit("embedding metadata does not match the config revision")
    if meta["chunking"] != CHUNKING_VERSION or meta["precision"] != "float32" or meta["normalize"] != "l2":
        raise SystemExit("embedding cache uses a different chunking, precision, or normalization")
    questions = _read_json(os.path.join(out, "questions.json"))
    if meta["text_hashes"] != questions["text_hashes"]:
        raise SystemExit("embedding cache does not match the validated question texts")
    expected = [
        cache_fingerprint(encoder=meta["encoder"], revision=meta["revision"], text_hash=text_hash,
                          max_seq_length=meta["max_seq_length"], tokenizer=meta["tokenizer"])
        for text_hash in questions["text_hashes"]
    ]
    if expected != meta["fingerprints"]:
        raise SystemExit("stale embedding fingerprint")
    embeddings = np.load(os.path.join(out, "embeddings.npz"))["embeddings"].astype(np.float64)
    if embeddings.shape != (len(questions["question_ids"]), 384):
        raise SystemExit(f"unexpected embedding shape {embeddings.shape}")
    return embeddings


def _load_folds(out: str, pilot, config: dict):
    path = os.path.join(out, "folds.json")
    if os.path.exists(path):
        raw = _read_json(path)
        folds = [(np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)) for train, test in raw["folds"]]
        return folds
    folds = stratified_group_folds(pilot.levels, pilot.group_ids, int(config["outer_folds"]), int(config["split_seed"]))
    payload = {
        "split_seed": int(config["split_seed"]),
        "outer_folds": int(config["outer_folds"]),
        "question_ids": pilot.question_ids,
        "folds": [(train.tolist(), test.tolist()) for train, test in folds],
        "test_question_ids": [[pilot.question_ids[int(i)] for i in test] for _train, test in folds],
    }
    _write_json(path, payload)
    return folds


def stage_evaluate(config: dict, out: str) -> None:
    if config["encoder_revision"] == PLACEHOLDER:
        raise SystemExit("encoder revision is not pinned; run --stage embed first")
    pilot = _pilot_from_saved(out)
    embeddings = _load_embeddings(out, config)
    folds = _load_folds(out, pilot, config)
    stamp = _sha(_canonical({
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "code": code_hashes(),
        "data_audit_config": pilot.audit.get("config_sha256"),
        "embedding_revision": config["encoder_revision"],
    }))
    _guard_completed(out, stamp)
    print("fitting nested routers", flush=True)
    t0 = time.perf_counter()
    fitted = evaluate_routers(pilot, config, embeddings, folds=folds)
    print(f"fit finished in {time.perf_counter() - t0:.1f}s", flush=True)
    table = build_policy_table(pilot, config, fitted["assignments"], fitted["selection"], folds)
    primary = primary_summaries_for_bootstrap(table)
    print("bootstrap", flush=True)
    bootstrap = bootstrap_intervals(pilot, primary, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]))
    strict_primary = {}
    for name, summary in primary.items():
        # Strict rescoring keeps the selected actions and replaces only the vote.
        # Per-question arrays in `primary` are already the primary vote. Rebuild from assignments.
        strict_primary[name] = summary
    strict_boot = _strict_bootstrap(pilot, fitted["assignments"], config)
    shuffles = _shuffles(pilot, fitted["assignments"], folds, config)
    calibration = calibration_table(pilot, fitted["test_predictions"])
    public_table = []
    score_arrays = {}
    for row in table:
        correct = row.pop("_per_question_correct", None)
        spent = row.pop("_per_question_cost", None)
        if correct is not None:
            score_arrays[row["method"].replace("@", "_at_") + "_correct"] = correct
            score_arrays[row["method"].replace("@", "_at_") + "_cost"] = spent
        public_table.append(_convert(row))
    if score_arrays:
        np.savez_compressed(os.path.join(out, "question_scores.npz"), **score_arrays)
    _write_jsonl(os.path.join(out, "predictions.jsonl"), fitted["predictions"])
    _write_joined(out, pilot, fitted["predictions"])
    results = {
        "stamp": stamp,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "code_hashes": code_hashes(),
        "repo": git_provenance(),
        "fitted_summary": fitted["fitted_summary"],
        "selection": fitted["selection"],
        "table": public_table,
        "calibration": calibration,
        "bootstrap_primary": bootstrap,
        "bootstrap_strict_cap": strict_boot,
        "shuffled": shuffles,
        "conclusion": conclusion_label(bootstrap),
        "elapsed_fit_seconds": time.perf_counter() - t0,
    }
    _write_json(os.path.join(out, "results.json"), _convert(results))
    _write_summary_csv(os.path.join(out, "summary.csv"), public_table)
    print(f"conclusion label: {results['conclusion']}", flush=True)


def _strict_bootstrap(pilot, assignments, config) -> dict:
    """Rescore frozen actions with length-capped candidates removed from the vote."""
    from looped.routing_evaluation import fixed_depth_assignment

    summaries = {}
    rho = float(config["primary_budget_ratio"])
    labels = [f"length@{rho}", f"tfidf@{rho}", f"semantic@{rho}", f"mixture@{rho}",
              f"tfidf_quality_length_cost@{rho}", f"semantic_quality_length_cost@{rho}"]
    for depth_index, depth in enumerate(pilot.depths):
        summaries[f"fixed_{depth}"] = summarize_assignment(
            pilot, fixed_depth_assignment(pilot.n, len(pilot.depths), depth_index),
            label=f"fixed_{depth}", vote=pilot.vote_correct_strict)
    for label in labels:
        if label not in assignments:
            continue
        summaries[label.split("@")[0]] = summarize_assignment(
            pilot, assignments[label], label=label, vote=pilot.vote_correct_strict)
    return bootstrap_intervals(pilot, summaries, int(config["bootstrap_replicates"]),
                               int(config["bootstrap_seed"]), vote=pilot.vote_correct_strict)


def _shuffles(pilot, assignments, folds, config) -> dict:
    rho = float(config["primary_budget_ratio"])
    report = {}
    for method in ("length", "tfidf", "semantic", "tfidf_quality_length_cost", "semantic_quality_length_cost"):
        key = f"{method}@{rho}"
        assignment = assignments.get(key)
        if assignment is None or not np.all(assignment["defined"]):
            report[method] = {"defined": False}
            continue
        accs, costs = [], []
        for shuffled in shuffled_assignment(assignment, folds, int(config["shuffle_repeats"]),
                                            int(config["bootstrap_seed"])):
            summary = summarize_assignment(pilot, shuffled, label=method)
            accs.append(summary["accuracy"])
            costs.append(summary["mean_loop_tokens"])
        report[method] = {
            "defined": True,
            "repeats": int(config["shuffle_repeats"]),
            "seed": int(config["bootstrap_seed"]),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_low": float(np.quantile(accs, 0.025)),
            "accuracy_high": float(np.quantile(accs, 0.975)),
            "mean_loop_tokens_mean": float(np.mean(costs)),
            "mean_loop_tokens_low": float(np.quantile(costs, 0.025)),
            "mean_loop_tokens_high": float(np.quantile(costs, 0.975)),
            "note": "Descriptive only. Not used to choose lambda, the encoder, or the budget.",
        }
    return report


def _write_jsonl(path: str, rows) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(_convert(row)) + "\n")


def _write_joined(out: str, pilot, predictions) -> None:
    by_q = {qid: i for i, qid in enumerate(pilot.question_ids)}
    with open(os.path.join(out, "joined.jsonl"), "w") as f:
        for row in predictions:
            i = by_q[row["question_id"]]
            joined = dict(row)
            joined["observed_vote_correct"] = pilot.vote_correct[i].tolist()
            joined["observed_vote_correct_strict_cap"] = pilot.vote_correct_strict[i].tolist()
            joined["observed_loop_tokens"] = pilot.loop_tokens[i].tolist()
            f.write(json.dumps(_convert(joined)) + "\n")


def _write_summary_csv(path: str, table) -> None:
    fields = ["method", "defined", "accuracy", "mean_loop_tokens", "cost_over_fixed_reference",
              "equal_cost_surplus", "retrospective_frontier_accuracy", "accuracy_minus_mixture",
              "mixture_accuracy", "mixture_mean_loop_tokens", "strict_cap_accuracy", "depth_counts"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow(row)


def stage_report(config: dict, out: str) -> None:
    results = _read_json(os.path.join(out, "results.json"))
    audit = _read_json(os.path.join(out, "data_audit.json"))
    meta = _read_json(os.path.join(out, "embedding_metadata.json"))
    pilot = _pilot_from_saved(out)
    _write_plot(os.path.join(out, "accuracy_cost.png"), results, pilot)
    text = render_report(results, audit, meta, pilot)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(text)
    doc = os.path.join(ROOT, "docs/looped_serving/SEMANTIC_ROUTING.md")
    with open(doc, "w") as f:
        f.write(text)
    _point_next_agent()
    print(f"wrote {doc}", flush=True)


def _write_plot(path: str, results: dict, pilot) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = pilot.vote_correct.mean(axis=(0, 1))
    cost = pilot.loop_tokens.mean(axis=(0, 1))
    order = np.argsort(cost)
    budgets = np.linspace(cost.min(), cost.max(), 40)
    frontier_x, frontier_y = [], []
    for budget in budgets:
        value = frontier_accuracy(acc, cost, float(budget))
        if value is not None:
            frontier_x.append(budget)
            frontier_y.append(value)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(frontier_x, frontier_y, linestyle="--", color="0.4",
            label="Retrospective blind frontier (uses held-out outcomes)")
    ax.scatter(cost, acc, s=36, color="black", label="Fixed depth", zorder=3)
    for depth, x, y in zip(pilot.depths, cost, acc):
        ax.annotate(f"{depth} loops", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    markers = {"length": "o", "tfidf": "s", "semantic": "D"}
    colors = {"0.75": "C0", "0.9": "C1", "1.0": "C3"}
    for row in results["table"]:
        method = row["method"]
        if not row.get("defined") or "@" not in method:
            continue
        name, rho = method.split("@", 1)
        if name not in markers:
            continue
        ax.scatter(row["mean_loop_tokens"], row["accuracy"], marker=markers[name], s=48,
                   color=colors.get(rho, "C2"),
                   label=f"{name} @ {rho} of training depth-3 cost")
    ax.set_xlabel("Mean loop-tokens of the selected saved actions")
    ax.set_ylabel("Vote accuracy, both runs")
    ax.set_title("Semantic loop router, 200-question development pilot")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.1f}%"


def _fmt_num(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def _interval_text(interval) -> str:
    if not interval or interval.get("low") is None:
        return "-"
    return f"{interval['low']:.4f} to {interval['high']:.4f}"


def render_report(results: dict, audit: dict, meta: dict, pilot) -> str:
    primary = str(results["config"]["primary_budget_ratio"])
    rows = [row for row in results["table"] if row.get("defined")]
    def find(name):
        for row in rows:
            if row["method"] == name:
                return row
        return None
    lines = [
        "# Semantic loop routing on the saved pilot",
        "",
        "Cross-validated development experiment on the 200-question router pilot. "
        "It does not retrain Ouro, generate answers, or change serving. "
        "The confirmation reserve and the final benchmarks were not read. "
        "One frozen encoder was used. No second encoder was tried.",
        "",
        "## Setup",
        "",
        f"Encoder `{meta['encoder']}` at revision `{meta['revision']}`, frozen, 384-d, L2-normalized. "
        f"{meta['n_long']} questions needed more than one 256-wordpiece window. "
        f"Device: {meta['device']} ({meta['hardware']}).",
        "",
        "Each question is one group. Both generation runs stay in that group. "
        "Five outer folds are stratified by MATH level. "
        "Logistic `C`, ridge `alpha`, and the utility penalty `lambda` are chosen on four inner folds "
        "of the outer-training questions. The reported accuracy uses the outer-test questions only.",
        "",
        "The action list is depths 1, 2, 3, and 4. "
        "The router maximizes predicted vote accuracy minus `lambda` times predicted loop-tokens / 10000. "
        "The primary budget is 0.9 times the outer-training mean cost of fixed 3 loops. "
        "Budgets 0.75 and 1.0 are reported and were not used to pick the primary.",
        "",
        "Length features are log(1 + characters) and log(1 + whitespace words). "
        "TF-IDF uses word unigrams and bigrams, `min_df=2`, sublinear tf, and L2 normalization, "
        "fit on training questions only. Semantic and TF-IDF both include the length features.",
        "",
        f"Repository HEAD at the run: `{results['repo']['head']}`. "
        "That is the inspected commit. Later deadline-study files in the worktree were not inputs.",
        "",
        "Primary vote counts reproduced before fitting: "
        f"run 0 {audit['reproduction']['primary_correct']['0']}, "
        f"run 1 {audit['reproduction']['primary_correct']['1']}. "
        "Strict cap exclusion, rescored and not refit: "
        f"run 0 {audit['reproduction']['strict_correct']['0']}, "
        f"run 1 {audit['reproduction']['strict_correct']['1']}.",
        "",
        audit["manifests"]["note"],
        "",
        "## Primary budget, 0.9",
        "",
        "| method | accuracy | loop-tokens | share of fixed-3 cost | equal-cost surplus | minus training mixture | strict-cap accuracy | depth counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = ["fixed_1", "fixed_2", "fixed_3", "fixed_4", f"mixture@{primary}",
             f"length@{primary}", f"tfidf@{primary}", f"semantic@{primary}",
             f"tfidf_quality_length_cost@{primary}", f"semantic_quality_length_cost@{primary}"]
    for name in order:
        row = find(name)
        if row is None:
            lines.append(f"| {name} | infeasible | | | | | | |")
            continue
        lines.append(
            f"| {name} | {_fmt_pct(row['accuracy'])} | {_fmt_num(row['mean_loop_tokens'])} | "
            f"{row['cost_over_fixed_reference']:.3f} | {None if row['equal_cost_surplus'] is None else round(row['equal_cost_surplus'], 4)} | "
            f"{None if row.get('accuracy_minus_mixture') is None else round(row['accuracy_minus_mixture'], 4)} | "
            f"{_fmt_pct(row['strict_cap_accuracy'])} | {row['depth_counts']} |"
        )
    boot = results["bootstrap_primary"]["methods"]
    gap = results["bootstrap_primary"].get("semantic_minus_tfidf_surplus")
    lines += [
        "",
        "Equal-cost surplus is accuracy minus the retrospective question-blind frontier at the same realized cost. "
        "That frontier is rebuilt from held-out outcomes and was not used to choose the router. "
        "The training mixture is the deployable blind baseline: its weights were fit on outer-training questions only.",
        "",
        "Budget miss is realized test cost divided by the budget fixed from that fold's training questions:",
    ]
    for name in order:
        row = find(name)
        miss = (row or {}).get("budget_miss") or {}
        if miss.get("mean_realized_over_budget") is not None:
            lines.append(
                f"- {name}: realized/budget {miss['mean_realized_over_budget']:.3f}, "
                f"folds over the training budget {miss['folds_over_budget']} of {len(miss['folds'])}."
            )
    lines += [
        "",
        "Bootstrap intervals resample questions, carry both runs, and use the same draws for every method. "
        "They describe these frozen predictions. They do not refit the heads.",
        "",
        f"- Semantic accuracy interval: {_interval_text(boot.get('semantic', {}).get('accuracy'))}.",
        f"- Semantic equal-cost surplus interval: {_interval_text(boot.get('semantic', {}).get('equal_cost_surplus'))}.",
        f"- TF-IDF equal-cost surplus interval: {_interval_text(boot.get('tfidf', {}).get('equal_cost_surplus'))}.",
        f"- Length equal-cost surplus interval: {_interval_text(boot.get('length', {}).get('equal_cost_surplus'))}.",
        f"- Semantic minus TF-IDF equal-cost surplus interval: {_interval_text(gap)}.",
        "",
        "## Other budgets",
        "",
        "| method | accuracy | loop-tokens | equal-cost surplus |",
        "|---|---:|---:|---:|",
    ]
    for rho in ("0.75", "1.0"):
        for name in (f"mixture@{rho}", f"length@{rho}", f"tfidf@{rho}", f"semantic@{rho}"):
            row = find(name)
            if row is None:
                lines.append(f"| {name} | infeasible | | |")
            else:
                lines.append(
                    f"| {name} | {_fmt_pct(row['accuracy'])} | {_fmt_num(row['mean_loop_tokens'])} | "
                    f"{None if row['equal_cost_surplus'] is None else round(row['equal_cost_surplus'], 4)} |"
                )
    lines += ["", "## Per run, primary budget", ""]
    semantic = find(f"semantic@{primary}") or {}
    tfidf = find(f"tfidf@{primary}") or {}
    for label, row in (("semantic", semantic), ("tfidf", tfidf)):
        per_run = row.get("per_run") or {}
        bits = [f"run {seed}: {_fmt_pct(payload['accuracy'])} at {_fmt_num(payload['mean_loop_tokens'])}"
                for seed, payload in per_run.items()]
        lines.append(f"- {label}: " + "; ".join(bits))
    lines += ["", "## Prediction quality", "",
              "Log loss and Brier score are held-out scores of the refit heads. A better score here does not by itself mean a better depth assignment.",
              "",
              "| method | depth | log loss | Brier | cost MAE | constant-cost MAE | predicted mean | observed mean |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("length", "tfidf", "semantic"):
        for depth_row in results["calibration"].get(method, []):
            lines.append(
                f"| {method} | {depth_row['depth']} | {depth_row['log_loss']:.3f} | {depth_row['brier']:.3f} | "
                f"{depth_row['cost_mae']:.0f} | {depth_row['constant_cost_mae']:.0f} | "
                f"{depth_row['mean_predicted_loop_tokens']:.0f} | {depth_row['mean_observed_loop_tokens']:.0f} |"
            )
    lines += ["", "## Shuffled depths", "",
              "Within each outer test fold, the learned depths were permuted across questions. "
              "This is descriptive. It was not a selection criterion.",
              ""]
    for method, payload in results["shuffled"].items():
        if not payload.get("defined"):
            lines.append(f"- {method}: not defined")
        else:
            lines.append(
                f"- {method}: shuffled accuracy {_fmt_pct(payload['accuracy_mean'])} "
                f"(interval {_fmt_pct(payload['accuracy_low'])} to {_fmt_pct(payload['accuracy_high'])}), "
                f"mean loop-tokens {_fmt_num(payload['mean_loop_tokens_mean'])}."
            )
    overhead = meta.get("overhead_ms", {})
    lines += [
        "",
        "## Encoder overhead",
        "",
        f"Hardware {overhead.get('hardware')}, device {overhead.get('device')}. "
        f"Model load {overhead.get('model_load_ms', 0):.0f} ms. "
        f"First batch-1 encode after load {overhead.get('cold_batch1_ms', 0):.1f} ms. "
        f"Warm batch-1, tokenization included, p50 {overhead.get('batch1_p50_ms', 0):.1f} ms, "
        f"p95 {overhead.get('batch1_p95_ms', 0):.1f} ms. "
        "These milliseconds are not loop-tokens and are not a serving claim.",
        "",
        "## Conclusion",
        "",
        conclusion_paragraph(results),
        "",
        "Artifacts: `artifacts/looped_serving/semantic_router/pilot_v1/`.",
        "",
    ]
    return "\n".join(lines)


def conclusion_paragraph(results: dict) -> str:
    label = results["conclusion"]
    primary = str(results["config"]["primary_budget_ratio"])
    table = {row["method"]: row for row in results["table"] if row.get("defined")}
    semantic = table.get(f"semantic@{primary}", {})
    tfidf = table.get(f"tfidf@{primary}", {})
    length = table.get(f"length@{primary}", {})
    fixed3 = table.get("fixed_3", {})
    gap = results["bootstrap_primary"].get("semantic_minus_tfidf_surplus") or {}
    surplus = (results["bootstrap_primary"].get("methods") or {}).get("semantic", {}).get("equal_cost_surplus") or {}
    mixture = table.get(f"mixture@{primary}", {})
    sentence = (
        f"The frozen MiniLM router is {label}. "
        f"At the precommitted 0.9 budget it scores {_fmt_pct(semantic.get('accuracy'))} "
        f"at {_fmt_num(semantic.get('mean_loop_tokens'))} loop-tokens. "
        f"TF-IDF scores {_fmt_pct(tfidf.get('accuracy'))} at {_fmt_num(tfidf.get('mean_loop_tokens'))}, "
        f"length scores {_fmt_pct(length.get('accuracy'))} at {_fmt_num(length.get('mean_loop_tokens'))}, "
        f"and the training-only blind mixture scores {_fmt_pct(mixture.get('accuracy'))} "
        f"at {_fmt_num(mixture.get('mean_loop_tokens'))}. "
        f"Fixed 3 loops scores {_fmt_pct(fixed3.get('accuracy'))} at {_fmt_num(fixed3.get('mean_loop_tokens'))}. "
        f"Semantic equal-cost surplus over the retrospective blind frontier has interval {_interval_text(surplus)}. "
        f"Semantic surplus minus TF-IDF surplus has interval {_interval_text(gap)}. "
    )
    if label == "promising":
        sentence += (
            "The interval sits above zero on this development pilot. "
            "The single next experiment is a separately frozen confirmation on questions that have not informed this design. "
            "That confirmation is not part of this run, and the reserve was not opened."
        )
    elif label == "negative":
        sentence += (
            "On this pilot the semantic router does not beat the blind equal-cost frontier, the TF-IDF router, or both. "
            "The single next experiment worth running is not another encoder on these same 200 questions. "
            "A later test needs either a different routing signal or a fresh question set declared before fitting."
        )
    else:
        sentence += (
            "The gap is small or the interval covers zero, so this pilot does not show that question text allocates loops better than the controls. "
            "The single next experiment is not a search over encoders on these same questions. "
            "Keep the confirmation reserve sealed."
        )
    return sentence


def _point_next_agent() -> None:
    path = os.path.join(ROOT, "docs/looped_serving/NEXT_AGENT.md")
    marker = "Semantic loop routing on the saved pilot"
    text = open(path).read()
    if marker in text:
        return
    note = (
        "\nSemantic loop routing on the saved pilot is a separate offline experiment. "
        "Protocol and result: `docs/looped_serving/SEMANTIC_ROUTING.md`. "
        "Artifacts: `artifacts/looped_serving/semantic_router/pilot_v1/`. "
        "It does not change the deadline-study decision in this file, and it does not open the confirmation reserve.\n"
    )
    parts = text.split("\n", 1)
    if len(parts) == 2:
        text = parts[0] + "\n" + note + "\n" + parts[1]
    else:
        text = text + note
    with open(path, "w") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["validate", "embed", "evaluate", "report", "all"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = load_config(os.path.join(ROOT, args.config) if not os.path.isabs(args.config) else args.config)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    stages = ["validate", "embed", "evaluate", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "validate":
            stage_validate(config, out)
        elif stage == "embed":
            stage_embed(config, out)
            config = load_config(os.path.join(ROOT, config["_path"]))
        elif stage == "evaluate":
            config = load_config(os.path.join(ROOT, config["_path"]))
            stage_evaluate(config, out)
        elif stage == "report":
            config = load_config(os.path.join(ROOT, config["_path"]))
            stage_report(config, out)


if __name__ == "__main__":
    main()
