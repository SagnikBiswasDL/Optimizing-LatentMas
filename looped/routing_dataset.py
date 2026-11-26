"""Strict loader for the router-pilot loop outcomes.

Each row is one question, generation run, and depth. The vote reuses
``looped.loop_select`` on structural answer keys. Question text is the only
field a later model may read. Generated answers, gold answers, keys, lengths,
and parse status stay in the audit and never become features.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from looped.loop_select import vote_correct

EXPECTED_QUESTION_SHA256 = "300d480f99cf1e73108027575c0cf07180079bdf2671ee45ea5dc0f1b50a0bad"
QUESTION_LIST = "artifacts/looped_serving/splits/router_pilot_questions.txt"
ITEMS_PATH = "artifacts/looped_serving/splits/math_train_calibration.items.jsonl"
SPLITS_PATH = "configs/looped_serving/splits.json"
PILOT_DIR = "artifacts/looped_serving/loop_routing/pilot"
SPLIT_NAME = "math_train_calibration"

PRIMARY_CORRECT = {
    0: (57, 152, 175, 179),
    1: (57, 156, 173, 176),
}
PRIMARY_COST = {
    0: (11487.260, 18987.350, 29965.890, 35406.420),
    1: (11422.970, 17506.980, 28653.015, 35826.620),
}
STRICT_CORRECT = {
    0: (57, 152, 175, 177),
    1: (57, 155, 172, 175),
}
ALLOWED_FINISH = frozenset({"eos", "length_cap"})
SAMPLES = (0, 1, 2, 3)

# Fields that may differ across arms. Everything else must match.
ALLOWED_MANIFEST_DIFFERENCES = frozenset({
    "decode_config.depth",
    "decode_config.slots",
    "run_seed",
    "config_hash",
    "extra.engine.full_depth",
    "extra.engine.max_num_seqs",
    # KV-plane count follows the depth-limited engine.
    "cache_semantics",
    # Seed 1 at depths 1, 3, and 4 was generated from a different engine tree.
    # Prompt, model, temperature, top-p, cap, width, and mode still match.
    "extra.verified.source_tree_sha256",
})

_CID_DEPTH_SAMPLE = re.compile(r":d(\d+):s(\d+)(?:$|:)")


class PilotDataError(ValueError):
    """The saved pilot does not match the frozen loader contract."""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise PilotDataError(f"{path}:{line_no} is not json: {exc}") from exc
    return rows


def load_question_ids(path: str) -> List[str]:
    ids = []
    with open(path) as f:
        for line in f:
            q = line.strip()
            if q:
                ids.append(q)
    if len(ids) != len(set(ids)):
        raise PilotDataError(f"duplicate question id in {path}")
    digest = file_sha256(path)
    if digest != EXPECTED_QUESTION_SHA256:
        raise PilotDataError(
            f"question list sha256 {digest} != {EXPECTED_QUESTION_SHA256}")
    return ids


def _flatten(value, prefix: str = "") -> Dict[str, str]:
    if isinstance(value, dict):
        out: Dict[str, str] = {}
        for key, child in value.items():
            out.update(_flatten(child, prefix + key + "."))
        return out
    return {prefix[:-1]: json.dumps(value, sort_keys=True)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotDataError(message)


def check_manifests(manifests: Sequence[Tuple[str, dict]], depths: Sequence[int],
                    run_seeds: Sequence[int]) -> dict:
    """Check generation settings. Return the common settings and the recorded differences."""
    _require(len(manifests) == len(depths) * len(run_seeds),
             f"expected {len(depths) * len(run_seeds)} manifests, found {len(manifests)}")
    flats = [(name, _flatten(manifest)) for name, manifest in manifests]
    keys = set()
    for _, flat in flats:
        keys |= set(flat)
    differing = {}
    for key in sorted(keys):
        vals = sorted({flat.get(key) for _, flat in flats})
        if len(vals) > 1:
            differing[key] = vals
    unexpected = sorted(set(differing) - ALLOWED_MANIFEST_DIFFERENCES)
    _require(not unexpected,
             "manifests differ in fields that are not allowed differences: " + ", ".join(unexpected))

    common = flats[0][1]
    def _const(path: str) -> str:
        raw = common[path]
        return json.loads(raw)

    _require(_const("decode_config.cap") == 8192, "cap is not 8192")
    _require(_const("decode_config.mode") == "natural", "answer mode is not natural")
    _require(_const("decode_config.temperature") == 0.6, "temperature is not 0.6")
    _require(_const("decode_config.top_p") == 0.95, "top_p is not 0.95")
    _require(_const("decode_config.width") == 4, "width is not 4")
    _require(_const("decode_config.kind") == "fixed", "decode kind is not fixed")
    _require(_const("decode_config.n_questions") == 200, "manifest n_questions is not 200")
    _require(_const("decode_config.question_subset_sha256") == EXPECTED_QUESTION_SHA256,
             "question subset hash does not match the pilot list")
    prompt = _const("prompt_hash")
    model = _const("model_revision")
    _require(isinstance(prompt, str) and prompt, "prompt hash missing")
    _require(isinstance(model, str) and model, "model revision missing")
    by_arm = []
    for (name, manifest), (_, flat) in zip(manifests, flats):
        by_arm.append({
            "arm": name,
            "depth": manifest["decode_config"]["depth"],
            "run_seed": manifest["run_seed"],
            "slots": manifest["decode_config"]["slots"],
            "config_hash": manifest["config_hash"],
            "source_tree_sha256": manifest["extra"]["verified"]["source_tree_sha256"],
            "model_revision": manifest["model_revision"],
            "prompt_hash": manifest["prompt_hash"],
        })
    return {
        "prompt_hash": prompt,
        "model_revision": model,
        "tokenizer_revision": _const("tokenizer_revision"),
        "dtype": _const("dtype"),
        "temperature": 0.6,
        "top_p": 0.95,
        "cap": 8192,
        "width": 4,
        "mode": "natural",
        "question_subset_sha256": EXPECTED_QUESTION_SHA256,
        "allowed_differences": {key: differing[key] for key in sorted(differing)},
        "arms": by_arm,
        "note": (
            "Seed 1 at depths 1, 3, and 4 records a different engine source-tree hash "
            "from the other arms. Seed 1 at depth 2 used 96 slots; every other arm used 32. "
            "Wait times are not comparable. Prompt, model revision, temperature, top-p, "
            "cap, width, and natural mode match."
        ),
    }


def eligible_key(parse_status: str, answer_key, finish_reason: str, *, strict_cap: bool) -> Optional[str]:
    """Return the structural key that may vote, or None when the candidate casts no vote.

    Primary rule: a complete structural key votes even if ``finish_reason`` is
    ``length_cap``. Strict rule: every ``length_cap`` candidate is dropped from
    the vote. Unexpected finish reasons are rejected by the caller.
    """
    if finish_reason not in ALLOWED_FINISH:
        raise PilotDataError(f"unexpected finish_reason {finish_reason!r}")
    if strict_cap and finish_reason == "length_cap":
        return None
    if parse_status == "ok":
        if not isinstance(answer_key, str) or answer_key == "":
            raise PilotDataError("parse_status ok without a structural answer key")
        return answer_key
    return None


def vote_from_candidates(rows: Sequence[dict], *, strict_cap: bool) -> bool:
    """``rows`` are the four joined candidates for one question, depth, and run."""
    samples = []
    for row in rows:
        key = eligible_key(row["parse_status"], row["answer_key"], row["finish_reason"],
                           strict_cap=strict_cap)
        samples.append((int(row["sample"]), key, bool(row["correct"])))
    return bool(vote_correct(samples))


def _same(a, b, label: str, candidate_id: str) -> None:
    if a != b:
        raise PilotDataError(f"{candidate_id}: {label} disagrees between candidates and grades ({a!r} vs {b!r})")


def join_arm(candidates: Sequence[dict], grades: Sequence[dict], *, question_ids: Sequence[str],
             depth: int, run_seed: int, arm: str) -> Dict[str, List[dict]]:
    """Join one arm on candidate id. Return question id -> four joined rows."""
    expected = set(question_ids)
    if len(grades) != len({g.get("candidate_id") for g in grades}):
        raise PilotDataError(f"{arm}: duplicate candidate_id in grades")
    grade_by = {}
    for grade in grades:
        cid = grade.get("candidate_id")
        _require(isinstance(cid, str), f"{arm}: grade without candidate_id")
        if cid in grade_by:
            raise PilotDataError(f"{arm}: duplicate grade for {cid}")
        grade_by[cid] = grade
    if len(candidates) != len({c.get("candidate_id") for c in candidates}):
        raise PilotDataError(f"{arm}: duplicate candidate_id in candidates")

    by_q: Dict[str, List[dict]] = defaultdict(list)
    seen_cand = set()
    for cand in candidates:
        cid = cand.get("candidate_id")
        _require(isinstance(cid, str), f"{arm}: candidate without candidate_id")
        if cid in seen_cand:
            raise PilotDataError(f"{arm}: duplicate candidate {cid}")
        seen_cand.add(cid)
        if cid not in grade_by:
            raise PilotDataError(f"{arm}: missing grade for {cid}")
        grade = grade_by[cid]
        qid = cand.get("question_id")
        _require(qid in expected, f"{arm}: unexpected question {qid}")
        _same(grade.get("question_id"), qid, "question_id", cid)
        _same(cand.get("parse_status"), grade.get("parse_status"), "parse_status", cid)
        _same(cand.get("finish_reason"), grade.get("finish_reason"), "finish_reason", cid)
        _same(cand.get("answer_key"), grade.get("answer_key"), "answer_key", cid)
        _same(cand.get("pred"), grade.get("pred"), "pred", cid)
        _require(grade.get("grader_error") in (None, ""), f"{arm}: grader_error on {cid}")
        _require(isinstance(grade.get("correct"), bool), f"{arm}: correct is not bool on {cid}")
        finish = cand.get("finish_reason")
        _require(finish in ALLOWED_FINISH, f"{arm}: unexpected finish_reason {finish!r} on {cid}")
        _require(cand.get("mode") == "natural", f"{arm}: mode is not natural on {cid}")
        _require(int(cand.get("depth")) == int(depth), f"{arm}: depth field {cand.get('depth')} != {depth} on {cid}")
        match = _CID_DEPTH_SAMPLE.search(cid)
        _require(match is not None, f"{arm}: candidate_id {cid} has no depth and sample")
        _require(int(match.group(1)) == int(depth), f"{arm}: candidate_id depth != {depth} for {cid}")
        _require(int(match.group(2)) == int(cand.get("sample")), f"{arm}: sample field != candidate_id for {cid}")
        cost = cand.get("logical_loop_tokens")
        _require(isinstance(cost, (int, float)) and not isinstance(cost, bool), f"{arm}: nonnumeric cost on {cid}")
        _require(np.isfinite(cost) and cost >= 0, f"{arm}: nonfinite or negative cost on {cid}")
        by_q[qid].append({
            "sample": int(cand["sample"]),
            "parse_status": cand["parse_status"],
            "finish_reason": finish,
            "answer_key": cand.get("answer_key"),
            "correct": grade["correct"],
            "logical_loop_tokens": float(cost),
        })
    extra_grades = set(grade_by) - seen_cand
    _require(not extra_grades, f"{arm}: grades with no candidate: {sorted(extra_grades)[:3]}")
    missing = [qid for qid in question_ids if qid not in by_q]
    _require(not missing, f"{arm}: missing questions, first {missing[:3]}")
    extra = [qid for qid in by_q if qid not in expected]
    _require(not extra, f"{arm}: unexpected questions, first {extra[:3]}")
    for qid, rows in by_q.items():
        samples = [row["sample"] for row in rows]
        _require(sorted(samples) == list(SAMPLES), f"{arm}: {qid} samples {sorted(samples)} != 0..3")
        _require(len(rows) == 4, f"{arm}: {qid} has {len(rows)} candidates")
    return by_q


def arm_dir(pilot_dir: str, run_seed: int, depth: int) -> str:
    return os.path.join(pilot_dir, f"seed{run_seed}", f"d{depth}_w4_cap8192")


@dataclass
class PilotDataset:
    question_ids: List[str]
    group_ids: List[str]
    texts: List[str]
    text_hashes: List[str]
    levels: np.ndarray
    subjects: List[str]
    depths: Tuple[int, ...]
    run_seeds: Tuple[int, ...]
    vote_correct: np.ndarray
    vote_correct_strict: np.ndarray
    loop_tokens: np.ndarray
    audit: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.question_ids)


def group_ids_for_texts(question_ids: Sequence[str], texts: Sequence[str]) -> Tuple[List[str], dict]:
    """Questions with the same whitespace-normalized text share a group. No paraphrase list is invented."""
    buckets: Dict[str, List[str]] = defaultdict(list)
    normalized = [normalize_whitespace(text) for text in texts]
    for qid, norm in zip(question_ids, normalized):
        buckets[norm].append(qid)
    groups = []
    shared = []
    for qid, norm in zip(question_ids, normalized):
        members = buckets[norm]
        if len(members) == 1:
            groups.append(qid)
        else:
            digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
            groups.append(f"norm:{digest}")
            shared.append(sorted(members))
    unique_shared = [list(item) for item in {tuple(item) for item in shared}]
    return groups, {
        "normalization": "strip and collapse whitespace",
        "n_distinct_normalized": len(buckets),
        "shared_groups": unique_shared,
        "paraphrase_list": None,
    }


def _outcome_arrays(question_ids, depths, run_seeds, joined) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, n_s, n_d = len(question_ids), len(run_seeds), len(depths)
    vote = np.zeros((n, n_s, n_d), dtype=np.float64)
    strict = np.zeros((n, n_s, n_d), dtype=np.float64)
    cost = np.zeros((n, n_s, n_d), dtype=np.float64)
    q_index = {qid: i for i, qid in enumerate(question_ids)}
    for s, seed in enumerate(run_seeds):
        for d, depth in enumerate(depths):
            for qid, rows in joined[(seed, depth)].items():
                i = q_index[qid]
                vote[i, s, d] = float(vote_from_candidates(rows, strict_cap=False))
                strict[i, s, d] = float(vote_from_candidates(rows, strict_cap=True))
                cost[i, s, d] = float(sum(row["logical_loop_tokens"] for row in rows))
    return vote, strict, cost


def verify_published(data: PilotDataset) -> dict:
    """Stop the experiment when the primary or strict counts do not reproduce."""
    report = {"primary_correct": {}, "primary_mean_loop_tokens": {}, "strict_correct": {}}
    for s, seed in enumerate(data.run_seeds):
        correct = tuple(int(data.vote_correct[:, s, d].sum()) for d in range(len(data.depths)))
        means = tuple(round(float(data.loop_tokens[:, s, d].mean()), 3) for d in range(len(data.depths)))
        strict = tuple(int(data.vote_correct_strict[:, s, d].sum()) for d in range(len(data.depths)))
        report["primary_correct"][str(seed)] = correct
        report["primary_mean_loop_tokens"][str(seed)] = means
        report["strict_correct"][str(seed)] = strict
        _require(correct == PRIMARY_CORRECT[seed],
                 f"run {seed} correct counts {correct} != {PRIMARY_CORRECT[seed]}")
        _require(means == PRIMARY_COST[seed],
                 f"run {seed} mean loop-tokens {means} != {PRIMARY_COST[seed]}")
        _require(strict == STRICT_CORRECT[seed],
                 f"run {seed} strict correct counts {strict} != {STRICT_CORRECT[seed]}")
    return report


def load_outcomes(pilot_dir: str, question_ids: Sequence[str], depths: Sequence[int],
                  run_seeds: Sequence[int]) -> Tuple[dict, List[Tuple[str, dict]], dict]:
    """Join every requested arm. A missing directory, extra question, or broken join is an error."""
    manifests = []
    joined = {}
    source_files = {}
    for seed in run_seeds:
        for depth in depths:
            directory = arm_dir(pilot_dir, seed, depth)
            name = f"seed{seed}/d{depth}_w4_cap8192"
            if not os.path.isdir(directory):
                raise PilotDataError(f"missing arm {name}")
            manifest_path = os.path.join(directory, "manifest.json")
            cand_path = os.path.join(directory, "candidates.jsonl")
            grade_path = os.path.join(directory, "graded.jsonl")
            for path in (manifest_path, cand_path, grade_path):
                _require(os.path.isfile(path), f"missing {path}")
            manifest = _read_json(manifest_path)
            _require(int(manifest["decode_config"]["depth"]) == int(depth), f"{name}: manifest depth mismatch")
            _require(int(manifest["run_seed"]) == int(seed), f"{name}: manifest seed mismatch")
            manifests.append((name, manifest))
            joined[(seed, depth)] = join_arm(
                _read_jsonl(cand_path), _read_jsonl(grade_path),
                question_ids=question_ids, depth=depth, run_seed=seed, arm=name)
            source_files[name] = {
                "manifest_sha256": file_sha256(manifest_path),
                "candidates_sha256": file_sha256(cand_path),
                "graded_sha256": file_sha256(grade_path),
            }
    return joined, manifests, source_files


def load_pilot(root: str, *, depths: Sequence[int] = (1, 2, 3, 4),
               run_seeds: Sequence[int] = (0, 1), reproduce: bool = True) -> PilotDataset:
    """Load the router pilot from the repository root. Set ``reproduce`` for the real experiment."""
    question_path = os.path.join(root, QUESTION_LIST)
    items_path = os.path.join(root, ITEMS_PATH)
    splits_path = os.path.join(root, SPLITS_PATH)
    pilot_dir = os.path.join(root, PILOT_DIR)
    question_ids = load_question_ids(question_path)
    _require(len(question_ids) == 200, f"expected 200 questions, found {len(question_ids)}")

    texts = {}
    for row in _read_jsonl(items_path):
        qid = row.get("question_id")
        if qid in texts:
            raise PilotDataError(f"duplicate question text row {qid}")
        _require(isinstance(row.get("text"), str) and row["text"], f"{qid} has no text")
        texts[qid] = row["text"]
    missing_text = [qid for qid in question_ids if qid not in texts]
    _require(not missing_text, f"question text missing for {missing_text[:3]}")

    splits = _read_json(splits_path)
    _require(SPLIT_NAME in splits.get("splits", {}), f"splits.json has no {SPLIT_NAME}")
    meta = {}
    for item in splits["splits"][SPLIT_NAME]["items"]:
        meta[item["question_id"]] = item
    missing_meta = [qid for qid in question_ids if qid not in meta]
    _require(not missing_meta, f"level metadata missing for {missing_meta[:3]}")

    source_files = {
        "question_list": {"path": QUESTION_LIST, "sha256": file_sha256(question_path)},
        "items": {"path": ITEMS_PATH, "sha256": file_sha256(items_path)},
        "splits": {"path": SPLITS_PATH, "sha256": file_sha256(splits_path)},
        "arms": {},
    }
    joined, manifests, arm_files = load_outcomes(pilot_dir, question_ids, depths, run_seeds)
    source_files["arms"] = arm_files

    manifest_report = check_manifests(manifests, depths, run_seeds)
    ordered_texts = [texts[qid] for qid in question_ids]
    groups, group_report = group_ids_for_texts(question_ids, ordered_texts)
    vote, strict, cost = _outcome_arrays(question_ids, depths, run_seeds, joined)
    _require(np.isfinite(cost).all(), "nonfinite aggregate cost")
    data = PilotDataset(
        question_ids=list(question_ids),
        group_ids=groups,
        texts=ordered_texts,
        text_hashes=[text_sha256(text) for text in ordered_texts],
        levels=np.array([int(meta[qid]["level"]) for qid in question_ids], dtype=np.int64),
        subjects=[str(meta[qid]["subject"]) for qid in question_ids],
        depths=tuple(int(d) for d in depths),
        run_seeds=tuple(int(s) for s in run_seeds),
        vote_correct=vote,
        vote_correct_strict=strict,
        loop_tokens=cost,
    )
    reproduction = verify_published(data) if reproduce else None
    data.audit = {
        "n_questions": data.n,
        "n_candidate_answers": data.n * len(run_seeds) * len(depths) * 4,
        "n_vote_outcomes": data.n * len(run_seeds) * len(depths),
        "depths": list(data.depths),
        "run_seeds": list(data.run_seeds),
        "source_files": source_files,
        "manifests": manifest_report,
        "groups": group_report,
        "reproduction": reproduction,
        "level_counts": {str(k): int(v) for k, v in zip(*np.unique(data.levels, return_counts=True))},
        "features_excluded": [
            "generated answer text", "gold answer", "answer key", "generated length",
            "parse status", "finish reason", "vote confidence", "recorded wait",
        ],
        "cost_definition": "sum of logical_loop_tokens over all four candidates, including wrong, unparseable, and capped outputs",
        "primary_vote": "parse_status ok with a structural answer key, including length_cap",
        "strict_vote": "length_cap candidates cast no vote; their loop-tokens still count",
    }
    return data


def dataset_rows(data: PilotDataset) -> Iterable[dict]:
    """One schema row per question, run, and depth. Text is the model input; labels are targets."""
    for i, qid in enumerate(data.question_ids):
        for s, seed in enumerate(data.run_seeds):
            for d, depth in enumerate(data.depths):
                yield {
                    "question_id": qid,
                    "group_id": data.group_ids[i],
                    "text": data.texts[i],
                    "depth": int(depth),
                    "run_seed": int(seed),
                    "vote_correct": int(data.vote_correct[i, s, d]),
                    "vote_correct_strict_cap": int(data.vote_correct_strict[i, s, d]),
                    "loop_tokens": float(data.loop_tokens[i, s, d]),
                    "source": {
                        "arm": f"seed{seed}/d{depth}_w4_cap8192",
                        "candidates": "candidates.jsonl",
                        "graded": "graded.jsonl",
                        "text": ITEMS_PATH,
                        "metadata": f"{SPLITS_PATH}:{SPLIT_NAME}",
                    },
                    "level": int(data.levels[i]),
                    "subject": data.subjects[i],
                }


def save_dataset_arrays(data: PilotDataset, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "outcomes.npz"),
        vote_correct=data.vote_correct,
        vote_correct_strict=data.vote_correct_strict,
        loop_tokens=data.loop_tokens,
        levels=data.levels,
    )
    with open(os.path.join(directory, "questions.json"), "w") as f:
        json.dump({
            "question_ids": data.question_ids,
            "group_ids": data.group_ids,
            "text_hashes": data.text_hashes,
            "subjects": data.subjects,
            "texts": data.texts,
            "depths": list(data.depths),
            "run_seeds": list(data.run_seeds),
        }, f)
    with open(os.path.join(directory, "dataset.jsonl"), "w") as f:
        for row in dataset_rows(data):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(directory, "data_audit.json"), "w") as f:
        json.dump(data.audit, f, indent=1)
        f.write("\n")
