"""Stable IDs, run manifests, resume checks and split manifests for the looped
serving study. Pure Python.

Why this exists. The Stage-0 runs identify a question by ``idx``, its position
in a list that ``data.load_math`` assembles from whichever Hugging Face dataset
happens to load, and they resume a run when the row *count* matches, after
rewriting ``env.json``. Both can silently mix incompatible data. Here:

* a question ID is a hash of (dataset, revision, split, question text);
* a run is described by a ``RunManifest`` whose ``config_hash`` covers code,
  model, tokenizer, data, prompt, split, decoding, backend and seed scheme;
* a run directory can only be resumed by a manifest with the same hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

MANIFEST_FILE = "manifest.json"
SEED_SCHEME = "sha256(json[run_seed,question_id,candidate_id])-63bit-v1"


# ------------------------------------------------------------------ hashing

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_tree_hash(root: str, paths: Sequence[str] = ("looped", "serving/looped", "scripts", "configs/looped_serving/pins.json")) -> str:
    """Content hash of the source a run executes: sha256 over the sorted
    (relative path, file sha256) pairs of every ``.py`` and ``.sh`` file under
    ``paths`` (a path may also name one file). Unlike a commit id typed into a
    file, this is computed from what is on disk, so uncommitted edits show."""
    found = []
    for p in paths:
        full = os.path.join(root, p)
        if os.path.isfile(full):
            found.append(p)
            continue
        for d, _, names in os.walk(full):
            found += [os.path.relpath(os.path.join(d, n), root) for n in names if n.endswith((".py", ".sh"))]
    return sha256_hex(canonical_json([[p.replace(os.sep, "/"), file_sha256(os.path.join(root, p))] for p in sorted(found)]))


def normalize_question_text(text: str) -> str:
    """Unicode NFC, trimmed, runs of whitespace collapsed. Nothing else: two
    questions that differ in any visible character stay different."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_hash(text: str) -> str:
    """Hash of the question text alone; equal across datasets and revisions, so it
    is the right key for leakage checks (train vs test, old run vs new split)."""
    return sha256_hex(normalize_question_text(text))[:20]


def question_id(dataset: str, revision: str, split: str, text: str) -> str:
    """Stable ID of one question in one pinned dataset revision."""
    blob = canonical_json([dataset, revision, split, normalize_question_text(text)])
    return "q_" + sha256_hex(blob)[:20]


def candidate_id(question: str, depth: int, sample: int) -> str:
    """ID of one candidate solution. ``sample`` -1 is the greedy candidate."""
    return f"{question}:d{int(depth)}:s{int(sample)}"


def candidate_seed(run_seed: int, question: str, candidate: str) -> int:
    """Seed of a candidate's own random stream (scheme ``SEED_SCHEME``).

    It depends only on the run seed and the two stable IDs, so a candidate's
    draws do not change with batch position, scheduling or which other
    candidates exist. Per-token draws must come from a generator seeded with
    this value and advanced only by this candidate.
    """
    return int(sha256_hex(canonical_json([int(run_seed), question, candidate]))[:16], 16) % (1 << 63)


# ------------------------------------------------------------------ run manifest

@dataclass(frozen=True)
class RunManifest:
    code_sha: str
    engine_sha: str                 # backend commit, or "reference-hf" for looped/ouro_adapter.py
    model_revision: str             # "<repo>@<commit>"
    tokenizer_revision: str
    dataset_revision: str           # "<repo>@<commit>"
    prompt_hash: str
    split_hash: str
    decode_config: Dict[str, Any]   # temperature, top_p, max_new_tokens, stop ids, depths, n_samples
    cache_semantics: str            # e.g. "per-request depth r for prompt and decode; depth-indexed KV"
    attention_backend: str
    dtype: str
    dependency_versions: Dict[str, str]
    hardware: str
    seed_scheme: str
    run_seed: int
    identity_version: str
    grader: Dict[str, str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return sha256_hex(canonical_json(self.payload()))[:16]

    @property
    def run_id(self) -> str:
        return "run_" + self.config_hash

    def to_json(self) -> Dict[str, Any]:
        return {**self.payload(), "config_hash": self.config_hash}

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "RunManifest":
        names = {f.name for f in fields(cls)}
        unknown = set(obj) - names - {"config_hash"}
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        m = cls(**{k: v for k, v in obj.items() if k in names})
        if "config_hash" in obj and obj["config_hash"] != m.config_hash:
            raise ValueError("stored config_hash does not match the manifest contents (file edited?)")
        return m


class IncompatibleResume(RuntimeError):
    pass


def manifest_diff(old: RunManifest, new: RunManifest) -> List[str]:
    a, b = old.payload(), new.payload()
    out = []
    for k in sorted(a):
        if a[k] == b[k]:
            continue
        if isinstance(a[k], dict) and isinstance(b[k], dict):
            for kk in sorted(set(a[k]) | set(b[k])):
                if a[k].get(kk) != b[k].get(kk):
                    out.append(f"{k}.{kk}: {a[k].get(kk)!r} -> {b[k].get(kk)!r}")
        else:
            out.append(f"{k}: {a[k]!r} -> {b[k]!r}")
    return out


def open_run_dir(path: str, manifest: RunManifest) -> str:
    """Create ``path`` for a new run, or confirm it belongs to ``manifest``.

    Returns ``"new"`` or ``"resume"``. Raises ``IncompatibleResume`` when the
    directory holds another configuration, or holds files but no manifest (rows
    of unknown provenance must not be adopted). The stored manifest is never
    rewritten.
    """
    mpath = os.path.join(path, MANIFEST_FILE)
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = RunManifest.from_json(json.load(f))
        if old.config_hash != manifest.config_hash:
            diff = manifest_diff(old, manifest)
            raise IncompatibleResume(
                f"{path} was written by {old.run_id}, not {manifest.run_id}. Changed: "
                + "; ".join(diff) + ". Use a new output directory.")
        return "resume"
    if os.path.isdir(path) and any(not n.startswith(".") for n in os.listdir(path)):
        raise IncompatibleResume(
            f"{path} has files but no {MANIFEST_FILE}; refusing to adopt rows of unknown provenance")
    os.makedirs(path, exist_ok=True)
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest.to_json(), f, indent=2, sort_keys=True)
    os.replace(tmp, mpath)
    return "new"


def check_rows(rows: Iterable[Mapping[str, Any]], manifest: RunManifest,
               expected_candidates: Optional[Sequence[str]] = None) -> None:
    """Rows reused on resume must carry this run's ID and, when the expected
    candidate list is given, be exactly a duplicate-free subset of it."""
    seen = set()
    allowed = set(expected_candidates) if expected_candidates is not None else None
    for r in rows:
        if r.get("run_id") != manifest.run_id:
            raise IncompatibleResume(f"row from {r.get('run_id')!r} found in a {manifest.run_id} directory")
        c = r.get("candidate_id")
        if c in seen:
            raise IncompatibleResume(f"duplicate candidate row {c!r}")
        if allowed is not None and c not in allowed:
            raise IncompatibleResume(f"row for unexpected candidate {c!r}")
        seen.add(c)


# ------------------------------------------------------------------ record contracts

CANDIDATE_RECORD_FIELDS = (
    "run_id", "question_id", "candidate_id", "depth", "seed", "prompt_tokens",
    "generated_token_ids", "finish_reason", "answer_key", "parse_status",
    "scalar_confidence_definition", "confidence_summary",
    "logical_loop_tokens", "executed_loop_tokens", "probe_work",
    "submit_time", "first_token_time", "finish_time", "cancel_time", "retire_time",
)
DECISION_EVENT_FIELDS = (
    "question_id", "timestamp", "observable_state_hash", "action", "candidates_considered",
    "estimated_gain", "estimated_incremental_ms", "deadline_slack", "KV_budget",
    "policy_version", "fallback_reason",
)
QUESTION_RESULT_FIELDS = (
    "scheduled_arrival", "selected_answer", "completion_time", "deadline",
    "all_candidate_ids", "total_charged_work", "deadline_met", "correctness",
)
FINISH_REASONS = ("eos", "length_cap", "cancelled", "deadline", "error")
# Gold-derived fields. They belong in evaluation records and offline calibration
# only; a runtime policy state containing any of them is a leak.
GOLD_FIELDS = ("correct", "correctness", "gold", "str_ok", "mv_ok", "level", "subject")


def check_record(rec: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [k for k in required if k not in rec]
    if missing:
        raise ValueError(f"record is missing fields: {missing}")


def assert_no_gold(state: Mapping[str, Any]) -> None:
    leaked = sorted(set(state) & set(GOLD_FIELDS))
    if leaked:
        raise ValueError(f"gold-derived fields in policy-visible state: {leaked}")


# ------------------------------------------------------------------ splits

def split_content_hash(splits: Mapping[str, Any]) -> str:
    """Hash over every split's ordered question IDs and the dataset pins."""
    body = {
        "datasets": splits["datasets"],
        "splits": {name: [e["question_id"] for e in s["items"]] for name, s in splits["splits"].items()},
    }
    return sha256_hex(canonical_json(body))[:16]


def load_splits(path: str) -> Dict[str, Any]:
    with open(path) as f:
        splits = json.load(f)
    if splits.get("split_hash") != split_content_hash(splits):
        raise ValueError(f"{path}: split_hash does not match its contents")
    check_split_disjointness(splits)
    return splits


def check_split_disjointness(splits: Mapping[str, Any]) -> None:
    """No question may appear twice, within a split or across splits. Compared by
    text hash, so the same question under another dataset or revision counts."""
    owner: Dict[str, str] = {}
    for name in sorted(splits["splits"]):
        for e in splits["splits"][name]["items"]:
            prev = owner.get(e["text_hash"])
            if prev is not None:
                where = f"twice in {name!r}" if prev == name else f"in both {prev!r} and {name!r}"
                raise ValueError(f"question {e['question_id']} appears {where}")
            owner[e["text_hash"]] = name


def split_ids(splits: Mapping[str, Any], name: str) -> List[str]:
    return [e["question_id"] for e in splits["splits"][name]["items"]]
