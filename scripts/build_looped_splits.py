#!/usr/bin/env python3
"""Build (or verify) the question-split manifest for the looped serving study.

Every question gets an ID derived from its dataset revision and text, so splits
do not depend on list positions. Questions the looped (Ouro) runs have already
examined are listed as exposed and are excluded from every fresh split:

  * the 1,000 MATH-Train questions of LoopDelta Stage 0a (300 of them also
    carry Stage-0b generations);
  * the 200 GSM8K test questions of the Ouro pilots (50 of them also used by the
    role pilot).

Fresh MATH-Train questions are drawn per difficulty level in the order of
``sha256(salt | question_id)`` and cut into consecutive blocks: smoke,
calibration, validation and a reserved confirmation block.

The script first proves it is looking at the same data as the historical runs
by reproducing their subset fingerprints, and stops if it cannot.

    python scripts/build_looped_splits.py            # write configs/looped_serving/splits.json
    python scripts/build_looped_splits.py --check    # rebuild and compare with the committed file
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import experiment_manifest as em  # noqa: E402

SALT = "looped-serving-splits-v1"
MATH_TRAIN = "EleutherAI/hendrycks_math"
MATH_TEST = "HuggingFaceH4/MATH-500"
GSM8K = "gsm8k"
# per-level block sizes for fresh MATH-Train questions, in draw order
BLOCKS = (("math_train_smoke", 20), ("math_train_calibration", 80), ("math_train_validation", 40),
          ("math_train_confirmation_reserve", 60))
# fingerprints recorded in the historical env.json files
STAGE0A_FINGERPRINT = "a809d749fce371ba"      # artifacts/loopdelta_run/.../stage0a_r4/env.json
STAGE0B_FINGERPRINT = "204800fffd5569db"      # .../stage0b/env.json
GSM8K_200_FINGERPRINT = "b8488b7fbc045f03"    # artifacts/looped*/**/env.json (n=200)
GSM8K_50_FINGERPRINT = "09f20eefc2b22c80"     # role pilots (n=50)


def revision_of(ds) -> str:
    """Commit of a loaded Hugging Face dataset, read from its cache path."""
    for f in ds.cache_files:
        m = re.search(r"/([0-9a-f]{40})/", f["filename"])
        if m:
            return m.group(1)
    raise RuntimeError("could not read the dataset revision from the cache path")


def item(dataset: str, revision: str, split: str, text: str, **extra) -> Dict:
    return {"question_id": em.question_id(dataset, revision, split, text), "text_hash": em.text_hash(text), **extra}


def build() -> Dict:
    from datasets import load_dataset

    from data import HENDRYCKS_MATH_CONFIGS, load_math
    from looped.loopdelta import fingerprint, load_math_train_stratified
    from looped.tasks import load_gsm8k_subset, subset_fingerprint

    # ---- pins, and proof that this is the data the historical runs saw
    math_revs = {revision_of(load_dataset(MATH_TRAIN, cfg, split="train")) for cfg in HENDRYCKS_MATH_CONFIGS}
    if len(math_revs) != 1:
        raise RuntimeError(f"MATH-Train configs come from different revisions: {math_revs}")
    math_rev = math_revs.pop()
    stage0a = load_math_train_stratified(200, 0)
    stage0b = load_math_train_stratified(60, 0)
    gsm200, gsm50 = load_gsm8k_subset(200, 0), load_gsm8k_subset(50, 0)
    got = (fingerprint(stage0a), fingerprint(stage0b), subset_fingerprint(gsm200), subset_fingerprint(gsm50))
    want = (STAGE0A_FINGERPRINT, STAGE0B_FINGERPRINT, GSM8K_200_FINGERPRINT, GSM8K_50_FINGERPRINT)
    if got != want:
        raise RuntimeError(f"historical subset fingerprints do not reproduce: got {got}, want {want}. "
                           "data.load_math may have fallen back to another dataset; refusing to build splits.")
    gsm_ds = load_dataset(GSM8K, "main", split="test")
    gsm_rev = revision_of(gsm_ds)
    m500 = load_dataset(MATH_TEST, split="test")
    m500_rev = revision_of(m500)

    # ---- MATH-Train
    train = list(load_math("train"))
    b_idx = {it["idx"] for it in stage0b}
    exposed_idx = {it["idx"] for it in stage0a}
    if not b_idx <= exposed_idx:
        raise RuntimeError("Stage-0b questions are not a subset of Stage-0a")
    # The same question can appear twice with different spacing. Keep one copy:
    # an exposed copy if there is one (so its twin can never enter a fresh split),
    # otherwise the first in loader order.
    copies = defaultdict(list)
    for i, r in enumerate(train):
        copies[em.text_hash(r["question"])].append(i)
    duplicate_rows = sorted(i for rows in copies.values() if len(rows) > 1
                            for i in rows if i != min(rows, key=lambda j: (j not in exposed_idx, j)))
    if exposed_idx & set(duplicate_rows):
        raise RuntimeError("two exposed rows share one question text")

    def math_item(i: int, **extra) -> Dict:
        r = train[i]
        return item(MATH_TRAIN, math_rev, "train", r["question"], level=r["level_int"], subject=r["subject"],
                    loader_index=i, **extra)

    splits: Dict[str, Dict] = {}
    splits["math_train_exposed"] = {
        "use": "never for fitting, selection or evaluation of the new study; history only",
        "items": [math_item(it["idx"], exposure=["stage0a", "stage0b"] if it["idx"] in b_idx else ["stage0a"])
                  for it in stage0a],
    }
    m500_hashes = {em.text_hash(r["problem"].strip()) for r in m500}
    fresh = defaultdict(list)
    dropped_for_test_overlap = 0
    for i, r in enumerate(train):
        if i in exposed_idx or i in duplicate_rows or not 1 <= r["level_int"] <= 5:
            continue
        if em.text_hash(r["question"]) in m500_hashes:
            dropped_for_test_overlap += 1
            continue
        fresh[r["level_int"]].append(math_item(i))
    for lv in fresh:
        fresh[lv].sort(key=lambda e: em.sha256_hex(f"{SALT}|{e['question_id']}"))
    uses = {
        "math_train_smoke": "correctness and runtime profiling only; not a quality claim",
        "math_train_calibration": "fit value tables, cost models and stopping rules",
        "math_train_validation": "choose policies, prices and thresholds",
        "math_train_confirmation_reserve": "locked: generate candidates only after the policy is frozen",
    }
    offset = 0
    for name, per_level in BLOCKS:
        items: List[Dict] = []
        for lv in range(1, 6):
            block = fresh[lv][offset:offset + per_level]
            if len(block) != per_level:
                raise RuntimeError(f"level {lv} has too few fresh questions for {name}")
            items += block
        splits[name] = {"use": uses[name], "per_level": per_level, "items": items}
        offset += per_level
    unassigned = {str(lv): len(fresh[lv]) - offset for lv in range(1, 6)}

    # ---- final test sets (listed in full; earlier exposure flagged per item)
    in200 = {it["idx"] for it in gsm200}
    in50 = {it["idx"] for it in gsm50}
    splits["gsm8k_test"] = {
        "use": "final evaluation, configurations frozen first; exposed items are reported, not hidden",
        "prior_use": "the whole split was evaluated repeatedly by the June-October Qwen experiments; "
                     "items tagged looped_pilots_n200 were also run on Ouro",
        "items": [item(GSM8K, gsm_rev, "test", row["question"].strip(), loader_index=i,
                       exposure=(["looped_pilots_n200"] if i in in200 else [])
                       + (["looped_role_pilot_n50"] if i in in50 else []))
                  for i, row in enumerate(gsm_ds)],
    }
    splits["math500_test"] = {
        "use": "final evaluation, configurations frozen first",
        "prior_use": "the whole split was evaluated by the June-October Qwen experiments; no Ouro run has used it",
        "items": [item(MATH_TEST, m500_rev, "test", row["problem"].strip(), loader_index=i,
                       level=int(row["level"]), subject=row["subject"], unique_id=row["unique_id"], exposure=[])
                  for i, row in enumerate(m500)],
    }

    out = {
        "version": 1,
        "salt": SALT,
        "id_scheme": "question_id = q_ + sha256(json[dataset, revision, split, NFC whitespace-collapsed text])[:20]; "
                     "text_hash = sha256(normalised text)[:20]",
        "selection": "fresh MATH-Train questions (not exposed, level 1-5, text not in MATH-500), per level, "
                     "ordered by sha256(salt|question_id), cut into consecutive blocks in the order of BLOCKS",
        "datasets": {
            "math_train": {"repo": MATH_TRAIN, "revision": math_rev, "split": "train",
                           "loader": "data.load_math('train'), 7 subject configs concatenated", "rows": len(train)},
            "gsm8k_test": {"repo": GSM8K, "config": "main", "revision": gsm_rev, "split": "test", "rows": len(gsm_ds)},
            "math500_test": {"repo": MATH_TEST, "revision": m500_rev, "split": "test", "rows": len(m500)},
        },
        "verified_historical_fingerprints": dict(zip(
            ("stage0a_n1000", "stage0b_n300", "gsm8k_pilot_n200", "gsm8k_role_pilot_n50"), want)),
        "checks": {
            "math_train_text_overlap_with_math500": dropped_for_test_overlap,
            "math_train_duplicate_text_loader_rows_dropped": duplicate_rows,
            "math_train_fresh_unassigned_per_level": unassigned,
            "caveat": "overlap is exact-text only; reworded or renumbered variants of a problem are not detected. "
                      "loader_index is a convenience for joining with old artifacts; the question_id is authoritative "
                      "and must be recomputed from the loaded text before a question is used.",
        },
        "splits": splits,
    }
    em.check_split_disjointness(out)
    out["split_hash"] = em.split_content_hash(out)
    return out


def dumps(obj: Dict) -> str:
    """JSON with one question per line, so diffs stay readable."""
    lines = ["{"]
    for k, v in obj.items():
        if k != "splits":
            lines.append(f" {json.dumps(k)}: {json.dumps(v, sort_keys=True)},")
    lines.append(' "splits": {')
    names = list(obj["splits"])
    for n, name in enumerate(names):
        s = obj["splits"][name]
        lines.append(f"  {json.dumps(name)}: {{")
        for k, v in s.items():
            if k != "items":
                lines.append(f"   {json.dumps(k)}: {json.dumps(v)},")
        lines.append('   "items": [')
        last = len(s["items"]) - 1
        lines += [f"    {json.dumps(e, sort_keys=True)}{'' if j == last else ','}" for j, e in enumerate(s["items"])]
        lines.append("   ]")
        lines.append("  }" + ("" if n == len(names) - 1 else ","))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(repo, "configs/looped_serving/splits.json"))
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the existing file; write nothing")
    args = ap.parse_args()
    text = dumps(build())
    built = json.loads(text)
    counts = {k: len(v["items"]) for k, v in built["splits"].items()}
    if args.check:
        with open(args.out) as f:
            same = f.read() == text
        print(f"{'identical' if same else 'DIFFERENT'}: {args.out} (split_hash {built['split_hash']})")
        sys.exit(0 if same else 1)
    if os.path.exists(args.out):
        old = em.load_splits(args.out)
        if old["split_hash"] != built["split_hash"]:
            sys.exit(f"{args.out} exists with split_hash {old['split_hash']}, this build gives "
                     f"{built['split_hash']}. Splits are frozen once used; write a new version to a new path.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {os.path.relpath(args.out, repo)}  split_hash {built['split_hash']}")
    print(counts)
    lv = Counter((name, e["level"]) for name, s in built["splits"].items() if name.startswith("math_train")
                 for e in s["items"])
    print({name: [lv[(name, k)] for k in range(1, 6)] for name in counts if name.startswith("math_train")})
    print(built["checks"])


if __name__ == "__main__":
    main()
