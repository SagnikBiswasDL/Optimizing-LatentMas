"""Reproducible GSM8K subsets, one answer extractor and one scorer for every arm."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from typing import Dict, List, Optional

from utils import extract_boxed_answer, extract_gold

_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def load_gsm8k_subset(n: int, seed: int = 0, split: str = "test") -> List[Dict]:
    """``n`` items drawn by a seeded permutation of the split; ``idx`` is the
    position in the HF split so subsets can be cross-referenced."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    out = []
    for i in order[:n]:
        row = ds[i]
        out.append({
            "idx": int(i),
            "question": row["question"].strip(),
            "solution": row["answer"],
            "gold": _canon(extract_gold(row["answer"])),
        })
    return out


def subset_fingerprint(items: List[Dict]) -> str:
    blob = json.dumps([(it["idx"], it["question"]) for it in items]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _canon(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
    except ValueError:
        return s.strip().lower() or None
    if not math.isfinite(v):
        return None
    return str(int(v)) if v == int(v) else repr(v)


def extract_answer(text: str) -> Optional[str]:
    """Last ``\\boxed{}`` (balanced braces); else the last number in the text."""
    if not text:
        return None
    box = extract_boxed_answer(text)
    if box is not None:
        nums = _NUM.findall(box.replace("\\!", ""))
        return _canon(nums[-1]) if nums else _canon(box)
    nums = _NUM.findall(text)
    return _canon(nums[-1]) if nums else None


def is_correct(text: str, gold: Optional[str]) -> bool:
    pred = extract_answer(text)
    return pred is not None and gold is not None and pred == gold


# ---------------------------------------------------------- corrupted solutions

_CALC = re.compile(r"<<([^=<>]*)=([^<>]*)>>")


def gold_steps(solution: str) -> List[str]:
    """GSM8K reference steps with calculator annotations removed, answer line dropped."""
    lines = [ln.strip() for ln in solution.split("\n") if ln.strip() and not ln.strip().startswith("####")]
    return [_CALC.sub("", ln) for ln in lines]


def corrupt_solution(item: Dict, seed: int) -> Optional[Dict]:
    """A deliberately wrong worked solution with a known first error.

    One step's stated result (the number after ``=``) is shifted by a small
    delta and later uses of that number are rewritten to the wrong value, so the
    first error is exactly that step. The stated final answer is guaranteed to
    differ from gold. Returns ``None`` when no step carries a numeric result.
    """
    rng = random.Random(seed * 7919 + item["idx"])
    steps = gold_steps(item["solution"])
    cands = []
    for k, ln in enumerate(steps):
        m = re.search(r"=\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)", ln)
        if m:
            cands.append((k, m))
    if not cands:
        return None
    k, m = rng.choice(cands)
    old = m.group(1).replace(",", "")
    try:
        ov = float(old)
    except ValueError:
        return None
    delta = rng.choice([1, 2, 3, 5, 10]) * rng.choice([-1, 1])
    nv = ov + delta if ov + delta > 0 else ov + abs(delta)
    new = str(int(nv)) if nv == int(nv) else f"{nv:g}"
    bad = list(steps)
    bad[k] = bad[k][: m.start(1)] + new + bad[k][m.end(1):]
    for j in range(k + 1, len(bad)):
        bad[j] = re.sub(rf"(?<![\d.]){re.escape(m.group(1))}(?![\d.])", new, bad[j])
    gold = item["gold"]
    if _canon(old) == gold:
        final = _canon(new)
    else:
        try:
            final = _canon(f"{float(gold) + delta:g}")
        except (TypeError, ValueError):
            final = _canon(new)
    if final == gold:
        final = _canon(f"{float(gold) + 1:g}")
    return {
        "steps": bad,
        "error_step": k + 1,          # 1-based, as shown to the critic
        "stated_answer": final,
        "text": "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(bad)) + f"\nFinal answer: {final}",
    }


def correct_solution(item: Dict) -> Dict:
    steps = gold_steps(item["solution"])
    return {
        "steps": steps,
        "error_step": None,
        "stated_answer": item["gold"],
        "text": "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps)) + f"\nFinal answer: {item['gold']}",
    }
