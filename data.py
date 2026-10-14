import re
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset

from utils import extract_gold, normalize_answer, extract_boxed_answer, normalize_math_answer


HENDRYCKS_MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def math_level_int(item: Dict[str, Any]) -> int:
    lv = item.get("level")
    if lv is None:
        return 0
    if isinstance(lv, bool):
        return 0
    if isinstance(lv, (int, float)):
        return int(lv)
    m = re.search(r"(\d+)", str(lv))
    return int(m.group(1)) if m else 0


def load_math(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    """MATH (competition math). 'test' -> MATH-500 (standard hard eval subset);
    'train' -> full MATH train (for steering-vector traces / pairs).

    Dataset ids are picked defensively (HF ids drift); verify on the pod and
    adjust if a load fails. Answers are the boxed final answer; grading uses the
    light math normalizer (see utils.normalize_math_answer).
    """
    ds = None
    if split == "test":
        for did in ("HuggingFaceH4/MATH-500",):
            try:
                ds = load_dataset(did, split="test", cache_dir=cache_dir)
                break
            except Exception:
                continue
    else:
        rows = []
        for cfg in HENDRYCKS_MATH_CONFIGS:
            try:
                part = load_dataset(
                    "EleutherAI/hendrycks_math", cfg, split="train", cache_dir=cache_dir)
            except Exception:
                continue
            for item in part:
                rec = dict(item)
                if not rec.get("type") and not rec.get("subject"):
                    rec["type"] = cfg.replace("_", " ")
                rows.append(rec)
        if rows:
            ds = rows
        else:
            for did, cfg in (("nlile/hendrycks-MATH-benchmark", None),
                             ("lighteval/MATH", "all")):
                try:
                    ds = (load_dataset(did, cfg, split="train", cache_dir=cache_dir)
                          if cfg else load_dataset(did, split="train", cache_dir=cache_dir))
                    break
                except Exception:
                    continue
    if ds is None:
        raise RuntimeError("Could not load a MATH dataset; check HF dataset ids in data.load_math")
    for item in ds:
        question = (item.get("problem") or item.get("question") or "").strip()
        solution = item.get("solution", item.get("answer", "")) or ""
        ans = item.get("answer")
        if ans is None or str(ans).strip() == "":
            ans = extract_boxed_answer(solution)
        gold = normalize_math_answer(str(ans) if ans is not None else None)
        if not question or not gold:
            continue
        subject = (item.get("subject") or item.get("type") or "").strip()
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
            "source": "math",
            "subject": subject,
            "level": item.get("level"),
            "level_int": math_level_int(item),
        }


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_aime2025(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
            "source": "aime2025",
        }


def load_aime2024(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
            "source": "aime2024",
        }


def load_aime_pooled(cache_dir: Optional[str] = None) -> Iterable[Dict]:
    """Pool AIME 2024 + 2025 for Gate 1 primary curves (not a probe-only set)."""
    for item in load_aime2024(split="train", cache_dir=cache_dir):
        yield item
    for item in load_aime2025(split="train", cache_dir=cache_dir):
        out = dict(item)
        out.setdefault("source", "aime2025")
        yield out


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        subject = (
            item.get("high_level_domain")
            or item.get("subdomain")
            or item.get("domain")
            or ""
        )
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
            "subject": str(subject).strip() if subject else "",
        }


def load_arc_easy(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        # Map choices
        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        # Map answers
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = 'Pickout proper choice that fits the _ in the following sentence:'
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""

        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace('candidate', item['entry_point'])
        answer += f'\n\ncheck({item["entry_point"]})'
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


# qa data from https://github.com/lupantech/AgentFlow/tree/main
from typing import Iterable, Dict, Optional
from datasets import load_dataset

# Frozen MedQA index splits (300 total). Gate1 used indices [0,100) as eval —
# keep that as the held-out test set for all steering claims.
MEDQA_TEST_END = 100
MEDQA_TRAIN_END = 220  # [100, 220) train/pairs
# [220, 300) development / role-budget mapping


def load_medqa(split=None, subset=None, cache_dir=None):
    """Load MedQA. split: None/all | 'test' | 'train' | 'dev'."""

    ds = load_dataset("json", data_files="./data/medqa.json", split='train')
    rows = []
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])

        choice_map = {"0":"A", "1":"B", "2":"C", "3":"D"}

        answer = None
        for idx, op in enumerate(item['options']):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break
        if answer is None:
            continue

        gold = normalize_answer(answer)
        rows.append({
            "question": question,
            "solution": answer,
            "gold": gold,
        })

    n = len(rows)
    if split in (None, "", "all"):
        selected = rows
    elif split == "test":
        selected = rows[: min(MEDQA_TEST_END, n)]
    elif split == "train":
        selected = rows[MEDQA_TEST_END: min(MEDQA_TRAIN_END, n)]
    elif split == "dev":
        selected = rows[MEDQA_TRAIN_END:n]
    else:
        raise ValueError(f"Unknown MedQA split {split!r}; use train|dev|test|all")

    for i, row in enumerate(selected):
        out = dict(row)
        # Absolute index into the full 300-row file (for provenance).
        if split == "test":
            out["idx"] = i
        elif split == "train":
            out["idx"] = MEDQA_TEST_END + i
        elif split == "dev":
            out["idx"] = MEDQA_TRAIN_END + i
        else:
            out["idx"] = i
        yield out

