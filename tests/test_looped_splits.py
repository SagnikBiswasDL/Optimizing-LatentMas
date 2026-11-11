"""The committed split manifest: sizes, balance, disjointness, and that the
exposed set is exactly what the saved artifacts contain."""
import glob
import json
import os
import sys
from collections import Counter

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped import experiment_manifest as em  # noqa: E402

SPLITS = os.path.join(ROOT, "configs/looped_serving/splits.json")
PINS = os.path.join(ROOT, "configs/looped_serving/pins.json")
ARTIFACTS = os.path.join(ROOT, "artifacts/loopdelta_run/artifacts/loopdelta")
FRESH = {"math_train_smoke": 100, "math_train_calibration": 400, "math_train_validation": 200,
         "math_train_confirmation_reserve": 300}


@pytest.fixture(scope="module")
def splits():
    return em.load_splits(SPLITS)            # also checks the hash and disjointness


def test_sizes_and_level_balance(splits):
    sizes = {name: len(s["items"]) for name, s in splits["splits"].items()}
    assert sizes == {"math_train_exposed": 1000, **FRESH, "gsm8k_test": 1319, "math500_test": 500}
    for name, n in FRESH.items():
        levels = Counter(e["level"] for e in splits["splits"][name]["items"])
        assert levels == {lv: n // 5 for lv in range(1, 6)}, name


def test_ids_are_unique_everywhere(splits):
    ids = [e["question_id"] for s in splits["splits"].values() for e in s["items"]]
    assert len(ids) == len(set(ids))
    hashes = [e["text_hash"] for s in splits["splits"].values() for e in s["items"]]
    assert len(hashes) == len(set(hashes))    # no question text in two places, train vs test included


def test_exposure_tags(splits):
    exposed = splits["splits"]["math_train_exposed"]["items"]
    assert sum("stage0b" in e["exposure"] for e in exposed) == 300
    assert all("stage0a" in e["exposure"] for e in exposed)
    gsm = splits["splits"]["gsm8k_test"]["items"]
    assert sum("looped_pilots_n200" in e["exposure"] for e in gsm) == 200
    assert sum("looped_role_pilot_n50" in e["exposure"] for e in gsm) == 50
    assert all("looped_pilots_n200" in e["exposure"] for e in gsm if "looped_role_pilot_n50" in e["exposure"])


def test_exposed_set_matches_the_saved_artifacts(splits):
    if not os.path.isdir(ARTIFACTS):
        pytest.skip("LoopDelta artifacts not present")

    def idx_of(pattern):
        out = set()
        for path in glob.glob(os.path.join(ARTIFACTS, pattern)):
            with open(path) as f:
                out |= {json.loads(line)["idx"] for line in f if line.strip()}
        return out

    exposed = splits["splits"]["math_train_exposed"]["items"]
    assert {e["loader_index"] for e in exposed} == idx_of("stage0a_r*/rows.jsonl")
    assert {e["loader_index"] for e in exposed if "stage0b" in e["exposure"]} == idx_of("stage0b/scored_r*.jsonl")
    seen = {e["loader_index"] for e in exposed}
    for name in FRESH:
        assert not seen & {e["loader_index"] for e in splits["splits"][name]["items"]}, name


def test_pins_agree_with_the_split_manifest(splits):
    pins = json.load(open(PINS))
    assert pins["splits"]["split_hash"] == splits["split_hash"]
    assert pins["datasets"]["EleutherAI/hendrycks_math"] == splits["datasets"]["math_train"]["revision"]
    assert pins["datasets"]["gsm8k"] == splits["datasets"]["gsm8k_test"]["revision"]
    assert pins["datasets"]["HuggingFaceH4/MATH-500"] == splits["datasets"]["math500_test"]["revision"]
    for sha in (pins["repository"]["base_commit"], pins["engine"]["commit"], *pins["models"].values()):
        assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")
