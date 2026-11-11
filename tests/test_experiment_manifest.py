"""Stable IDs, resume rejection and split discipline."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped import experiment_manifest as em  # noqa: E402


def manifest(**over):
    base = dict(
        code_sha="641d891", engine_sha="reference-hf",
        model_revision="ByteDance/Ouro-1.4B@574fa66", tokenizer_revision="ByteDance/Ouro-1.4B@574fa66",
        dataset_revision="EleutherAI/hendrycks_math@21a5633", prompt_hash="p0", split_hash="s0",
        decode_config={"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 2048, "depths": [2, 3, 4]},
        cache_semantics="per-request depth for prompt and decode", attention_backend="sdpa",
        dtype="bfloat16", dependency_versions={"torch": "2.8.0"}, hardware="test",
        seed_scheme=em.SEED_SCHEME, run_seed=0, identity_version="structural-v2",
        grader={"grader": "math-verify", "version": "0"},
    )
    base.update(over)
    return em.RunManifest(**base)


# ---------------------------------------------------------------- IDs

def test_question_id_depends_on_text_and_pin_not_on_whitespace():
    a = em.question_id("ds", "rev1", "train", "What is  2+2?\n")
    assert a == em.question_id("ds", "rev1", "train", " What is 2+2? ")
    assert a != em.question_id("ds", "rev1", "train", "What is 2+3?")
    assert a != em.question_id("ds", "rev2", "train", "What is 2+2?")
    assert a != em.question_id("ds", "rev1", "test", "What is 2+2?")
    assert em.text_hash("What is  2+2?") == em.text_hash("What is 2+2?")
    assert em.text_hash("What is 2+2?") != em.text_hash("what is 2+2?")


def test_candidate_seed_is_independent_of_other_candidates():
    q = em.question_id("ds", "rev", "train", "q")
    c0, c1 = em.candidate_id(q, 3, 0), em.candidate_id(q, 3, 1)
    assert c0 != c1 and em.candidate_id(q, 4, 0) != c0
    s0 = em.candidate_seed(7, q, c0)
    assert s0 == em.candidate_seed(7, q, c0)                    # no hidden state
    assert len({s0, em.candidate_seed(7, q, c1), em.candidate_seed(8, q, c0),
                em.candidate_seed(7, "q_other", c0)}) == 4
    assert 0 <= s0 < 2 ** 63


# ---------------------------------------------------------------- resume

@pytest.mark.parametrize("change,needle", [
    (dict(tokenizer_revision="ByteDance/Ouro-1.4B@other"), "tokenizer_revision"),
    (dict(seed_scheme="position-in-batch"), "seed_scheme"),
    (dict(decode_config={"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 4096, "depths": [2, 3, 4]}),
     "decode_config.max_new_tokens"),
    (dict(run_seed=1), "run_seed"),
    (dict(split_hash="s1"), "split_hash"),
    (dict(identity_version="legacy-v1"), "identity_version"),
    (dict(engine_sha="e0a5515"), "engine_sha"),
])
def test_resume_with_a_changed_configuration_fails_and_names_the_change(tmp_path, change, needle):
    run = str(tmp_path / "run")
    m = manifest()
    assert em.open_run_dir(run, m) == "new"
    assert em.open_run_dir(run, manifest()) == "resume"
    before = open(os.path.join(run, em.MANIFEST_FILE)).read()
    with pytest.raises(em.IncompatibleResume) as err:
        em.open_run_dir(run, manifest(**change))
    assert needle in str(err.value)
    assert open(os.path.join(run, em.MANIFEST_FILE)).read() == before     # never rewritten


def test_rows_without_a_manifest_are_not_adopted(tmp_path):
    run = tmp_path / "old"
    run.mkdir()
    (run / "gen_r3.jsonl").write_text("{}\n")
    with pytest.raises(em.IncompatibleResume):
        em.open_run_dir(str(run), manifest())


def test_edited_manifest_is_detected(tmp_path):
    run = str(tmp_path / "run")
    em.open_run_dir(run, manifest())
    path = os.path.join(run, em.MANIFEST_FILE)
    obj = json.load(open(path))
    obj["run_seed"] = 99
    json.dump(obj, open(path, "w"))
    with pytest.raises(ValueError):
        em.open_run_dir(run, manifest())


def test_row_checks():
    m = manifest()
    ok = [{"run_id": m.run_id, "candidate_id": "a"}, {"run_id": m.run_id, "candidate_id": "b"}]
    em.check_rows(ok, m, expected_candidates=["a", "b", "c"])
    with pytest.raises(em.IncompatibleResume):
        em.check_rows([{"run_id": "run_other", "candidate_id": "a"}], m)
    with pytest.raises(em.IncompatibleResume):
        em.check_rows(ok + ok[:1], m)
    with pytest.raises(em.IncompatibleResume):
        em.check_rows(ok, m, expected_candidates=["a"])


# ---------------------------------------------------------------- contracts

def test_record_contracts_and_gold_guard():
    rec = {k: None for k in em.CANDIDATE_RECORD_FIELDS}
    em.check_record(rec, em.CANDIDATE_RECORD_FIELDS)
    del rec["retire_time"]
    with pytest.raises(ValueError):
        em.check_record(rec, em.CANDIDATE_RECORD_FIELDS)
    em.assert_no_gold({"answer_histogram": {}, "elapsed_ms": 3})
    for leak in ("correct", "gold", "level"):
        with pytest.raises(ValueError):
            em.assert_no_gold({"answer_histogram": {}, leak: 1})


# ---------------------------------------------------------------- splits

def toy_splits():
    def items(texts):
        return [{"question_id": em.question_id("ds", "rev", "train", t), "text_hash": em.text_hash(t)}
                for t in texts]

    s = {"datasets": {"ds": "rev"},
         "splits": {"exposed": {"items": items(["a", "b"])}, "calibration": {"items": items(["c"])},
                    "validation": {"items": items(["d"])}}}
    s["split_hash"] = em.split_content_hash(s)
    return s


def test_split_overlap_and_tampering_are_rejected(tmp_path):
    s = toy_splits()
    path = str(tmp_path / "splits.json")
    json.dump(s, open(path, "w"))
    assert em.split_ids(em.load_splits(path), "calibration") == [em.question_id("ds", "rev", "train", "c")]

    leak = toy_splits()
    leak["splits"]["validation"]["items"].append(leak["splits"]["exposed"]["items"][0])
    leak["split_hash"] = em.split_content_hash(leak)
    json.dump(leak, open(path, "w"))
    with pytest.raises(ValueError, match="in both"):
        em.load_splits(path)

    # the same question text under another revision is still a leak
    other = toy_splits()
    other["splits"]["validation"]["items"].append(
        {"question_id": em.question_id("ds", "rev2", "train", "a"), "text_hash": em.text_hash("a")})
    other["split_hash"] = em.split_content_hash(other)
    with pytest.raises(ValueError, match="in both"):
        em.check_split_disjointness(other)

    edited = toy_splits()
    edited["splits"]["calibration"]["items"].pop()
    json.dump(edited, open(path, "w"))
    with pytest.raises(ValueError, match="split_hash"):
        em.load_splits(path)
