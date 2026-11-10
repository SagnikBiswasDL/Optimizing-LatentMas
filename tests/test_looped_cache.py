"""Cache-correctness and compute-accounting checks for the Ouro adapter.

Uses a tiny random-weight model built from Ouro's own remote modeling code at the
pinned revision (so the loop / cache logic under test is the real one), in fp32 on
CPU. Skips if that code is not in the local HF cache.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped.ouro_adapter import (  # noqa: E402
    OURO_ID, OURO_REVISION, OuroRunner, cache_depth, cache_len, slice_cache,
)

L = 3
T = 4


@pytest.fixture(scope="module")
def runner():
    tr = pytest.importorskip("transformers")
    try:
        cfg = tr.AutoConfig.from_pretrained(OURO_ID, revision=OURO_REVISION, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - offline without cache
        pytest.skip(f"Ouro remote code unavailable: {exc}")
    cfg.update(dict(
        num_hidden_layers=L, hidden_size=64, intermediate_size=128, num_attention_heads=4,
        num_key_value_heads=4, head_dim=16, vocab_size=97, layer_types=["full_attention"] * L,
        max_window_layers=L, total_ut_steps=T, max_position_embeddings=512,
    ))
    torch.manual_seed(0)
    model = tr.AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    )
    with torch.no_grad():  # random init leaves the loop nearly an identity; make loops matter
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return OuroRunner(model=model, tokenizer=None, device="cpu")


def ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, 97, (1, n), generator=g)


def test_depth_executes_exactly_r_loops(runner):
    x = ids(7, 1)
    for r in range(1, T + 1):
        c0, t0 = runner.counter.snapshot()
        _, _, cache = runner.forward(depth=r, input_ids=x)
        c1, t1 = runner.counter.snapshot()
        assert c1 - c0 == r
        assert t1 - t0 == 7 * r
        assert cache_depth(cache, L) == r
    assert runner.model.model.total_ut_steps == T  # restored


def test_upstream_exit_at_step_still_runs_every_loop(runner):
    """The trap: selecting an earlier readout does not reduce executed compute."""
    c0, _ = runner.counter.snapshot()
    runner.model(input_ids=ids(5, 2), exit_at_step=0, use_cache=False)
    c1, _ = runner.counter.snapshot()
    assert c1 - c0 == T


def test_readout_is_last_executed_loop(runner):
    """Depth-r logits equal the upstream full-depth run's loop-r readout."""
    x = ids(6, 3)
    for r in range(1, T + 1):
        lg, _, _ = runner.forward(depth=r, input_ids=x)
        ref = runner.model(input_ids=x, exit_at_step=r - 1, use_cache=False).logits[:, -1].float()
        torch.testing.assert_close(lg, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("r", [1, 2, 3])
def test_first_r_loop_sets_of_full_run_equal_depth_r_run(runner, r):
    prefix, cont = ids(11, 4), ids(5, 5)
    _, _, rec = runner.forward(depth=r, input_ids=prefix)
    _, _, full = runner.forward(depth=T, input_ids=prefix)
    cut = slice_cache(full, r, L)
    assert cache_len(cut) == cache_len(rec) == 11
    for a, b in zip(rec.key_cache + rec.value_cache, cut.key_cache + cut.value_cache):
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    la, _, _ = runner.forward(depth=r, input_ids=cont, cache=rec)
    lb, _, _ = runner.forward(depth=r, input_ids=cont, cache=cut)
    torch.testing.assert_close(la, lb, atol=1e-5, rtol=1e-5)


def test_split_prefill_matches_single_pass(runner):
    """Reusing a stored depth-r cache = processing prefix+continuation in one pass."""
    prefix, cont = ids(9, 6), ids(4, 7)
    whole, _, _ = runner.forward(depth=2, input_ids=torch.cat([prefix, cont], 1))
    _, _, c = runner.forward(depth=2, input_ids=prefix)
    part, _, _ = runner.forward(depth=2, input_ids=cont, cache=c)
    torch.testing.assert_close(whole, part, atol=1e-5, rtol=1e-5)


def test_depth_mismatch_is_rejected(runner):
    _, _, full = runner.forward(depth=T, input_ids=ids(4, 8))
    with pytest.raises(ValueError):
        runner.forward(depth=2, input_ids=ids(2, 9), cache=full)


def test_continuing_a_slice_does_not_mutate_source(runner):
    _, _, full = runner.forward(depth=T, input_ids=ids(6, 10))
    before = [k.clone() for k in full.key_cache]
    cut = slice_cache(full, 2, L)
    runner.forward(depth=2, input_ids=ids(3, 11), cache=cut)
    assert cache_len(full) == 6 and cache_len(cut) == 9
    for a, b in zip(before, full.key_cache):
        assert torch.equal(a, b)


def test_latent_replay_equals_truncation_but_regeneration_differs(runner):
    x = ids(8, 12)
    _, h4, c4 = runner.forward(depth=T, input_ids=x)
    c4, _, e4 = runner.latent_rollout(c4, h4[:, -1], depth=T, steps=3)
    r = 2
    _, hr, cr = runner.forward(depth=r, input_ids=x)
    replay, _, er = runner.latent_rollout(cr, hr[:, -1], depth=r, steps=3, forced=e4)
    torch.testing.assert_close(er, e4)
    cut = slice_cache(c4, r, L)
    for a, b in zip(replay.key_cache, cut.key_cache):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    _, hr2, cr2 = runner.forward(depth=r, input_ids=x)
    _, _, eg = runner.latent_rollout(cr2, hr2[:, -1], depth=r, steps=3)
    assert not torch.allclose(eg, e4, atol=1e-3)  # depth-r latents are different inputs


def test_work_accounting_matches_executed_loops(runner):
    p = ids(10, 13)[0].tolist()
    for r in (1, 3):
        g = runner.generate([p], depth=r, max_new_tokens=6)
        n = len(g.token_ids[0])
        assert g.work[0].prompt_tokens == 10 and g.work[0].gen_tokens == n
        assert g.work[0].loop_tokens == r * (10 + n - 1)
        assert g.executed_token_loops == g.work[0].loop_tokens
        assert g.executed_calls == r * n


def test_finished_rows_leave_batch_without_disturbing_others(runner):
    a, b, c = (ids(n, s)[0].tolist() for n, s in ((7, 16), (9, 17), (4, 18)))
    first = runner.generate([a], depth=2, max_new_tokens=1).token_ids[0][0]
    old = runner.stop_ids
    runner.stop_ids = [first]  # row a stops at its first token, forcing compaction
    try:
        batch = runner.generate([a, b, c], depth=2, max_new_tokens=8)
        assert batch.token_ids[0] == [first] and batch.hit_eos[0]
        for i, p in ((1, b), (2, c)):
            solo = runner.generate([p], depth=2, max_new_tokens=8)
            assert batch.token_ids[i] == solo.token_ids[0]
            assert batch.work[i].loop_tokens == solo.work[0].loop_tokens
    finally:
        runner.stop_ids = old


def test_left_padded_batch_matches_unbatched_greedy(runner):
    a, b = ids(9, 14)[0].tolist(), ids(5, 15)[0].tolist()
    batch = runner.generate([a, b], depth=2, max_new_tokens=8)
    for i, p in enumerate((a, b)):
        solo = runner.generate([p], depth=2, max_new_tokens=8)
        n = len(solo.token_ids[0])
        assert batch.token_ids[i][:n] == solo.token_ids[0]


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_slot_pool_matches_unbatched_greedy(runner, slots):
    """Refilling finished rows (join + left-trim) leaves every request's tokens and
    work identical to decoding it alone."""
    prompts = [ids(n, 20 + n)[0].tolist() for n in (6, 11, 3, 8, 5)]
    old = runner.stop_ids
    # stop on a token some rows hit early, so rows finish at different steps
    runner.stop_ids = [runner.generate([prompts[2]], depth=2, max_new_tokens=3).token_ids[0][-1]]
    try:
        pool = runner.generate_pool(prompts, depth=2, max_new_tokens=7, slots=slots)
        lens = set()
        for i, p in enumerate(prompts):
            solo = runner.generate([p], depth=2, max_new_tokens=7)
            assert pool.token_ids[i] == solo.token_ids[0]
            assert pool.hit_eos[i] == solo.hit_eos[0]
            assert pool.work[i].as_dict() == solo.work[0].as_dict()
            lens.add(len(solo.token_ids[0]))
        assert len(lens) > 1 or slots == 1
    finally:
        runner.stop_ids = old
