"""Jiayi's diagonal schedules, driven by a scripted cache instead of Ouro."""
import torch

import looped.delta_exit as de
from looped.delta_exit import _PipeTok, _run_convergence, _run_wavefront
from looped.ouro_adapter import Work


class _Runner:
    def __init__(self, layers):
        self.device = torch.device("cpu")
        self.L = layers
        self.stop_ids = []
        self.tok = None


class _Cache:
    def __init__(self, slots, prompt_len):
        self.key_cache = [torch.zeros(1, 1, prompt_len, 1) for _ in range(slots)]
        self.value_cache = [torch.zeros(1, 1, prompt_len, 1) for _ in range(slots)]

    def get_seq_length(self, layer_idx=0):
        key = self.key_cache[layer_idx]
        return 0 if key is None else int(key.shape[2])


def _logits(token_id, vocab=64):
    out = torch.full((1, 1, vocab), -10.0)
    out[0, 0, int(token_id)] = 10.0
    return out


def _install(advance):
    previous = (de._advance_diagonal, de._open_pipe_token)

    def open_token(runner, token_id, rope_pos):
        hidden = torch.zeros(1, 1, 1)
        tok = _PipeTok(token_id, hidden)
        tok.embed = hidden
        tok.rope_pos = int(rope_pos)
        return tok

    de._advance_diagonal = advance
    de._open_pipe_token = open_token
    return previous


def _restore(previous):
    de._advance_diagonal, de._open_pipe_token = previous


def _append(cache, loop, layers, value):
    piece = torch.full((1, 1, 1, 1), float(value))
    for layer in range(layers):
        idx = loop * layers + layer
        cache.key_cache[idx] = torch.cat([cache.key_cache[idx], piece], dim=2)
        cache.value_cache[idx] = torch.cat([cache.value_cache[idx], piece], dim=2)


def test_convergence_diagonal_matches_figure_6():
    # Halt depths 2/4/3/2, the grid in the proposal's Figure 6.
    depth, layers, prompt_len = 4, 2, 2
    halt_at = {0: 1, 1: None, 2: 2, 3: 1}
    seen = []

    def value(ordinal, loop):
        halt = halt_at[ordinal]
        if halt is not None and loop >= halt:
            return float(halt - 1)
        return float(loop)

    def advance(runner, toks, cache, pad_count=0):
        for tok in toks:
            loop = tok.ut_done
            past = cache.get_seq_length(loop * runner.L)
            assert past == tok.pos_index, (tok.ordinal, loop, past, tok.pos_index)
            if loop == 0 and tok.ordinal > 0:
                seen.append(float(cache.key_cache[0][0, 0, tok.pos_index - 1, 0]))
            _append(cache, loop, runner.L, value(tok.ordinal, loop))
            tok.ut_done = loop + 1
            tok.hidden = tok.embed
            tok.logits = _logits(tok.ordinal + 1)

    previous = _install(advance)
    try:
        out = _run_convergence(
            _Runner(layers), _Cache(depth * layers, prompt_len), _logits(0),
            max_new_tokens=4, depth=depth, epsilon=0.1, pad_count=0, rope0=prompt_len,
            work=Work(), t0=0.0,
        )
    finally:
        _restore(previous)

    assert out["schedule"] == [
        [(0, 0, "keep")],
        [(0, 1, "halt"), (1, 0, "keep")],
        [(1, 1, "keep"), (2, 0, "reject")],
        [(1, 2, "keep"), (2, 0, "reject")],
        [(1, 3, "keep"), (2, 0, "reject")],
        [(2, 0, "keep")],
        [(2, 1, "keep"), (3, 0, "reject")],
        [(2, 2, "halt"), (3, 0, "keep")],
        [(3, 1, "halt")],
    ]
    assert out["token_ids"] == [0, 1, 2, 3]
    assert out["chosen_loops"] == [2, 4, 3, 2]
    assert out["rejected_loops"] == 4
    assert out["wavefront_steps"] == 9
    # First try reads the parent's previous repeat. A recompute reads the newer one.
    # The restart after a full-depth parent reads that parent's real first repeat again.
    assert seen == [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]


def test_convergence_with_no_early_halt_restarts_every_word():
    depth, layers = 3, 1

    def advance(runner, toks, cache, pad_count=0):
        for tok in toks:
            loop = tok.ut_done
            assert cache.get_seq_length(loop * runner.L) == tok.pos_index
            _append(cache, loop, runner.L, loop + 1)
            tok.ut_done = loop + 1
            tok.hidden = tok.embed
            tok.logits = _logits(tok.ordinal + 1)

    previous = _install(advance)
    try:
        out = _run_convergence(
            _Runner(layers), _Cache(depth * layers, 1), _logits(0),
            max_new_tokens=3, depth=depth, epsilon=0.0, pad_count=0, rope0=1,
            work=Work(), t0=0.0,
        )
    finally:
        _restore(previous)

    assert out["token_ids"] == [0, 1, 2]
    assert out["chosen_loops"] == [3, 3, 3]
    assert out["rejected_loops"] == 4  # two handoffs, R-1 wasted cells each


def test_wavefront_accepts_matching_drafts_and_restarts_a_miss():
    depth, draft_depth, layers, prompt_len = 3, 1, 1, 2
    gold = [7, 8, 9, 10]

    def run(bad_at):
        cache = _Cache(depth * layers, prompt_len)

        def advance(runner, toks, cache, pad_count=0):
            for tok in toks:
                loop = tok.ut_done
                assert cache.get_seq_length(loop * runner.L) == tok.pos_index, (
                    tok.ordinal, loop, cache.get_seq_length(loop * runner.L), tok.pos_index,
                )
                _append(cache, loop, runner.L, tok.token_id)
                tok.ut_done = loop + 1
                tok.hidden = tok.embed
                nxt = gold[tok.ordinal + 1] if tok.ordinal + 1 < len(gold) else 1
                if tok.ut_done == draft_depth and tok.ordinal in bad_at:
                    nxt = 50
                tok.logits = _logits(nxt)

        previous = _install(advance)
        try:
            out = _run_wavefront(
                _Runner(layers), cache, _logits(gold[0]),
                max_new_tokens=len(gold), depth=depth, draft_depth=draft_depth,
                pad_count=0, rope0=prompt_len, work=Work(), t0=0.0,
            )
        finally:
            _restore(previous)
        return out, cache

    accepted, accepted_cache = run(set())
    assert accepted["token_ids"] == gold
    assert accepted["rejected_loops"] == 0
    assert accepted["drafts_checked"] > 0
    assert accepted["drafts_accepted"] == accepted["drafts_checked"]
    for i, token_id in enumerate(gold):
        assert float(accepted_cache.key_cache[0][0, 0, prompt_len + i, 0]) == token_id

    missed, missed_cache = run({0})
    assert missed["token_ids"] == gold
    assert missed["rejected_loops"] > 0
    for i, token_id in enumerate(gold):
        assert float(missed_cache.key_cache[0][0, 0, prompt_len + i, 0]) == token_id
