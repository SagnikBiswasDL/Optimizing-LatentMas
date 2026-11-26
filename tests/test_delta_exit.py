"""Exit-rule checks that do not load Ouro."""
import torch

from looped.delta_exit import (
    attach_token,
    copy_forward_last_token,
    detach_token,
    gate_exit_index,
    halt_loop_index,
    last_token_delta,
)
from looped.loopdelta import kv_deltas


def test_gate_picks_the_first_loop_whose_cumulative_mass_reaches_the_threshold():
    # Hazards 0.2, 0.5, then the remainder. pdf = 0.2, 0.4, 0.4.
    gates = [torch.tensor([[[logit]]]) for logit in (_logit(0.2), _logit(0.5), 0.0)]
    assert int(gate_exit_index(gates, 0.5)[0, 0]) == 1
    assert int(gate_exit_index(gates, 0.15)[0, 0]) == 0
    assert int(gate_exit_index(gates, 1.0)[0, 0]) == 2


def test_delta_halt_respects_zero_and_infinite_epsilon():
    deltas = [0.4, 0.2, 0.05]
    assert halt_loop_index(deltas, 0.0, 4) == 3
    assert halt_loop_index(deltas, float("inf"), 4) == 0
    assert halt_loop_index(deltas, 0.1, 4) == 3
    assert halt_loop_index(deltas, 0.3, 4) == 2
    assert halt_loop_index(deltas, 0.5, 4) == 1


def test_copy_forward_appends_only_the_halted_token_onto_deeper_loops():
    class Cache:
        pass

    cache = Cache()
    cache.key_cache = []
    cache.value_cache = []
    for loop in range(3):
        for _layer in range(2):
            length = 4 if loop == 0 else 3
            cache.key_cache.append(torch.full((1, 1, length, 1), float(loop)))
            cache.value_cache.append(torch.full((1, 1, length, 1), float(loop + 10)))
    written = copy_forward_last_token(cache, src_loop=0, depth=3, num_layers=2)
    assert written == 4
    assert cache.key_cache[2].shape[2] == 4
    assert float(cache.key_cache[2][0, 0, -1, 0]) == 0.0
    assert float(cache.value_cache[3][0, 0, -1, 0]) == 10.0
    assert float(cache.key_cache[2][0, 0, 0, 0]) == 1.0


def test_detach_and_attach_round_trip_the_last_token():
    class Cache:
        pass

    cache = Cache()
    cache.key_cache = [torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1) for _ in range(2)]
    cache.value_cache = [torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1) + 10 for _ in range(2)]
    saved = detach_token(cache, num_layers=2, loops=[0])
    assert cache.key_cache[0].shape[2] == 5
    assert float(saved[0][1][0, 0, 0, 0]) == 5.0
    attach_token(cache, saved)
    assert cache.key_cache[0].shape[2] == 6
    assert float(cache.value_cache[1][0, 0, -1, 0]) == 15.0


def test_last_token_delta_matches_the_masked_kv_delta():
    class Cache:
        pass

    cache = Cache()
    gen = torch.Generator().manual_seed(0)
    cache.key_cache = [torch.randn(1, 2, 4, 3, generator=gen) for _ in range(4)]
    cache.value_cache = [torch.randn(1, 2, 4, 3, generator=gen) for _ in range(4)]
    mask = torch.zeros(1, 4)
    mask[0, -1] = 1
    block = kv_deltas(cache, num_layers=2, depth=2, pos_mask=mask)
    assert abs(last_token_delta(cache, 2, 1, 0) - float(block[0, 0].mean())) < 1e-5


def _logit(probability: float) -> float:
    p = torch.tensor(probability)
    return float(torch.log(p / (1 - p)))
