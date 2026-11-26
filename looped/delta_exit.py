"""Token-level exit for one frozen Ouro model.

Two readouts, both on the ordinary token-major schedule:

* ``gate`` is Ouro's built-in cumulative-exit rule. Every loop still runs. The
  gate only chooses which loop's hidden state feeds the LM head.
* ``kv_delta`` halts a generated token at the first loop whose cross-loop KV
  delta is below epsilon, skips the remaining loops, and appends that token's
  last computed key/value into the skipped loop slots. Prompt tokens still run
  every loop.

Two decode schedules from the looped-exit proposal sit on top of that:

* ``wfd`` is diagonal wavefront decoding (Ha & Kim): one batched step advances
  every in-flight word by one repeat, the newest word one repeat behind the
  word before it. A word drafted at repeat ``draft_depth`` is kept only when
  the word before it, at full depth, predicts that same id. A miss deletes
  every in-flight word behind the verifier and restarts from the full-depth id.
  Greedy output matches token-major full depth.
* ``diagonal`` is the same one-repeat diagonal with convergence validation.
  The child word's first repeat runs beside the parent's next repeat and is
  kept only when that repeat settles (KV delta below epsilon). Otherwise the
  child repeat is thrown away and recomputed against the parent's newer state.
  If the parent never settles, the child starts over from the parent's final
  repeat and runs one word at a time. The parent's own repeats never depend on
  the child, so its halt depth matches token-major order.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence

import torch

from looped.ouro_adapter import GenResult, Work, sync


def gate_exit_index(gates: Sequence[torch.Tensor], threshold: float) -> torch.Tensor:
    """0-based loop chosen by Ouro's cumulative exit rule.

    ``gates`` are the per-loop early-exit logits, each ``[B, S, 1]`` or ``[B, S]``.
    This matches ``OuroForCausalLM.forward``: a sigmoid hazard, a survival product,
    the cumulative mass, and the first step at or above ``threshold``. If the mass
    never reaches the threshold, the last loop is used.
    """
    if not gates:
        raise ValueError("gate list is empty")
    pdfs = []
    remaining = torch.ones_like(gates[0].squeeze(-1), dtype=torch.float32)
    for idx, gate in enumerate(gates):
        lam = torch.sigmoid(gate.squeeze(-1).float())
        if idx < len(gates) - 1:
            pdfs.append(lam * remaining)
            remaining = remaining * (1.0 - lam)
        else:
            pdfs.append(remaining)
    cumulative = torch.cumsum(torch.stack(pdfs, dim=-1), dim=-1)
    reached = cumulative >= float(threshold)
    steps = torch.argmax(reached.float(), dim=-1)
    last = cumulative.shape[-1] - 1
    return steps.masked_fill(~reached.any(dim=-1), last)


def halt_loop_index(deltas: Sequence[float], epsilon: float, n_loops: int) -> int:
    """0-based loop to keep. ``deltas[k]`` compares loop ``k+1`` with loop ``k``.

    Epsilon 0 never halts early. An infinite epsilon keeps the first loop.
    Otherwise the kept loop is the later side of the first delta below epsilon.
    """
    if n_loops < 1:
        raise ValueError("n_loops must be positive")
    if epsilon == float("inf"):
        return 0
    for k, delta in enumerate(deltas):
        if k + 1 >= n_loops:
            break
        if float(delta) < float(epsilon):
            return k + 1
    return n_loops - 1


def last_token_delta(cache, num_layers: int, newer_loop: int, older_loop: int) -> float:
    """Mean over layers and K/V of the normalised last-position change between two loops.

    Same reduction as ``kv_deltas`` on a mask that keeps only the last position.
    One reduction covers every layer, and the GPU syncs once when the scalar is read.
    """
    ratios = []
    for store in (cache.key_cache, cache.value_cache):
        cur = torch.stack(
            [store[newer_loop * num_layers + layer][:, :, -1, :] for layer in range(num_layers)]
        ).float()
        prev = torch.stack(
            [store[older_loop * num_layers + layer][:, :, -1, :] for layer in range(num_layers)]
        ).float()
        num = (cur - prev).square().sum(dim=(1, 2, 3))
        den = prev.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
        ratios.append(num / den)
    return float(torch.cat(ratios).mean())


def copy_forward_last_token(cache, src_loop: int, depth: int, num_layers: int, position: Optional[int] = None) -> int:
    """Append one token's key/value onto each deeper loop that does not have it yet."""
    src = cache.key_cache[src_loop * num_layers]
    if src is None:
        raise RuntimeError("cannot copy forward a loop that has no keys")
    index = -1 if position is None else position
    src_len = int(index if index >= 0 else src.shape[2] - 1) + 1
    written = 0
    for loop in range(src_loop + 1, depth):
        for layer in range(num_layers):
            idx = loop * num_layers + layer
            dst_k = cache.key_cache[idx]
            dst_v = cache.value_cache[idx]
            if dst_k is None or dst_k.shape[2] >= src_len:
                continue
            if dst_k.shape[2] != src_len - 1:
                raise RuntimeError(
                    f"loop {loop} cache length {dst_k.shape[2]} is not one behind loop {src_loop} ({src_len})"
                )
            take_k = cache.key_cache[src_loop * num_layers + layer][:, :, index:index + 1, :]
            take_v = cache.value_cache[src_loop * num_layers + layer][:, :, index:index + 1, :]
            if index == -1:
                take_k = cache.key_cache[src_loop * num_layers + layer][:, :, -1:, :]
                take_v = cache.value_cache[src_loop * num_layers + layer][:, :, -1:, :]
            cache.key_cache[idx] = torch.cat([dst_k, take_k], dim=2)
            cache.value_cache[idx] = torch.cat([dst_v, take_v], dim=2)
            written += 1
    return written


def detach_token(cache, num_layers: int, loops: Sequence[int]):
    """Remove one token's key/value from the given loops. Returns slices that can be attached again."""
    saved = []
    for ut in loops:
        for layer in range(num_layers):
            idx = ut * num_layers + layer
            key = cache.key_cache[idx]
            value = cache.value_cache[idx]
            if key is None or key.shape[2] < 1:
                raise RuntimeError(f"cannot detach an empty cache slot {idx}")
            saved.append((idx, key[:, :, -1:, :].contiguous(), value[:, :, -1:, :].contiguous()))
            cache.key_cache[idx] = key[:, :, :-1, :].contiguous()
            cache.value_cache[idx] = value[:, :, :-1, :].contiguous()
    return saved


def attach_token(cache, saved) -> None:
    """Put a detached token back on the end of each loop it was removed from."""
    for idx, key, value in saved:
        cache.key_cache[idx] = torch.cat([cache.key_cache[idx], key], dim=2)
        cache.value_cache[idx] = torch.cat([cache.value_cache[idx], value], dim=2)


def _cache_len(cache) -> int:
    if cache is None or not cache.key_cache or cache.key_cache[0] is None:
        return 0
    return int(cache.key_cache[0].shape[2])


def _attention_masks(inner, inputs_embeds, attention_mask, cache_position, past, position_ids):
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    if isinstance(attention_mask, dict):
        return attention_mask
    kwargs = {
        "config": inner.config,
        "input_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past,
        "position_ids": position_ids,
    }
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if getattr(inner, "has_sliding_layers", False):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    return masks


def _matched_random_depth(position: int, mean: float = 3.35) -> int:
    """Halt depth in {2, 3, 4} with mean 3.35. Seeded by position so a rerun matches."""
    draw = random.Random(position).random()
    if draw < 0.15:
        return 2
    if draw < 0.50:
        return 3
    return 4


@torch.no_grad()
def _forward_kv_delta(
    runner, input_ids, cache, attention_mask, position_ids, *,
    depth: int, epsilon: float, rule: str = "delta", reuse: bool = True,
):
    """Run one new word under a token-level halt rule.

    ``delta`` stops when the KV change falls below epsilon. ``accel`` is the
    second-order rule: the change of that delta is below epsilon twice in a
    row, which needs three repeats before it can fire. ``random`` stops at a
    depth drawn to average 3.35 repeats. With ``reuse``, skipped repeats copy
    the kept key/value forward. Without it, those repeats still run so later
    words see real keys, and the next word is still chosen at the halt.
    """
    inner = runner.model.model
    input_ids = input_ids.to(runner.device)
    hidden = inner.embed_tokens(input_ids)
    past = _cache_len(cache)
    n = input_ids.shape[1]
    cache_position = torch.arange(past, past + n, device=runner.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    masks = _attention_masks(inner, hidden, attention_mask, cache_position, cache, position_ids)
    position_embeddings = inner.rotary_emb(hidden, position_ids)
    layers = inner.layers[: inner.config.num_hidden_layers]
    deltas: List[float] = []
    ran = 0
    halt_at = None
    halt_hidden = None
    accel_hits = 0
    random_depth = _matched_random_depth(int(cache_position[-1])) if rule == "random" else None
    for ut in range(depth):
        if epsilon == float("inf") and ut > 0 and halt_at is None:
            break
        for layer in layers:
            hidden = layer(
                hidden,
                attention_mask=masks[layer.attention_type],
                position_ids=position_ids,
                past_key_value=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                current_ut=ut,
            )
        hidden = inner.norm(hidden)
        ran = ut + 1
        if halt_at is None and ut > 0 and epsilon != float("inf"):
            delta = last_token_delta(cache, runner.L, ut, ut - 1)
            deltas.append(delta)
            stop = False
            if rule == "delta" and delta < epsilon:
                stop = True
            elif rule == "accel" and len(deltas) >= 2:
                if abs(deltas[-1] - deltas[-2]) < epsilon:
                    accel_hits += 1
                else:
                    accel_hits = 0
                stop = accel_hits >= 2 and ran >= 3
            elif rule == "random" and ran >= random_depth:
                stop = True
            if stop:
                halt_at = ran - 1
                halt_hidden = hidden
                if reuse:
                    break
        elif halt_at is None and ut == 0 and epsilon == float("inf"):
            halt_at = 0
            halt_hidden = hidden
            break
    chosen = ran - 1 if halt_at is None else halt_at
    if reuse and chosen < depth - 1:
        copy_forward_last_token(cache, chosen, depth, runner.L)
    read_hidden = hidden if halt_hidden is None else halt_hidden
    logits = runner.model.lm_head(read_hidden[:, -1, :]).float()
    return logits, cache, chosen, ran, deltas


@torch.no_grad()
def _forward_step(
    runner, input_ids, cache, attention_mask, position_ids, *,
    depth: int, mode: str, threshold: float, epsilon: float,
    rule: str = "delta", reuse: bool = True,
):
    inner = runner.model.model
    past = _cache_len(cache)
    n = input_ids.shape[1]
    decoding = n == 1 and past > 0
    if mode == "kv_delta" and decoding:
        return _forward_kv_delta(
            runner, input_ids, cache, attention_mask, position_ids,
            depth=depth, epsilon=epsilon, rule=rule, reuse=reuse,
        )
    cache_position = torch.arange(past, past + n, device=runner.device)
    with runner.depth(depth):
        base, hs_list, gate_list = inner(
            input_ids=input_ids.to(runner.device),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    if mode == "gate" and hs_list:
        chosen = int(gate_exit_index(gate_list, threshold)[0, -1])
        executed = depth
        deltas: List[float] = []
    else:
        chosen = depth - 1
        executed = depth
        deltas = []
    hidden = hs_list[chosen][:, -1, :]
    logits = runner.model.lm_head(hidden).float()
    return logits, base.past_key_values, chosen, executed, deltas


class _PipeTok:
    """One generated word. ``ut_done`` repeats have been written and kept."""

    def __init__(self, token_id: int, hidden: torch.Tensor):
        self.token_id = token_id
        self.hidden = hidden
        self.embed = hidden
        self.ctx = None
        self.pos_emb = None
        self.ut_done = 0
        self.logits = None
        self.deltas: List[float] = []
        self.halted = False
        self.copied = False
        self.pos_index = 0
        self.rope_pos = 0
        self.ordinal = 0


def _greedy(logits: torch.Tensor) -> int:
    flat = logits.reshape(-1, logits.shape[-1])
    return int(torch.argmax(flat, dim=-1)[-1])


def _open_pipe_token(runner, token_id: int, rope_pos: int) -> _PipeTok:
    inner = runner.model.model
    ids = torch.tensor([[token_id]], device=runner.device)
    embed = inner.embed_tokens(ids)
    position_ids = torch.tensor([[rope_pos]], device=runner.device)
    tok = _PipeTok(token_id, embed)
    tok.pos_emb = inner.rotary_emb(embed, position_ids)
    tok.rope_pos = int(rope_pos)
    return tok


def _masks_for_loop(inner, hidden, pad_count: int, past: int):
    """Attention masks for one new word whose repeat-cache already holds ``past`` keys.

    Loop caches have different lengths on a diagonal, so the mask is rebuilt for
    the repeat about to run. With no padding and one query, the mask is empty and
    the word attends to every cached key.
    """
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        causal_mask_function,
        sliding_window_causal_mask_function,
    )

    q = hidden.shape[1]
    kv_length = past + q
    cache_position = torch.arange(past, past + q, device=hidden.device)
    attn_2d = None
    if pad_count:
        attn_2d = torch.ones((hidden.shape[0], kv_length), dtype=torch.long, device=hidden.device)
        attn_2d[:, :pad_count] = 0
    interface = ALL_MASK_ATTENTION_FUNCTIONS[inner.config._attn_implementation]

    def build(mask_function, local_size=None):
        return interface(
            batch_size=hidden.shape[0],
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=0,
            mask_function=mask_function,
            attention_mask=attn_2d,
            allow_is_causal_skip=local_size is None,
            local_size=local_size,
            dtype=hidden.dtype,
            config=inner.config,
        )

    masks = {"full_attention": build(causal_mask_function)}
    if getattr(inner, "has_sliding_layers", False):
        window = int(inner.config.sliding_window)
        masks["sliding_attention"] = build(sliding_window_causal_mask_function(window), window)
    return masks, cache_position


def _bind_loop(inner, tok: _PipeTok, cache, pad_count: int) -> None:
    """Point this word's attention at the repeat it is about to write."""
    past = int(cache.get_seq_length(tok.ut_done * inner.config.num_hidden_layers))
    if past != tok.pos_index:
        raise RuntimeError(
            f"word {tok.ordinal} repeat {tok.ut_done} would write cache index {past}, "
            f"not its position {tok.pos_index}"
        )
    masks, cache_position = _masks_for_loop(inner, tok.hidden, pad_count, past)
    position_ids = torch.full((1, tok.hidden.shape[1]), tok.rope_pos, device=tok.hidden.device, dtype=torch.long)
    tok.ctx = (masks, tok.pos_emb, position_ids, cache_position)


def _delta_at(cache, num_layers: int, newer_loop: int, older_loop: int, position: int) -> float:
    """Normalised change of one token between two repeats. Same reduction as ``last_token_delta``."""
    ratios = []
    for store in (cache.key_cache, cache.value_cache):
        cur = torch.stack(
            [store[newer_loop * num_layers + layer][:, :, position, :] for layer in range(num_layers)]
        ).float()
        prev = torch.stack(
            [store[older_loop * num_layers + layer][:, :, position, :] for layer in range(num_layers)]
        ).float()
        num = (cur - prev).square().sum(dim=(1, 2, 3))
        den = prev.square().sum(dim=(1, 2, 3)).clamp_min(1e-12)
        ratios.append(num / den)
    return float(torch.cat(ratios).mean())


def _note_halt(runner, tok: _PipeTok, cache, epsilon: float) -> None:
    """Record the cross-loop delta for the repeat just written, and stop if it settled."""
    ut = tok.ut_done - 1
    if ut > 0 and epsilon != float("inf"):
        delta = _delta_at(cache, runner.L, ut, ut - 1, tok.pos_index)
        tok.deltas.append(delta)
        if delta < epsilon:
            tok.halted = True
    elif epsilon == float("inf"):
        tok.halted = True


def _diagonal_layer(layer, hidden, toks, cache):
    """One decoder layer for every word on the current diagonal.

    The MLP runs as one batch. Attention stays per word, because each word is
    writing a different repeat slot.
    """
    residual = hidden
    normed = layer.input_layernorm(hidden)
    pieces = []
    for i, tok in enumerate(toks):
        masks, pos_emb, position_ids, cache_position = tok.ctx
        attn_out, _ = layer.self_attn(
            hidden_states=normed[i:i + 1],
            attention_mask=masks[layer.attention_type],
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=pos_emb,
            current_ut=tok.ut_done,
        )
        pieces.append(attn_out)
    attn = torch.cat(pieces, dim=0)
    hidden = residual + layer.input_layernorm_2(attn)
    residual = hidden
    hidden = layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
    return residual + hidden


def _advance_diagonal(runner, toks, cache, pad_count: int = 0) -> None:
    """Advance every word on this diagonal by one repeat, in one pass through the stack."""
    if not toks:
        return
    uts = [tok.ut_done for tok in toks]
    if len(uts) != len(set(uts)):
        raise RuntimeError(f"diagonal step has two words on the same repeat: {uts}")
    inner = runner.model.model
    for tok in toks:
        _bind_loop(inner, tok, cache, pad_count)
    hidden = torch.cat([tok.hidden for tok in toks], dim=0)
    for layer in inner.layers[: runner.L]:
        hidden = _diagonal_layer(layer, hidden, toks, cache)
    hidden = inner.norm(hidden)
    logits = runner.model.lm_head(hidden).float()
    for i, tok in enumerate(toks):
        tok.hidden = hidden[i:i + 1]
        tok.ut_done += 1
        tok.logits = logits[i:i + 1]


def _erase_token(cache, tok: _PipeTok, num_layers: int) -> int:
    """Delete the repeats this word has written. The word is left ready to run its first repeat again."""
    ran = tok.ut_done
    if ran > 0:
        detach_token(cache, num_layers, list(range(ran)))
    tok.ut_done = 0
    tok.hidden = tok.embed
    tok.logits = None
    tok.deltas = []
    tok.halted = False
    tok.copied = False
    return ran


def _borrow_parent_state(cache, parent: _PipeTok, child_loop: int, num_layers: int):
    """Show the parent's latest repeat in the cache slot the child is about to read.

    The child always writes its first repeat, but a recomputed cell has to attend
    to the parent's newer repeat. Restoring afterwards leaves the parent's own
    repeats untouched, which is what later words and the halt rule both read.
    """
    latest = parent.ut_done - 1
    if latest < 0 or latest == child_loop:
        return lambda: None
    index = parent.pos_index
    saved = []
    for layer in range(num_layers):
        src = latest * num_layers + layer
        dst = child_loop * num_layers + layer
        saved.append((
            dst,
            cache.key_cache[dst][:, :, index:index + 1, :].clone(),
            cache.value_cache[dst][:, :, index:index + 1, :].clone(),
        ))
        cache.key_cache[dst][:, :, index:index + 1, :] = cache.key_cache[src][:, :, index:index + 1, :]
        cache.value_cache[dst][:, :, index:index + 1, :] = cache.value_cache[src][:, :, index:index + 1, :]

    def restore() -> None:
        for dst, key, value in saved:
            cache.key_cache[dst][:, :, index:index + 1, :] = key
            cache.value_cache[dst][:, :, index:index + 1, :] = value

    return restore


def _spawn(runner, cache, token_id: int, rope_pos: int, ordinal: int) -> _PipeTok:
    tok = _open_pipe_token(runner, token_id, rope_pos)
    tok.rope_pos = int(rope_pos)
    tok.ordinal = int(ordinal)
    tok.pos_index = int(cache.get_seq_length(0))
    return tok


def _prefill_fixed(runner, prompt: Sequence[int], depth: int):
    """Prompt words run every repeat. Both schedules need that cache to match token-major decoding."""
    ids, mask = runner._pad_left([list(prompt)])
    work = Work()
    work.add(len(prompt), 0, depth, "prompt")
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    use_mask = mask if bool((mask == 0).any()) else None
    sync(runner.device)
    t0 = time.perf_counter()
    logits, cache, _, _, _ = _forward_step(
        runner, ids, runner.new_cache(depth), use_mask, pos,
        depth=depth, mode="fixed", threshold=1.0, epsilon=0.0,
    )
    return {
        "logits": logits,
        "cache": cache,
        "work": work,
        "t0": t0,
        "pad_count": int((mask[0] == 0).sum()),
        "rope0": int(pos[0, -1]) + 1,
    }


def _pack_generation(
    runner, out_ids, hit_eos, t0, work, chosen, executed, deltas,
    rejected, steps, checked, accepted, schedule,
) -> Dict:
    stop = set(runner.stop_ids)
    text = ""
    if getattr(runner, "tok", None) is not None:
        text = runner.tok.decode([t for t in out_ids if t not in stop], skip_special_tokens=True)
    sync(runner.device)
    work.loop_tokens += sum(executed) + rejected
    work.gen_tokens = len(out_ids)
    wall = time.perf_counter() - t0
    return {
        "text": text,
        "token_ids": out_ids,
        "hit_eos": hit_eos,
        "wall_s": wall,
        "work": work.as_dict(),
        "chosen_loops": chosen,
        "executed_loops": executed,
        "mean_chosen_loop": float(sum(chosen) / len(chosen)) if chosen else None,
        "mean_executed_loops": float(sum(executed) / len(executed)) if executed else None,
        "token_deltas": deltas,
        "rejected_loops": rejected,
        "wavefront_steps": steps,
        "drafts_checked": checked,
        "drafts_accepted": accepted,
        "schedule": schedule,
        "result": GenResult(
            texts=[text], token_ids=[out_ids], hit_eos=[hit_eos], work=[work],
            wall_s=wall, executed_calls=steps, executed_token_loops=sum(executed) + rejected,
        ),
    }


def _depths_for(by_ord: Dict[int, _PipeTok], n: int) -> List[int]:
    return [0 if by_ord.get(i) is None else by_ord[i].ut_done for i in range(n)]


@torch.no_grad()
def _run_convergence(
    runner, cache, first_logits, *,
    max_new_tokens: int, depth: int, epsilon: float, pad_count: int, rope0: int, work, t0,
) -> Dict:
    """Figure 6. One child repeat runs beside the parent's next repeat, and is kept only if that repeat settles."""
    stop = set(runner.stop_ids)
    nxt = _greedy(first_logits)
    nxt_rope = rope0
    nxt_ordinal = 0
    active = None
    draft = None
    kept = False
    out_ids: List[int] = []
    schedule = []
    rejected = 0
    steps = 0
    hit_eos = False
    by_ord: Dict[int, _PipeTok] = {}

    while len(out_ids) < max_new_tokens:
        out_ids.append(nxt)
        if nxt in stop:
            hit_eos = True
            break
        if draft is not None and kept and draft.token_id == nxt:
            active = draft
            by_ord[active.ordinal] = active
            draft = None
            kept = False
        else:
            if draft is not None:
                _erase_token(cache, draft, runner.L)
                draft = None
                kept = False
            active = _spawn(runner, cache, nxt, nxt_rope, nxt_ordinal)
            by_ord[active.ordinal] = active

        while active.ut_done < depth and not active.halted:
            if steps > (max_new_tokens + 1) * depth:
                raise RuntimeError("diagonal schedule did not finish")
            run = [active]
            restore = lambda: None
            if draft is not None:
                _erase_token(cache, draft, runner.L)
                restore = _borrow_parent_state(cache, active, 0, runner.L)
                run.append(draft)
            elif active.ut_done >= 1 and len(out_ids) < max_new_tokens:
                draft = _spawn(
                    runner, cache, _greedy(active.logits), active.rope_pos + 1, active.ordinal + 1,
                )
                restore = _borrow_parent_state(cache, active, 0, runner.L)
                run.append(draft)
            _advance_diagonal(runner, run, cache, pad_count)
            restore()
            _note_halt(runner, active, cache, epsilon)
            steps += 1
            kind = "halt" if active.halted else "keep"
            step = [(active.ordinal, active.ut_done - 1, kind)]
            if draft is not None and draft in run:
                if active.halted:
                    kept = True
                    step.append((draft.ordinal, 0, "keep"))
                else:
                    rejected += 1
                    step.append((draft.ordinal, 0, "reject"))
                    kept = False
                    if active.ut_done >= depth:
                        _erase_token(cache, draft, runner.L)
                        draft = None
            schedule.append(step)

        if active.halted and active.ut_done < depth and not active.copied:
            copy_forward_last_token(cache, active.ut_done - 1, depth, runner.L, active.pos_index)
            active.copied = True
        if len(out_ids) >= max_new_tokens:
            break
        if kept and draft is not None:
            nxt = draft.token_id
            nxt_rope = draft.rope_pos
            nxt_ordinal = draft.ordinal
        else:
            nxt = _greedy(active.logits)
            nxt_rope = active.rope_pos + 1
            nxt_ordinal = active.ordinal + 1

    chosen = _depths_for(by_ord, len(out_ids))
    deltas = [list(by_ord[i].deltas) if by_ord.get(i) is not None else [] for i in range(len(out_ids))]
    return _pack_generation(
        runner, out_ids, hit_eos, t0, work, chosen, list(chosen), deltas,
        rejected, steps, 0, 0, schedule,
    )


@torch.no_grad()
def _run_wavefront(
    runner, cache, first_logits, *,
    max_new_tokens: int, depth: int, draft_depth: int, pad_count: int, rope0: int, work, t0,
) -> Dict:
    """WFD Algorithm 1. Draft at ``draft_depth``, verify at full depth, flush the suffix on a miss."""
    if not 1 <= draft_depth <= depth:
        raise ValueError(f"draft_depth {draft_depth} is outside 1..{depth}")
    stop = set(runner.stop_ids)
    first = _greedy(first_logits)
    out_ids: List[int] = [first]
    pending = [] if first in stop or max_new_tokens <= 1 else [(first, rope0, 0)]
    wave: List[_PipeTok] = []
    by_ord: Dict[int, _PipeTok] = {}
    schedule = []
    rejected = 0
    steps = 0
    checked = 0
    accepted = 0
    hit_eos = first in stop

    while len(out_ids) < max_new_tokens and (pending or wave):
        for token_id, rope, ordinal in pending:
            tok = _spawn(runner, cache, token_id, rope, ordinal)
            by_ord[ordinal] = tok
            wave.append(tok)
        pending = []
        live = [tok for tok in wave if tok.ut_done < depth]
        if not live:
            break
        if steps > (max_new_tokens + depth) * depth:
            raise RuntimeError("wavefront did not finish")
        _advance_diagonal(runner, live, cache, pad_count)
        steps += 1
        schedule.append([(tok.ordinal, tok.ut_done - 1, "keep") for tok in live])
        verifiers = sorted((tok for tok in wave if tok.ut_done == depth), key=lambda tok: tok.rope_pos)
        if len(verifiers) > 1:
            raise RuntimeError(f"one chain has two words ready to verify: {[t.ordinal for t in verifiers]}")
        missed = False
        if verifiers:
            verifier = verifiers[0]
            wave.remove(verifier)
            pred = _greedy(verifier.logits)
            succ = next((tok for tok in wave if tok.rope_pos == verifier.rope_pos + 1), None)
            if len(out_ids) < max_new_tokens:
                out_ids.append(pred)
                if succ is not None:
                    checked += 1
                    if succ.token_id == pred:
                        accepted += 1
                    else:
                        missed = True
                else:
                    missed = True
            if missed:
                suffix = sorted(
                    (tok for tok in wave if tok.rope_pos > verifier.rope_pos),
                    key=lambda tok: tok.pos_index,
                    reverse=True,
                )
                for tok in suffix:
                    rejected += _erase_token(cache, tok, runner.L)
                    wave.remove(tok)
                pending.append((pred, verifier.rope_pos + 1, verifier.ordinal + 1))
            if pred in stop or len(out_ids) >= max_new_tokens:
                hit_eos = pred in stop
                break
        if missed or len(out_ids) >= max_new_tokens:
            continue
        sources = [tok for tok in wave if tok.ut_done == draft_depth]
        if draft_depth == depth and verifiers:
            sources = verifiers
        for source in sources:
            nxt_rope = source.rope_pos + 1
            if any(tok.rope_pos == nxt_rope for tok in wave):
                continue
            if any(rope == nxt_rope for _, rope, _ in pending):
                continue
            pending.append((_greedy(source.logits), nxt_rope, source.ordinal + 1))

    chosen = _depths_for(by_ord, len(out_ids))
    deltas = [list(by_ord[i].deltas) if by_ord.get(i) is not None else [] for i in range(len(out_ids))]
    return _pack_generation(
        runner, out_ids, hit_eos, t0, work, chosen, list(chosen), deltas,
        rejected, steps, checked, accepted, schedule,
    )


@torch.no_grad()
def generate_diagonal(
    runner,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    depth: int,
    epsilon: float = 0.1,
) -> Dict:
    """Greedy decode on the convergence diagonal (proposal Figure 6).

    Prompt words run every repeat. Each new word runs one repeat behind its parent.
    The child's first repeat is kept when the parent's new repeat settles, recomputed
    when it does not, and started over from the parent's final repeat when the parent
    never settles.
    """
    pre = _prefill_fixed(runner, prompt, depth)
    return _run_convergence(
        runner, pre["cache"], pre["logits"],
        max_new_tokens=max_new_tokens, depth=depth, epsilon=epsilon,
        pad_count=pre["pad_count"], rope0=pre["rope0"], work=pre["work"], t0=pre["t0"],
    )


@torch.no_grad()
def generate_wavefront(
    runner,
    prompt: Sequence[int],
    *,
    max_new_tokens: int,
    depth: int,
    draft_depth: int = 1,
) -> Dict:
    """Greedy diagonal wavefront decoding with full-depth verification (proposal B4).

    One step advances every in-flight word by one repeat. A word drafted at
    ``draft_depth`` is committed only when the word before it, at full depth,
    predicts that same id. Greedy output matches token-major decoding at ``depth``.
    """
    pre = _prefill_fixed(runner, prompt, depth)
    return _run_wavefront(
        runner, pre["cache"], pre["logits"],
        max_new_tokens=max_new_tokens, depth=depth, draft_depth=draft_depth,
        pad_count=pre["pad_count"], rope0=pre["rope0"], work=pre["work"], t0=pre["t0"],
    )


@torch.no_grad()
def generate_exit(
    runner,
    prompt: Sequence[int],
    *,
    mode: str,
    max_new_tokens: int,
    depth: int,
    threshold: float = 1.0,
    epsilon: float = 0.1,
    rule: str = "delta",
    reuse: bool = True,
    draft_depth: int = 1,
) -> Dict:
    """Greedy batch-1 generation.

    ``mode`` is ``fixed``, ``gate``, ``kv_delta``, ``diagonal`` (convergence diagonal),
    or ``wfd`` (full-depth diagonal wavefront).
    """
    if mode == "diagonal":
        return generate_diagonal(
            runner, prompt, max_new_tokens=max_new_tokens, depth=depth, epsilon=epsilon,
        )
    if mode == "wfd":
        return generate_wavefront(
            runner, prompt, max_new_tokens=max_new_tokens, depth=depth, draft_depth=draft_depth,
        )
    if mode not in ("fixed", "gate", "kv_delta"):
        raise ValueError(f"unknown readout {mode}")
    ids, mask = runner._pad_left([list(prompt)])
    work = Work()
    work.add(len(prompt), 0, depth, "prompt")
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    use_mask = mask if bool((mask == 0).any()) else None
    sync(runner.device)
    t0 = time.perf_counter()
    logits, cache, prefill_chosen, prefill_executed, prefill_deltas = _forward_step(
        runner, ids, runner.new_cache(depth), use_mask, pos,
        depth=depth, mode=mode, threshold=threshold, epsilon=epsilon,
    )
    full_mask = mask
    last_pos = pos[:, -1]
    saw_pad = bool((mask == 0).any())
    out_ids: List[int] = []
    chosen_loops: List[int] = []
    executed_loops: List[int] = []
    token_deltas: List[List[float]] = []
    hit_eos = False
    stop = set(runner.stop_ids)
    for step in range(max_new_tokens):
        nxt = int(torch.argmax(logits, dim=-1)[0])
        out_ids.append(nxt)
        if not chosen_loops:
            chosen_loops.append(prefill_chosen + 1)
            executed_loops.append(prefill_executed if mode == "kv_delta" else depth)
            token_deltas.append(prefill_deltas)
        if nxt in stop:
            hit_eos = True
            break
        if step == max_new_tokens - 1:
            break
        work.add(1, int(full_mask.sum()), depth, "decode")
        full_mask = torch.cat([full_mask, torch.ones((1, 1), dtype=torch.long, device=runner.device)], 1)
        last_pos = last_pos + 1
        use_mask = full_mask if saw_pad else None
        step_ids = torch.tensor([[nxt]], device=runner.device)
        logits, cache, chosen, executed, deltas = _forward_step(
            runner, step_ids, cache, use_mask, last_pos.view(1, 1),
            depth=depth, mode=mode, threshold=threshold, epsilon=epsilon,
            rule=rule, reuse=reuse,
        )
        chosen_loops.append(chosen + 1)
        executed_loops.append(executed)
        token_deltas.append(deltas)
        if mode == "kv_delta":
            # The ledger above assumed a full-depth decode. Replace that token's share.
            work.loop_tokens += executed - depth
            work.attn_kv_loop += (executed - depth) * int(full_mask.sum())
    sync(runner.device)
    text = runner.tok.decode([t for t in out_ids if t not in stop], skip_special_tokens=True)
    work.gen_tokens = len(out_ids)
    return {
        "text": text,
        "token_ids": out_ids,
        "hit_eos": hit_eos,
        "wall_s": time.perf_counter() - t0,
        "work": work.as_dict(),
        "chosen_loops": chosen_loops,
        "executed_loops": executed_loops,
        "mean_chosen_loop": float(sum(chosen_loops) / len(chosen_loops)) if chosen_loops else None,
        "mean_executed_loops": float(sum(executed_loops) / len(executed_loops)) if executed_loops else None,
        "token_deltas": token_deltas,
        "result": GenResult(
            texts=[text], token_ids=[out_ids], hit_eos=[hit_eos], work=[work],
            wall_s=time.perf_counter() - t0, executed_calls=0, executed_token_loops=0,
        ),
    }
