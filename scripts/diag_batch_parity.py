"""CPU probe: why does grouped Judger decoding diverge from batch-1 decoding?

Runs the real `pad_caches_left` / `decode_batch` wiring from `seal.latent_eval`
against a tiny randomly-initialised Qwen3 so it can execute on CPU, and reports
what the model actually receives: prepared `position_ids`, the attention mask,
and the first-step logits at batch 1 versus batch >1.

The documented explanation for the divergence is that a shared `cache_position`
opens a per-sequence RoPE gap. That is a claim about *positions*, so it is
checkable directly rather than by inference from downstream accuracy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seal.latent_eval import pad_caches_left, to_legacy  # noqa: E402


def tiny_model(dtype, n_layer=4, hidden=128, vocab=512):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layer,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=hidden // 8,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(dtype).eval()
    return m


def make_cache(model, ids, dtype):
    """Real KV cache produced by actually running the prefix, as in the experiment."""
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


def captured_position_ids(model):
    """Record the position_ids each decoder layer's rotary embedding actually sees."""
    seen = {}

    def hook(_mod, args, kwargs):
        pid = kwargs.get("position_ids")
        if pid is None and len(args) > 1:
            pid = args[1]
        if pid is not None and "pos" not in seen:
            seen["pos"] = pid.detach().clone()
        return None

    h = model.model.rotary_emb.register_forward_pre_hook(hook, with_kwargs=True)
    return seen, h


def greedy(model, ids, past, mask, cp, n_new):
    with torch.no_grad():
        o = model.generate(
            input_ids=ids, attention_mask=mask, past_key_values=past,
            cache_position=cp, max_new_tokens=n_new, do_sample=False, pad_token_id=0,
        )
    return o[:, ids.shape[1]:]


def trial(model, dtype, seed, n_items, n_new=16, ragged_cache=True):
    """One solo-vs-batched greedy comparison. Returns per-item (shift, identical)."""
    torch.manual_seed(seed)
    if ragged_cache:
        cache_lens = torch.randint(4, 20, (n_items,)).tolist()
    else:
        cache_lens = [12] * n_items
    prompt_lens = torch.randint(3, 8, (n_items,)).tolist()
    pre = [torch.randint(10, 400, (1, n)) for n in cache_lens]
    prm = [torch.randint(10, 400, (1, n)) for n in prompt_lens]

    solo = []
    for i in range(n_items):
        c = [make_cache(model, pre[i], dtype)]
        past, pm, pmx = pad_caches_left(c, torch.device("cpu"))
        L = prompt_lens[i]
        fm = torch.cat([pm, torch.ones(1, L, dtype=torch.long)], 1)
        solo.append(greedy(model, prm[i], past, fm, torch.arange(pmx, pmx + L), n_new)[0].tolist())

    P = max(prompt_lens)
    jids = torch.zeros(n_items, P, dtype=torch.long)
    jmask = torch.zeros(n_items, P, dtype=torch.long)
    for i in range(n_items):
        L = prompt_lens[i]
        jids[i, P - L:] = prm[i][0]
        jmask[i, P - L:] = 1
    cs = [make_cache(model, p, dtype) for p in pre]
    past, pmask, pmax = pad_caches_left(cs, torch.device("cpu"))
    bat = greedy(model, jids, past, torch.cat([pmask, jmask], 1),
                 torch.arange(pmax, pmax + P), n_new)
    return [(pmax - cache_lens[i], bat[i].tolist() == solo[i]) for i in range(n_items)]


def run_trials(n_trials, n_items, seeds_from=100):
    print("## 5. Divergence rate across seeds: is it positions, or is it precision?")
    print("If shifted-cache items diverge and unshifted ones do not, the cause is positions.")
    print("If divergence tracks dtype and hits items with shift=0 too, the cause is precision.\n")
    for dt_name in ("float32", "bfloat16"):
        dtype = getattr(torch, dt_name)
        model = tiny_model(dtype)
        shifted = [0, 0]   # [diverged, total]
        unshifted = [0, 0]
        for t in range(n_trials):
            for shift, same in trial(model, dtype, seeds_from + t, n_items):
                bucket = unshifted if shift == 0 else shifted
                bucket[1] += 1
                if not same:
                    bucket[0] += 1
        def pct(b):
            return f"{b[0]}/{b[1]}" + (f" ({100*b[0]/b[1]:.0f}%)" if b[1] else "")
        print(f"{dt_name:9s} batch={n_items}: "
              f"shift>0 diverged {pct(shifted):14s}  shift==0 diverged {pct(unshifted)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--trials", type=int, default=0,
                    help="run the multi-seed divergence-rate study instead of the single probe")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    if args.trials:
        run_trials(args.trials, args.batch)
        return
    dtype = getattr(torch, args.dtype)

    print(f"# Batched-decode parity probe (dtype={args.dtype})")
    import transformers

    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    print(f"attn impl: {tiny_model(dtype).config._attn_implementation}\n")

    model = tiny_model(dtype)

    # Two items with DIFFERENT cache lengths and DIFFERENT prompt lengths -- the
    # ragged case that left-padding has to reconcile.
    torch.manual_seed(1)
    cache_lens = [5, 11]
    prompt_lens = [3, 6]
    pre = [torch.randint(10, 400, (1, n)) for n in cache_lens]
    prm = [torch.randint(10, 400, (1, n)) for n in prompt_lens]

    caches = [make_cache(model, p, dtype) for p in pre]

    # ---- batch-1 reference: each item alone, no padding anywhere ----
    solo_logits, solo_pos = [], []
    for i in range(2):
        seen, h = captured_position_ids(model)
        with torch.no_grad():
            past = caches[i]
            L = prm[i].shape[1]
            cp = torch.arange(cache_lens[i], cache_lens[i] + L)
            out = model(
                input_ids=prm[i],
                attention_mask=torch.ones(1, cache_lens[i] + L, dtype=torch.long),
                past_key_values=past,
                cache_position=cp,
            )
        h.remove()
        solo_logits.append(out.logits[0, -1].float().clone())
        solo_pos.append(seen.get("pos"))

    # ---- batched: the production path ----
    caches = [make_cache(model, p, dtype) for p in pre]  # fresh, prior call mutated them
    pmax_pad = max(prompt_lens)
    jids = torch.zeros(2, pmax_pad, dtype=torch.long)
    jmask = torch.zeros(2, pmax_pad, dtype=torch.long)
    for i in range(2):
        L = prompt_lens[i]
        jids[i, pmax_pad - L :] = prm[i][0]  # LEFT pad, as the fixed code now does
        jmask[i, pmax_pad - L :] = 1

    past, past_mask, pmax = pad_caches_left(caches, torch.device("cpu"))
    full_mask = torch.cat([past_mask, jmask], dim=1)
    cache_position = torch.arange(pmax, pmax + pmax_pad)

    seen, h = captured_position_ids(model)
    with torch.no_grad():
        out = model(
            input_ids=jids,
            attention_mask=full_mask,
            past_key_values=past,
            cache_position=cache_position,
        )
    h.remove()
    batch_logits = out.logits[:, -1].float().clone()
    batch_pos = seen.get("pos")

    print("## 1. Positions actually used")
    print(f"cache lens {cache_lens}, prompt lens {prompt_lens}, pmax={pmax}")
    print(f"shared cache_position passed in: {cache_position.tolist()}")
    print(f"full attention_mask:\n{full_mask}")
    print()
    for i in range(2):
        sp = solo_pos[i][0].tolist() if solo_pos[i] is not None else None
        bp = batch_pos[i].tolist() if batch_pos is not None and batch_pos.shape[0] > 1 else (
            batch_pos[0].tolist() if batch_pos is not None else None
        )
        print(f"item {i}: solo position_ids   = {sp}")
        print(f"item {i}: batched position_ids= {bp}")
        # the real prompt occupies the LAST prompt_lens[i] slots
        L = prompt_lens[i]
        if sp is not None and bp is not None:
            got = bp[-L:]
            want = sp[-L:] if len(sp) >= L else sp
            verdict = "MATCH" if got == want else "MISMATCH"
            print(f"item {i}: real-token positions solo={want} batched={got} -> {verdict}")
        print()

    print("## 2. First-step logits, batch 1 vs batch 2")
    for i in range(2):
        d = (batch_logits[i] - solo_logits[i]).abs()
        am_s = int(solo_logits[i].argmax())
        am_b = int(batch_logits[i].argmax())
        print(
            f"item {i}: max|dlogit|={d.max():.3e}  mean|dlogit|={d.mean():.3e}  "
            f"argmax solo={am_s} batched={am_b} {'SAME' if am_s == am_b else 'FLIPPED'}"
        )
    print()

    print("## 3. Through generate(), which is what the experiment actually calls")
    # generate() may prepare position_ids itself, so the direct-forward result above
    # is not conclusive for the production path. Compare greedy token sequences.
    NEW = 12
    solo_seqs = []
    for i in range(2):
        cs = [make_cache(model, pre[i], dtype)]
        past, pm, pm_max = pad_caches_left(cs, torch.device("cpu"))
        L = prompt_lens[i]
        fm = torch.cat([pm, torch.ones(1, L, dtype=torch.long)], dim=1)
        with torch.no_grad():
            o = model.generate(
                input_ids=prm[i],
                attention_mask=fm,
                past_key_values=past,
                cache_position=torch.arange(pm_max, pm_max + L),
                max_new_tokens=NEW,
                do_sample=False,
                pad_token_id=0,
            )
        solo_seqs.append(o[0, L:].tolist())

    caches2 = [make_cache(model, p, dtype) for p in pre]
    past, past_mask, pmax = pad_caches_left(caches2, torch.device("cpu"))
    full_mask = torch.cat([past_mask, jmask], dim=1)
    with torch.no_grad():
        ob = model.generate(
            input_ids=jids,
            attention_mask=full_mask,
            past_key_values=past,
            cache_position=torch.arange(pmax, pmax + pmax_pad),
            max_new_tokens=NEW,
            do_sample=False,
            pad_token_id=0,
        )
    for i in range(2):
        bseq = ob[i, pmax_pad:].tolist()
        same = bseq == solo_seqs[i]
        print(f"item {i} (cache_len={cache_lens[i]}, pmax={pmax}, shift={pmax - cache_lens[i]}):")
        print(f"  solo   : {solo_seqs[i]}")
        print(f"  batched: {bseq}")
        print(f"  -> {'IDENTICAL' if same else 'DIVERGED'}")
    print()

    print("## 4. Does the padded cache region leak?")
    leg = to_legacy(past)
    k0 = leg[0][0]
    pad_rows = (pmax - min(cache_lens))
    print(f"item 0 has {pad_rows} zero-padded cache rows; mask marks them "
          f"{past_mask[0, :pad_rows].tolist()}")
    print(f"norm of padded K region (should be 0): {k0[0, :, :pad_rows].norm():.3e}")
    print(f"norm of real   K region             : {k0[0, :, pad_rows:].norm():.3e}")


if __name__ == "__main__":
    main()
