"""Ouro (looped LM) adapter with real depth control and loop-cache surgery.

Facts about the upstream implementation (``modeling_ouro.py`` at the pinned
revision) that this adapter relies on, each covered by ``tests/test_looped_cache.py``:

* ``OuroModel.forward`` executes ``for current_ut in range(self.total_ut_steps)``
  where ``total_ut_steps`` is an *instance attribute* read at every call. Setting it
  to ``r`` executes exactly ``r`` passes over the 24 shared layers. The upstream
  ``exit_at_step`` / ``exit_threshold`` arguments only choose which loop's hidden
  state feeds ``lm_head`` *after all loops ran*, so they do not save compute and
  are not used here.
* Between loops the hidden state goes through the final ``norm``; the next loop's
  input is that normed state. ``last_hidden_state`` is the normed output of the
  last executed loop. Logits here are ``lm_head(last_hidden_state)``, i.e. the
  readout of loop ``r``, bypassing the early-exit gate entirely.
* ``UniversalTransformerCache`` is a flat list of K/V tensors indexed
  ``loop * num_hidden_layers + layer``. Loop ``i`` of token ``t`` reads only loop
  ``i`` keys of tokens ``<= t``, so "the first ``r`` loop-cache sets" is the slice
  ``[0, r * L)`` of that list.
* One RoPE table (``position_ids``) is shared by every loop: a token has the same
  position in all loop-cache sets.
"""
from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

OURO_ID = "ByteDance/Ouro-1.4B-Thinking"
OURO_BASE_ID = "ByteDance/Ouro-1.4B"
OURO_REVISION = "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d"
OURO_BASE_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
OURO_REVISIONS = {
    OURO_ID: OURO_REVISION,
    OURO_BASE_ID: OURO_BASE_REVISION,
}
SUPPORTED_DEPTHS = (1, 2, 3, 4)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def allocated_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return 0


def peak_bytes(device: torch.device) -> Optional[int]:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# --------------------------------------------------------------------------- cache


def cache_class_for(model) -> type:
    """``UniversalTransformerCache`` from the model's own (remote) module."""
    return sys.modules[type(model.model).__module__].UniversalTransformerCache


def install_mask_size_fix(cache_cls) -> None:
    """The base Ouro checkpoint's cache omits ``get_mask_sizes``.

    Hugging Face then builds the attention mask for ``len(cache_position)`` keys
    instead of the cached length. Prefill of a left-padded batch happens to look
    right for the longest row and corrupts every shorter row. The Thinking
    checkpoint already carries this fix; applying it here makes both checkpoints
    batch correctly.
    """

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0):
        query_length = cache_position.shape[0]
        return self.get_seq_length(layer_idx) + query_length, 0

    cache_cls.get_mask_sizes = get_mask_sizes


def cache_num_entries(cache) -> int:
    return sum(1 for k in cache.key_cache if k is not None)


def cache_depth(cache, num_layers: int) -> int:
    n = cache_num_entries(cache)
    if n % num_layers:
        raise ValueError(f"cache holds {n} entries, not a multiple of {num_layers} layers")
    return n // num_layers


def cache_len(cache) -> int:
    return int(cache.get_seq_length(0)) if cache is not None else 0


def slice_cache(cache, r: int, num_layers: int, *, clone: bool = False):
    """New cache holding only the first ``r`` loop-cache sets of ``cache``.

    Tensors are shared unless ``clone``; that is safe because ``update`` replaces
    entries with ``torch.cat`` results instead of writing in place.
    """
    have = cache_depth(cache, num_layers)
    if r > have:
        raise ValueError(f"asked for {r} loop-cache sets, cache has {have}")
    out = type(cache)(max_cache_size=r * num_layers)
    n = r * num_layers
    out.key_cache = [k.clone() if clone else k for k in cache.key_cache[:n]]
    out.value_cache = [v.clone() if clone else v for v in cache.value_cache[:n]]
    out._seen_tokens = cache_len(cache)
    return out


def clone_cache(cache, num_layers: int):
    return slice_cache(cache, cache_depth(cache, num_layers), num_layers, clone=True)


def cache_bytes(cache) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for k, v in zip(cache.key_cache, cache.value_cache) if k is not None)


def cache_summary(cache, num_layers: int) -> Dict:
    shapes = sorted({tuple(k.shape) for k in cache.key_cache if k is not None})
    lens = sorted({int(k.shape[2]) for k in cache.key_cache if k is not None})
    return {
        "depth": cache_depth(cache, num_layers),
        "entries": cache_num_entries(cache),
        "seq_len": cache_len(cache),
        "distinct_lengths": lens,
        "shapes": [list(s) for s in shapes],
        "mb": round(cache_bytes(cache) / 2 ** 20, 2),
    }


# ------------------------------------------------------------------ work accounting


class LoopCounter:
    """Counts what the shared stack actually executed, via a hook on layer 0.

    Layer 0 runs once per loop per forward, so ``calls`` is executed loops and
    ``token_loops`` is the sum of (batch x query tokens) over those calls. This is
    measured, not derived from the configured depth.
    """

    def __init__(self, model) -> None:
        self.calls = 0
        self.token_loops = 0
        self._h = model.model.layers[0].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, _mod, args, kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        self.calls += 1
        self.token_loops += int(hs.shape[0] * hs.shape[1])

    def snapshot(self) -> Tuple[int, int]:
        return self.calls, self.token_loops

    def remove(self) -> None:
        self._h.remove()


@dataclass
class Work:
    """Per-request compute ledger. ``loop_tokens`` counts real (unpadded) tokens."""

    loop_tokens: int = 0          # sum over processed tokens of executed depth
    attn_kv_loop: int = 0         # sum over processed tokens of kv_len * depth
    prompt_tokens: int = 0
    gen_tokens: int = 0           # output tokens (the last one is never processed)
    latent_steps: int = 0

    def add(self, n_tokens: int, past_len: int, depth: int, kind: str) -> None:
        self.loop_tokens += n_tokens * depth
        # token i of the chunk attends to past_len + i + 1 keys per layer per loop
        self.attn_kv_loop += depth * (n_tokens * past_len + n_tokens * (n_tokens + 1) // 2)
        if kind == "prompt":
            self.prompt_tokens += n_tokens
        elif kind == "latent":
            self.latent_steps += n_tokens

    def merged(self, other: "Work") -> "Work":
        return Work(**{k: getattr(self, k) + getattr(other, k) for k in self.__dataclass_fields__})

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class GenResult:
    texts: List[str]
    token_ids: List[List[int]]
    hit_eos: List[bool]
    work: List[Work]
    wall_s: float
    executed_calls: int
    executed_token_loops: int
    cache: Optional[object] = None
    last_hidden: Optional[torch.Tensor] = None
    extra: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- runner


class OuroRunner:
    def __init__(
        self,
        model_id: str = OURO_ID,
        revision: Optional[str] = None,
        think: Optional[bool] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = "sdpa",
        model=None,
        tokenizer=None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        revision = revision or OURO_REVISIONS.get(model_id)
        self.model_id, self.revision = model_id, revision
        if model is None:
            kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, trust_remote_code=True,
                torch_dtype=getattr(torch, dtype), **kw,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tok = tokenizer
        if self.tok is not None and self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.L = int(self.model.config.num_hidden_layers)
        self.max_depth = int(self.model.config.total_ut_steps)
        self.CacheCls = cache_class_for(self.model)
        install_mask_size_fix(self.CacheCls)
        self.counter = LoopCounter(self.model)
        self.stop_ids = self._stop_ids()
        self._embed_norm = None
        tpl = (self.tok.chat_template or "") if self.tok is not None else ""
        self.supports_thinking = "enable_thinking" in tpl
        # Base Ouro has no thinking tag. Default on only when the template supports it.
        self.think = self.supports_thinking if think is None else bool(think)

    # -- basics -----------------------------------------------------------

    def _stop_ids(self) -> List[int]:
        ids = set()
        if self.tok is None:
            return [int(self.model.config.eos_token_id)]
        if self.tok.eos_token_id is not None:
            ids.add(int(self.tok.eos_token_id))
        for t in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                ids.add(i)
        return sorted(ids)

    @contextlib.contextmanager
    def depth(self, r: int):
        """Execute exactly ``r`` loops inside the block."""
        if not 1 <= r <= self.max_depth:
            raise ValueError(f"depth {r} outside 1..{self.max_depth}")
        old = self.model.model.total_ut_steps
        self.model.model.total_ut_steps = int(r)
        try:
            yield
        finally:
            self.model.model.total_ut_steps = old

    def new_cache(self, r: int):
        return self.CacheCls(max_cache_size=r * self.L)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(ids.to(self.device))

    def chat_ids(self, messages: List[Dict], *, think: bool = True, add_generation_prompt: bool = True) -> List[int]:
        kw = {"enable_thinking": think} if self.supports_thinking else {}
        return self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kw,
        )

    # -- one forward ------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        *,
        depth: int,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        all_logits: bool = False,
    ):
        """Run ``depth`` loops over new tokens appended to ``cache``.

        Returns ``(logits, last_hidden, cache)``; ``logits`` are fp32 for the last
        position (or every position with ``all_logits``), read out of loop ``depth``.
        ``attention_mask`` must span past + new positions when given.
        """
        if cache is None:
            cache = self.new_cache(depth)
        elif cache_num_entries(cache) and cache_depth(cache, self.L) != depth:
            raise ValueError(
                f"cache has {cache_depth(cache, self.L)} loop-cache sets but depth={depth}; "
                "slice_cache() it first, otherwise loops beyond the cache depth would see "
                "stale or missing keys"
            )
        n = (input_ids if input_ids is not None else inputs_embeds).shape[1]
        past = cache_len(cache)
        cache_position = torch.arange(past, past + n, device=self.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        with self.depth(depth):
            out = self.model.model(
                input_ids=None if input_ids is None else input_ids.to(self.device),
                inputs_embeds=None if inputs_embeds is None else inputs_embeds.to(self.device),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
        base = out[0] if isinstance(out, tuple) else out
        h = base.last_hidden_state
        hs = h if all_logits else h[:, -1:, :]
        logits = self.model.lm_head(hs).float()
        return (logits if all_logits else logits[:, -1, :]), h, base.past_key_values

    # -- generation -------------------------------------------------------

    def _pad_left(self, seqs: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        T = max(len(s) for s in seqs)
        pad = int(self.tok.pad_token_id if self.tok is not None else 0)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, T - len(s):] = torch.tensor(list(s), dtype=torch.long)
            mask[i, T - len(s):] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        depth: int,
        max_new_tokens: int,
        cache=None,
        keep_cache: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> GenResult:
        """Greedy (default) batched decode at fixed depth with left padding.

        With ``cache`` (batch 1 only) the prompt is appended to it. Rows leave the
        batch as they stop; ``executed_token_loops`` also counts prompt padding,
        which per-row ``work`` does not.
        """
        B = len(prompts)
        if cache is not None and B != 1:
            raise ValueError("continuing from a cache is supported at batch size 1")
        gen = None
        if temperature > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed or 0))
        ids, mask = self._pad_left(prompts)
        past = cache_len(cache)
        works = [Work() for _ in range(B)]
        for i, p in enumerate(prompts):
            works[i].add(len(p), past, depth, "prompt")
        full_mask = torch.cat([torch.ones((B, past), dtype=torch.long, device=self.device), mask], 1)
        pos = (past + mask.cumsum(-1) - 1).clamp_min(past)
        c0, t0l = self.counter.snapshot()
        sync(self.device)
        t0 = time.perf_counter()
        use_mask = full_mask if bool((mask == 0).any()) else None
        logits, h, cache = self.forward(depth=depth, input_ids=ids, cache=cache,
                                        attention_mask=use_mask, position_ids=pos)
        last_pos = pos[:, -1]
        out_ids: List[List[int]] = [[] for _ in range(B)]
        hit_eos = [False] * B
        stop = set(self.stop_ids)
        rows = list(range(B))  # original index of each live batch row
        for step in range(max_new_tokens):
            nxt = self._pick(logits, temperature, top_p, gen)
            keep = []
            for j, i in enumerate(rows):
                t = int(nxt[j])
                out_ids[i].append(t)
                if t in stop:
                    hit_eos[i] = True
                else:
                    keep.append(j)
            if not keep or step == max_new_tokens - 1:
                break
            if len(keep) < len(rows):
                # Finished rows leave the batch so their cache memory is released.
                kt = torch.tensor(keep, device=self.device)
                for e in range(len(cache.key_cache)):
                    if cache.key_cache[e] is not None:
                        cache.key_cache[e] = cache.key_cache[e].index_select(0, kt)
                        cache.value_cache[e] = cache.value_cache[e].index_select(0, kt)
                full_mask, last_pos = full_mask[kt], last_pos[kt]
                nxt = nxt[torch.tensor(keep)]
                rows = [rows[j] for j in keep]
            # The last sampled token of a row is never fed back, so it costs nothing.
            for j, i in enumerate(rows):
                works[i].add(1, int(full_mask[j].sum()), depth, "decode")
            n = len(rows)
            full_mask = torch.cat([full_mask, torch.ones((n, 1), dtype=torch.long, device=self.device)], 1)
            last_pos = last_pos + 1
            use_mask = full_mask if bool((full_mask == 0).any()) else None
            logits, h, cache = self.forward(
                depth=depth, input_ids=nxt.view(n, 1).to(self.device), cache=cache,
                attention_mask=use_mask, position_ids=last_pos.view(n, 1),
            )
        sync(self.device)
        wall = time.perf_counter() - t0
        c1, t1l = self.counter.snapshot()
        for w, o in zip(works, out_ids):
            w.gen_tokens = len(o)
        texts = [self.tok.decode([t for t in o if t not in stop], skip_special_tokens=True) if self.tok else "" for o in out_ids]
        return GenResult(
            texts=texts, token_ids=out_ids, hit_eos=hit_eos, work=works, wall_s=wall,
            executed_calls=c1 - c0, executed_token_loops=t1l - t0l,
            cache=cache if keep_cache else None,
            last_hidden=h[:, -1, :] if keep_cache else None,
        )

    @torch.no_grad()
    def generate_pool(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        depth: int,
        max_new_tokens: int,
        slots: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> GenResult:
        """Decode many prompts through ``slots`` batch rows, refilling a row as soon
        as its request stops. New requests are prefilled on their own and joined to
        the live batch by left-padding the shorter cache; columns no live row
        attends to are trimmed. Positions are per row, so RoPE is unaffected.
        """
        N = len(prompts)
        gen = None
        if temperature > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
        stop = set(self.stop_ids)
        out_ids: List[List[int]] = [[] for _ in range(N)]
        hit_eos = [False] * N
        works = [Work() for _ in range(N)]
        pending = list(range(N))[::-1]
        rows: List[int] = []
        cache = mask = last_pos = logits = None
        c0, t0l = self.counter.snapshot()
        sync(self.device)
        t0 = time.perf_counter()
        while pending or rows:
            if pending and len(rows) < slots:
                take = [pending.pop() for _ in range(min(slots - len(rows), len(pending)))]
                ids, m = self._pad_left([prompts[i] for i in take])
                pos = (m.cumsum(-1) - 1).clamp_min(0)
                for i in take:
                    works[i].add(len(prompts[i]), 0, depth, "prompt")
                lg, _, nc = self.forward(depth=depth, input_ids=ids, position_ids=pos,
                                         attention_mask=m if bool((m == 0).any()) else None)
                cache, mask = self._join(cache, mask, nc, m)
                last_pos = pos[:, -1] if last_pos is None else torch.cat([last_pos, pos[:, -1]])
                logits = lg if logits is None else torch.cat([logits, lg])
                rows += take
            nxt = self._pick(logits, temperature, top_p, gen)
            keep = []
            for j, i in enumerate(rows):
                t = int(nxt[j])
                out_ids[i].append(t)
                if t in stop:
                    hit_eos[i] = True
                elif len(out_ids[i]) < max_new_tokens:
                    keep.append(j)
            if len(keep) < len(rows):
                kt = torch.tensor(keep, dtype=torch.long, device=self.device)
                for e in range(len(cache.key_cache)):
                    if cache.key_cache[e] is not None:
                        cache.key_cache[e] = cache.key_cache[e].index_select(0, kt)
                        cache.value_cache[e] = cache.value_cache[e].index_select(0, kt)
                mask, last_pos = mask[kt], last_pos[kt]
                nxt = nxt[torch.tensor(keep, dtype=torch.long)]
                rows = [rows[j] for j in keep]
                if not rows:
                    cache = mask = last_pos = logits = None
                    continue
                mask = self._trim_left(cache, mask)
            for j, i in enumerate(rows):
                works[i].add(1, int(mask[j].sum()), depth, "decode")
            n = len(rows)
            mask = torch.cat([mask, torch.ones((n, 1), dtype=torch.long, device=self.device)], 1)
            last_pos = last_pos + 1
            logits, _, cache = self.forward(
                depth=depth, input_ids=nxt.view(n, 1).to(self.device), cache=cache,
                attention_mask=mask if bool((mask == 0).any()) else None,
                position_ids=last_pos.view(n, 1),
            )
        sync(self.device)
        wall = time.perf_counter() - t0
        c1, t1l = self.counter.snapshot()
        for w, o in zip(works, out_ids):
            w.gen_tokens = len(o)
        texts = [self.tok.decode([t for t in o if t not in stop], skip_special_tokens=True) if self.tok else ""
                 for o in out_ids]
        return GenResult(texts=texts, token_ids=out_ids, hit_eos=hit_eos, work=works, wall_s=wall,
                         executed_calls=c1 - c0, executed_token_loops=t1l - t0l)

    def _join(self, cache, mask, new_cache, new_mask):
        if cache is None:
            return new_cache, new_mask
        La, Lb = mask.shape[1], new_mask.shape[1]
        L = max(La, Lb)

        def lpad(x, n):
            if n == 0:
                return x
            z = torch.zeros((*x.shape[:2], n, x.shape[3]), dtype=x.dtype, device=x.device)
            return torch.cat([z, x], 2)

        for e in range(len(cache.key_cache)):
            cache.key_cache[e] = torch.cat([lpad(cache.key_cache[e], L - La), lpad(new_cache.key_cache[e], L - Lb)], 0)
            cache.value_cache[e] = torch.cat([lpad(cache.value_cache[e], L - La), lpad(new_cache.value_cache[e], L - Lb)], 0)
        zeros = lambda b, n: torch.zeros((b, n), dtype=mask.dtype, device=mask.device)  # noqa: E731
        mask = torch.cat([torch.cat([zeros(mask.shape[0], L - La), mask], 1),
                          torch.cat([zeros(new_mask.shape[0], L - Lb), new_mask], 1)], 0)
        return cache, mask

    @staticmethod
    def _trim_left(cache, mask) -> torch.Tensor:
        """Drop leading cache columns that no live row attends to; returns the mask."""
        live = (mask.sum(0) > 0).nonzero()
        k = int(live[0]) if len(live) else 0
        if k > 0:
            for e in range(len(cache.key_cache)):
                if cache.key_cache[e] is not None:
                    cache.key_cache[e] = cache.key_cache[e][:, :, k:].contiguous()
                    cache.value_cache[e] = cache.value_cache[e][:, :, k:].contiguous()
            mask = mask[:, k:]
        return mask

    def split_think(self, ids: Sequence[int]) -> Tuple[str, str, bool]:
        """(reasoning, answer, closed): text before/after the last ``</think>``."""
        end = self.tok.convert_tokens_to_ids("</think>")
        stop = set(self.stop_ids)
        ids = [t for t in ids if t not in stop]
        if end in ids:
            k = len(ids) - 1 - ids[::-1].index(end)
            return (self.tok.decode(ids[:k], skip_special_tokens=True),
                    self.tok.decode(ids[k + 1:], skip_special_tokens=True).strip(), True)
        return self.tok.decode(ids, skip_special_tokens=True), "", False

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float, top_p: float, gen) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1).cpu()
        probs = torch.softmax(logits / temperature, -1).cpu()
        if top_p < 1.0:
            sp, si = probs.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            probs = torch.zeros_like(probs).scatter(-1, si, sp)
        return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=gen).view(-1)

    # -- LatentMAS-style latent steps ------------------------------------

    def input_embed_norm(self) -> torch.Tensor:
        if self._embed_norm is None:
            w = self.model.model.embed_tokens.weight.detach().float()
            self._embed_norm = w.norm(dim=1).mean()
        return self._embed_norm

    def realign(self, h: torch.Tensor) -> torch.Tensor:
        """The repo's default LatentMAS realignment: identity map, then rescale
        to the mean input-embedding norm (``latent_space_realign=False`` path)."""
        hf = h.float()
        out = hf * (self.input_embed_norm().to(hf.device) / hf.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return out.to(self.model.dtype)

    @torch.no_grad()
    def latent_rollout(
        self,
        cache,
        last_hidden: torch.Tensor,
        *,
        depth: int,
        steps: int,
        forced: Optional[torch.Tensor] = None,
        work: Optional[Work] = None,
    ):
        """Append ``steps`` latent positions: ``realign(h_last)`` fed back as the next
        input embedding (batch 1). ``forced`` ([steps, D]) replays saved vectors
        instead of regenerating them. Returns ``(cache, last_hidden, embeds)``."""
        h = last_hidden
        embeds = []
        for s in range(steps):
            e = forced[s].view(1, 1, -1).to(self.device, self.model.dtype) if forced is not None \
                else self.realign(h).view(1, 1, -1)
            embeds.append(e.view(-1).detach().clone())
            past = cache_len(cache)
            if work is not None:
                work.add(1, past, depth, "latent")
            _, hh, cache = self.forward(depth=depth, inputs_embeds=e, cache=cache)
            h = hh[:, -1, :]
        return cache, h, (torch.stack(embeds, 0) if embeds else None)
