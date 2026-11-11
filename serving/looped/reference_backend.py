"""The Hugging Face reference harness (``OuroRunner``) behind the backend contract.

This is the computation the saved Stage-0 evidence came from: prompt and every
generated token at the requested depth. It is slow (one request at a time, no
batching across requests) and exists to say what the right answer is.

``prompt_depth="full"`` emulates the pinned engine's stock behaviour instead:
the prompt runs every loop, the first token is read from the full-depth state,
and later tokens run ``depth`` loops over the first ``depth`` KV planes.

``teacher_forced`` scores a token sequence produced elsewhere, so two backends
can be compared position by position without their sequences having to stay
identical after the first near-tie.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import torch

from looped.ouro_adapter import OuroRunner, slice_cache

from .backend import EOS, LENGTH_CAP, Backend, Completion, GenerationRequest, logical_loop_tokens


class ReferenceBackend(Backend):
    name = "hf-reference"

    def __init__(self, runner: OuroRunner, *, prompt_depth: str = "request", clock=time.perf_counter) -> None:
        if prompt_depth not in ("request", "full"):
            raise ValueError("prompt_depth must be 'request' or 'full'")
        self.runner, self.prompt_depth, self.clock = runner, prompt_depth, clock
        self._queue: List[GenerationRequest] = []
        self._submit: Dict[str, float] = {}
        self.semantics = {
            "engine": "transformers + looped.ouro_adapter.OuroRunner", "prompt_depth": prompt_depth,
            "first_token_depth": prompt_depth, "full_depth": runner.max_depth,
            "dtype": str(runner.model.dtype).replace("torch.", ""), "stop_ids": list(runner.stop_ids),
            "sampling": "CPU torch.Generator seeded per request",
        }

    def submit(self, request: GenerationRequest) -> None:
        self._queue.append(request)
        self._submit[request.request_id] = self.clock()

    def active(self) -> int:
        return len(self._queue)

    def step(self) -> List[Completion]:
        if not self._queue:
            return []
        req = self._queue.pop(0)
        tokens, first = self._generate(req)
        now = self.clock()
        r, full = self.runner, self.runner.max_depth
        stopped = bool(tokens) and tokens[-1] in set(r.stop_ids)
        prompt_d = req.depth if self.prompt_depth == "request" else full
        charged = prompt_d * len(req.prompt_ids) + req.depth * (len(tokens) - 1)
        return [Completion(
            request_id=req.request_id, depth=req.depth, prompt_tokens=len(req.prompt_ids), token_ids=tokens,
            finish_reason=EOS if stopped else LENGTH_CAP, submit_time=self._submit.pop(req.request_id),
            first_token_time=first, finish_time=now,
            logical_loop_tokens=logical_loop_tokens(req.depth, len(req.prompt_ids), len(tokens)),
            charged_loop_tokens=int(charged),
        )]

    def cancel(self, request_id: str) -> Completion:
        raise NotImplementedError("the reference backend runs each request to completion")

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def _generate(self, req: GenerationRequest):
        r = self.runner
        gen = None
        if req.temperature > 0:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(req.seed))
        stop = set(r.stop_ids)
        ids = torch.tensor([list(req.prompt_ids)], dtype=torch.long, device=r.device)
        prompt_d = req.depth if self.prompt_depth == "request" else r.max_depth
        logits, _, cache = r.forward(depth=prompt_d, input_ids=ids)
        if prompt_d != req.depth:
            cache = slice_cache(cache, req.depth, r.L)
        out: List[int] = []
        first: Optional[float] = None
        while True:
            t = int(r._pick(logits, req.temperature, req.top_p, gen)[0])
            out.append(t)
            if first is None:
                first = self.clock()
            if t in stop or len(out) >= req.max_new_tokens:
                return out, first
            logits, _, cache = r.forward(depth=req.depth, input_ids=torch.tensor([[t]], device=r.device), cache=cache)

    # ------------------------------------------------------------------ scoring

    @torch.no_grad()
    def teacher_forced(self, prompt_ids: Sequence[int], token_ids: Sequence[int], depth: int,
                       prompt_depth: Optional[int] = None) -> torch.Tensor:
        """fp32 logits ``[len(token_ids), vocab]``: row ``j`` is this model's
        distribution for generated position ``j`` given the prompt and
        ``token_ids[:j]``. ``prompt_depth`` defaults to ``depth``."""
        r = self.runner
        prompt_depth = depth if prompt_depth is None else prompt_depth
        P = len(prompt_ids)
        if prompt_depth == depth:
            ids = torch.tensor([list(prompt_ids) + list(token_ids[:-1])], dtype=torch.long, device=r.device)
            logits, _, _ = r.forward(depth=depth, input_ids=ids, all_logits=True)
            return logits[0, P - 1:]
        ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=r.device)
        first, _, cache = r.forward(depth=prompt_depth, input_ids=ids)
        if len(token_ids) == 1:
            return first
        cache = slice_cache(cache, depth, r.L)
        rest = torch.tensor([list(token_ids[:-1])], dtype=torch.long, device=r.device)
        logits, _, _ = r.forward(depth=depth, input_ids=rest, cache=cache, all_logits=True)
        return torch.cat([first, logits[0]], 0)


def agreement(logits: torch.Tensor, token_ids: Sequence[int]) -> Dict:
    """How well ``token_ids`` (greedy output of another backend) matches the
    argmax of teacher-forced ``logits``. A mismatch margin is the reference logit
    gap between its own argmax and the other backend's token."""
    toks = torch.tensor(list(token_ids), dtype=torch.long, device=logits.device)
    top = logits.argmax(-1)
    miss = (top != toks).nonzero().view(-1)
    margins = (logits.max(-1).values - logits.gather(1, toks.view(-1, 1)).view(-1))[miss]
    top2 = logits.topk(2, -1).values
    return {
        "positions": int(toks.numel()),
        "mismatches": int(miss.numel()),
        "first_mismatch": int(miss[0]) if miss.numel() else None,
        "first_token_match": bool(top[0] == toks[0]),
        "first_token_margin": float(logits[0].max() - logits[0, toks[0]]),
        "max_mismatch_margin": float(margins.max()) if miss.numel() else 0.0,
        "median_top2_gap": float((top2[:, 0] - top2[:, 1]).median()),
    }
