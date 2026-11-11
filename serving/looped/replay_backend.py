"""A backend that replays saved candidates on a token clock.

Every live request advances one token per ``step()``, whatever else is running,
so time is counted in decode steps of an ideal batch. That is a length proxy,
not a latency: it ignores prefill, how step time grows with batch size and
depth, and waiting for memory. It exists so that the same policy objects and
the same driver that run live can be evaluated on a saved pool, and so that
policy parameters can be fitted without touching a GPU. Latency claims need a
live run.

A saved candidate is ``(tokens to generate, finish reason, payload)``. A request
whose cap is shorter than the saved candidate is cut at the cap and reported as
``length_cap``; the caller decides what a cut candidate's answer is (normally
none). A request can never be served past what was saved: asking for a longer
cap than the pool was generated with raises unless the saved candidate ended
with EOS.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .backend import CANCELLED, EOS, LENGTH_CAP, Backend, Completion, GenerationRequest, logical_loop_tokens

Saved = Tuple[int, str, object]          # generated tokens, finish reason, payload


class ReplayBackend(Backend):
    name = "replay"

    def __init__(self, lookup: Callable[[GenerationRequest], Saved], pool_cap: Callable[[GenerationRequest], int]) -> None:
        self.lookup, self.pool_cap = lookup, pool_cap
        self.t = 0
        self._live: Dict[str, List] = {}
        self.payload: Dict[str, object] = {}
        self.semantics = {"engine": "replay", "clock": "decode steps of an ideal batch", "prompt_depth": "request"}

    def clock(self) -> float:
        return float(self.t)

    def submit(self, request: GenerationRequest) -> None:
        if request.request_id in self._live:
            raise ValueError(f"duplicate live request {request.request_id}")
        n, reason, payload = self.lookup(request)
        if reason not in (EOS, LENGTH_CAP):
            raise ValueError(f"{request.request_id}: a saved candidate must have finished or hit its cap, not {reason!r}")
        if reason == LENGTH_CAP and request.max_new_tokens > self.pool_cap(request):
            raise ValueError(f"{request.request_id}: the pool was cut at {self.pool_cap(request)} tokens; "
                             f"it cannot serve a cap of {request.max_new_tokens}")
        cut = n > request.max_new_tokens
        self._live[request.request_id] = [request, 0, min(n, request.max_new_tokens), LENGTH_CAP if cut else reason,
                                          self.t, cut]
        self.payload[request.request_id] = payload

    def active(self) -> int:
        return len(self._live)

    def _completion(self, st, reason: str) -> Completion:
        req, done, _, _, start, cut = st
        return Completion(
            request_id=req.request_id, depth=req.depth, prompt_tokens=len(req.prompt_ids), token_ids=[0] * done,
            finish_reason=reason, submit_time=float(start), first_token_time=float(start + 1) if done else None,
            finish_time=float(self.t), logical_loop_tokens=logical_loop_tokens(req.depth, len(req.prompt_ids), done),
            charged_loop_tokens=logical_loop_tokens(req.depth, len(req.prompt_ids), done), detail={"cut_by_cap": cut},
        )

    def step(self) -> List[Completion]:
        self.t += 1
        out = []
        for rid in list(self._live):
            st = self._live[rid]
            st[1] += 1
            if st[1] >= st[2]:
                del self._live[rid]
                out.append(self._completion(st, st[3]))
        return out

    def cancel(self, request_id: str) -> Completion:
        return self._completion(self._live.pop(request_id), CANCELLED)

    def partial(self, request_id: str) -> List[int]:
        return [0] * self._live[request_id][1]

    def stats(self) -> Dict:
        return {"live": len(self._live), "step": self.t}
