"""Request/completion contract shared by every looped generation backend.

A request asks for one candidate at one recurrent depth. What "depth" covers is
a property of the backend, not of the request, and is declared in
``Backend.semantics``:

* ``prompt_depth == "request"``: prompt tokens and every generated token use the
  requested depth. This is what the saved Stage-0 evidence was generated with.
* ``prompt_depth == "full"``: prompt tokens run the model's full depth and the
  first generated token is read from that full-depth state; only later tokens
  use the requested depth.

Policies must not mix evidence from the two without saying so.

Work is reported twice. ``logical_loop_tokens`` is the uniform-depth formula
``depth * (prompt + generated - 1)`` used by the saved runs. ``charged_loop_tokens``
is what the backend's semantics make it execute for the tokens it returned. Work
an engine started on a token it never returned (after a stop or a cancel) is not
in either number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EOS, LENGTH_CAP, CANCELLED, ERROR = "eos", "length_cap", "cancelled", "error"


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt_ids: Tuple[int, ...]
    depth: int
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.prompt_ids:
            raise ValueError("a request needs an id and a non-empty prompt")
        if self.depth < 1 or self.max_new_tokens < 1:
            raise ValueError("depth and max_new_tokens must be positive")


@dataclass
class Completion:
    request_id: str
    depth: int
    prompt_tokens: int
    token_ids: List[int]
    finish_reason: str
    submit_time: float
    first_token_time: Optional[float]
    finish_time: float
    logical_loop_tokens: int
    charged_loop_tokens: int
    detail: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return asdict(self)


def logical_loop_tokens(depth: int, prompt_tokens: int, generated: int) -> int:
    """Uniform-depth work; the last generated token is never fed back."""
    return depth * (prompt_tokens + max(generated, 1) - 1)


class Backend:
    """Submit requests, advance the engine, collect completions, cancel."""

    name = "backend"
    semantics: Dict = {}

    def submit(self, request: GenerationRequest) -> None:
        raise NotImplementedError

    def step(self) -> List[Completion]:
        """Advance once; return the requests that finished during this call."""
        raise NotImplementedError

    def cancel(self, request_id: str) -> Completion:
        raise NotImplementedError

    def active(self) -> int:
        raise NotImplementedError

    def partial(self, request_id: str) -> List[int]:
        """Tokens returned so far for a live request."""
        raise NotImplementedError

    def stats(self) -> Dict:
        return {}

    def close(self) -> None:
        pass


def run_all(backend: Backend, requests: Iterable[GenerationRequest]) -> Dict[str, Completion]:
    """Submit everything, then step until every request has finished."""
    for r in requests:
        backend.submit(r)
    done: Dict[str, Completion] = {}
    while backend.active():
        for c in backend.step():
            done[c.request_id] = c
    return done


def strip_stop(token_ids: Sequence[int], stop_ids: Iterable[int]) -> List[int]:
    stop = set(stop_ids)
    return [t for t in token_ids if t not in stop]
