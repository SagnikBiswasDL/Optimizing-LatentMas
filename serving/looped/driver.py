"""Closed-loop executor: run an allocation policy live on a backend.

Questions are served first come, first served. At most ``slots`` candidates run
at once. Follow-up launches of questions already in flight take free slots
before a new question is admitted, and a new question is admitted only when its
whole initial launch fits. A question's latency runs from its admission to the
decision that fixes its answer; candidates cancelled by that decision stop at
once and their slots are free for the next launch.

The driver never sees gold. ``keyer`` turns a completion into the observable
answer key the policy votes on.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from looped import allocation as al

from .backend import CANCELLED, Backend, Completion, GenerationRequest


@dataclass
class QuestionState:
    question_id: str
    prompt: Tuple[int, ...]
    arrival: float
    candidates: Dict[int, Dict] = field(default_factory=dict)     # sample -> record
    done: bool = False

    def views(self) -> List[al.CandidateView]:
        return [al.CandidateView(s, c["depth"], c["status"], c.get("answer_key"), c.get("finish_reason"),
                                 c.get("gen_tokens", 0)) for s, c in sorted(self.candidates.items())]


def run_closed_loop(
    backend: Backend,
    policy: al.Policy,
    questions: Sequence[Tuple[str, Sequence[int]]],
    *,
    slots: int,
    make_request: Callable[[str, Sequence[int], al.Launch], GenerationRequest],
    keyer: Callable[[Completion], Optional[str]],
    on_candidate: Callable[[QuestionState, int, Completion, Optional[float]], None] = lambda *a: None,
    on_decision: Callable[[QuestionState, al.Decision, float, int], None] = lambda *a: None,
    on_question: Callable[[QuestionState, Optional[al.CandidateView], float], None] = lambda *a: None,
    clock=time.perf_counter,
) -> Dict:
    if len(policy.initial()) > slots:
        raise ValueError(f"policy launches {len(policy.initial())} candidates at once but only {slots} slots exist")
    pending = deque(questions)
    follow: deque = deque()                       # (question state, Launch)
    live: Dict[str, Tuple[QuestionState, int]] = {}
    stats = {"launched": 0, "cancelled": 0, "questions": 0, "peak_live": 0}

    def launch(q: QuestionState, l: al.Launch) -> None:
        req = make_request(q.question_id, q.prompt, l)
        q.candidates[l.sample] = {"depth": l.depth, "status": al.RUNNING, "request_id": req.request_id, "request": req}
        backend.submit(req)
        live[req.request_id] = (q, l.sample)
        stats["launched"] += 1

    while pending or follow or live:
        while follow and len(live) < slots:
            launch(*follow.popleft())
        while not follow and pending and slots - len(live) >= len(policy.initial()):
            qid, prompt = pending.popleft()
            q = QuestionState(qid, tuple(prompt), clock())
            for l in policy.initial():
                launch(q, l)
        stats["peak_live"] = max(stats["peak_live"], len(live))
        if not live:
            raise RuntimeError("nothing is running but work remains; slots too small for the policy")
        for comp in backend.step():
            q, sample = live.pop(comp.request_id)
            c = q.candidates[sample]
            c.update(status=al.DONE, finish_reason=comp.finish_reason, gen_tokens=len(comp.token_ids),
                     answer_key=keyer(comp))
            on_candidate(q, sample, comp, None)
            if q.done:
                continue
            decision = policy.decide(q.views())
            now = clock()
            on_decision(q, decision, now, slots - len(live))
            for s in decision.cancel:
                rid = q.candidates[s]["request_id"]
                cancelled = backend.cancel(rid)
                live.pop(rid)
                q.candidates[s].update(status=al.CANCELLED, finish_reason=CANCELLED,
                                       gen_tokens=len(cancelled.token_ids), answer_key=None)
                stats["cancelled"] += 1
                on_candidate(q, s, cancelled, clock())
            follow.extend((q, l) for l in decision.launch)
            if decision.done:
                q.done = True
                stats["questions"] += 1
                on_question(q, al.selected(q.views()), now)
    return stats
