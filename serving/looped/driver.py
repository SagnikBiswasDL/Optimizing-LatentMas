"""Closed-loop executor: run an allocation policy live on a backend.

Questions are served first come, first served. At most ``slots`` candidates run
at once. Follow-up launches of questions already in flight take free slots
before a new question is admitted, and a new question is admitted only when its
whole initial launch fits. A question's latency runs from its admission to the
decision that fixes its answer; candidates cancelled by that decision stop at
once and their slots are free for the next launch. Follow-up launches that are
still waiting for a slot when their question is answered are dropped: they
never run and leave no candidate. Every admitted question must end with exactly
one final decision; a policy that leaves a question with nothing running and
no answer is an error, as is a run that ends with an unanswered question.

The driver never sees gold. ``keyer`` turns a completion into the observable
answer key the policy votes on; ``think_of`` says whether the answer opened a
reasoning block. The answer returned is the plurality winner of the finished
candidates unless the policy's final decision names a candidate.
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
        """Candidates in sample order; a launch waiting for a slot has status ``queued``."""
        return [al.CandidateView(s, c["depth"], c["status"], c.get("answer_key"), c.get("finish_reason"),
                                 c.get("gen_tokens", 0), c["mode"], c.get("think"))
                for s, c in sorted(self.candidates.items())]


def run_closed_loop(
    backend: Backend,
    policy: al.Policy,
    questions: Sequence[Tuple[str, Sequence[int]]],
    *,
    slots: int,
    make_request: Callable[[str, Sequence[int], al.Launch], GenerationRequest],
    keyer: Callable[[Completion], Optional[str]],
    think_of: Callable[[Completion, al.Launch], Optional[bool]] = lambda comp, launch: None,
    on_candidate: Callable[[QuestionState, int, Completion, Optional[float]], None] = lambda *a: None,
    on_decision: Callable[[QuestionState, al.Decision, float, int], None] = lambda *a: None,
    on_question: Callable[[QuestionState, Optional[al.CandidateView], float], None] = lambda *a: None,
    clock=time.perf_counter,
) -> Dict:
    if len(policy.initial()) > slots:
        raise ValueError(f"policy launches {len(policy.initial())} candidates at once but only {slots} slots exist")
    pending = deque(questions)
    follow: deque = deque()                       # (question state, Launch) waiting for a slot
    live: Dict[str, Tuple[QuestionState, int]] = {}
    admitted: List[QuestionState] = []
    stats = {"launched": 0, "cancelled": 0, "questions": 0, "peak_live": 0, "dropped_queued": 0}

    def enqueue(q: QuestionState, l: al.Launch) -> None:
        """Register a follow-up launch. The policy sees it as queued from now on."""
        if l.sample in q.candidates:
            raise ValueError(f"{q.question_id}: sample {l.sample} launched twice")
        q.candidates[l.sample] = {"depth": l.depth, "status": al.QUEUED, "mode": l.mode, "launch": l}
        follow.append((q, l))

    def launch(q: QuestionState, l: al.Launch) -> None:
        if q.done:
            raise RuntimeError(f"{q.question_id}: launch after the question was answered")
        c = q.candidates.get(l.sample)
        if c is not None and c["status"] != al.QUEUED:
            raise ValueError(f"{q.question_id}: sample {l.sample} launched twice")
        req = make_request(q.question_id, q.prompt, l)
        q.candidates[l.sample] = {"depth": l.depth, "status": al.RUNNING, "request_id": req.request_id, "request": req,
                                  "mode": l.mode, "launch": l}
        backend.submit(req)
        live[req.request_id] = (q, l.sample)
        stats["launched"] += 1

    def drop_queued(q: QuestionState, samples=None) -> None:
        """Remove launches of ``q`` that never started; they cost nothing and leave no candidate."""
        keep = []
        for qq, l in follow:
            if qq is q and (samples is None or l.sample in samples):
                del q.candidates[l.sample]
                stats["dropped_queued"] += 1
            else:
                keep.append((qq, l))
        follow.clear()
        follow.extend(keep)

    while pending or follow or live:
        while follow and len(live) < slots:
            launch(*follow.popleft())
        while not follow and pending and slots - len(live) >= len(policy.initial()):
            qid, prompt = pending.popleft()
            q = QuestionState(qid, tuple(prompt), clock())
            admitted.append(q)
            for l in policy.initial():
                launch(q, l)
        stats["peak_live"] = max(stats["peak_live"], len(live))
        if not live:
            raise RuntimeError("nothing is running but work remains; slots too small for the policy")
        # Record everything that finished in this step before deciding anything:
        # two candidates of one question can finish together, and a decision made
        # after the first must not treat the second as still running.
        touched: List[QuestionState] = []
        for comp in backend.step():
            q, sample = live.pop(comp.request_id)
            c = q.candidates[sample]
            c.update(status=al.DONE, finish_reason=comp.finish_reason, gen_tokens=len(comp.token_ids),
                     answer_key=keyer(comp), think=think_of(comp, c["launch"]))
            on_candidate(q, sample, comp, None)
            if q not in touched:
                touched.append(q)
        for q in touched:
            if q.done:
                continue
            decision = policy.decide(q.views())
            now = clock()
            on_decision(q, decision, now, slots - len(live))
            if decision.done and decision.launch:
                raise ValueError(f"{q.question_id}: a decision cannot both answer and launch")
            queued = {s for s in decision.cancel if q.candidates[s]["status"] == al.QUEUED}
            drop_queued(q, queued)
            for s in decision.cancel:
                if s in queued:
                    continue
                rid = q.candidates[s]["request_id"]
                cancelled = backend.cancel(rid)
                live.pop(rid)
                q.candidates[s].update(status=al.CANCELLED, finish_reason=CANCELLED,
                                       gen_tokens=len(cancelled.token_ids), answer_key=None)
                stats["cancelled"] += 1
                on_candidate(q, s, cancelled, clock())
            for l in decision.launch:
                enqueue(q, l)
            if decision.done:
                q.done = True
                drop_queued(q)        # work that never started must not run for an answered question
                stats["questions"] += 1
                views = q.views()
                if decision.select is None:
                    chosen = al.selected(views)
                else:
                    chosen = next(v for v in views if v.sample == decision.select)
                    if chosen.status != al.DONE or chosen.answer_key is None:
                        raise ValueError(f"{q.question_id}: policy selected sample {decision.select}, which has no answer")
                on_question(q, chosen, now)
            elif not any(c["status"] in (al.RUNNING, al.QUEUED) for c in q.candidates.values()):
                raise RuntimeError(f"{q.question_id}: the policy neither answered nor left any work running "
                                   f"(action {decision.action!r})")
    unanswered = [q.question_id for q in admitted if not q.done]
    if unanswered or len(admitted) != len(questions):
        raise RuntimeError(f"{len(unanswered)} admitted questions have no final decision; "
                           f"{len(questions) - len(admitted)} were never admitted")
    return stats
