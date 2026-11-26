"""Open-loop executor for the deadline study.

Questions arrive on a fixed trace. A question is admitted only once it has
arrived and its whole first launch fits in the free slots; follow-up launches
of a question already admitted take a free slot first. Time is the caller's
clock. When nothing is running and the next arrival or deadline is in the
future, the driver waits instead of stepping the engine, so an idle gap is not
a batch of empty forwards.

A question whose decision time is past its deadline is not an answer. One that
has not been admitted by its deadline is expired without a launch. Every
arrival gets one outcome, and an expired arrival stays in the list the scorer
divides by. The driver never sees gold.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from looped import allocation as al
from looped.deadline import Arm

from .backend import CANCELLED, Backend, Completion, GenerationRequest


@dataclass
class QuestionState:
    question_id: str
    prompt: Tuple[int, ...]
    arrival: float
    deadline: float
    index: int
    candidates: Dict[int, Dict] = field(default_factory=dict)
    policy: Optional[al.Policy] = None
    choice: Optional[str] = None
    done: bool = False
    admitted: bool = False
    launched_long: bool = False
    outcome: Optional[dict] = None

    def views(self) -> List[al.CandidateView]:
        return [al.CandidateView(s, c["depth"], c["status"], c.get("answer_key"), c.get("finish_reason"),
                                 c.get("gen_tokens", 0), c["mode"], c.get("think"))
                for s, c in sorted(self.candidates.items())]


def run_deadline(
    backend: Backend,
    arm: Arm,
    questions: Sequence[Tuple[str, Sequence[int], float]],
    *,
    slots: int,
    deadline_s: float,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
    make_request: Callable[[str, Sequence[int], al.Launch], GenerationRequest],
    keyer: Callable[[Completion], Optional[str]],
    think_of: Callable[[Completion, al.Launch], Optional[bool]] = lambda comp, launch: None,
    on_candidate: Callable[[QuestionState, int, Completion, Optional[float]], None] = lambda *a: None,
    on_decision: Callable[[QuestionState, al.Decision, float, int], None] = lambda *a: None,
    on_outcome: Callable[[QuestionState], None] = lambda q: None,
) -> Dict:
    """Serve ``questions`` of ``(question_id, prompt, arrival_offset)``.

    Offsets are seconds from the clock reading at entry. ``sleeper(dt)`` must
    move the same clock forward by ``dt`` seconds. On a live run that is
    ``time.sleep`` and ``time.perf_counter``.
    """
    if deadline_s <= 0 or slots < 1:
        raise ValueError("deadline and slots must be positive")
    origin = clock()

    def now() -> float:
        return clock() - origin

    ordered = [QuestionState(qid, tuple(prompt), float(arrival), float(arrival) + deadline_s, i)
               for i, (qid, prompt, arrival) in enumerate(questions)]
    if any(q.arrival < 0 for q in ordered):
        raise ValueError("arrival offsets must be >= 0")
    if len({q.question_id for q in ordered}) != len(ordered):
        raise ValueError("duplicate question id")
    ordered.sort(key=lambda q: (q.arrival, q.index))
    waiting: deque = deque(ordered)
    admitted: List[QuestionState] = []
    follow: deque = deque()
    live: Dict[str, Tuple[QuestionState, int]] = {}
    stats = {"launched": 0, "cancelled": 0, "dropped_queued": 0, "expired": 0, "peak_live": 0}

    def finish(q: QuestionState, t: float, reason: str, chosen: Optional[al.CandidateView], met: bool) -> None:
        q.done = True
        q.outcome = {
            "question_id": q.question_id,
            "scheduled_arrival": q.arrival,
            "deadline": q.deadline,
            "completion_time": t,
            "deadline_met": met,
            "admitted": q.admitted,
            "choice": q.choice,
            "launched_long": q.launched_long,
            "decision_reason": reason,
            "selected_answer": None if chosen is None else chosen.answer_key,
            "selected_candidate_id": None if chosen is None else q.candidates[chosen.sample].get("request_id"),
            "all_candidate_ids": [c["request_id"] for _, c in sorted(q.candidates.items()) if "request_id" in c],
        }
        if not met:
            stats["expired"] += 1
        on_outcome(q)

    def drop_queued(q: QuestionState) -> None:
        keep = []
        for qq, launch in follow:
            if qq is q:
                del q.candidates[launch.sample]
                stats["dropped_queued"] += 1
            else:
                keep.append((qq, launch))
        follow.clear()
        follow.extend(keep)

    def cancel_running(q: QuestionState) -> None:
        for sample, c in list(q.candidates.items()):
            if c["status"] != al.RUNNING:
                continue
            rid = c["request_id"]
            cancelled = backend.cancel(rid)
            live.pop(rid, None)
            c.update(status=al.CANCELLED, finish_reason=CANCELLED, gen_tokens=len(cancelled.token_ids), answer_key=None)
            stats["cancelled"] += 1
            on_candidate(q, sample, cancelled, clock())

    def expire(q: QuestionState, t: float) -> None:
        drop_queued(q)
        cancel_running(q)
        finish(q, t, "deadline", None, False)
        on_decision(q, al.Decision(done=True, action="expire", reason="deadline"), t, slots - len(live))

    def launch(q: QuestionState, l: al.Launch) -> None:
        if q.done:
            raise RuntimeError(f"{q.question_id}: launch after the question was closed")
        c = q.candidates.get(l.sample)
        if c is not None and c["status"] != al.QUEUED:
            raise ValueError(f"{q.question_id}: sample {l.sample} launched twice")
        req = make_request(q.question_id, q.prompt, l)
        q.candidates[l.sample] = {"depth": l.depth, "status": al.RUNNING, "request_id": req.request_id, "request": req,
                                  "mode": l.mode, "launch": l}
        if l.mode == al.NATURAL:
            q.launched_long = True
        backend.submit(req)
        live[req.request_id] = (q, l.sample)
        stats["launched"] += 1

    def enqueue(q: QuestionState, l: al.Launch) -> None:
        if l.sample in q.candidates:
            raise ValueError(f"{q.question_id}: sample {l.sample} launched twice")
        q.candidates[l.sample] = {"depth": l.depth, "status": al.QUEUED, "mode": l.mode, "launch": l}
        follow.append((q, l))

    def apply(q: QuestionState, decision: al.Decision, t: float) -> None:
        on_decision(q, decision, t, slots - len(live))
        if decision.done and decision.launch:
            raise ValueError(f"{q.question_id}: a decision cannot both answer and launch")
        queued = {s for s in decision.cancel if q.candidates[s]["status"] == al.QUEUED}
        if queued:
            keep = []
            for qq, l in follow:
                if qq is q and l.sample in queued:
                    del q.candidates[l.sample]
                    stats["dropped_queued"] += 1
                else:
                    keep.append((qq, l))
            follow.clear()
            follow.extend(keep)
        for s in decision.cancel:
            if s in queued:
                continue
            rid = q.candidates[s]["request_id"]
            cancelled = backend.cancel(rid)
            live.pop(rid, None)
            q.candidates[s].update(status=al.CANCELLED, finish_reason=CANCELLED,
                                   gen_tokens=len(cancelled.token_ids), answer_key=None)
            stats["cancelled"] += 1
            on_candidate(q, s, cancelled, clock())
        for l in decision.launch:
            enqueue(q, l)
        if decision.done:
            views = q.views()
            if decision.select is None:
                chosen = al.selected(views)
            else:
                chosen = next(v for v in views if v.sample == decision.select)
                if chosen.status != al.DONE or chosen.answer_key is None:
                    raise ValueError(f"{q.question_id}: selected sample {decision.select} has no answer")
            finish(q, t, decision.reason or "answer", chosen, True)
            drop_queued(q)
        elif not any(c["status"] in (al.RUNNING, al.QUEUED) for c in q.candidates.values()):
            raise RuntimeError(f"{q.question_id}: the policy neither answered nor left any work running")

    for _ in range(1_000_000):
        if all(q.done for q in ordered):
            break
        acted = False
        t = now()
        for q in list(waiting):
            if not q.done and t > q.deadline:
                waiting.remove(q)
                expire(q, t)
                acted = True
        for q in admitted:
            if not q.done and t > q.deadline:
                expire(q, t)
                acted = True
        while follow and len(live) < slots:
            q, l = follow.popleft()
            if q.done or l.sample not in q.candidates:
                continue
            launch(q, l)
            acted = True
        while waiting and not follow and len(live) < slots:
            q = waiting[0]
            if q.done:
                waiting.popleft()
                continue
            if q.arrival > now():
                break
            unfinished = sum(not other.done for other in admitted)
            policy = arm.policy_for(q.question_id, slack_s=q.deadline - now(), unfinished=unfinished)
            first = policy.initial()
            if not first:
                raise ValueError(f"{q.question_id}: a question must launch something")
            if len(first) > slots:
                raise ValueError(f"{arm.name} launches {len(first)} candidates but only {slots} slots exist")
            if len(live) + len(first) > slots:
                break
            waiting.popleft()
            q.policy = policy
            q.choice = policy.choice
            q.admitted = True
            admitted.append(q)
            for l in first:
                launch(q, l)
            acted = True
        stats["peak_live"] = max(stats["peak_live"], len(live))
        if live:
            touched: List[QuestionState] = []
            for comp in backend.step():
                q, sample = live.pop(comp.request_id)
                c = q.candidates[sample]
                c.update(status=al.DONE, finish_reason=comp.finish_reason, gen_tokens=len(comp.token_ids),
                         answer_key=keyer(comp), think=think_of(comp, c["launch"]))
                on_candidate(q, sample, comp, None)
                if q not in touched:
                    touched.append(q)
            t = now()
            for q in admitted:
                if q.done or t <= q.deadline:
                    continue
                expire(q, t)
                acted = True
            for q in touched:
                if q.done:
                    continue
                apply(q, q.policy.decide(q.views()), now())
                acted = True
            continue
        nxt = [q.arrival for q in waiting if not q.done]
        nxt += [q.deadline for q in admitted if not q.done]
        nxt += [q.deadline for q in waiting if not q.done]
        if not nxt:
            if not acted:
                break
            continue
        target = min(nxt)
        delay = target - now()
        if delay > 0:
            sleeper(delay)
            acted = True
        if not acted:
            raise RuntimeError("deadline driver made no progress")
    else:
        raise RuntimeError("deadline driver did not finish")
    missing = [q.question_id for q in ordered if not q.done]
    if missing:
        raise RuntimeError(f"{len(missing)} questions have no outcome")
    by_index = sorted(ordered, key=lambda q: q.index)
    return {"questions": [q.outcome for q in by_index], "stats": stats}
