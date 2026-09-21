"""Frozen MATH-1k cache + one live K=10 agent → Judger.

Distinct from ``frozen_k2`` / ``frozen_k5`` (all three roles at small K).
This keeps recurrent depth K=10 and drops the other two live roles.

  frozen                    Mean-Replay prefix, no live agent
  frozen_planner_k10        prefix + Planner at K=10  (11 forwards)
  frozen_refiner_k10        prefix + Refiner at K=10
  frozen_critic_k10         prefix + Critic at K=10
  real                      all three roles at K=10, no frozen prefix

Do not train. Kill after MATH n=20 or AIME n=6/12 if paired results are flat.
Judger SEAL is a later stage: only after AIME n=30 GO.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROLES = ("planner", "refiner", "critic")  # test order: Planner, then Refiner, Critic control
ROLE_PREF = ("planner", "refiner", "critic")
K_DEFAULT = 10
FORWARDS_ONE_ROLE = K_DEFAULT + 1  # 11
FORWARDS_REAL = 3 * (K_DEFAULT + 1)  # 33

# Nested AIME slices: n=12 contains n=6, n=30 contains n=12.
AIME6_IDX = tuple(range(6))
AIME12_IDX = tuple(range(12))


def parse_one_role_arm(name: str) -> Dict[str, Any]:
    name = str(name).strip()
    if name == "frozen":
        return {"kind": "frozen", "role": None, "k": 0, "name": name}
    if name == "real":
        return {"kind": "real", "role": "all", "k": K_DEFAULT, "name": name}
    m = re.fullmatch(r"frozen_(planner|critic|refiner)_k(\d+)", name)
    if not m:
        raise ValueError(
            f"unknown arm {name!r}; expected frozen, frozen_planner_k10, "
            f"frozen_refiner_k10, frozen_critic_k10, real"
        )
    k = int(m.group(2))
    if k <= 0:
        raise ValueError(f"live role arm {name!r} needs K>=1")
    return {"kind": "one_role", "role": m.group(1), "k": k, "name": name}


def arm_name(role: Optional[str], k: int = K_DEFAULT) -> str:
    if not role:
        return "frozen"
    if role == "all":
        return "real"
    return f"frozen_{role}_k{int(k)}"


def expand_one_role_arms(
    requested: Optional[Iterable[str]] = None,
    *,
    k: int = K_DEFAULT,
    include_real: bool = False,
) -> List[str]:
    names = ["frozen"] + [arm_name(r, k) for r in ROLES]
    if include_real:
        names.append("real")
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    if requested:
        want = [str(x).strip() for x in requested if str(x).strip()]
        missing = [w for w in want if w not in set(ordered)]
        # allow a custom K that isn't in the default expansion
        extra = []
        for w in missing:
            spec = parse_one_role_arm(w)
            extra.append(spec["name"])
        ordered = [w for w in want]
        for e in extra:
            if e not in ordered:
                ordered.append(e)
    return ordered


def upstream_forwards(spec: Dict[str, Any]) -> int:
    if spec["kind"] == "frozen":
        return 0
    if spec["kind"] == "real":
        return 3 * (int(spec["k"]) + 1)
    return int(spec["k"]) + 1


def parse_indices(raw: str) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return []
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def select_items(items: Sequence[dict], n: int, indices: Optional[Sequence[int]] = None):
    if indices:
        out = []
        for i in indices:
            if i < 0 or i >= len(items):
                raise IndexError(f"index {i} out of range n_pool={len(items)}")
            rec = dict(items[i])
            rec["idx"] = int(i)
            out.append(rec)
        return out
    out = []
    for i, it in enumerate(list(items)[: int(n)]):
        rec = dict(it)
        rec["idx"] = i
        out.append(rec)
    return out


def paired_flags(frozen_correct: Sequence[bool], live_correct: Sequence[bool]) -> Dict[str, Any]:
    if len(frozen_correct) != len(live_correct):
        raise ValueError("paired_flags: length mismatch")
    recovered, lost, kept, both_miss = [], [], [], []
    for i, (f, live) in enumerate(zip(frozen_correct, live_correct)):
        if live and not f:
            recovered.append(i)
        elif f and not live:
            lost.append(i)
        elif f and live:
            kept.append(i)
        else:
            both_miss.append(i)
    return {
        "n": len(frozen_correct),
        "n_recovered": len(recovered),
        "n_lost": len(lost),
        "n_kept": len(kept),
        "n_both_miss": len(both_miss),
        "recovered": recovered,
        "lost": lost,
        "net": len(recovered) - len(lost),
        "frozen_acc": float(sum(bool(x) for x in frozen_correct) / max(len(frozen_correct), 1)),
        "live_acc": float(sum(bool(x) for x in live_correct) / max(len(live_correct), 1)),
    }


def math_role_ok(
    frozen_acc: Optional[float],
    live_acc: Optional[float],
    *,
    slack: float = 0.15,
    floor: float = 0.50,
) -> bool:
    """Checkpoint 2: live stays near Frozen (n=20 is noisy)."""
    if live_acc is None or frozen_acc is None:
        return False
    if live_acc + 1e-9 < float(floor):
        return False
    return live_acc + 1e-9 >= float(frozen_acc) - float(slack)


def pick_best_role(
    arms: Dict[str, Dict[str, Any]],
    *,
    frozen_acc: Optional[float],
    k: int = K_DEFAULT,
    slack: float = 0.15,
) -> Dict[str, Any]:
    """Prefer Planner, then Refiner, then Critic on ties."""
    candidates = []
    for role in ROLE_PREF:
        name = arm_name(role, k)
        blk = arms.get(name) or {}
        acc = blk.get("acc")
        if acc is None:
            continue
        ok = math_role_ok(frozen_acc, acc, slack=slack)
        tokens = (blk.get("tokens") or {}).get("mean") if isinstance(blk.get("tokens"), dict) else blk.get("tokens")
        candidates.append({
            "role": role,
            "arm": name,
            "acc": float(acc),
            "ok": ok,
            "tokens": float(tokens) if tokens is not None else 1e9,
        })
    passing = [c for c in candidates if c["ok"]]
    pool = passing or []
    if not pool:
        return {
            "role": None,
            "arm": None,
            "acc": None,
            "ok": False,
            "reason": "no live role stayed within slack of Frozen",
            "candidates": candidates,
        }
    pool.sort(key=lambda c: (-c["acc"], c["tokens"], ROLE_PREF.index(c["role"])))
    best = pool[0]
    return {
        "role": best["role"],
        "arm": best["arm"],
        "acc": best["acc"],
        "ok": True,
        "reason": f"{best['arm']} acc={best['acc']:.3f} near frozen={frozen_acc}",
        "candidates": candidates,
    }


def aime_subset_verdict(
    frozen_correct: Sequence[bool],
    live_correct: Sequence[bool],
    *,
    stage: str,
) -> Dict[str, Any]:
    """Checkpoints 3–5. Kill if there is no paired gain or regressions dominate."""
    flags = paired_flags(frozen_correct, live_correct)
    out = dict(flags)
    out["stage"] = stage
    rec, lost, net = flags["n_recovered"], flags["n_lost"], flags["net"]
    if stage == "aime6":
        if rec >= 1 and lost == 0:
            out["recommend"] = "go"
            out["reason"] = f"recovered {rec} Frozen miss, lost 0"
        elif rec >= 1 and lost <= 1 and net >= 0:
            out["recommend"] = "borderline"
            out["reason"] = (
                f"recovered {rec} lost {lost}; continue to n=12 but watch regressions"
            )
        else:
            out["recommend"] = "stop"
            out["reason"] = (
                f"no clean paired gain (recovered={rec}, lost={lost}); kill one-role"
            )
        return out
    if stage == "aime12":
        if net >= 1 and rec >= 1:
            out["recommend"] = "go"
            out["reason"] = f"net +{net} on n=12 (recovered={rec}, lost={lost})"
        else:
            out["recommend"] = "stop"
            out["reason"] = f"gain did not persist (recovered={rec}, lost={lost})"
        return out
    # aime30
    live_n = int(round(flags["live_acc"] * flags["n"]))
    frozen_n = int(round(flags["frozen_acc"] * flags["n"]))
    if live_n >= frozen_n + 2 and rec > lost:
        out["recommend"] = "go"
        out["reason"] = (
            f"{live_n}/30 vs Frozen {frozen_n}/30; SEAL next, do not train an MLP"
        )
    elif net >= 1:
        out["recommend"] = "borderline"
        out["reason"] = f"{live_n}/30 vs Frozen {frozen_n}/30; net +{net}, weak for SEAL"
    else:
        out["recommend"] = "stop"
        out["reason"] = f"{live_n}/30 vs Frozen {frozen_n}/30; one live K=10 did not recover AIME"
    return out


def smoke_match(pass_a: Sequence[dict], pass_b: Sequence[dict]) -> Tuple[bool, str]:
    """Checkpoint 1: two Frozen passes must match pred and correctness."""
    if len(pass_a) != len(pass_b):
        return False, f"length {len(pass_a)} vs {len(pass_b)}"
    for a, b in zip(pass_a, pass_b):
        if int(a["idx"]) != int(b["idx"]):
            return False, f"idx mismatch {a.get('idx')} vs {b.get('idx')}"
        if bool(a["correct"]) != bool(b["correct"]):
            return False, f"idx {a['idx']} correct {a['correct']} vs {b['correct']}"
        if str(a.get("pred") or "") != str(b.get("pred") or ""):
            return False, (
                f"idx {a['idx']} pred mismatch {a.get('pred')!r} vs {b.get('pred')!r}"
            )
    return True, f"exact match on {len(pass_a)} items"


def render_checkin(gate: Dict[str, Any]) -> str:
    rec = gate.get("recommend") or "running"
    nxt = gate.get("next") or ""
    lines = [
        "# One-role ladder check-in",
        "",
        f"**recommend:** `{rec}`",
        "",
        gate.get("reason") or "",
        "",
    ]
    if nxt:
        lines += [f"**next:** {nxt}", ""]
    smoke = gate.get("smoke")
    if smoke:
        lines.append(f"- smoke match: {smoke.get('ok')} ({smoke.get('reason')})")
    math = gate.get("math")
    if math:
        lines.append(
            f"- MATH n={math.get('n')}: frozen={math.get('frozen_acc')} "
            f"best={math.get('best_arm')} acc={math.get('best_acc')}"
        )
    for key in ("aime6", "aime12", "aime30"):
        blk = gate.get(key)
        if blk:
            lines.append(
                f"- {key}: live={blk.get('live_acc')} frozen={blk.get('frozen_acc')} "
                f"recovered={blk.get('n_recovered')} lost={blk.get('n_lost')} "
                f"→ {blk.get('recommend')}"
            )
    lines += [
        "",
        "Locked: Frozen AIME 17/30, Real K=10 20/30. Do not train. "
        "Judger SEAL only after aime30 GO.",
        "",
        "This is not frozen_k2/k5 (those cut K). One live role at K=10 is 11 forwards "
        "vs Real's 33. Judger decode still dominates wall-clock.",
        "",
    ]
    return "\n".join(lines)
