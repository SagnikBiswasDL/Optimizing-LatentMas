"""Frozen MATH-1k cache + Judger-only SEAL. Latency head. No training.

This is the lock-in campaign. Do not mix in live agents, DeltaBridge, or AIME
accuracy hunting. Kill if MATH/GSM8K accuracy drops or AIME n=6 loses solves.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

DEFAULT_VECTOR = "artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt"
FALLBACK_VECTOR = "artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt"
DEFAULT_COEF = 40.0
DEFAULT_LAYER = 28


def iso_acc_token_gate(
    frozen_acc: Optional[float],
    seal_acc: Optional[float],
    frozen_tok: Optional[float],
    seal_tok: Optional[float],
    *,
    acc_slack: float = 0.05,
    tok_cut: float = 0.10,
) -> Dict[str, Any]:
    """Continue if accuracy holds and tokens drop by at least tok_cut."""
    out = {
        "frozen_acc": frozen_acc,
        "seal_acc": seal_acc,
        "frozen_tok": frozen_tok,
        "seal_tok": seal_tok,
        "acc_ok": False,
        "tok_ok": False,
        "recommend": "stop",
        "reason": "",
    }
    if frozen_acc is None or seal_acc is None:
        out["reason"] = "missing accuracy"
        return out
    out["acc_ok"] = float(seal_acc) + 1e-9 >= float(frozen_acc) - float(acc_slack)
    if frozen_tok is None or seal_tok is None or float(frozen_tok) <= 0:
        out["reason"] = "missing tokens"
        return out
    out["tok_ok"] = float(seal_tok) <= float(frozen_tok) * (1.0 - float(tok_cut))
    delta_tok = (float(seal_tok) - float(frozen_tok)) / float(frozen_tok)
    if out["acc_ok"] and out["tok_ok"]:
        out["recommend"] = "go"
        out["reason"] = (
            f"acc {seal_acc:.3f} holds vs frozen {frozen_acc:.3f}; "
            f"tokens {delta_tok:+.1%}"
        )
        return out
    if out["acc_ok"] and not out["tok_ok"]:
        out["recommend"] = "stop"
        out["reason"] = (
            f"acc held but token cut too small ({delta_tok:+.1%}; need ≤{-tok_cut:.0%})"
        )
        return out
    out["recommend"] = "stop"
    out["reason"] = (
        f"accuracy drop {seal_acc:.3f} vs frozen {frozen_acc:.3f} "
        f"(slack {acc_slack:.2f}); do not go to AIME"
    )
    return out


def aime_latency_verdict(
    frozen_correct: Sequence[bool],
    seal_correct: Sequence[bool],
    frozen_tok: Sequence[float],
    seal_tok: Sequence[float],
    *,
    stage: str,
) -> Dict[str, Any]:
    """AIME: do not spend more GPU if SEAL loses net solves."""
    n = len(frozen_correct)
    rec = sum(1 for f, s in zip(frozen_correct, seal_correct) if s and not f)
    lost = sum(1 for f, s in zip(frozen_correct, seal_correct) if f and not s)
    f_acc = float(sum(frozen_correct) / max(n, 1))
    s_acc = float(sum(seal_correct) / max(n, 1))
    f_tok = float(sum(frozen_tok) / max(len(frozen_tok), 1)) if frozen_tok else 0.0
    s_tok = float(sum(seal_tok) / max(len(seal_tok), 1)) if seal_tok else 0.0
    out: Dict[str, Any] = {
        "stage": stage,
        "n": n,
        "n_recovered": rec,
        "n_lost": lost,
        "net": rec - lost,
        "frozen_acc": f_acc,
        "seal_acc": s_acc,
        "frozen_tok": f_tok,
        "seal_tok": s_tok,
        "hit_cap_frozen": sum(1 for t in frozen_tok if t >= 8192),
        "hit_cap_seal": sum(1 for t in seal_tok if t >= 8192),
    }
    if lost > rec:
        out["recommend"] = "stop"
        out["reason"] = f"SEAL lost {lost} and recovered {rec}; latency head is hurting AIME"
        return out
    if s_tok >= f_tok * 0.95:
        out["recommend"] = "stop"
        out["reason"] = f"no token cut on AIME ({s_tok:.0f} vs {f_tok:.0f})"
        return out
    if rec >= lost:
        out["recommend"] = "go" if stage != "aime6" or lost <= 1 else "borderline"
        out["reason"] = (
            f"net {rec - lost:+d} solves, tokens {s_tok:.0f} vs frozen {f_tok:.0f}"
        )
        if stage == "aime6" and lost == 1 and rec == 0:
            out["recommend"] = "borderline"
            out["reason"] = "lost 1, recovered 0; continue to n=12 only if tokens dropped"
        return out
    out["recommend"] = "stop"
    out["reason"] = f"recovered={rec} lost={lost}"
    return out


def render_checkin(gate: Dict[str, Any]) -> str:
    rec = gate.get("recommend") or "running"
    lines = [
        "# Frozen + Judger SEAL check-in",
        "",
        f"**recommend:** `{rec}`",
        "",
        gate.get("reason") or "",
        "",
    ]
    if gate.get("next"):
        lines += [f"**next:** {gate['next']}", ""]
    for key in ("gsm8k", "math", "aime6", "aime12", "aime30"):
        blk = gate.get(key)
        if not blk:
            continue
        lines.append(
            f"- {key}: frozen={blk.get('frozen_acc')} seal={blk.get('seal_acc')} "
            f"tok {blk.get('frozen_tok')}→{blk.get('seal_tok')} → {blk.get('recommend')}"
        )
    lines += [
        "",
        "Lock-in campaign. Coef 40, layer 28, GSM8K exec−reflection vector.",
        "No live agents. Do not train. Stop the pod if recommend=stop.",
        "",
    ]
    return "\n".join(lines)
