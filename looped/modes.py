"""Answer modes of the Ouro chat models.

With the solver prompt the model's first answer token decides how it answers.
``<think>`` opens a long free-form reasoning block ("Okay, so I need to ...");
anything else, almost always ``To``, starts a short worked solution
("To determine ...").

``natural``  the model picks its own mode with its first token.
``think``    the answer is started with ``<think>``.
``direct``   the answer is started with ``To``, the opening of 96% of the
             model's own direct answers.
``nothink``  the model picks its own first token, except that ``<think>`` is
             not allowed: the first token is drawn from the model's own
             first-token distribution with ``<think>`` removed. This separates
             "no reasoning block" from "the word To".

A forced mode is an assistant prefill: the prefix tokens are appended to the
prompt and reported as the first tokens of the answer, so a forced candidate
has the same record shape as a natural one that happened to start that way.
``nothink`` has no fixed prefix; the caller draws the first token per candidate
(``scripts/looped_profile.py``: one-token requests, redrawn while they return
``<think>``) and then prefills it like any other prefix.
The cap counts the prefix. The token after the prefix is read from whatever
state the backend gives the last prompt token; on a backend with a full-depth
prompt that differs from the uniform-depth computation unless the request runs
at full depth (see ``serving/looped/rlt_backend.py``).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

NATURAL, DIRECT, THINK, NOTHINK = "natural", "direct", "think", "nothink"
MODES = (NATURAL, DIRECT, THINK, NOTHINK)
POLICY_MODES = (NATURAL, DIRECT, THINK)     # modes a live policy can request: those with a fixed prefix
THINK_TOKEN = "<think>"
DIRECT_OPENER = "To"


def think_id(tok) -> int:
    i = tok.convert_tokens_to_ids(THINK_TOKEN)
    if not isinstance(i, int) or i < 0 or i == tok.unk_token_id:
        raise ValueError(f"the tokenizer has no {THINK_TOKEN} token")
    return i


def prefix_ids(tok, mode: str) -> Tuple[int, ...]:
    """Assistant-prefill tokens for ``mode`` (empty for ``natural``)."""
    if mode == NATURAL:
        return ()
    if mode == THINK:
        return (think_id(tok),)
    if mode == DIRECT:
        ids = tok.encode(DIRECT_OPENER, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{DIRECT_OPENER!r} is not a single token for this tokenizer")
        return (int(ids[0]),)
    if mode == NOTHINK:
        raise ValueError("nothink has no fixed prefix; its first token is drawn per candidate")
    raise ValueError(f"unknown mode {mode!r}")


def observed_think(token_ids: Sequence[int], think_token_id: int) -> Optional[bool]:
    """Whether an answer (prefix included) opened a reasoning block; ``None``
    if it has no tokens yet."""
    return None if not token_ids else int(token_ids[0]) == int(think_token_id)
