"""Vote identity for math answers: which generated answers count as the same vote.

Vote identity and grading are different operations. Identity groups candidate
answers without ever seeing the gold answer; grading compares one selected answer
with gold. This module owns identity (two versions) and a pinned grader wrapper.

``legacy-v1`` is a verbatim copy of ``utils.normalize_math_answer`` as used by the
Stage-0 runs. It deletes every brace, so ``\\frac{1}{23}`` and ``\\frac{12}{3}`` both
become ``\\frac123``. It is kept only to reproduce historical numbers.

``structural-v2`` keeps argument structure:

* ``\\frac``, ``\\binom``, ``\\sqrt``, ``^`` and ``_`` take TeX arguments (a group or
  one token; one *digit* of a number), which are always written back braced, so
  ``\\frac12`` equals ``\\frac{1}{2}`` and differs from ``\\frac{1}{23}``.
* Other braces are kept as written; nothing is flattened.
* An answer that is a plain rational (integer, decimal, ``a/b``, ``\\frac{a}{b}``,
  optionally signed) gets an exact-value key, so ``1/2``, ``0.5`` and
  ``\\frac{2}{4}`` vote together. ``0.333`` and ``1/3`` do not: they are not equal.
* Unbalanced braces or a missing macro argument are parse failures. A parse
  failure has no key and casts no vote; its status says why.

Pure Python, no torch, so the CPU audit and the tests can import it anywhere.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union

LEGACY = "legacy-v1"
STRUCTURAL = "structural-v2"
IDENTITY_VERSIONS = (LEGACY, STRUCTURAL)

# parse_status values
OK = "ok"
NO_ANSWER = "no_answer"          # nothing was extracted from the generation
EMPTY = "empty"                  # extracted, but nothing left after cleaning
MALFORMED = "malformed"          # unbalanced braces or a macro without its argument


@dataclass(frozen=True)
class AnswerIdentity:
    key: Optional[str]            # None casts no vote
    status: str
    version: str


# ------------------------------------------------------------------ extraction

def extract_final_answer(text: Optional[str]) -> Tuple[Optional[str], str]:
    """Content of the last ``\\boxed{...}`` and how the extraction ended.

    Status is ``ok``, ``no_boxed`` or ``unterminated_boxed``. An unterminated box
    (a generation cut off inside the answer) returns the partial content so it
    can be logged, but callers should treat it as a parse failure, not a vote.
    """
    if not text:
        return None, "no_boxed"
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None, "no_boxed"
    i = idx + len("\\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None, "no_boxed"
    depth, start = 0, i + 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j], OK
    return text[start:], "unterminated_boxed"


# ------------------------------------------------------------------ legacy-v1

def legacy_key(ans: Optional[str]) -> Optional[str]:
    """Verbatim behaviour of ``utils.normalize_math_answer`` (tests assert parity)."""
    if ans is None:
        return None
    s = str(ans)
    if "\\boxed" in s:
        b, _ = extract_final_answer(s)
        if b is not None:
            s = b
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", s)
    for tok in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "\\$"]:
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\%", "").replace("\\degree", "").replace("^\\circ", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(" ", "")
    s = s.strip().rstrip(".").rstrip()
    s = s.lower()
    return s or None


# ------------------------------------------------------------------ structural-v2

_TOKEN = re.compile(
    r"\\[a-zA-Z]+"            # control word
    r"|\\."                   # control symbol (\\, \{, \% ...)
    r"|\d+\.\d+|\.\d+|\d+"    # number
    r"|\s+"                   # whitespace (dropped, but it separates tokens)
    r"|.",                    # any other single character
    re.DOTALL,
)
_NUMBER = re.compile(r"\d+\.\d+|\.\d+|\d+")
_TEXT_MACROS = {"\\text", "\\mbox", "\\textrm", "\\textbf", "\\textit", "\\textnormal"}
_ARGS = {"\\frac": 2, "\\binom": 2, "\\sqrt": 1, "^": 1, "_": 1}

Node = Union[str, "Group", "Macro"]


@dataclass(frozen=True)
class Group:
    items: Tuple[Node, ...]


@dataclass(frozen=True)
class Macro:
    name: str
    optional: Optional[Tuple[Node, ...]]     # \sqrt[n]{x}
    args: Tuple[Group, ...]


class _Malformed(ValueError):
    pass


def _clean(s: str) -> str:
    s = s.strip()
    if "\\boxed" in s:
        inner, status = extract_final_answer(s)
        if status != OK:
            raise _Malformed("unterminated \\boxed")
        s = inner
    return s.replace("\u2212", "-")                        # unicode minus


_DROP = {"\\!", "\\,", "\\;", "\\:", "\\ ", "~", "\\quad", "\\qquad",    # spacing
         "\\left", "\\right",                                            # delimiter sizing
         "$", "\\$", "\\%", "\\degree"}                                  # units the benchmarks ignore
_RENAME = {"\\dfrac": "\\frac", "\\tfrac": "\\frac", "\\dbinom": "\\binom", "\\tbinom": "\\binom"}


def _tokenize(s: str) -> List[str]:
    """Tokens with layout noise removed. Works on tokens, not on the raw string,
    so ``\\\\`` (a matrix row break) is never mistaken for ``\\`` plus a space and
    ``\\leftarrow`` is never mistaken for ``\\left``."""
    raw = [t for t in _TOKEN.findall(s) if not t.isspace()]
    out: List[str] = []
    i = 0
    while i < len(raw):
        t = raw[i]
        nxt = raw[i + 1:i + 4]
        sep = 2 if (t == "," and nxt[:1] == ["\\!"]) else 3 if (t == "{" and nxt[:2] == [",", "}"]) else 0
        if sep:                                             # thousands: 10,\!000 and 10{,}000
            i += sep
            if out and out[-1].isdigit() and i < len(raw) and raw[i].isdigit():
                out[-1] += raw[i]                           # rejoin the digit groups
                i += 1
        elif t == "^" and nxt[:1] == ["\\circ"]:            # 90^\circ
            i += 2
        elif t == "^" and nxt[:3] == ["{", "\\circ", "}"]:  # 90^{\circ}
            i += 4
        elif t in _DROP:
            i += 1
        else:
            out.append(_RENAME.get(t, t))
            i += 1
    return out


class _Parser:
    def __init__(self, tokens: List[str]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def sequence(self, closer: Optional[str]) -> Tuple[Node, ...]:
        out: List[Node] = []
        while True:
            t = self.peek()
            if t is None:
                if closer is not None:
                    raise _Malformed(f"missing {closer}")
                return tuple(out)
            if t == closer:
                self.i += 1
                return tuple(out)
            if t == "}":
                raise _Malformed("unmatched }")
            out.extend(self.node())

    def node(self) -> List[Node]:
        t = self.toks[self.i]
        self.i += 1
        if t == "{":
            return [Group(self.sequence("}"))]
        if t in _TEXT_MACROS:
            if self.peek() != "{":
                raise _Malformed(f"{t} without a group")
            self.i += 1
            return [x.lower() if isinstance(x, str) and not x.startswith("\\") else x
                    for x in self.sequence("}")]
        if t in _ARGS:
            optional = None
            if t == "\\sqrt" and self.peek() == "[":
                self.i += 1
                optional = self.sequence("]")
            return [Macro(t, optional, tuple(self.argument(t) for _ in range(_ARGS[t])))]
        return [t]

    def argument(self, owner: str) -> Group:
        t = self.peek()
        if t is None or t == "}":
            raise _Malformed(f"{owner} is missing an argument")
        if t == "{":
            self.i += 1
            return Group(self.sequence("}"))
        if _NUMBER.fullmatch(t) and len(t) > 1:             # TeX takes one digit: \frac123
            self.toks[self.i] = t[1:]
            return Group((t[0],))
        return Group(tuple(self.node()))


def _is_word(tok: Node) -> bool:
    return isinstance(tok, str) and tok.startswith("\\") and tok[1:].isalpha()


def _serialize(items: Tuple[Node, ...]) -> str:
    parts: List[str] = []
    prev: Optional[Node] = None
    for it in items:
        if isinstance(it, Group):
            s = "{" + _serialize(it.items) + "}"
        elif isinstance(it, Macro):
            s = it.name
            if it.optional is not None:
                s += "[" + _serialize(it.optional) + "]"
            s += "".join("{" + _serialize(a.items) + "}" for a in it.args)
        else:
            s = it
            first = s[0]
            # keep apart what whitespace kept apart: "\pi r" is not "\pir", "1 2" is not "12"
            if (_is_word(prev) and first.isalpha()) or (
                    isinstance(prev, str) and _NUMBER.fullmatch(prev) and _NUMBER.fullmatch(s)):
                s = " " + s
        parts.append(s)
        prev = it
    return "".join(parts)


def _integer(items: Tuple[Node, ...]) -> Optional[int]:
    sign, rest = _sign(items)
    if len(rest) == 1 and isinstance(rest[0], str) and rest[0].isdigit():
        return sign * int(rest[0])
    return None


def _sign(items: Tuple[Node, ...]) -> Tuple[int, Tuple[Node, ...]]:
    if items and items[0] in ("-", "+"):
        return (-1 if items[0] == "-" else 1), items[1:]
    return 1, items


def _rational(items: Tuple[Node, ...]) -> Optional[Fraction]:
    """Exact value when the whole answer is one plain rational, else None."""
    sign, rest = _sign(items)
    if len(rest) == 1:
        x = rest[0]
        if isinstance(x, str) and _NUMBER.fullmatch(x):
            return sign * Fraction(x if not x.startswith(".") else "0" + x)
        if isinstance(x, Macro) and x.name == "\\frac":
            p, q = _integer(x.args[0].items), _integer(x.args[1].items)
            if p is not None and q:
                return sign * Fraction(p, q)
        return None
    if len(rest) == 3 and rest[1] == "/":
        a, b = rest[0], rest[2]
        if isinstance(a, str) and a.isdigit() and isinstance(b, str) and b.isdigit() and int(b):
            return sign * Fraction(int(a), int(b))
    return None


def structural_identity(ans: Optional[str]) -> AnswerIdentity:
    if ans is None:
        return AnswerIdentity(None, NO_ANSWER, STRUCTURAL)
    try:
        items = _Parser(_tokenize(_clean(str(ans)))).sequence(None)
    except _Malformed:
        return AnswerIdentity(None, MALFORMED, STRUCTURAL)
    while len(items) == 1 and isinstance(items[0], Group):   # {5} -> 5 (whole answer only)
        items = items[0].items
    while items and items[-1] == ".":                         # trailing sentence period
        items = items[:-1]
    if not items:
        return AnswerIdentity(None, EMPTY, STRUCTURAL)
    value = _rational(items)
    if value is not None:
        return AnswerIdentity(f"Q({value.numerator}/{value.denominator})", OK, STRUCTURAL)
    if all(isinstance(x, str) and x.isalpha() for x in items):   # a word answer: "Yes", "East"
        items = tuple(x.lower() for x in items)
    return AnswerIdentity("S(" + _serialize(items) + ")", OK, STRUCTURAL)


# ------------------------------------------------------------------ public API

def identify(ans: Optional[str], version: str = STRUCTURAL) -> AnswerIdentity:
    """Vote identity of an extracted answer under a named identity version."""
    if version == STRUCTURAL:
        return structural_identity(ans)
    if version == LEGACY:
        key = legacy_key(ans)
        status = OK if key is not None else (NO_ANSWER if ans is None else EMPTY)
        return AnswerIdentity(key, status, LEGACY)
    raise ValueError(f"unknown identity version {version!r}; known: {IDENTITY_VERSIONS}")


def answer_key(ans: Optional[str], version: str = STRUCTURAL) -> Optional[str]:
    return identify(ans, version).key


def answer_from_text(text: Optional[str], version: str = STRUCTURAL) -> Tuple[Optional[str], AnswerIdentity]:
    """(answer, identity) of a whole generation.

    Only a completely extracted box is an answer. A generation without a box,
    or cut off inside one, returns ``(None, identity)`` with no key and the
    extraction status (``no_boxed`` or ``unterminated_boxed``): it casts no vote
    and is not graded. The partial content stays in the generation text.
    """
    ans, status = extract_final_answer(text)
    if status != OK:
        return None, AnswerIdentity(None, status, version)
    return ans, identify(ans, version)


# ------------------------------------------------------------------ pinned grading

class GraderUnavailable(RuntimeError):
    pass


def grader_info(name: str = "math-verify") -> Dict[str, str]:
    """Name and version of the grader, for the run manifest. Raises if it is missing,
    so a run never silently falls back to a different grader (the Stage-0
    ``grade_math`` falls back to string match when ``math_verify`` is absent)."""
    if name == "string-legacy-v1":
        return {"grader": name, "version": "1"}
    if name == "math-verify":
        try:
            from importlib.metadata import version
            import math_verify  # noqa: F401
        except Exception as exc:
            raise GraderUnavailable(
                "math-verify is the pinned grader and is not installed here "
                "(it needs Python >= 3.10); refusing to fall back to string match") from exc
        return {"grader": name, "version": version("math-verify")}
    raise ValueError(f"unknown grader {name!r}")


def grade(pred: Optional[str], gold: str, grader: str = "math-verify") -> Dict:
    """Correctness of one extracted answer under the pinned grader.

    A grader exception is recorded as ``error`` with ``correct=False``; it is never
    replaced by another grader's verdict.
    """
    info = grader_info(grader)
    if pred is None:
        return {**info, "correct": False, "error": None}
    if grader == "string-legacy-v1":
        return {**info, "correct": legacy_key(pred) == legacy_key(gold), "error": None}
    from math_verify import parse, verify
    try:
        return {**info, "correct": bool(verify(parse(f"${gold}$"), parse(f"${pred}$"))), "error": None}
    except Exception as exc:
        return {**info, "correct": False, "error": type(exc).__name__}
