"""The only file the autoresearch loop may edit.

`build_prefix` returns the token ids placed at the start of every request.
The eval calls it once and reuses that list, so the prefix is identical across
the batch and vLLM can keep it in the prefix cache.
"""
from __future__ import annotations


PREFIX_TEXT = (
    "Solve the question step by step, then give the final answer as \\boxed{answer}.\n"
    "Example: A shop has 3 boxes with 12 pencils each. "
    "3 times 12 is 36. \\boxed{36}\n"
)


def build_prefix(tokenizer) -> list[int]:
    ids = tokenizer.encode(PREFIX_TEXT, add_special_tokens=False)
    return [int(token) for token in ids]
