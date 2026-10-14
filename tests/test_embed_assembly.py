"""CPU checks for hybrid Judger embedding assembly.

These do not load a model. They pin two facts the GPU run has to start from:
batch size 1 crashes in the original assembly, and a padded batch does not
carry the same embeddings as the same requests run alone.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.embed_assembly import (  # noqa: E402
    assemble_original,
    assemble_repaired,
    left_insert_lengths,
    valid_agent_embeds,
)


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = [ord(ch) % 97 for ch in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def test_original_insert_length_counts_the_default_special_token():
    tok = _CharTokenizer()
    prompt = "SYS<|im_start|>user\nQuestion"
    with_special = left_insert_lengths(tok, [prompt], add_special_tokens=True)[0]
    without = left_insert_lengths(tok, [prompt], add_special_tokens=False)[0]
    assert with_special == without + 1


def test_original_assembly_crashes_at_batch_size_one():
    curr = torch.randn(1, 6, 4)
    past = torch.randn(1, 3, 4)
    with pytest.raises(ValueError):
        assemble_original(curr, past, [2])


def test_original_batch_pads_with_float32_zeros_and_keeps_pad_rows():
    curr = torch.ones(2, 4, 2, dtype=torch.bfloat16)
    # Column 3 of item 0 is a pad embedding that original assembly keeps.
    curr[0, 3, :] = 9
    past = torch.full((2, 2, 2), 3, dtype=torch.bfloat16)
    stacked = assemble_original(curr, past, [2, 2])
    assert stacked.dtype == torch.float32
    assert stacked.shape[0] == 2
    assert torch.any(stacked[0].float() == 9)


def test_repaired_batch_matches_each_request_run_alone():
    torch.manual_seed(0)
    hidden = 4
    # Left-padded text. Item 0 has 3 real tokens, item 1 has 5.
    text = torch.randn(2, 5, hidden)
    text[0, :2, :] = 0
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    latent = torch.randn(2, 2, hidden)
    previous = torch.cat([text, latent], dim=1)
    parts = valid_agent_embeds(previous, mask)

    alone = []
    for i in range(2):
        prev_i = torch.cat([text[i:i + 1, mask[i].bool(), :], latent[i:i + 1]], dim=1)
        part_i = valid_agent_embeds(prev_i, torch.ones(1, int(mask[i].sum()), dtype=torch.long))
        alone.append(part_i[0])
        assert torch.equal(parts[i], alone[i])

    curr = torch.randn(2, 6, hidden)
    curr[0, 4:, :] = 0
    curr_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    insert = [2, 2]
    batched = assemble_repaired(curr, curr_mask, parts, insert)
    for i in range(2):
        single_curr = curr[i:i + 1, curr_mask[i].bool(), :]
        single_mask = torch.ones(1, single_curr.shape[1], dtype=torch.long)
        single = assemble_repaired(single_curr, single_mask, [alone[i]], [insert[i]])
        assert torch.equal(batched[i], single[0])
        assert batched[i].dtype == curr.dtype


def test_repaired_keeps_a_three_dimensional_batch_of_one():
    curr = torch.ones(1, 5, 3, dtype=torch.bfloat16)
    mask = torch.ones(1, 5, dtype=torch.long)
    past = [torch.full((2, 3), 4, dtype=torch.bfloat16)]
    rows = assemble_repaired(curr, mask, past, [3])
    assert len(rows) == 1
    assert rows[0].shape == (7, 3)
    assert rows[0].dtype == torch.bfloat16
    assert torch.all(rows[0][3:5].float() == 4)


def test_source_default_path_still_contains_the_squeeze():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "methods", "latent_mas.py"), encoding="utf-8").read()
    assert "squeeze(0)" in src
    assert 'getattr(self.args, "embed_assembly", "original")' in src
