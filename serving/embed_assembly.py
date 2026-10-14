"""Judger prompt-embedding assembly for the hybrid path.

``original`` reproduces ``LatentMASMethod.run_batch_vllm`` exactly, including
the batch-size-1 squeeze crash and the padded batch that does not match
per-request execution.

``repaired`` is the minimum change that lets one request run and makes a
batch carry the same embeddings as those requests run alone:

- keep the ``[batch, length, hidden]`` prompt-embedding tensor
- drop positions the attention mask marks as padding, whichever side they sit on
- do not append float32 zeros (that upcast promotes bf16 embeddings to fp32)
- pass each request at its own length

The insertion index uses the same tokenizer call as the original code.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

USER_MARKER = "<|im_start|>user\n"


def left_insert_lengths(tokenizer, prompts: Sequence[str], *, add_special_tokens: bool = True) -> List[int]:
    """Token length of the chat prefix up to and including the user marker.

    The original hybrid path calls ``tokenizer(left)`` with the tokenizer
    default, which is ``add_special_tokens=True``. Pass False only to measure
    the off-by-special-token gap; do not change the index the pipeline uses.
    """
    lengths: List[int] = []
    for prompt in prompts:
        idx = prompt.find(USER_MARKER)
        if idx < 0:
            raise ValueError("Qwen user marker is missing from the Judger prompt")
        left = prompt[: idx + len(USER_MARKER)]
        ids = tokenizer(left, add_special_tokens=add_special_tokens)["input_ids"]
        lengths.append(len(ids))
    return lengths


def assemble_original(
    curr_prompt_emb: torch.Tensor,
    past_embedding: torch.Tensor,
    len_of_left: Sequence[int],
) -> torch.Tensor:
    """Return ``[B, max_len, H]`` the way ``run_batch_vllm`` builds it.

    Raises ``ValueError`` at batch size 1, because ``squeeze(0)`` removes the
    batch dimension and the following three-way unpack fails.
    """
    curr_prompt_emb = curr_prompt_emb.squeeze(0)
    batch, _length, hidden = curr_prompt_emb.shape
    _batch_p, _past_len, hidden = past_embedding.shape
    rows = []
    for i in range(batch):
        insert_idx = int(len_of_left[i])
        left = curr_prompt_emb[i, :insert_idx, :]
        right = curr_prompt_emb[i, insert_idx:, :]
        combined = torch.cat([left, past_embedding[i], right], dim=0)
        rows.append(combined)
    max_len = max(row.shape[0] for row in rows)
    return torch.stack([
        torch.cat([
            row,
            torch.zeros(max_len - row.shape[0], hidden, device=row.device),
        ], dim=0)
        for row in rows
    ])


def valid_agent_embeds(
    previous_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """Per-request agent embeddings with padding positions removed.

    ``previous_hidden`` is ``[B, text_len + latent_steps, H]`` as returned by
    ``generate_latent_batch_hidden_state``: token embeddings of the (possibly
    padded) prompt, then one realigned latent vector per step. Latent steps
    are not padded. The mask covers only the text portion.
    """
    if previous_hidden.dim() != 3:
        raise ValueError("previous_hidden must be [batch, length, hidden]")
    if attention_mask.shape[0] != previous_hidden.shape[0]:
        raise ValueError("attention mask batch does not match embeddings")
    text_len = int(attention_mask.shape[1])
    if text_len > previous_hidden.shape[1]:
        raise ValueError("attention mask is longer than the embedding sequence")
    text = previous_hidden[:, :text_len, :]
    latent = previous_hidden[:, text_len:, :]
    parts: List[torch.Tensor] = []
    for i in range(previous_hidden.shape[0]):
        valid = text[i, attention_mask[i].bool()]
        parts.append(torch.cat([valid, latent[i]], dim=0).contiguous())
    return parts


def assemble_repaired(
    curr_prompt_emb: torch.Tensor,
    curr_mask: torch.Tensor,
    past_per_item: Sequence[torch.Tensor],
    len_of_left: Sequence[int],
) -> List[torch.Tensor]:
    """Per-request Judger embeddings at their true lengths.

    ``len_of_left`` counts tokens in the unpadded prompt, so it indexes the
    mask-selected rows rather than the padded tensor.
    """
    if curr_prompt_emb.dim() != 3:
        raise ValueError("curr_prompt_emb must stay [batch, length, hidden]")
    batch, _length, _hidden = curr_prompt_emb.shape
    if len(past_per_item) != batch or len(len_of_left) != batch:
        raise ValueError("per-request assembly arguments disagree on batch size")
    rows: List[torch.Tensor] = []
    for i in range(batch):
        valid = curr_prompt_emb[i, curr_mask[i].bool()]
        insert_idx = int(len_of_left[i])
        if insert_idx < 0 or insert_idx > valid.shape[0]:
            raise ValueError(
                f"insert index {insert_idx} is outside the {valid.shape[0]} valid tokens"
            )
        past = past_per_item[i].to(device=valid.device, dtype=valid.dtype)
        combined = torch.cat([valid[:insert_idx], past, valid[insert_idx:]], dim=0)
        rows.append(combined.contiguous())
    return rows
