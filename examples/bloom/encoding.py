import re
from typing import Iterable, List, Sequence

import torch
from transformers.tokenization_utils import PreTrainedTokenizerBase


def remove_blanks(string: str) -> str:
    r""" Substitute all blanks with single space. """
    return re.sub(r"\s+", " ", string)


def split(_list: Iterable, part_length: int) -> Iterable:
    r""" Split an Iterable `_list` in parts of length `part_length`. """

    # checks
    if not isinstance(_list, Iterable):
        raise ValueError("`_list` must be an iterable")
    if not isinstance(part_length, int) or not part_length > 0:
        raise ValueError("`part_length` must be a positive integer")

    if not isinstance(_list, Sequence):
        _list = list(_list)

    for i in range(0, len(_list), part_length):
        yield _list[i:i + part_length]


def batch_encode_with_prefix_and_postfix(
    sequences: List[str],
    prefix: str = "",
    postfix: str = "",
    max_sequence_length: int = None,
    tokenizer: PreTrainedTokenizerBase = None,
):
    r""" Encode a batch of sentences for generation. """
    num_prefix_tokens = len(tokenizer.encode(prefix))
    num_postfix_tokens = len(tokenizer.encode(postfix))
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    # compute max length for real text
    reduced_max_sequence_length = max_sequence_length - (
        num_prefix_tokens + num_postfix_tokens + num_special_tokens
    )
    assert reduced_max_sequence_length > 0

    # limiting sequences to clipped reduced_max_sequence_length
    sequences = tokenizer(sequences).input_ids
    sequences = [seq[:reduced_max_sequence_length] for seq in sequences]
    sequences = tokenizer.batch_decode(sequences)

    # finally adding prefix and postfix
    sequences = [prefix + " " + seq + " " + postfix for seq in sequences]

    # final encoding
    sequences_encoding = tokenizer(
        sequences,
        padding='longest',
        truncation=True,
        max_length=max_sequence_length,
        return_tensors='pt',
    )
    return dict(sequences_encoding)


def decode_batch(
    results: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    num_return_sequences: int,
    original_input_length: int
):
    r""" Decode a tensor into a batch of sentences after generation. """
    results = results[:, original_input_length:].contiguous()
    sequences = tokenizer.batch_decode(results, skip_special_tokens=True)
    sequences = [remove_blanks(seq.strip()) for seq in sequences]
    sequences = list(split(sequences, part_length=num_return_sequences))
    return sequences
