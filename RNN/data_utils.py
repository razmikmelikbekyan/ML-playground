from typing import Tuple, Dict

import numpy as np
import torch


def read_in_chunks(data_path: str, chunk_size: int, offset: int, full_sequences: bool):
    """
    Lazy function (generator) to read a file piece by piece. It is needed in order to optimize
    memory consumption due to big files.

    If full_sequences is True, it will read file with the following order:
        [: chunk_size]
        [1: chunk_size + 1]
        [2: chunk_size + 2]
        ...
    If full_sequences is False, the reading order will be the following:
        [: chunk_size]
        [chunk_size: 2 * chunk_size]
        [2 * chunk_size: 3 * chunk_size]
        ...

    If the offset is given the reading starting point will be the character located at
    index = offset.
    """
    file_object = open(data_path, 'r')

    counter = 0
    if offset:
        file_object.seek(offset, 0)
        counter += offset

    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

        # if full_sequences is True, i
        if full_sequences:
            counter += 1
            file_object.seek(counter, 0)


def get_support_data(data_path: str) -> Tuple[int, Dict[str, int], Dict[int, str]]:
    """Reads the file and returns vocabulary size and characters to indexes mapping dicts."""
    chars = set()

    for x in read_in_chunks(data_path, 1000, 0, False):
        chars.update(set(x.lower()))

    chars = sorted(list(chars))

    vocab_size = len(chars)
    char_to_ix = dict(zip(chars, range(vocab_size)))
    ix_to_char = dict(zip(range(vocab_size), chars))
    return vocab_size, char_to_ix, ix_to_char


def get_inputs_targets(data_path: str,
                       sequence_length: int,
                       char_to_ix: Dict[str, int],
                       full_sequences: bool,
                       package: str):
    """Generates inputs and targets from given data and sequence length."""
    assert package in ('numpy', 'pytorch')
    inputs_gen = read_in_chunks(data_path, sequence_length, 0, full_sequences)
    targets_gen = read_in_chunks(data_path, sequence_length, 1, full_sequences)
    for x, y in zip(inputs_gen, targets_gen):
        # handling the last item
        if len(y) < sequence_length:
            x = x[:-1]

        x, y = [char_to_ix[ch] for ch in x.lower()], [char_to_ix[ch] for ch in y.lower()]
        if package == 'numpy':
            yield np.array(x), np.array(y)
        else:
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
