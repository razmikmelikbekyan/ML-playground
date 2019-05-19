from typing import Tuple, Dict

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return (x > 0) * x


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def read_in_chunks(data_path: str, chunk_size: int, offset: int):
    """Lazy function (generator) to read a file piece by piece. It is needed in order to
    optimize memory consumption due to big files."""
    file_object = open(data_path, 'r')
    if offset:
        file_object.seek(offset, 0)
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data
    file_object.close()


def get_support_data(data_path: str) -> Tuple[int, Dict[str, int], Dict[int, str]]:
    """Reads the file and returns vocabulary size and characters to indexes mapping dicts."""
    chars = set()

    for x in read_in_chunks(data_path, 1000, 0):
        chars.update(set(x.lower()))

    chars = sorted(list(chars))

    vocab_size = len(chars)
    char_to_ix = dict(zip(chars, range(vocab_size)))
    ix_to_char = dict(zip(range(vocab_size), chars))
    return vocab_size, char_to_ix, ix_to_char


def get_inputs_targets(data_path: str,
                       sequence_length: int,
                       char_to_ix: Dict[str, int]):
    """Generates inputs and targets from given data and sequence length."""
    inputs_gen = read_in_chunks(data_path, sequence_length, 0)
    targets_gen = read_in_chunks(data_path, sequence_length, 1)
    for x, y in zip(inputs_gen, targets_gen):
        # handling the last item
        if len(y) < sequence_length:
            x = x[:-1]
        yield [char_to_ix[ch] for ch in x.lower()], [char_to_ix[ch] for ch in y.lower()]
