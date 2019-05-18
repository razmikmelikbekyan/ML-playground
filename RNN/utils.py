from typing import IO, Tuple, Dict, Generator

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return (x > 0) * x


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def read_in_chunks(file_object: IO, chunk_size: int):
    """Lazy function (generator) to read a file piece by piece."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def get_support_data(file_object: IO) -> Tuple[int, Dict[str, int], Dict[int, str]]:
    """Reads the file and returns vocabulary size and characters to indexes mapping dicts."""
    chars = set()

    for x in read_in_chunks(file_object, 1000):
        chars.update(set(x))

    chars = sorted(list(chars))

    vocab_size = len(chars)
    char_to_ix = dict(zip(chars, range(vocab_size)))
    ix_to_char = dict(zip(range(vocab_size), chars))
    return vocab_size, char_to_ix, ix_to_char


def get_inputs_targets(file_object: IO, sequence_length: int) -> Tuple[Generator, Generator]:
    """Returns inputs and targets generator objects."""
    return (
        read_in_chunks(file_object, sequence_length),
        read_in_chunks(file_object, sequence_length + 1)
    )


