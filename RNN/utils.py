from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return (x > 0) * x


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dsigmoid(y: np.ndarray) -> np.ndarray:
    # y = sigmoid(x)
    # dy / dx = sigmoid(x)(1 - sigmoid(x)) = y * (1 - y)
    return y * (1 - y)


def drelu(y: np.ndarray) -> np.ndarray:
    # y = relu(x) = (x > 0) * x
    # dy / dx = (x > 0) * 1, so we can just replace all non zero elements in y with 1
    dy = y.copy()
    dy[dy != 0] = 1.
    return dy


def dtanh(y: np.ndarray) -> np.ndarray:
    # y = tanh(x)
    # dy / dx = (1 - tanh^2(x)) = 1 - y^2
    return 1 - y ** 2


def one_hot_encode(x: np.ndarray, size: int) -> np.ndarray:
    """
    Given the x array of inputs or labels, where each item is the index of character. Performs
    one hot encoding, aka returns the matrix with dimensions (len(x), size). Each row of the matrix
    consists of 0s and only one 1. The 1 is located at the index of the corresponding correct
    character.
    """
    n_rows = len(x)

    # here we manually add 1 at the end, in order to have each row of the matrix as a
    # matrix, instead of vector, for properly calculating dot products
    # for example, if size = 15, then the each row should have the
    # size - (15, 1), (without additional 1, it will have size - (15, )
    one_hot_encoded = np.zeros((n_rows, size, 1))
    one_hot_encoded[(np.arange(n_rows), x)] = 1
    return one_hot_encoded


def check_relative_difference(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    """Returns True if (|a - b| / (|a| + |b|)) > threshold else False."""
    return np.any(np.abs(a - b) > threshold * (np.abs(a) + np.abs(b)))


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
                       full_sequences: bool):
    """Generates inputs and targets from given data and sequence length."""
    inputs_gen = read_in_chunks(data_path, sequence_length, 0, full_sequences)
    targets_gen = read_in_chunks(data_path, sequence_length, 1, full_sequences)
    for x, y in zip(inputs_gen, targets_gen):
        # handling the last item
        if len(y) < sequence_length:
            x = x[:-1]
        yield [char_to_ix[ch] for ch in x.lower()], [char_to_ix[ch] for ch in y.lower()]


def plot_loss(losses: List):
    """PLots losses."""
    _, epoch, loss = zip(*losses)
    plt.figure(figsize=(15, 15))
    plt.title('Loss over epochs')
    plt.plot(epoch, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()
