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
