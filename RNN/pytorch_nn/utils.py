import torch
import torch.nn.functional as F

def softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=0)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.sigmoid(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def dsigmoid(y: torch.Tensor) -> torch.Tensor:
    # y = sigmoid(x)
    # dy / dx = sigmoid(x)(1 - sigmoid(x)) = y * (1 - y)
    return y * (1 - y)


def drelu(y: torch.Tensor) -> torch.Tensor:
    # y = relu(x) = (x > 0) * x
    # dy / dx = (x > 0) * 1, so we can just replace all non zero elements in y with 1
    dy = torch.tensor(y)
    dy[dy != 0] = 1.
    return dy


def dtanh(y: torch.Tensor) -> torch.Tensor:
    # y = tanh(x)
    # dy / dx = (1 - tanh^2(x)) = 1 - y^2
    return 1 - y ** 2


def one_hot_encode(x: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Given the x array of inputs or labels, where each item is the index of character. Performs
    one hot encoding, aka returns the matrix with dimensions (len(x), size). Each row of the matrix
    consists of 0s and only one 1. The 1 is located at the index of the corresponding correct
    character.
    """
    n_rows = len(x)

    # here we manually add 1 at the end, in order to have each row of the matrix as a
    # matrix, instead of vector, for properly calculating dot products
    # for example, if size = 15, then the each row should have the size - (15, )
    one_hot_encoded = torch.zeros((n_rows, size))
    one_hot_encoded[(torch.arange(n_rows), x)] = 1.
    return one_hot_encoded.type(dtype)


def check_relative_difference(a: torch.Tensor, b: torch.Tensor, threshold: float) -> bool:
    """Returns True if (|a - b| / (|a| + |b|)) > threshold else False."""
    return torch.any(torch.abs(a - b) > threshold * (torch.abs(a) + torch.abs(b)))
