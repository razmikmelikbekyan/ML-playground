from typing import Tuple, Generator

import torch.utils.data
from torchvision import datasets, transforms


def get_mnist_data(batch_size: int = 64) -> Tuple[Generator, Generator]:
    """
    Downloads MNIST data set. Returns iterators of train and test data sets with given batch size.

    The size of each batch of images will be the tensor with shape: (batch_size, 784) .
    The size of each batch of labels will be the tensor with shape: (batch_size, 10) .

    """
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                              download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    trainloader = ((flatten_mnist(im), one_hot_encode_mnist(l)) for im, l in trainloader)

    # Download and load the training data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/',
                             download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
    testloader = ((flatten_mnist(im), one_hot_encode_mnist(l)) for im, l in testloader)

    return trainloader, testloader


def one_hot_encode_mnist(labels: torch.Tensor) -> torch.Tensor:
    """
    Makes one hot encoding of mnist data set labels.

    input_shape: (batch_size, )
    output_shape: (batch_size, 10)
    """
    n = labels.shape[0]
    one_hot_encoded = torch.zeros(n, 10)
    one_hot_encoded[(torch.arange(n), labels)] = 1
    return one_hot_encoded.view(n, 10)


def flatten_mnist(images: torch.Tensor) -> torch.Tensor:
    """
    Flattens input images batch in order to have image like a vector.

    input_shape: (batch_size, 1, 28, 28)
    output_shape: (batch_size, 1 * 28 * 28) =  (batch_size, 784)
    """
    return images.view(images.shape[0], -1)
