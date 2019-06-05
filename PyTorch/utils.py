from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


def check_relative_difference(a: torch.Tensor, b: torch.Tensor, threshold: float) -> bool:
    """Returns True if (|a - b| / (|a| + |b|)) > threshold else False."""
    numerator = torch.abs(a - b)
    denominator = torch.abs(a) + torch.abs(b)
    result = numerator / denominator
    result[torch.isnan(result)] = 0
    return bool(torch.any(result > threshold))


def plot_loss(train_loss: List, test_loss: List):
    """PLots losses."""
    epochs = range(len(train_loss))

    plt.figure(figsize=(15, 15))
    plt.title('Loss over epochs')
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def imshow(image: torch.Tensor, normalize: bool = True):
    """Imshow for Tensor."""

    fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax


def view_classify(image: torch.Tensor, ps: torch.Tensor):
    """Function for viewing an image and it's predicted classes."""

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
