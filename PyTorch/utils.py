from typing import List

import matplotlib.pyplot as plt
import torch


def check_relative_difference(a: torch.tensor, b: torch.tensor, threshold: float) -> bool:
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
