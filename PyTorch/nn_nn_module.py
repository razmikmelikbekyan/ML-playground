import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class NeuralNetwork(nn.Module):
    """
    Simple Neural Network with one hidden layer for classification.
    It uses PyTorch built in nn module for performing backpropagation.

    It uses sigmoid as an activation function for hidden layer and log_softmax for output layer.
    Loss function is a negative log likelihood loss.
    """

    loss_function = nn.NLLLoss(reduction='sum')

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dtype: torch.dtype):
        super().__init__()
        self.w_1 = nn.Parameter(torch.randn(input_size, hidden_size, dtype=dtype) * 0.01)
        self.w_2 = nn.Parameter(torch.randn(hidden_size, output_size, dtype=dtype) * 0.01)

    def forward(self, x: torch.Tensor):
        """
        Forward pass function.

        x shape: (batch_size, input_size)
        Returns log prediction.
        """
        z_1 = torch.sigmoid(torch.matmul(x, self.w_1))
        z_2 = F.log_softmax(torch.matmul(z_1, self.w_2), dim=1)
        return z_2

    def loss(self, x: torch.Tensor, label: torch.Tensor):
        """
        Cross entropy loss function.

        x shape: (batch_size, input_size)
        label shape: (batch_size, output_size)
        """
        log_prediction = self.forward(x)
        return self.loss_function(log_prediction, torch.max(label, 1)[1])

    def sgd_step(self, x: torch.Tensor, label: torch.Tensor, lr: float):
        """Performs simple stochastic gradient descent step."""
        loss = self.loss(x, label)
        loss.backward()

        with torch.no_grad():
            for p in self.parameters():
                p -= lr * p.grad
                p.grad.zero_()


if __name__ == "__main__":
    print('Testing implementation.')
    dtype = torch.float32
    model = NeuralNetwork(10, 20, 3, dtype)

    x = torch.arange(10, dtype=dtype).view(1, 10)
    label = torch.tensor([0, 0, 1], dtype=torch.long).reshape(1, 3)

    log_pred = model.forward(x)
    pred = torch.exp(log_pred)

    assert pred.shape == label.shape == (1, 3)
    assert abs(torch.sum(pred).detach().item() - 1.) < 1e-6

    loss = model.loss(x, label)
    assert torch.equal(loss, -log_pred[0, 2])
    print('\nShapes are correct.')
