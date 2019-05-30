import torch
import torch.nn.functional as F


class NeuralNetwork:
    """
    Simple Neural Network with one hidden layer for classification.
    The backpropagation is done using PyTorch automatic gradient calculation.

    It uses sigmoid as an activation function for hidden layer and log_softmax for output layer.
    Loss function is a cross entropy loss.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dtype: torch.dtype):
        self.w_1 = torch.randn(hidden_size, input_size, dtype=dtype) * 0.01
        self.w_2 = torch.randn(output_size, hidden_size, dtype=dtype) * 0.01

        self.w_1.requires_grad = True
        self.w_2.requires_grad = True

    def forward(self, x: torch.Tensor):
        """
        Forward pass function.

        x shape: (batch_size, input_size)
        Returns log prediction.
        """
        z_1 = torch.sigmoid(torch.matmul(x, self.w_1.t()))
        z_2 = F.log_softmax(torch.matmul(z_1, self.w_2.t()), dim=1)
        return z_2

    def loss(self, x: torch.Tensor, label: torch.Tensor):
        """
        Cross entropy loss function.

        x shape: (batch_size, input_size)
        label shape: (batch_size, output_size)
        """
        log_prediction = self.forward(x)
        return -torch.sum(label * log_prediction)

    def sgd_step(self, x: torch.Tensor, label: torch.Tensor, lr: float):
        """Performs simple stochastic gradient descent step."""
        loss = self.loss(x, label)
        loss.backward()

        with torch.no_grad():
            self.w_1 -= lr * self.w_1.grad
            self.w_2 -= lr * self.w_2.grad

            # Manually zero the gradients after updating weights
            self.w_1.grad.zero_()
            self.w_2.grad.zero_()


if __name__ == "__main__":
    print('Testing implementation.')
    dtype = torch.float32
    model = NeuralNetwork(10, 20, 3, dtype)

    x = torch.arange(10, dtype=dtype).view(1, 10)
    label = torch.tensor([0, 0, 1.], dtype=dtype).reshape(1, 3)

    log_pred = model.forward(x)
    pred = torch.exp(log_pred)

    assert pred.shape == label.shape == (1, 3)
    assert abs(torch.sum(pred).detach().item() - 1.) < 1e-6

    loss = model.loss(x, label)
    assert torch.equal(loss, -log_pred[0, 2])
    print('\nShapes are correct.')
