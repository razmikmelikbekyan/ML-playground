import torch
import torch.nn.functional as F


class NeuralNetwork:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dtype: torch.dtype):
        self.w_1 = torch.randn(hidden_size, input_size, dtype=dtype, requires_grad=True)
        self.w_2 = torch.randn(output_size, hidden_size, dtype=dtype, requires_grad=True)

    def forward(self, x: torch.Tensor):


        z_1 = torch.sigmoid(torch.matmul(self.w_1, x))
        z_2 = F.log_softmax(torch.matmul(self.w_2, z_1), dim=0)
        return z_2

    def loss(self, x: torch.Tensor, label: torch.Tensor):
        log_prediction = self.forward(x)
        return -torch.sum(label * log_prediction)

    def sgd_step(self, x: torch.Tensor, label: torch.Tensor, lr: float):
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

    model = NeuralNetwork(10, 20, 3)

    x = torch.arange(10, dtype=dtype).view(10, 1)
    label = torch.tensor([0, 0, 1.], dtype=dtype).reshape(3, 1)

    log_pred = model.forward(x)
    pred = torch.exp(log_pred)

    assert pred.shape == label.shape == (3, 1)
    assert bool(abs(sum(pred) - 1.) < 1e-6)

    loss = model.loss(x, label)
    assert torch.equal(loss, -log_pred[2, 0])
    print('\nShapes are correct.')
