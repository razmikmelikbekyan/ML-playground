import numpy as np
import torch
import torch.nn.functional as F
from utils import check_relative_difference

dtype = torch.float64
device = torch.device("cpu")


class NeuralNetwork:

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.w_1 = torch.randn(hidden_size, input_size, dtype=dtype, device=device) * 0.01
        self.w_2 = torch.randn(output_size, hidden_size, dtype=dtype, device=device) * 0.01

        self.cache = {}

    def forward(self, x: torch.Tensor):
        h_1 = torch.matmul(self.w_1, x)
        z_1 = torch.sigmoid(h_1)
        h_2 = torch.matmul(self.w_2, z_1)
        z_2 = F.log_softmax(h_2, dim=0)

        self.cache['z_1'] = z_1
        self.cache['z_2'] = z_2
        return z_2

    def loss(self, x: torch.Tensor, label: torch.Tensor):
        log_prediction = self.forward(x)
        return -torch.sum(label * log_prediction)

    def backward(self, x: torch.Tensor, label: torch.Tensor):
        self.forward(x)

        z_1, z_2 = self.cache['z_1'], self.cache['z_2']

        dh_2 = torch.exp(z_2) - label
        dw_2 = torch.matmul(dh_2, z_1.t())
        dh_1 = torch.matmul(self.w_2.t(), dh_2) * (z_1 * (1 - z_1))
        dw_1 = torch.matmul(dh_1, x.t())
        return dw_1, dw_2

    def sgd_step(self, x: torch.Tensor, label: torch.Tensor, lr: float):
        dw_1, dw_2 = self.backward(x, label)
        self.w_1 -= lr * dw_1
        self.w_2 -= lr * dw_2

    def numerical_gradients(self, x: torch.Tensor, label: torch.Tensor, epsilon: float):
        d_params = (
            torch.zeros_like(self.w_1, dtype=dtype, device=device),
            torch.zeros_like(self.w_2, dtype=dtype, device=device)
        )
        params = (self.w_1, self.w_2)

        # calculating numerical gradients for each parameter
        for d_param, param in zip(d_params, params):

            # iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

                # keeping the original value so we can reset it later
                original_value = param[ix].item()

                # estimating numeric gradients

                # x + epsilon
                param[ix] = original_value + epsilon
                loss_plus = self.loss(x, label)

                # x - epsilon
                param[ix] = original_value - epsilon
                loss_minus = self.loss(x, label)

                # numeric_gradient = (f(x + delta) - f(x - delta)) / (2 * delta)
                d_param[ix] = ((loss_plus - loss_minus) / (2 * epsilon)).item()

                # resetting parameter to original value
                param[ix] = original_value
                it.iternext()

        return d_params

    def gradient_check(self,
                       x: torch.Tensor,
                       label: torch.Tensor,
                       epsilon: float = 1e-1,
                       threshold: float = 1e-5):
        """
        Performs gradient checking for model parameters:
         - computes the analytic gradients using our back-propagation implementation
         - computes the numerical gradients using the two-sided epsilon method
         - computes the relative difference between numerical and analytical gradients
         - checks that the relative difference is less than threshold
         - if the last check is failed, then raises an error
        """
        params = ('w_1', 'w_2')

        # calculating the gradients using backpropagation, aka analytic gradients
        self.cache = {}
        analytic_gradients = self.backward(x, label)

        # calculating numerical gradients
        self.cache = {}
        numeric_gradients = self.numerical_gradients(x, label, epsilon)

        # gradient check for each parameter
        for p_name, d_analytic, d_numeric in zip(params, analytic_gradients, numeric_gradients):
            print(f"\nPerforming gradient check for parameter {p_name} "
                  f"with size = {np.prod(d_analytic.shape)}.")

            if (not d_analytic.shape == d_numeric.shape or
                    check_relative_difference(d_analytic, d_numeric, threshold)):
                raise ValueError(f'Gradient check for {p_name} is failed.')

            print(f"Gradient check for parameter {p_name} is passed.")


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

    dw_1, dw_2 = model.backward(x, label)
    assert dw_1.shape == model.w_1.shape == (20, 10)
    assert dw_2.shape == model.w_2.shape == (3, 20)

    print('\nShapes are correct.')

    model.gradient_check(x, label, epsilon=1e-3, threshold=1e-4)