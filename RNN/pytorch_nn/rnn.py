from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from .utils import dsigmoid, drelu, dtanh, one_hot_encode, check_relative_difference
except ModuleNotFoundError:
    from utils import dsigmoid, drelu, dtanh, one_hot_encode, check_relative_difference


class RNN:
    """
    This class represents simple Recurrent Neural Network implementation for
    character-level language model. The purpose of the network is correctly predicting the next
    character, given the previous sequence of characters.

    This implementation is purely PyTorch based. It uses PyTorch automatic gradient calculation.
    """

    activations = {
        'tanh': (torch.tanh, dtanh),
        'sigmoid': (torch.sigmoid, dsigmoid),
        'relu': (F.relu, drelu)
    }

    params = ('w_hx', 'w_hh', 'w_hy')

    def __init__(self,
                 vocabulary_size: int,
                 hidden_size: int,
                 non_linearity: str = 'tanh',
                 dtype: torch.dtype = torch.float64):
        """
        :param vocabulary_size: the size of vocabulary, aka the number of unique characters in
                                vocabulary
        :param hidden_size: the size of hidden state
        """

        if non_linearity not in self.activations:
            raise ValueError(f'Non linearity must be one of the followings: '
                             f'{tuple(self.activations)}.')

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.dtype = dtype

        # activation function and its dervivate w.r.t. its direct input
        self.f, self.f_prime = self.activations[non_linearity]

        # randomly initializing weights

        self.w_hx = torch.Tensor(hidden_size, vocabulary_size).uniform_(
            -np.sqrt(1. / vocabulary_size), np.sqrt(1. / vocabulary_size)
        )
        self.w_hx = self.w_hx.type(self.dtype)
        self.w_hx.requires_grad = True

        self.w_hh = torch.Tensor(hidden_size, hidden_size).uniform_(
            -np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size)
        )
        self.w_hh = self.w_hh.type(self.dtype)
        self.w_hh.requires_grad = True

        self.w_hy = torch.Tensor(vocabulary_size, hidden_size).uniform_(
            -np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size)
        )
        self.w_hy = self.w_hy.type(self.dtype)
        self.w_hy.requires_grad = True

        # setting the current state
        self.current_state = torch.zeros(self.hidden_size, dtype=self.dtype)

    def reset_current_state(self):
        """Resets current state to zeros."""
        self.current_state = torch.zeros(self.hidden_size, dtype=self.dtype)

    # ### Forward pass ###

    def forward(self, x: Tensor, update_state: bool) -> Tuple[Tensor, Tensor]:
        """
        The basic forward pass:

        z_{t} = w_hh * h_{t-1} + w_hx * x_{t}
        h_{t} = f(z_{t})
        y_{t} = w_hy * h_{t}
        p_{t} = softmax(y_{t})

        Makes forward pass through network. self.w_hx.requires_grad()
        :param x: the array of integers, where each item is the index of character, the size of
                  array will be the sequence length
        :param update_state: bool, if True updates current state with last state
        :return: the tuple of states and predicted_probabilities
                 states - tensor of states, size = (sequence length, hidden size)
                 predicted_probabilities - tensor of predicted probabilities for each character in
                                           vocabulary, size = (sequence length, vocabulary size)
        """
        n = len(x)

        # one hot encoding of input
        inputs_matrix = one_hot_encode(x, self.vocabulary_size, self.dtype)

        log_ps = torch.zeros(n, self.vocabulary_size, dtype=self.dtype)
        hs = torch.zeros(n, self.hidden_size, dtype=self.dtype)

        for t in range(len(x)):
            # state at t - 1, dim : (self.hidden_size, 1)
            h_t_1 = self.current_state.clone() if t == 0 else hs[t - 1].clone()

            # state at t, dim : (self.hidden_size, 1)
            h_t = self.f(
                torch.matmul(self.w_hh, h_t_1) + torch.matmul(self.w_hx, inputs_matrix[t])
            )

            # prediction from hidden state at t,
            # log probabilities for next chars,  dim : (self.vocabulary_size, 1)
            p_t = F.log_softmax(torch.matmul(self.w_hy, h_t), dim=0)

            # updating hidden state and and predicted_probabilities keepers
            hs[t], log_ps[t] = h_t, p_t

        if update_state:
            self.current_state = hs[-1].clone()  # updating the current state

        return hs, log_ps

    def _loss(self, x: Tensor, labels: Tensor, update_state: bool) -> Tensor:
        """
        Calculates cross entropy loss using target characters indexes and network predictions for
        all characters: loss = ∑ -label_{t} * log(predicted_probability_{t}). It keeps the
        computational graph for automatic gradient calculation.
        """
        _, log_ps = self.forward(x, update_state)
        return -torch.sum(log_ps[(torch.arange(len(labels)), labels)])

    def calculate_loss(self, x: Tensor, labels: Tensor, update_state: bool) -> float:
        """
        Calculates cross entropy loss using target characters indexes and network predictions for
        all characters: loss = ∑ -label_{t} * log(predicted_probability_{t}). It returns loss
        value (float).
        """
        return self._loss(x, labels, update_state).item()

    # ### Gradient descent ###

    def sgd_step(self, x: Tensor, labels: Tensor, lr: float):
        """
        Performs gradient descent step for model parameters: w_hx, w_hh, w_hy.

        - forward pass: calculating next char probabilities, given previous sequence of chars
        - backward pass: calculating loss and its gradient w.r.t. model params
        - sgd update: update params in the opposite of the gradient direction
        """
        # Backward pass
        loss = self._loss(x, labels, False)
        loss.backward()

        with torch.no_grad():
            # w <-- w - lr * dloss / dw
            self.w_hx -= lr * torch.clamp(self.w_hx.grad, -5., 5.)
            self.w_hh -= lr * torch.clamp(self.w_hh.grad, -5., 5.)
            self.w_hy -= lr * torch.clamp(self.w_hy.grad, -5., 5.)

            # Manually zero the gradients after running the backward pass
            self.w_hx.grad.zero_()
            self.w_hh.grad.zero_()
            self.w_hy.grad.zero_()

    # ### Sampling ###

    def generate(self, seed_ix: int, n: int) -> List[int]:
        """
        Sample a sequence of integers from the model.
        :param seed_ix: seed letter for first time step
        :param n: number of samples
        :return: list of indexes
        """
        assert isinstance(seed_ix, int) and self.vocabulary_size > seed_ix >= 0
        self.reset_current_state()

        possible_indexes = np.arange(self.vocabulary_size)

        sample_indexes = []
        ix = seed_ix
        for t in range(n):
            _, ps = self.forward(torch.tensor([ix], dtype=torch.long), True)
            ix = np.random.choice(possible_indexes, p=torch.exp(ps[0]).detach().numpy().ravel())
            sample_indexes.append(possible_indexes[ix])
        return sample_indexes

    # ### Saving and Loading model ###

    def save(self, saving_path: str):
        """Saves model."""
        np.save(saving_path, {p: getattr(self, p) for p in self.params})
        print(f'Model has been save at the following path: {saving_path}.')

    @staticmethod
    def load(model_path: str):
        """Loads saved model and returns it."""
        params = dict(np.load(model_path).item())
        w_hx, w_hh, w_hy = params['w_hx'], params['w_hh'], params['w_hy']
        _, vocabulary_size = w_hx.shape
        hidden_size, hidden_size = w_hh.shape
        model = RNN(vocabulary_size, hidden_size)
        model.w_hx, model.w_hh, model.w_hy = w_hx, w_hh, w_hy
        return model

    # implementation by Andrej Kharpaty, for performing check

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = hprev.clone().detach()
        loss = torch.tensor([0], dtype=self.dtype)

        # forward pass
        for t in range(len(inputs)):
            xs[t] = torch.zeros((vocab_size, 1), dtype=self.dtype)
            xs[t][inputs[t]] = 1

            # hidden state
            hs[t] = torch.tanh(
                torch.matmul(self.w_hx, xs[t]) + torch.matmul(self.w_hh, hs[t - 1]))
            ys[t] = torch.matmul(self.w_hy, hs[t])  # unnormalized log probabilities for next chars
            ps[t] = torch.exp(ys[t]) / torch.sum(torch.exp(ys[t]))  # probabilities for next chars
            loss += -torch.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)

        # backward pass: compute gradients going backwards
        dWxh = torch.zeros_like(self.w_hx, dtype=self.dtype)
        dWhh = torch.zeros_like(self.w_hh, dtype=self.dtype)
        dWhy = torch.zeros_like(self.w_hy, dtype=self.dtype)

        dhnext = torch.zeros_like(hs[0], dtype=self.dtype)
        for t in reversed(range(len(inputs))):
            dy = ps[t].clone().detach()
            dy[targets[t]] -= 1

            # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad
            dWhy += torch.matmul(dy, hs[t].t())
            dh = torch.matmul(self.w_hy.t(), dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dWxh += torch.matmul(dhraw, xs[t].t())
            dWhh += torch.matmul(dhraw, hs[t - 1].t())
            dhnext = torch.matmul(self.w_hh.t(), dhraw)

        # clip to mitigate exploding gradients
        dWxh = torch.clamp(dWxh, -5., 5.)
        dWhh = torch.clamp(dWhh, -5., 5.)
        dWhy = torch.clamp(dWhy, -5., 5.)

        return loss, dWxh, dWhh, dWhy, hs[len(inputs) - 1]


if __name__ == '__main__':
    dtype = torch.float64

    # should be simple plain text file
    file = open('data/datasets/simple.txt', 'r')
    data = file.read()
    file.close()

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    print(f'Data has {data_size} characters, {vocab_size} unique.')
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    n = 25
    inputs = torch.tensor([char_to_ix[x] for x in data[:n]], dtype=torch.long)
    labels = torch.tensor([char_to_ix[x] for x in data[1:n + 1]], dtype=torch.long)

    rnn = RNN(vocab_size, 10, dtype=dtype)

    # current implementation
    hs, log_ps = rnn.forward(inputs, update_state=False)

    assert all(abs(torch.sum(torch.exp(x)).item() - 1) < 1e-6 for x in log_ps)

    # pytorch autograd backpropagation
    l_1 = rnn._loss(inputs, labels, False)
    l_1.backward()
    dw_hx_1, dw_hh_1, dw_hy_1 = rnn.w_hx.grad, rnn.w_hh.grad, rnn.w_hy.grad
    dw_hx_1 = torch.clamp(dw_hx_1, -5., 5.)
    dw_hh_1 = torch.clamp(dw_hh_1, -5., 5.)
    dw_hy_1 = torch.clamp(dw_hy_1, -5., 5.)

    # Karpathy implementation
    l_2, dw_hx_2, dw_hh_2, dw_hy_2, _ = rnn.lossFun(inputs, labels,
                                                    torch.zeros((10, 1), dtype=dtype))

    print()
    print('Checking current implementation with Karpathy implementation.')

    print()
    print('loss_1={:.5f}'.format(l_1.item()))
    print('loss_2={:.5f}'.format(l_2.item()))
    assert abs(l_1.item() - l_2.item()) < 1e-6
    print('loss check is passed')

    print()
    assert torch.allclose(dw_hx_1, dw_hx_2, atol=1e-6)
    assert torch.allclose(dw_hx_1, dw_hx_2, atol=1e-6)

    print('dWxh check is passed')

    print()
    assert torch.allclose(dw_hh_1, dw_hh_2, atol=1e-6)
    assert torch.allclose(dw_hh_1, dw_hh_2, atol=1e-6)

    print('dWhh check is passed')

    print()
    assert torch.allclose(dw_hy_1, dw_hy_2, atol=1e-6)
    assert torch.allclose(dw_hy_1, dw_hy_2, atol=1e-6)
    print('dWhy check is passed')
