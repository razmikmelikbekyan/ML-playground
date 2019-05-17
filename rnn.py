from typing import Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))

state = np.random.RandomState(seed=15)

class RNN:
    """
    This class represents simple Recurrent Neural Network implementation for  for
    character-level language model.
    """

    def __init__(self,
                 vocabulary_size: int,
                 hidden_size: int,
                 non_linearity: str = 'tanh'):
        """
        :param vocabulary_size: the size of vocabulary, aka the number of unique characters in
                                vocabulary
        :param hidden_size: the size of hidden state
        """

        if non_linearity not in ('tanh', 'sigmoid'):
            raise ValueError('Non linearity must be sigmoid or tanh.')

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.non_linearity = non_linearity

        # randomly initializing weights

        self.w_hx = np.random.uniform(
            -np.sqrt(1. / vocabulary_size), np.sqrt(1. / vocabulary_size),
            (hidden_size, vocabulary_size)
        )

        self.w_hh = np.random.uniform(
            -np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size),
            (hidden_size, hidden_size)
        )

        self.w_hy = np.random.uniform(
            -np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size),
            (vocabulary_size, hidden_size)
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Makes forward pass through network.
        :param x: the array of integers, where each item is the index of character, the size of
                  array will be the sequence length
        :return: the tuple of states and predicted_probabilities
                 states - array of states, size = (sequence length, hidden size)
                 predicted_probabilities - array of predicted probabilities for each character in
                                           vocabulary, size = (sequence length, vocabulary size)
        """
        # sequence length
        n = x.shape[0]

        # non linear function
        f = np.tanh if self.non_linearity == 'tanh' else sigmoid

        states = np.zeros((n, self.hidden_size))
        predicted_probabilities = np.zeros((n, self.vocabulary_size))
        for t in range(n):
            # state at t - 1
            h_t_1 = np.zeros(self.hidden_size) if t == 0 else states[t - 1]

            # TODO: make more smart
            # one hot encoding input
            input_x = np.zeros(self.vocabulary_size)
            input_x[x[t]] = 1

            # state at t
            z_t = np.dot(self.w_hh, h_t_1) + np.dot(self.w_hx, input_x)
            h_t = f(z_t)

            # prediction from hidden state at t
            y_t = np.dot(self.w_hy, h_t)
            p_t = softmax(y_t)

            # updating hidden state and and predicted_probabilities keepers
            states[t] = h_t
            predicted_probabilities[t] = p_t

        return states, predicted_probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes prediction based on forward pass. It returns the array of integers,
        where each item is the index of predicted character.
        """
        _, outputs = self.forward(x)
        return np.argmax(outputs, axis=1)

    def calculate_loss(self, predicted_probabilities: np.ndarray, labels: np.ndarray) -> float:
        return -sum(np.log(p[labels[i]]) for i, p in enumerate(predicted_probabilities))

    def calculate_h_grad_wrt_z(self, h_t: np.ndarray) -> np.ndarray:
        """
        Calculates state gradient w.r.t. z, dh / dz.
        Note that h_{t} = tanh(z_{t}) or h_{t} = sigmoid(z_{t})"""
        return 1 - h_t ** 2 if self.non_linearity == 'tanh' else h_t * (1 - h_t)

    def calculate_loss_grad_wrt_z(self, dh_t: np.ndarray, states: np.ndarray, t: int):
        """
        Calculates dloss_{t} / dz_{k} for all k from 1 to t. Please note that h_{t} = f(z_{t})

        :param dh_t: dloss_{t} / dh_{t}, aka loss gradient w.r.t state for the same time index t
        :param states: array of states
        :param t: time index
        :return: list of arrays, where each array is the calculated gradient:
                 [dloss_{t} / dz_{t}, dloss_{t} / dz_{t - 1}, ..., dloss_{t} / dz_{1}]
        """

        # dl / dz = (dl / dh) * (dh / dz)
        dz_t = dh_t * self.calculate_h_grad_wrt_z(states[t])

        grads = [dz_t]
        for i in reversed(range(1, t + 1)):
            # dl / dz_(t-1) = (dl / dz_{t})(dz_{t} / dh_{t-1}) * (dh_{t-1) / dz_{t-1})
            dz_i = np.dot(self.w_hh, grads[-1]) * self.calculate_h_grad_wrt_z(states[i - 1])
            grads.append(dz_i)
        return grads

    def calculate_w_hh_grad(self, dh_t: np.ndarray, states: np.ndarray, t: int):
        """
        Calculates dloss_{t} / dd_whh, aka gradient of loss at time t with respect to W_hh.
        :param dh_t: dloss_{t} / dh_{t}, aka gradient of loss at time t with respect to h_{t}
        :param states_gradients: the array of state gradients, where each item represents the state
                                 gradient with respect to its previous state
        :param t: time index
        :return: the gradient of loss w.r.t. W_hh
        """
        dw_hh = np.zeros_like(self.w_hh)

        tmp = dh_t * (1 - states[t] ** 2)
        for i in reversed(range(1, t + 1)):
            dw_hh += np.outer(tmp, states[i - 1])
            tmp = np.dot(self.w_hh, tmp) * (1 - states[i - 1] ** 2)
        return dw_hh

    def backward(self, x: np.ndarray, labels: np.ndarray):
        """

        :param x:
        :param labels:
        :return:
        """

        n = x.shape[0]
        inputs = np.zeros((n, self.vocabulary_size))
        inputs[(np.arange(n), x)] = 1


        dw_hx = np.zeros_like(self.w_hx)
        dw_hh = np.zeros_like(self.w_hh)
        dw_hy = np.zeros_like(self.w_hy)

        states, predicted_probabilities = self.forward(x)

        for t in reversed(range(n)):
            input_x = np.zeros(self.vocabulary_size)
            input_x[x[t]] = 1

            one_hot_labels = np.zeros(self.vocabulary_size)
            one_hot_labels[labels[t]] = 1

            dy_t = predicted_probabilities[t] - one_hot_labels
            dh_t = np.dot(self.w_hy.T, dy_t)
            dz_t_0 = self.calculate_loss_grad_wrt_z(dh_t, states, t)

            # dl / dw_hy = (dl / dy) * (dl * dw_hy)
            dw_hy += np.outer(dy_t, states[t].T)

            # dl / dw_hh = ∑ (dl / dz_{k}) * (dz_{k} / dw_hy) for all k from 0 to t
            dw_hh += sum(
                np.outer(dz_i, states[i - 1]) for dz_i, i in zip(dz_t_0, reversed(range(1, t + 1)))
            )

            # dl / dw_hx = ∑ (dl / dz_{k}) * (dz_{k} / dw_hx) for all k from 0 to t
            dw_hx += sum(
                np.outer(dz_i, inputs[i].T) for dz_i, i in zip(dz_t_0, reversed(range(1, t + 1)))
            )

        return dw_hx, dw_hh, dw_hy

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(
                np.dot(self.w_hx, xs[t]) + np.dot(self.w_hh, hs[t - 1]))  # hidden state
            ys[t] = np.dot(self.w_hy, hs[t])  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
            loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.w_hx), np.zeros_like(self.w_hh), np.zeros_like(
            self.w_hy)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[
                t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dh = np.dot(self.w_hy.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.w_hh.T, dhraw)
        # for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        #     np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, hs[len(inputs) - 1]


n = 25
data = open('input.txt', 'r').read()  # should be simple plain text file

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

inputs = np.array([char_to_ix[x] for x in data[:n]])
labels = np.array([char_to_ix[x] for x in data[1:n + 1]])

rnn = RNN(vocab_size, 10)
_, p_probs = rnn.forward(inputs)

loss = rnn.calculate_loss(p_probs, labels)
print('loss_1={:.5f}'.format(loss))

whx, whh, why = rnn.w_hx.copy(), rnn.w_hh.copy(), rnn.w_hy.copy()

dw_hx_1, dw_hh_1, dw_hy_1 = rnn.backward(inputs, labels)

rnn = RNN(vocab_size, 10)
rnn.w_hx, rnn.w_hh, rnn.w_hy = whx, whh, why

loss, dWxh_2, dWhh_2, dWhy_2, _ = rnn.lossFun(inputs, labels, np.zeros((10, 1)))
print('loss_2={:.5f}'.format(loss))
#
print()
print('dWxh')
print(np.isclose(dw_hx_1, dWxh_2))
#
print()
print('dWhh')
print(np.isclose(dw_hh_1, dWhh_2))

# print()
# print('dWhy')
# print(np.isclose(dw_hy_1, dWhy_2))
