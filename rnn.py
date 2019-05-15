from typing import Tuple

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


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
            input_x[t] = x[t]

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

    def backward(self, x: np.ndarray, labels: np.ndarray):
        """

        :param x:
        :param labels:
        :return:
        """
        n = x.shape[0]

        dw_hx = np.zeros_like(self.w_hx)
        dw_hh = np.zeros_like(self.w_hh)
        dw_hy = np.zeros_like(self.w_hy)

        states, predicted_probabilities = self.forward(x)

        for t in reversed(range(n)):
            one_hot_labels = np.zeros(self.vocabulary_size)
            one_hot_labels[labels[t]] = 1

            dy = predicted_probabilities[t] - one_hot_labels
            dw_hy += np.outer(dy, states[t].T)

            dh = np.dot(self.w_hy.T, dy)

            dh_dh_ks = []
            for k in range(t):
                dh_dz = 1 - states[t] ** 2
                dz_dh_prev = self.w_hh
                dh_dh_prev = dh_dz * dz_dh_prev


n = 25
data = open('input.txt', 'r').read()  # should be simple plain text file

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

inputs = [char_to_ix[x] for x in data[:n]]
labels = [char_to_ix[x] for x in data[1:n + 1]]

rnn = RNN(len(char_to_ix), 10)
a, b = rnn.forward(np.array(inputs))
loss = rnn.calculate_loss(b, np.array(labels))
rnn.backward(np.array(inputs), np.array(labels))
print(a.shape, b.sum(axis=1))
