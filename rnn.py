import operator
from typing import Tuple, Dict

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1. / (1 + np.exp(-x))


class RNN:
    """
    This class represents simple Recurrent Neural Network implementation for  for
    character-level language model. In this model the purpose of the network is correctly
    predicting the next character, given the previous one.
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

    def one_hot_encode(self, x: np.ndarray) -> np.ndarray:
        """
        Given the array of inputs or labels, where each item is the index of character. Performs
        one hot encoding, aka returns the matrix with dimensions (len(x), vocabulary_size). Each
        row of the matrix consists of 0s and only one 1. The 1 is located at the index of the cor-
        responding correct character.
        """
        n_rows = len(x)

        # here we manually add 1 at the end, in order to have each row of the matrix as a
        # matrix, instead of vector, for properly calculating dot products
        # for example, if self.vocabulary_size = 15, then the each row should have the
        # size - (15, 1), (without additional 1, it will have size - (15, )
        one_hot_encoded = np.zeros((n_rows, self.vocabulary_size, 1))
        one_hot_encoded[(np.arange(n_rows), x)] = 1
        return one_hot_encoded

    def forward(self, x: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Makes forward pass through network.
        :param x: the array of integers, where each item is the index of character, the size of
                  array will be the sequence length
        :return: the tuple of states and predicted_probabilities
                 states - array of states, size = (sequence length, hidden size)
                 predicted_probabilities - array of predicted probabilities for each character in
                                           vocabulary, size = (sequence length, vocabulary size)
        """
        # non linear function
        f = np.tanh if self.non_linearity == 'tanh' else sigmoid

        # one hot encoding of input
        inputs_matrix = self.one_hot_encode(x)

        predicted_probabilities, states = {}, {}
        states[-1] = np.zeros((self.hidden_size, 1))
        for t in range(len(x)):
            # state at t - 1
            h_t_1 = states[t - 1]

            # state at t
            z_t = np.dot(self.w_hh, h_t_1) + np.dot(self.w_hx, inputs_matrix[t])
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

    @staticmethod
    def calculate_loss(predicted_probabilities: Dict, labels: np.ndarray) -> float:
        """
        Calculates cross entropy loss using target characters indexes and network predictions for
        all characters: loss = ∑ -label_{t} * log(predicted_probability_{t})
        """
        return -sum(np.log(predicted_probabilities[i][labels[i], 0]) for i in range(len(labels)))

    def calculate_h_grad_wrt_z(self, h_t: np.ndarray) -> np.ndarray:
        """
        Calculates state gradient w.r.t. z, dh / dz.
        Note that h_{t} = tanh(z_{t}) or h_{t} = sigmoid(z_{t})"""
        return 1 - h_t ** 2 if self.non_linearity == 'tanh' else h_t * (1 - h_t)

    def calculate_loss_grad_wrt_z(self, dh_t: np.ndarray, states: Dict, t: int):
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
        for i in reversed(range(t)):
            # dl / dz_(t-1) = (dl / dz_{t})(dz_{t} / dh_{t-1}) * (dh_{t-1) / dz_{t-1})
            dz_i = np.dot(self.w_hh.T, grads[-1]) * self.calculate_h_grad_wrt_z(states[i])
            grads.append(dz_i)
        return grads

    def backward(self,
                 x: np.ndarray,
                 labels: np.ndarray,
                 states: Dict,
                 predicted_probabilities: Dict):
        """
        Makes backward pass through the network. Returns the gradients of loss w.r.t. network
        parameters -  w_hx, w_hh, w_hy.

        :param x: the array of input characters, where each item is the index of character, the
                  size of array will be the sequence length
        :param labels: the array of target characters, where each item is the index of character,
                       the size of array will be the sequence length
        :param states: the hidden states of network, (the first output of the self.forward method)
        :param predicted_probabilities: network predictions for given inputs,
                                        (the second output of the self.forward method)
        :return: gradients of w_hx, w_hh, w_hy
        """
        inputs_matrix = self.one_hot_encode(x)
        labels_matrix = self.one_hot_encode(labels)

        dw_hx = np.zeros_like(self.w_hx)
        dw_hh = np.zeros_like(self.w_hh)
        dw_hy = np.zeros_like(self.w_hy)

        for t in reversed(range(len(x))):
            # dl / dy = p - label
            dy_t = predicted_probabilities[t] - labels_matrix[t]

            # dl / dh = (dl / dy) (dy / dh) = (p - label)w_hy
            dh_t = np.dot(self.w_hy.T, dy_t)

            # dl / dz
            dz_t_0 = self.calculate_loss_grad_wrt_z(dh_t, states, t)

            # dl / dw_hy = (dl / dy) * (dl * dw_hy)
            dw_hy += np.dot(dy_t, states[t].T)

            # dl / dw_hh = ∑ (dl / dz_{k}) * (dz_{k} / dw_hy) for all k from 0 to t
            dw_hh += sum(
                np.dot(dz_i, states[i - 1].T)
                for dz_i, i in zip(dz_t_0, reversed(range(t + 1)))
            )

            # dl / dw_hx = ∑ (dl / dz_{k}) * (dz_{k} / dw_hx) for all k from 0 to t
            dw_hx += sum(
                np.dot(dz_i, inputs_matrix[i].T)
                for dz_i, i in zip(dz_t_0, reversed(range(t + 1)))
            )

        return dw_hx, dw_hh, dw_hy

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        states, predicted_probabilities = self.forward(x)
        bptt_gradients = self.backward(x, y, states, predicted_probabilities)

        # List of all parameters we want to check.
        model_parameters = ['w_hx', 'w_hh', 'w_hy']

        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print(f"Performing gradient check for parameter {pname} "
                  f"with size = {np.prod(parameter.shape)}")

            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

                # Save the original value so we can reset it later
                original_value = parameter[ix]

                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)

                parameter[ix] = original_value + h
                _, ps = self.forward(x)
                gradplus = self.calculate_loss(ps, y)

                parameter[ix] = original_value - h
                _, ps = self.forward(x)
                gradminus = self.calculate_loss(ps, y)

                estimated_gradient = (gradplus - gradminus) / (2 * h)

                # Reset parameter to original value
                parameter[ix] = original_value

                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]

                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = (
                        np.abs(backprop_gradient - estimated_gradient) /
                        (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                )
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print(f"Gradient Check ERROR: parameter = {pname} ix = {ix}")
                    print(f"+ h Loss: {gradplus}")
                    print(f"- h Loss: {gradminus}")
                    print(f"Estimated_gradient: {estimated_gradient}")
                    print(f"Backpropagation gradient: {backprop_gradient}")
                    print(f"Relative Error: {relative_error}")
                    return
                it.iternext()
            print(f"Gradient check for parameter {pname} passed.")

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
            dy[targets[t]] -= 1

            # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad
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
states, p_probs = rnn.forward(inputs)

l_1 = rnn.calculate_loss(p_probs, labels)
print('loss_1={:.5f}'.format(l_1))

dw_hx_1, dw_hh_1, dw_hy_1 = rnn.backward(inputs, labels, states, p_probs)

l_2, dWxh_2, dWhh_2, dWhy_2, _ = rnn.lossFun(inputs, labels, np.zeros((10, 1)))
print('loss_2={:.5f}'.format(l_2))

print()
print('loss')
print(l_1 == l_2)

#
print()
print('dWxh')
assert np.all(np.isclose(dw_hx_1, dWxh_2, atol=1e-10))
# print(dw_hx_1 - dWxh_2)
#
print()
print('dWhh')
assert np.all(np.isclose(dw_hh_1, dWhh_2, atol=1e-10))

#
print()
print('dWhy')
assert np.array_equal(dw_hy_1, dWhy_2)

# http://willwolf.io/2016/10/18/recurrent-neural-network-gradients-and-lessons-learned-therein/
