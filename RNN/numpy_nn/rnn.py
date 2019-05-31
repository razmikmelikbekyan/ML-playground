from typing import Tuple, Dict, List

import numpy as np

try:
    from .utils import (softmax, sigmoid, tanh, relu, dsigmoid, drelu, dtanh,
                        one_hot_encode, check_relative_difference)
except ModuleNotFoundError:
    from utils import (softmax, sigmoid, tanh, relu, dsigmoid, drelu, dtanh,
                       one_hot_encode, check_relative_difference)


class RNN:
    """
    This class represents simple Recurrent Neural Network implementation for
    character-level language model. The purpose of the network is correctly predicting the next
    character, given the previous sequence of characters.

    This implementation is purely numpy based.
    """

    activations = {
        'tanh': (tanh, dtanh),
        'sigmoid': (sigmoid, dsigmoid),
        'relu': (relu, drelu)
    }

    def __init__(self,
                 vocabulary_size: int,
                 hidden_size: int,
                 non_linearity: str = 'tanh'):
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

        # activation function and its dervivate w.r.t. its direct input
        self.f, self.f_prime = self.activations[non_linearity]

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

        # setting the current state
        self.current_state = np.zeros((self.hidden_size, 1))

    def reset_current_state(self):
        """Resets current state to zeros."""
        self.current_state = np.zeros((self.hidden_size, 1))

    # ### Forward pass ###

    def forward(self, x: np.ndarray, update_state: bool) -> Tuple[Dict, Dict]:
        """
        The basic forward pass:

        z_{t} = w_hh * h_{t-1} + w_hx * x_{t}
        h_{t} = f(z_{t})
        y_{t} = w_hy * h_{t}
        p_{t} = softmax(y_{t})

        Makes forward pass through network.
        :param x: the array of integers, where each item is the index of character, the size of
                  array will be the sequence length
        :param update_state: bool, if True updates current state with last state
        :return: the tuple of states and predicted_probabilities
                 states - array of states, size = (sequence length, hidden size)
                 predicted_probabilities - array of predicted probabilities for each character in
                                           vocabulary, size = (sequence length, vocabulary size)
        """
        # one hot encoding of input
        inputs_matrix = one_hot_encode(x, self.vocabulary_size)

        ps, hs = {}, {}  # predicted probabilities and hidden states
        hs[-1] = self.current_state  # setting the current state
        for t in range(len(x)):
            # state at t - 1
            h_t_1 = hs[t - 1]  # dim : (self.hidden_size, 1)

            # state at t
            z_t = np.dot(self.w_hh, h_t_1) + np.dot(self.w_hx, inputs_matrix[t])
            h_t = self.f(z_t)  # dim : (self.hidden_size, 1)

            # prediction from hidden state at t
            y_t = np.dot(self.w_hy, h_t)  # unnormalized log probabilities for next chars
            p_t = softmax(y_t)  # probabilities for next chars,  dim : (self.vocabulary_size, 1)

            # updating hidden state and and predicted_probabilities keepers
            hs[t], ps[t] = h_t, p_t

        if update_state:
            self.current_state = hs[len(x) - 1]  # updating the current state
        return hs, ps

    def calculate_loss(self, x: np.ndarray, labels: np.ndarray, update_state: bool) -> float:
        """
        Calculates cross entropy loss using target characters indexes and network predictions for
        all characters: loss = ∑ -label_{t} * log(predicted_probability_{t})
        """
        _, ps = self.forward(x, update_state)
        return -sum(np.log(ps[i][labels[i], 0]) for i in range(len(labels)))

    # ### Backward pass ###

    def backward(self, x: np.ndarray, labels: np.ndarray, hs: Dict, ps: Dict):
        """
        Makes backward pass through the network.
        Returns the gradients of loss w.r.t. network parameters -  w_hx, w_hh, w_hy.

        :param x: the array of input characters, where each item is the index of character, the
                  size of array will be the sequence length
        :param labels: the array of target characters, where each item is the index of character,
                       the size of array will be the sequence length
        :param hs: the hidden states of network, (the first output of the self.forward method)
        :param ps: network predictions for given inputs,
                   (the second output of the self.forward method)
        :return: gradients of w_hx, w_hh, w_hy
        """
        inputs_matrix = one_hot_encode(x, self.vocabulary_size)
        labels_matrix = one_hot_encode(labels, self.vocabulary_size)

        dw_hx = np.zeros_like(self.w_hx)
        dw_hh = np.zeros_like(self.w_hh)
        dw_hy = np.zeros_like(self.w_hy)

        for t in reversed(range(len(x))):
            # dl / dy = p - label
            dy_t = ps[t] - labels_matrix[t]

            # dl / dw_hy = (dl / dy) * (dy / dw_hy)
            dw_hy += np.dot(dy_t, hs[t].T)

            # dl / dh = (dl / dy) * (dy / dh) = (p - label) * w_hy
            dh_t = np.dot(self.w_hy.T, dy_t)

            # dl / dz_{k} = (dl / dh_{k}) * (dh_{k} / dz_{k}) = dh_{t} * (dh_{k} / dz_{k})
            dz_k = dh_t * self.f_prime(hs[t])

            # dl / dw_hh = ∑ (dl / dz_{k}) * (dz_{k} / dw_hh) for all k from 1 to t
            # dl / dw_hx = ∑ (dl / dz_{k}) * (dz_{k} / dw_hx) for all k from 1 to t
            for k in reversed(range(t + 1)):
                # (dl / dz_{k}) (dz_{k} / dw_hh) = dz_k * h_{k-1}
                dw_hh += np.dot(dz_k, hs[k - 1].T)

                # (dl / dz_{k}) (dz_{k} / dw_h) = dz_k * x_{k}
                dw_hx += np.dot(dz_k, inputs_matrix[k].T)

                # updating dz_k using all previous derivatives (from t to t - k)
                # dl / dz_(k-1) = (dl / dz_{k})(dz_{k} / dh_{k-1}) * (dh_{k-1) / dz_{k-1})
                dz_k = np.dot(self.w_hh.T, dz_k) * self.f_prime(hs[k - 1])

        # clip to mitigate exploding gradients
        for d_param in (dw_hx, dw_hh, dw_hy):
            np.clip(d_param, -5, 5, out=d_param)

        return dw_hx, dw_hh, dw_hy

    def calculate_numeric_gradients(self, x: np.ndarray,
                                    labels: np.ndarray,
                                    epsilon: float) -> Tuple:
        """
        Calculates numeric gradients w.r.t. model all parameters.

        (dL / dtheta) ≈ (L(theta + epsilon) - L(theta - epsilon)) / (2 * epsilon)

        :return: numeric gradients for dw_hx, dw_hh, dw_hy
        """
        self.reset_current_state()
        dw_hx_numeric = np.zeros_like(self.w_hx)
        dw_hh_numeric = np.zeros_like(self.w_hh)
        dw_hy_numeric = np.zeros_like(self.w_hy)

        d_params_numeric = (dw_hx_numeric, dw_hh_numeric, dw_hy_numeric)
        params = (self.w_hx, self.w_hh, self.w_hy)

        # calculating numerical gradients for each parameter
        for d_param, param in zip(d_params_numeric, params):

            # iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

                # keeping the original value so we can reset it later
                original_value = param[ix]

                # estimating numeric gradients

                # x + epsilon
                param[ix] = original_value + epsilon
                loss_plus = self.calculate_loss(x, labels, False)

                # x - epsilon
                param[ix] = original_value - epsilon
                loss_minus = self.calculate_loss(x, labels, False)

                # numeric_gradient = (f(x + delta) - f(x - delta)) / (2 * delta)
                d_param[ix] = (loss_plus - loss_minus) / (2 * epsilon)

                # resetting parameter to original value
                param[ix] = original_value

                it.iternext()

        return d_params_numeric

    def gradient_check(self,
                       x: np.ndarray,
                       labels: np.ndarray,
                       epsilon: float = 1e-3,
                       threshold: float = 1e-6):
        """
        Performs gradient checking for model parameters:

         - computes the analytic gradients using our back-propagation implementation
         - computes the numerical gradients using the two-sided epsilon method
         - computes the relative difference between numerical and analytical gradients
         - checks that the relative difference is less than threshold
         - if the last check is failed, then raises an error

        """
        params = ('w_hx', 'w_hh', 'w_hy')

        # calculating the gradients using backpropagation, aka analytic gradients
        self.reset_current_state()
        hs, ps = self.forward(x, False)
        analytic_gradients = self.backward(x, labels, hs, ps)

        # calculating numerical gradients
        numeric_gradients = self.calculate_numeric_gradients(x, labels, epsilon)

        # gradient check for each parameter
        for p_name, d_analytic, d_numeric in zip(params, analytic_gradients, numeric_gradients):
            print(f"\nPerforming gradient check for parameter {p_name} "
                  f"with size = {np.prod(d_analytic.shape)}.")

            if (not d_analytic.shape == d_numeric.shape or
                    check_relative_difference(d_analytic, d_numeric, threshold)):
                raise ValueError(f'Gradient check for {p_name} is failed.')

            print(f"Gradient check for parameter {p_name} is passed.")

    # ### Gradient descent ###

    def sgd_step(self, x: np.ndarray, labels: np.ndarray, lr: float):
        """
        Performs gradient descent step for model parameters: w_hx, w_hh, w_hy.

        - forward pass: calculating next char probabilities, given previous sequence of chars
        - backward pass: calculating loss and its gradient w.r.t. model params
        - sgd update: update params in the opposite of the gradient direction
        """
        hs, ps = self.forward(x, True)
        dw_hx, dw_hh, dw_hy = self.backward(x, labels, hs, ps)

        # w <-- w - lr * dloss / dw
        self.w_hx -= lr * dw_hx
        self.w_hh -= lr * dw_hh
        self.w_hy -= lr * dw_hy

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
            _, ps = self.forward(np.array([ix]), True)
            ix = np.random.choice(possible_indexes, p=ps[0].ravel())
            sample_indexes.append(possible_indexes[ix])
        return sample_indexes

    # ### Saving and Loading model ###

    def save(self, saving_path: str):
        """Saves model."""
        params = [self.w_hx, self.w_hh, self.w_hy, self.current_state]
        np.save(saving_path, params)
        print(f'Model has been save at the following path: {saving_path}.')

    @staticmethod
    def load(model_path: str):
        """Loads saved model and returns it."""
        w_hx, w_hh, w_hy, current_state = np.load(model_path)
        _, vocabulary_size = w_hx.shape
        hidden_size, hidden_size = w_hh.shape
        model = RNN(vocabulary_size, hidden_size)
        model.w_hx, model.w_hh, model.w_hy, model.current_state = w_hx, w_hh, w_hy, current_state
        return model

    # implementation by Andrej Kharpaty, for performing check

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
        dWxh = np.zeros_like(self.w_hx)
        dWhh = np.zeros_like(self.w_hh)
        dWhy = np.zeros_like(self.w_hy)

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

        # clip to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy]:
            np.clip(dparam, -5, 5, out=dparam)
        return loss, dWxh, dWhh, dWhy, hs[len(inputs) - 1]


if __name__ == '__main__':
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
    inputs = np.array([char_to_ix[x] for x in data[:n]])
    labels = np.array([char_to_ix[x] for x in data[1:n + 1]])

    rnn = RNN(vocab_size, 10)

    # current implementation
    hs, ps = rnn.forward(inputs, False)
    l_1 = rnn.calculate_loss(inputs, labels, False)
    dw_hx_1, dw_hh_1, dw_hy_1 = rnn.backward(inputs, labels, hs, ps)

    # Karpathy implementation
    l_2, dw_hx_2, dw_hh_2, dw_hy_2, _ = rnn.lossFun(inputs, labels, np.zeros((10, 1)))

    print()
    print('Checking current implementation with Karpathy implementation.')

    print()
    print('loss_1={:.5f}'.format(l_1))
    print('loss_2={:.5f}'.format(l_2))
    assert l_1 == l_2
    print('loss check is passed')

    print()
    assert np.allclose(dw_hx_1, dw_hx_2, atol=1e-10)
    print('dWxh check is passed')

    print()
    assert np.allclose(dw_hh_1, dw_hh_2, atol=1e-10)
    print('dWhh check is passed')

    print()
    assert np.array_equal(dw_hy_1, dw_hy_2)
    print('dWhy check is passed')

    rnn.reset_current_state()
    rnn.gradient_check(np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5]),
                       epsilon=1e-5, threshold=1e-3)
