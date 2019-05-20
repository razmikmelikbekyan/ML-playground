from typing import Tuple, Dict, List

import numpy as np

from utils import softmax, sigmoid, tanh, relu


class RNN:
    """
    This class represents simple Recurrent Neural Network implementation for  for
    character-level language model. In this model the purpose of the network is correctly
    predicting the next character, given the previous one.
    """

    activations = {
        'tanh': tanh,
        'sigmoid': sigmoid,
        'relu': relu
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

        # setting the current state
        self.current_state = np.zeros((self.hidden_size, 1))

    def _one_hot_encode(self, x: np.ndarray) -> np.ndarray:
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

    def reset_current_state(self):
        """Resets current state to zeros."""
        self.current_state = np.zeros((self.hidden_size, 1))

    # ### Forward pass ###

    def forward(self, x: np.ndarray) -> Tuple[Dict, Dict]:
        """
        The basic forward pass:

        z_{t} = w_hh * h_{t-1} + w_hx * x_{t}
        h_{t} = f(z_{t})
        y_{t} = w_hy * h_{t}
        p_{t} = softmax(y_{t})

        Makes forward pass through network.
        :param x: the array of integers, where each item is the index of character, the size of
                  array will be the sequence length
        :return: the tuple of states and predicted_probabilities
                 states - array of states, size = (sequence length, hidden size)
                 predicted_probabilities - array of predicted probabilities for each character in
                                           vocabulary, size = (sequence length, vocabulary size)
        """
        # one hot encoding of input
        inputs_matrix = self._one_hot_encode(x)

        # activation function
        f = self.activations[self.non_linearity]

        ps, hs = {}, {}  # predicted probabilities and hidden states
        hs[-1] = self.current_state  # setting the current state
        for t in range(len(x)):
            # state at t - 1
            h_t_1 = hs[t - 1]  # dim : (self.hidden_size, 1)

            # state at t
            z_t = np.dot(self.w_hh, h_t_1) + np.dot(self.w_hx, inputs_matrix[t])
            h_t = f(z_t)  # dim : (self.hidden_size, 1)

            # prediction from hidden state at t
            y_t = np.dot(self.w_hy, h_t)  # unnormalized log probabilities for next chars
            p_t = softmax(y_t)  # probabilities for next chars,  dim : (self.vocabulary_size, 1)

            # updating hidden state and and predicted_probabilities keepers
            hs[t], ps[t] = h_t, p_t

        self.current_state = hs[len(x) - 1]  # updating the current state
        return hs, ps

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes prediction based on forward pass. It returns the array of integers,
        where each item is the index of predicted character.
        """
        _, ps = self.forward(x)
        return np.argmax(ps, axis=1)

    def calculate_loss(self, x: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculates cross entropy loss using target characters indexes and network predictions for
        all characters: loss = ∑ -label_{t} * log(predicted_probability_{t})
        """
        _, ps = self.forward(x)
        return -sum(np.log(ps[i][labels[i], 0]) for i in range(len(labels)))

    # ### Backward pass ###

    def _calculate_h_grad_wrt_z(self, h_t: np.ndarray) -> np.ndarray:
        """
        Calculates state gradient w.r.t. z, dh / dz.
        Note that h_{t} = tanh(z_{t}) or h_{t} = sigmoid(z_{t}) or h_{t} = relu(z_{t})"""

        if self.non_linearity == 'tanh':
            # h = tanh(z)
            # dh / dz = (1 - tanh^2(z)) = 1 - h^2
            return 1 - h_t ** 2
        elif self.non_linearity == 'sigmoid':
            # h = sigmoid(z)
            # dh / dz = sigmoid(z)(1 - sigmoid(z)) = h * (1 - h)
            return h_t * (1 - h_t)
        else:
            # h = relu(z) = (z > 0) * z
            # dh / dz = (z > 0) * 1, so if we can just replace all non zero elements in h with 1
            y = h_t.copy()
            y[y != 0] = 1
            return y

    def _calculate_loss_grad_wrt_z(self, dh_t: np.ndarray, hs: Dict, t: int):
        """
        Calculates dloss_{t} / dz_{k} for all k from 1 to t.

        The basic forward pass and loss are the following:

        z_{t} = w_hh * h_{t-1} + w_hx * x_{t}
        h_{t} = f(z_{t})
        y_{t} = w_hy * h_{t}
        p_{t} = softmax(y_{t})
        loss_{t} = -labels_{t} * log(p_{t})

        If we change w_hh or w_hx, then it will affect on all z_{k} for all k from 1 to t. So in
        order to calculate dloss_{t} / dw_hh we will need dloss_{t} / dz_{k} for all k from 1 to t
        gradients. So this function calculates the gradient of loss at time t w.r.t. all z_{k}
        for all k from 1 to t.

        :param dh_t: dloss_{t} / dh_{t}, aka loss gradient w.r.t state for the same time index t
        :param hs: array of states
        :param t: time index
        :return: list of arrays, where each array is the calculated gradient:
                 [dloss_{t} / dz_{t}, dloss_{t} / dz_{t - 1}, ..., dloss_{t} / dz_{1}]
        """

        # dl / dz = (dl / dh) * (dh / dz)
        dz_t = dh_t * self._calculate_h_grad_wrt_z(hs[t])

        grads = [dz_t]
        for i in reversed(range(t)):
            # dl / dz_(t-1) = (dl / dz_{t})(dz_{t} / dh_{t-1}) * (dh_{t-1) / dz_{t-1})
            dz_i = np.dot(self.w_hh.T, grads[-1]) * self._calculate_h_grad_wrt_z(hs[i])
            grads.append(dz_i)
        return grads

    def backward(self, x: np.ndarray, labels: np.ndarray, hs: Dict, ps: Dict):
        """
        Makes backward pass through the network. Returns the gradients of loss w.r.t. network
        parameters -  w_hx, w_hh, w_hy.

        :param x: the array of input characters, where each item is the index of character, the
                  size of array will be the sequence length
        :param labels: the array of target characters, where each item is the index of character,
                       the size of array will be the sequence length
        :param hs: the hidden states of network, (the first output of the self.forward method)
        :param ps: network predictions for given inputs,
                   (the second output of the self.forward method)
        :return: gradients of w_hx, w_hh, w_hy
        """
        inputs_matrix = self._one_hot_encode(x)
        labels_matrix = self._one_hot_encode(labels)

        dw_hx = np.zeros_like(self.w_hx)
        dw_hh = np.zeros_like(self.w_hh)
        dw_hy = np.zeros_like(self.w_hy)

        for t in reversed(range(len(x))):
            # dl / dy = p - label
            dy_t = ps[t] - labels_matrix[t]

            # dl / dh = (dl / dy) (dy / dh) = (p - label)w_hy
            dh_t = np.dot(self.w_hy.T, dy_t)

            # dl / dz_{k} for all k from 1 to t
            dz_t_1 = self._calculate_loss_grad_wrt_z(dh_t, hs, t)

            # dl / dw_hy = (dl / dy) * (dl * dw_hy)
            dw_hy += np.dot(dy_t, hs[t].T)

            # dl / dw_hh = ∑ (dl / dz_{k}) * (dz_{k} / dw_hy) for all k from 1 to t
            # (dz_{k} / dw_hy) = h_{t-1}
            dw_hh += sum(
                np.dot(dz_i, hs[i - 1].T)
                for dz_i, i in zip(dz_t_1, reversed(range(t + 1)))
            )

            # dl / dw_hx = ∑ (dl / dz_{k}) * (dz_{k} / dw_hx) for all k from 1 to t
            # (dz_{k} / dw_hx) = x_{t}
            dw_hx += sum(
                np.dot(dz_i, inputs_matrix[i].T)
                for dz_i, i in zip(dz_t_1, reversed(range(t + 1)))
            )

        # clip to mitigate exploding gradients
        for d_param in (dw_hx, dw_hh, dw_hy):
            np.clip(d_param, -5, 5, out=d_param)

        return dw_hx, dw_hh, dw_hy

    def gradient_check(self,
                       x: np.ndarray,
                       labels: np.ndarray,
                       delta: float = 0.001,
                       rel_error: float = 0.01,
                       debug: bool = True):
        """
        Whenever you implement backpropagation it is good idea to also implement gradient checking,
        which is a way of verifying that your implementation is correct. The idea behind gradient
        checking is that derivative of a parameter is equal to the slope at the point, which we can
        approximate by slightly changing the parameter and then dividing by the change:

        (dL / dtheta) ≈ (L(theta + delta) - L(theta - delta)) / (2 * delta)

        So we will change each parameter 'w_hx', 'w_hh', 'w_hy' by some small ±delta, calculate
        the loss difference, divide it 2 * delta and compare with analytic gradient. Their
        relative difference should be very small.

        :return: True if numerical gradient and analytic gradient are almost equal, else False
        """

        def check_relative_difference(a, b, threshold):
            """Returns True if (|a - b| / (|a| + |b|)) > threshold else False."""
            return np.abs(a - b) > threshold * (np.abs(a) + np.abs(b))

            # calculating the gradients using backpropagation, aka analytic gradients

        hs, ps = self.forward(x)
        analytic_gradients = self.backward(x, labels, hs, ps)
        self.reset_current_state()

        # parameters we want to check
        model_parameters = ('w_hx', 'w_hh', 'w_hy')

        # gradient check for each parameter
        for p_idx, p_name in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = getattr(self, p_name)

            if debug:
                print()
                print(f"Performing gradient check for parameter {p_name} "
                      f"with size = {np.prod(parameter.shape)}.")

            # iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

                # keeping the original value so we can reset it later
                original_value = parameter[ix]

                # estimating numeric gradients
                # numeric_grad = (f(x + delta) - f(x - delta)) / (2 * delta)

                # x + delta
                parameter[ix] = original_value + delta
                loss_plus = self.calculate_loss(x, labels)
                self.reset_current_state()

                # x - delta
                parameter[ix] = original_value - delta
                loss_minus = self.calculate_loss(x, labels)
                self.reset_current_state()

                numeric_gradient = (loss_plus - loss_minus) / (2 * delta)

                # resetting parameter to original value
                parameter[ix] = original_value

                # the analytic_gradient for this parameter calculated using backpropagation
                analytic_gradient = analytic_gradients[p_idx][ix]

                # if the error is to large fail the gradient check
                if check_relative_difference(analytic_gradient, numeric_gradient, rel_error):
                    if debug:
                        print(f"Gradient Check ERROR: parameter = {p_name} ix = {ix}")
                        print(f"+ delta Loss: {loss_plus}")
                        print(f"- delta Loss: {loss_minus}")
                        print(f"Numeric gradient: {numeric_gradient}")
                        print(f"Analytic gradient: {analytic_gradient}")
                    return False
                it.iternext()

            if debug:
                print(f"Gradient check for parameter {p_name} passed.")

        return True

    # implementation by Andrej Kharpaty
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

    # ### Gradient descent ###

    def sgd_step(self,
                 dw_hx: np.ndarray,
                 dw_hh: np.ndarray,
                 dw_hy: np.ndarray,
                 lr: float):
        """
        Performs gradient descent step, for w_hx, w_hh, w_hy.

        w_new = w_old - lr * dloss / dw_old
        """
        self.w_hx -= lr * dw_hx
        self.w_hh -= lr * dw_hh
        self.w_hy -= lr * dw_hy

    def train(self, x: np.ndarray, labels: np.ndarray, lr: float):
        """
        Performs training of model:
         - forward pass
         - backward pass
         - sgd update
        """
        hs, ps = self.forward(x)
        dw_hx, dw_hh, dw_hy = self.backward(x, labels, hs, ps)
        self.sgd_step(dw_hx, dw_hh, dw_hy, lr)

    def sample(self, seed_ix: int, n: int) -> List[int]:
        """
        Sample a sequence of integers from the model.
        :param seed_ix: seed letter for first time step
        :param n: number of samples
        :return: list of indexes
        """
        assert isinstance(seed_ix, int) and self.vocabulary_size > seed_ix >= 0

        possible_indexes = np.arange(self.vocabulary_size)

        sample_indexes = []
        ix = seed_ix
        for t in range(n):
            _, ps = self.forward(np.array([ix]))
            ix = np.random.choice(possible_indexes, p=ps[0].ravel())
            sample_indexes.append(ix)
        return sample_indexes

if __name__ == '__main__':
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

    rnn.reset_current_state()
    l_1 = rnn.calculate_loss(inputs, labels)
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

    rnn.current_state = np.zeros_like(rnn.current_state)
    rnn.gradient_check(np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5]))

    # http://willwolf.io/2016/10/18/recurrent-neural-network-gradients-and-lessons-learned-therein/
    # https://github.com/dennybritz/rnn-tutorial-rnnlm
    # https://github.com/dennybritz/nn-theano/blob/master/nn-theano.ipynb
    # https://github.com/sar-gupta/rnn-from-scratch/blob/master/rnn.py
    # http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    # http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
    # https://towardsdatascience.com/character-level-language-model-1439f5dd87fe
    # https://peterroelants.github.io/posts/rnn-implementation-part01/
    
    print(rnn.sample(10, 5))
