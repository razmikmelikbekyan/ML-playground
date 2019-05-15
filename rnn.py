import numpy as np


def softmax(x: np.ndarray):
    return np.exp(x) / np.exp(x).sum()


def sigmoid(x: np.ndarray):
    return 1. / (1 + np.exp(-x))


class RNN:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_size: int,
                 non_linearity: str = 'tanh'):
        """

        :param input_size:
        :param output_size:
        :param hidden_size:
        """
        if not non_linearity in ('tanh', 'sigmoid'):
            raise ValueError('Non linearity must be sigmoid or tanh.')

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.non_linearity = non_linearity

        # randomly initializing weights

        self.w_hx = np.random.uniform(
            -np.sqrt(1. / input_size), np.sqrt(1. / input_size),
            (hidden_size, input_size)
        )

        self.w_hh = np.random.uniform(
            -np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size),
            (hidden_size, hidden_size)
        )

        self.w_hy = np.random.uniform(
            -np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size),
            (output_size, hidden_size)
        )

    def forward(self, x: np.ndarray):
        """

        :param x:
        :return:
        """
        # sequence length
        n = x.shape[0]

        # non linear function
        f = np.tanh if self.non_linearity == 'tanh' else sigmoid

        states = np.zeros((n, self.hidden_size))
        outputs = np.zeros((n, self.output_size))
        for t in range(n):
            # state at t - 1
            h_t_1 = np.zeros(self.hidden_size) if t == 0 else states[t - 1]

            # TODO: make more smart
            # one hot encoding input
            input_x = np.zeros(self.input_size)
            input_x[t] = x[t]

            # state at t
            z_t = np.dot(self.w_hh, h_t_1) + np.dot(self.w_hx, input_x)
            h_t = f(z_t)

            # prediction from hidden state at t
            y_t = np.dot(self.w_hy, h_t)
            p_t = softmax(y_t)

            # updating hidden state and and outputs keepers
            states[t] = h_t
            outputs[t] = p_t

        return states, outputs

    def backward(self, x: np.ndarray, labels: np.ndarray):
        """

        :param x:
        :param labels:
        :return:
        """

        pass


n = 25
data = open('input.txt', 'r').read()  # should be simple plain text file

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

inputs = [char_to_ix[x] for x in data[:n]]
labels = [char_to_ix[x] for x in data[1:n + 1]]

rnn = RNN(len(char_to_ix), len(char_to_ix), 10)
a, b = rnn.forward(np.array(inputs))
print(a.shape, b.sum(axis=1))