from typing import IO

from rnn import RNN
from utils import get_inputs_targets


def train_with_sgd(model: RNN,
                   data_set: IO,
                   lr: float = 0.005,
                   epochs: int = 100,
                   evaluate_loss_after: int = 5,
                   sequence_length: int = 25):
    """
    Trains given model using given data set.

    :param data_set:
    :param model: the RNN model instance
    :param x_train: the training data set inputs
    :param y_train: the training data set labels
    :param lr: initial learning rate for SGD
    :param epochs: number of times to iterate through the complete dataset
    :param evaluate_loss_after: evaluate the loss after this many epochs
    :return:
    """
    # We keep track of the losses so we can plot them later
    losses = []

    num_examples_seen = 0
    for epoch in range(epochs):
        # if epoch % evaluate_loss_after == 0:
        #     loss = model.calculate_loss(x_train, y_train)
        #     losses.append((num_examples_seen, loss))
        #     print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time,
        #                                                                 num_examples_seen,
        #                                                                 epoch, loss))
        #     # Adjust the learning rate if loss increases
        #     if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
        #         lr *= 0.5
        #         print(f"Setting learning rate to {lr}.")
        #     sys.stdout.flush()

        # For each training example...
        x_gen, targets_gen = get_inputs_targets(data_set, sequence_length)
        for x, targets in x_gen, targets_gen:
            pass

        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(x_train[i], y_train[i], lr)
            num_examples_seen += 1
