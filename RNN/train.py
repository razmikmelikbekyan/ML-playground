import argparse
import json
from typing import Dict

from rnn import RNN
from utils import get_support_data, get_inputs_targets


def train_with_sgd(params: Dict,
                   data_path: str,
                   lr: float = 0.005,
                   epochs: int = 100,
                   evaluate_loss_after: int = 5):
    """
    Initializes RNN model with given params, rains given model using given data set.
    :param params: the RNN model params dict,
                         {'hidden_size' : int, 'non_linearity': str, 'sequence_length': int}

    :param data_path: the data set path, it should be a txt file
    :param lr: initial learning rate for SGD
    :param epochs: number of times to iterate through the complete data set
    :param evaluate_loss_after: evaluate the loss after this many epochs
    :return:
    """

    vocabulary_size, char_to_ix, ix_to_char = get_support_data(data_path)
    hidden_size, non_linearity, sequence_length = (params['hidden_size'],
                                                   params['non_linearity'],
                                                   params['sequence_length'])

    model = RNN(vocabulary_size, params['hidden_size'], non_linearity=params['non_linearity'])

    # keep track of the losses so we can plot them later
    losses = []

    num_examples_seen = 0
    for epoch in range(epochs):
        if epoch % evaluate_loss_after == 0:
            epoch_loss = sum([
                model.calculate_loss(x, y)
                for x, y in get_inputs_targets(data_path, sequence_length, char_to_ix)
            ])
            losses.append((num_examples_seen, epoch_loss))
            print(f"Loss after num_examples_seen={num_examples_seen} epoch={epoch}: {epoch_loss}")

            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                lr *= 0.5
                print(f"Setting learning rate to {lr}.")

        # performing training of model
        model.reset_current_state()
        for x, y in get_inputs_targets(data_path, params['sequence_length'], char_to_ix):
            model.train(x, y, lr)
            num_examples_seen += 1


parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, required=True, help='The model params json path.')
parser.add_argument('--data', type=str, required=True, help='The data set path.')
parser.add_argument('--lr', type=float, default=0.05, help='The learning rate for SGD.')
parser.add_argument('--epochs', type=int, default=100,
                    help='The number of epochs to train the model.')
parser.add_argument('--loss-evaluation-epochs', type=int, default=5,
                    help='evaluate the loss after this many epochs')


def main(args):
    with open(args.params, 'r') as f:
        params = json.load(f)

    train_with_sgd(params, args.data, lr=args.lr, epochs=args.epochs,
                   evaluate_loss_after=args.loss_evaluation_epochs)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
