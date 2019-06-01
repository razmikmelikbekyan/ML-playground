import argparse
import json
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt

from data_utils import get_support_data, get_inputs_targets
from numpy_nn.rnn import RNN as NumpyRNN
from pytorch_nn.rnn import RNN as PytorchRNN


def plot_loss(losses: List):
    """PLots losses."""
    _, epoch, loss = zip(*losses)
    plt.figure(figsize=(15, 15))
    plt.title('Loss over epochs')
    plt.plot(epoch, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()


def train_with_sgd(params: Dict,
                   data_path: str,
                   lr: float = 0.005,
                   full_sequences: bool = False,
                   epochs: int = 100,
                   evaluate_loss_after: int = 5,
                   package: str = 'numpy') -> Tuple:
    """
    Initializes RNN model with given params, rains given model using given data set.

    :param params: the RNN model params dict,
                         {'hidden_size' : int, 'non_linearity': str, 'sequence_length': int}
    :param data_path: the data set path, it should be a txt file
    :param lr: initial learning rate for SGD
    :param full_sequences: if True generates full sequences from data
    :param epochs: number of times to iterate through the complete data set
    :param evaluate_loss_after: evaluate the loss after this many epochs
    :param package: which package implementation to use, numpy or pytorch
    :return: trained model and list of losses
    """

    vocabulary_size, char_to_ix, ix_to_char = get_support_data(data_path)
    print('Vocabulary size={}'.format(vocabulary_size))
    hidden_size, non_linearity, sequence_length = (params['hidden_size'],
                                                   params['non_linearity'],
                                                   params['sequence_length'])

    model_class = NumpyRNN if package == 'numpy' else PytorchRNN
    model = model_class(vocabulary_size, hidden_size, non_linearity=non_linearity)

    # keep track of the losses so we can plot them later
    losses = []

    num_examples_seen = 0
    for epoch in range(epochs):
        if epoch % evaluate_loss_after == 0:
            epoch_loss = sum([
                model.calculate_loss(x, y, True)
                for x, y in get_inputs_targets(data_path, sequence_length,
                                               char_to_ix, full_sequences, package)
            ])
            losses.append((num_examples_seen, epoch, epoch_loss))

            print("Loss after num_examples_seen={} epoch={}: {:.2f}".format(
                num_examples_seen, epoch, epoch_loss))

            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][-1] > losses[-2][-1]:
                lr *= 0.5
                print(f"Setting learning rate to {lr}.")

        # performing training of model for current epoch
        model.reset_current_state()
        for x, y in get_inputs_targets(data_path, sequence_length,
                                       char_to_ix, full_sequences, package):
            model.sgd_step(x, y, lr)
            num_examples_seen += 1

    print('\nGenerated sample from model:')
    print(''.join(ix_to_char[ix] for ix in model.generate(20, 200)))
    return model, losses


parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, required=True, help='The model params json path.')
parser.add_argument('--data', type=str, required=True, help='The data set path.')
parser.add_argument('--lr', type=float, default=0.05, help='The learning rate for SGD.')
parser.add_argument('--package', type=str, choices=['numpy', 'pytorch'], default='numpy')
parser.add_argument('--full-sequences', action='store_true',
                    help='If True generates full sequences from data.')
parser.add_argument('--epochs', type=int, default=100,
                    help='The number of epochs to train the model.')
parser.add_argument('--loss-evaluation-epochs', type=int, default=5,
                    help='evaluate the loss after this many epochs')
parser.add_argument('--saving-path', type=str, default='data/trained_model.npy',
                    help='Model saving path.')


def main(args):
    with open(args.params, 'r') as f:
        params = json.load(f)

    model, losses, = train_with_sgd(params, args.data,
                                    lr=args.lr,
                                    full_sequences=args.full_sequences,
                                    epochs=args.epochs,
                                    evaluate_loss_after=args.loss_evaluation_epochs,
                                    package=args.package)
    model.save(args.saving_path)

    plot_loss(losses)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
