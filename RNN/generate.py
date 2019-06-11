import argparse

from data_utils import get_support_data
from numpy_nn.rnn import RNN as NumpyRNN
from pytorch_nn.rnn import RNN as PytorchRNN

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='The saved model path.')
parser.add_argument('--package', type=str, choices=['numpy', 'pytorch'], default='numpy')
parser.add_argument('--data', type=str, required=True, help='The data set path.')
parser.add_argument('--seed-ix', type=int, default=5,
                    help='The sedd for starting generation of text.')
parser.add_argument('--n', type=int, default=1000,
                    help='The number of characters to be generated.')


def main(args):
    model_class = NumpyRNN if args.package == 'numpy' else PytorchRNN
    model = model_class.load(args.model)
    _, _, ix_to_char = get_support_data(args.data)

    generated_indexes = model.generate(args.seed_ix, args.n)

    print('\nGenerated sample from model:')
    print(''.join(ix_to_char[ix] for ix in generated_indexes))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
