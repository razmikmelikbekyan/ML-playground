from argparse import ArgumentParser
from itertools import tee

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score

from dataset import get_mnist_data
from nn_auto_grad import NeuralNetwork as AutoGradNN
from nn_manual_grad import NeuralNetwork as ManualGradNN
from nn_nn_module import NeuralNetwork as NNModuleNN
from utils import plot_loss

nns = {
    '1': ManualGradNN,
    '2': AutoGradNN,
    '3': NNModuleNN,
}

parser = ArgumentParser()
parser.add_argument('nn_type', choices=list(nns))

if __name__ == '__main__':
    train_data, test_data = get_mnist_data()

    args = parser.parse_args()
    nn = nns[args.nn_type](784, 20, 10, torch.float32)
    epochs = 100

    # Training
    train_loss, test_loss = [], []
    for epoch in range(epochs):
        train_data, train_data_copy = tee(train_data)
        test_data, test_data_copy = tee(test_data)

        train_loss.append(np.mean([nn.loss(x, l).item() for x, l in train_data_copy]))
        test_loss.append(np.mean([nn.loss(x, l).item() for x, l in test_data_copy]))

        # if epoch % 10 == 0:
        print('\nepoch: {}, train loss: {:.5f}, test loss: {:.5f}'.format(
            epoch, train_loss[-1], test_loss[-1]))

        train_data, train_data_copy = tee(train_data)
        for x, l in train_data_copy:
            nn.sgd_step(x, l, 1e-4)

    plot_loss(train_loss, test_loss)

    # Testing
    test_predictions_labels = [
        (np.argmax(torch.exp(nn.forward(x).detach()), axis=1), np.argmax(l, axis=1))
        for x, l in test_data
    ]
    predictions, labels = zip(*test_predictions_labels)
    predictions, labels = torch.cat(predictions), torch.cat(labels)

    print('\n**************** Classification report ****************')
    print(classification_report(labels, predictions))
    print('Accuracy: {:.2f}'.format(accuracy_score(labels, predictions)))
