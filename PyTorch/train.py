from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from nn_from_scratch import NeuralNetwork as ManualGradNN
from nn_autograd import NeuralNetwork as AutoGradNN
from utils import plot_loss

nns = {
    '1': ManualGradNN,
    '2': AutoGradNN,
}

parser = ArgumentParser()
parser.add_argument('nn_type', choices=list(nns))

if __name__ == '__main__':
    args = parser.parse_args()

    # Data preparation
    data = load_breast_cancer()
    xs = data['data']
    labels = data['target']
    print(f'Data shape: {xs.shape}')
    print(f'Targets shape: {labels.shape}')

    xs = xs.reshape(xs.shape[0], xs.shape[1], 1)
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    labels = one_hot_encoder.fit_transform(labels.reshape(-1, 1))
    labels = labels.reshape(labels.shape[0], labels.shape[1], 1)

    xs_train, xs_test, labels_train, labels_test = train_test_split(xs, labels, test_size=0.25)
    print(xs_train.shape, labels_train.shape, xs_test.shape, labels_test.shape)

    xs_train = torch.from_numpy(xs_train)
    xs_test = torch.from_numpy(xs_test)
    labels_train = torch.from_numpy(labels_train)
    labels_test = torch.from_numpy(labels_test)

    # Training
    nn = nns[args.nn_type](30, 20, 2)

    epochs = 50

    train_loss, test_loss = [], []
    for epoch in range(epochs):
        train_loss.append(np.mean([nn.loss(x, l).item() for x, l in zip(xs_train, labels_train)]))
        test_loss.append(np.mean([nn.loss(x, l).item() for x, l in zip(xs_test, labels_test)]))
        for x, y in zip(xs_train, labels_train):
            nn.sgd_step(x, y, 1e-4)

        if epoch % 10 == 0:
            print('\nepoch: {}, train loss: {:.5f}, test loss: {:.5f}'.format(
                epoch, train_loss[-1], test_loss[-1]))
    plot_loss(train_loss, test_loss)

    # Testing
    test_predictions = [torch.exp(nn.forward(x)) for x in xs_test]

    test_predictions = list(map(torch.argmax, test_predictions))
    test_labels = list(map(torch.argmax, labels_test))

    print('\n**************** Classification report ****************')
    print(classification_report(test_labels, test_predictions))
    print('Accuracy: {:.2f}'.format(accuracy_score(test_labels, test_predictions)))
