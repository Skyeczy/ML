import sys
import argparse

import itertools
from collections import Counter
import numpy as np

import solution


class Opts:
    def __init__(self):
        self.train = './percept_data/spam_train.txt'
        self.test = './percept_data/spam_test.txt'
        self.__dict__.update(solution.opts)


def read_data(filename):
    """
    Read the dataset from the file given by name $filename.
    The returned object should be a list of pairs of data, such as
        [
            (True , ['a', 'b', 'c']),
            (False, ['d', 'e', 'f']),
            ...
        ]
    """

    with open(filename, 'r') as f:
        return [(bool(int(y)), x) for y, *x in [line.strip().split() for line in f]]


def split_train(original_train_data):
    """
    Split the original training set into two sets:
        * the training set, and
        * the validation set.
    """

    return original_train_data[:4000], original_train_data[4000:]


def create_wordlist(original_train_data, threshold=26):
    """
    Create a word list from the original training set.
    Only get a word if it appears in at least $threshold emails.
    """

    return [w for w, c in Counter(itertools.chain(*[set(x) for y, x in original_train_data])).items() if c >= threshold]


class Vectorization:
    def __init__(self, wordlist):
        self.wordlist = wordlist

    def vectorize_data_point(self, x):
        x = set(x)
        return np.array([1.0] + [float(word in x) for word in self.wordlist], dtype=np.float)

    def __call__(self, data):
        data = [(y, self.vectorize_data_point(x)) for y, x in data]

        xdata = [x for y, x in data]
        ydata = [y for y, x in data]
        return np.array(ydata, dtype=np.bool), np.array(xdata, dtype=np.float)


class LinearLayer:
    def __init__(self, num_features, init_weight_scale=1.0):
        self.weights = np.random.normal(init_weight_scale, size=(num_features, ))
        self._forward = solution.LinearLayerForward()
        self._backward = solution.LinearLayerBackward()
        self._update = solution.LinearLayerUpdate()

    def forward(self, xs, ctx=None):
        return self._forward(self.weights, xs, ctx=ctx)

    def backward(self, ctx, dlogits):
        return self._backward(ctx, dlogits)

    def update(self, dw, learning_rate=1.0):
        self.weights = self._update(self.weights, dw, learning_rate=learning_rate)


class SigmoidCrossEntropyLoss:
    def __init__(self):
        self._forward = solution.SigmoidCrossEntropyForward()
        self._backward = solution.SigmoidCrossEntropyBackward()

    def forward(self, logits, ys, ctx=None):
        return self._forward(logits, ys, ctx=ctx)

    def backward(self, ctx, dloss):
        return self._backward(ctx, dloss)


class LogisticRegression:
    def __init__(self, num_features, init_weight_scale=1.0):
        self.linear_layer = LinearLayer(num_features, init_weight_scale=init_weight_scale)
        self.sigmoid_cross_entropy_loss = SigmoidCrossEntropyLoss()
        self._predict = solution.Prediction()

    def step(self, xs, ys, learning_rate=None):
        linear_ctx = dict()
        logits = self.linear_layer.forward(xs, ctx=linear_ctx)

        loss_ctx = dict()
        average_loss = self.sigmoid_cross_entropy_loss.forward(logits, ys, ctx=loss_ctx)

        dlogits = self.sigmoid_cross_entropy_loss.backward(loss_ctx, 1.0)
        dw = self.linear_layer.backward(linear_ctx, dlogits)
        self.linear_layer.update(dw, learning_rate=learning_rate)

        return average_loss

    def fit(self, xdata, ydata, batch_size=None, learning_rate=None):
        length = len(xdata)
        indices = np.random.permutation(range(length))

        total_loss = 0.0

        for i in range(0, length, batch_size):
            l, r = i, min(i + batch_size, length)
            ind = indices[l:r]

            xs = xdata[ind]
            ys = ydata[ind]

            average_loss = self.step(xs, ys, learning_rate=learning_rate)
            total_loss += average_loss * (r - l)

        return total_loss / length

    def predict(self, xs):
        """
        Predict whether email $x is a spam or not.
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """

        logits = self.linear_layer.forward(xs)
        predictions = self._predict(logits)

        return predictions


class Phase:
    def __init__(self, original_train_data, train_data, val_data, threshold, init_weight_scale=1.0):
        # Create the word list.
        self.wordlist = create_wordlist(original_train_data, threshold)
        vectorize = Vectorization(self.wordlist)
        self.train_ydata, self.train_xdata = vectorize(train_data)
        self.val_ydata, self.val_xdata = vectorize(val_data)

        num_features = len(self.wordlist) + 1
        self.model = LogisticRegression(num_features, init_weight_scale=init_weight_scale)

    def apply(self, xdata, ydata):
        num_data, *_ = xdata.shape

        error_count = (ydata != self.model.predict(xdata)).sum()
        error_percentage = error_count / num_data * 100

        return error_count, error_percentage
