import numpy as np
import solution
from utils import Opts, read_data, split_train, Phase


np.seterr(over='raise')


class LinearLayerForward:
    def __init__(self):
        self.batch_size = 3
        self.num_features = 5

    def __iter__(self):
        for i in range(10):
            weights = np.random.normal(size=(self.num_features, ))
            xs = np.random.normal(size=(self.batch_size, self.num_features))
            yield weights, xs

    def __call__(self, weights, xs):
        forward = solution.LinearLayerForward()

        alpha_xs = np.random.normal(size=(self.batch_size, self.num_features))
        beta_xs = np.random.normal(size=(self.batch_size, self.num_features))
        kalpha = np.random.normal() * 50
        kbeta = np.random.normal() * 50
        assert np.all(np.absolute(forward(weights, kalpha * alpha_xs + kbeta * beta_xs) - (kalpha * forward(weights, alpha_xs) + kbeta * forward(weights, beta_xs))) < 1e-10), 'The linear layer should be linear with the input data.'

        alpha_weights = np.random.normal(size=(self.num_features, ))
        beta_weights = np.random.normal(size=(self.num_features, ))
        kalpha = np.random.normal() * 50
        kbeta = np.random.normal() * 50
        assert np.all(np.absolute(forward(kalpha * alpha_weights + kbeta * beta_weights, xs) - (kalpha * forward(alpha_weights, xs) + kbeta * forward(beta_weights, xs))) < 1e-10), 'The linear layer should be linear with the weights.'


linear_layer_forward = LinearLayerForward()
for i, p in enumerate(linear_layer_forward, start=1):
    try:
        linear_layer_forward(*p)
    except:
        print("Checkpoint {i:d} for LinearLayerForward fails.".format(i=i))
        raise
print("LinearLayerForward test succeeds.")


class LinearLayerBackward:
    def __init__(self):
        self.batch_size = 100
        self.num_features = 10
        self.delta = 1e-5

    def __iter__(self):
        for i in range(10):
            weights = np.random.normal(size=(self.num_features, ))
            xs = np.random.normal(size=(self.batch_size, self.num_features))
            dlogits = np.random.normal(size=(self.batch_size, ))
            yield weights, xs, dlogits

    def __call__(self, weights, xs, dlogits):
        forward = solution.LinearLayerForward()
        backward = solution.LinearLayerBackward()

        ctx = dict()
        logits = forward(weights, xs, ctx=ctx)
        dw = backward(ctx, dlogits)

        for d in range(self.num_features):
            real_dw = np.zeros_like(weights)
            real_dw[d] = self.delta

            moved_weights = weights + real_dw
            moved_logits = forward(moved_weights, xs)

            assert np.all(np.absolute((moved_logits - logits).dot(dlogits) / self.delta - dw[d]) < 1e-5), 'Value mismatch in dimension {}.'.format(d)


linear_layer_backward = LinearLayerBackward()
for i, p in enumerate(linear_layer_backward, start=1):
    try:
        linear_layer_backward(*p)
    except:
        print("Checkpoint {i:d} for LinearLayerBackward fails.".format(i=i))
        raise
print("LinearLayerBackward test succeeds.")


class LinearLayerUpdate:
    def __init__(self):
        self.batch_size = 20
        self.num_features = 100

    def __iter__(self):
        for i in range(10):
            weights = np.random.normal(size=(self.num_features, ))
            dw = np.random.normal(size=(self.num_features, ))
            learning_rate = np.random.poisson() / 10 + 0.1
            yield weights, dw, learning_rate

    def __call__(self, weights, dw, learning_rate):
        update = solution.LinearLayerUpdate()

        delta_weights = np.random.normal(size=weights.shape)
        alpha_new_weights = update(weights, dw, learning_rate=learning_rate)
        beta_new_weights = update(weights + delta_weights, dw, learning_rate=learning_rate)
        assert np.all(np.absolute((beta_new_weights - alpha_new_weights) - delta_weights) < 1e-8), 'The update rule should be linear with the original weights.'

        delta_dw = np.random.normal(size=dw.shape)
        beta_new_weights = update(weights, dw + delta_dw, learning_rate=learning_rate)
        assert np.all(np.absolute((beta_new_weights - alpha_new_weights) + delta_dw * learning_rate) < 1e-8)


linear_layer_update = LinearLayerUpdate()
for i, p in enumerate(linear_layer_update, start=1):
    try:
        linear_layer_update(*p)
    except:
        print("Checkpoint {i:d} for LinearLayerUpdate fails.".format(i=i))
        raise
print("LinearLayerUpdate test succeeds.")


class SigmoidCrossEntropyForward:
    def __init__(self):
        self.num_checkpoints = 11
        self.midpoint = (self.num_checkpoints - 1) // 2
        self.batch_size = 10

        self.left_inf = -10000
        self.right_inf = 10000

    def positive_monotonically_decreasing(self, forward):
        checkpoints = np.linspace(self.left_inf, self.right_inf, num=self.num_checkpoints, endpoint=True)
        ys = np.ones(shape=(self.batch_size, ), dtype=np.bool)

        average_loss_list = np.array([forward(np.ones(shape=(self.batch_size, )) * ckpt, ys) for ckpt in checkpoints])
        for average_loss in average_loss_list:
            assert average_loss >= 0, 'The sigma cross entropy values should be non-negative.'
        for left, right in zip(average_loss_list[:-1], average_loss_list[1:]):
            assert right <= left, 'The sigma cross entropy function should be monotonically decreasing when labels are positive.'

        assert average_loss_list[0] >= - self.left_inf, 'The sigma cross entropy function should be asymptotically approaching when the logit goes to negative infinity, with labels being positive.'
        assert average_loss_list[0] <= - self.left_inf + 1e-10, 'The sigma cross entropy function should be asymptotically approaching when the logit goes to negative infinity, with labels being positive.'

        assert np.all(np.absolute(average_loss_list[self.midpoint] + np.log(0.5)) < 1e-10), 'The sigma cross entropy function should be -log(0.5) when logits are zeros.'

        assert average_loss_list[-1] <= 1e-10, 'The sigma cross entropy function should be asymptotically approaching zero when the logit goes to positive infinity, with labels being positive.'

    def negative_monotonically_increasing(self, forward):
        checkpoints = np.linspace(self.left_inf, self.right_inf, num=self.num_checkpoints, endpoint=True)
        ys = np.zeros(shape=(self.batch_size, ), dtype=np.bool)

        average_loss_list = np.array([forward(np.ones(shape=(self.batch_size, )) * ckpt, ys) for ckpt in checkpoints])
        for average_loss in average_loss_list:
            assert average_loss >= 0, 'The sigma cross entropy values should be non-negative.'
        for left, right in zip(average_loss_list[:-1], average_loss_list[1:]):
            assert right >= left, 'The sigma cross entropy function should be monotonically increasing when labels are negative.'

        assert average_loss_list[-1] >= - self.left_inf, 'The sigma cross entropy function should be asymptotically approaching when the logit goes to positive infinity, with labels being negative.'
        assert average_loss_list[-1] <= - self.left_inf + 1e-10, 'The sigma cross entropy function should be asymptotically approaching when the logit goes to positive infinity, with labels being negative.'

        assert np.all(np.absolute(average_loss_list[self.midpoint] + np.log(0.5)) < 1e-10), 'The sigma cross entropy function should be -log(0.5) when logits are zeros.'

        assert average_loss_list[0] <= 1e-10, 'The sigma cross entropy function should be asymptotically approaching zero when the logit goes to negative infinity, with labels being negative.'

    def __call__(self):
        forward = solution.SigmoidCrossEntropyForward()
        self.positive_monotonically_decreasing(forward)
        self.negative_monotonically_increasing(forward)


sigmoid_cross_entropy_forward = SigmoidCrossEntropyForward()
try:
    sigmoid_cross_entropy_forward()
except:
    print("Test for SigmoidCrossEntropyForward fails.")
    raise
print("SigmoidCrossEntropyForward test succeeds.")


class SigmoidCrossEntropyBackward:
    def __init__(self):
        self.batch_size = 100
        self.left_inf = - 10000
        self.right_inf = 10000
        self.delta = 1e-5

    def __iter__(self):
        for i in range(10):
            logits = np.random.normal(size=(self.batch_size, ))
            ys = np.random.binomial(1, 0.5, size=(self.batch_size, ))
            yield logits, ys
        yield np.ones(shape=(self.batch_size, )) * self.left_inf, np.ones(shape=(self.batch_size, ))
        yield np.ones(shape=(self.batch_size, )) * self.right_inf, np.zeros(shape=(self.batch_size, ))

    def __call__(self, logits, ys):
        forward = solution.SigmoidCrossEntropyForward()
        backward = solution.SigmoidCrossEntropyBackward()

        ctx = dict()
        average_loss = forward(logits, ys, ctx=ctx)
        dlogits = backward(ctx, 1.0)

        for d in range(self.batch_size):
            real_dlogits = np.zeros_like(logits)
            real_dlogits[d] = self.delta

            moved_logits = logits + real_dlogits
            moved_average_loss = forward(moved_logits, ys)

            assert np.all(np.absolute((moved_average_loss - average_loss) / self.delta - dlogits[d]) < 1e-5), 'Value mismatch in dimension {}.'.format(d)


sigmoid_cross_entropy_backward = SigmoidCrossEntropyBackward()
for i, p in enumerate(sigmoid_cross_entropy_backward, start=1):
    try:
        sigmoid_cross_entropy_backward(*p)
    except:
        print("Checkpoint {i:d} for SigmoidCrossEntropyBackward fails.".format(i=i))
        raise
print("SigmoidCrossEntropyBackward test succeeds.")


class Prediction:
    def __init__(self):
        pass

    def __iter__(self):
        for i in range(10):
            yield np.random.normal(size=(100, )),

    def __call__(self, logits):
        predict = solution.Prediction()
        predictions = predict(logits)
        assert isinstance(predictions, np.ndarray), 'Predictions should form a numpy array.'
        assert predictions.dtype == np.bool, 'Predictions should be booleans.'


prediction = Prediction()
for i, p in enumerate(prediction, start=1):
    try:
        prediction(*p)
    except:
        print("Checkpoint {i:d} for Prediction fails.".format(i=i))
        raise
print("Prediction test succeeds.")


class EvaluationPhase(Phase):
    """
    Train the model on the training set, and apply the model on the validation set.
    """
    def __init__(self, opts):
        original_train_data = read_data(opts.train)
        train_data, val_data = split_train(original_train_data)
        super().__init__(original_train_data, train_data, val_data, opts.threshold, init_weight_scale=opts.init_weight_scale)

    def __call__(self):
        error_count, error_percentage = self.apply(self.val_xdata, self.val_ydata)
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))

        training_error_count = None

        for i in range(opts.num_epochs):
            average_loss = self.model.fit(self.train_xdata, self.train_ydata, batch_size=opts.batch_size, learning_rate=opts.learning_rate)
            error_count, error_percentage = self.apply(self.train_xdata, self.train_ydata)
            print("Epoch: {:d}, average loss = {:>8.4f}, trianing error, # = {:>4d}, % = {:>8.4f}%.".format(i + 1, average_loss, error_count, error_percentage))
            training_error_count = error_count

            error_count, error_percentage = self.apply(self.val_xdata, self.val_ydata)
            print("Epoch: {:d}, validation error, # = {:>4d}, % = {:>8.4f}%.".format(i + 1, error_count, error_percentage))

        assert training_error_count == 0, 'The training error should be zero.'


opts = Opts()
phrase = EvaluationPhase(opts)
try:
    phrase()
except:
    print("Checkpoint for training fails.")
    raise
print("Model test succeeds.")
