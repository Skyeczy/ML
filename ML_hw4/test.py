from utils import Opts, read_data, Phase


class TestPhase(Phase):
    """
    Train the model on the original training set, and apply the model on the test set.
    """
    def __init__(self, opts):
        train_data = read_data(opts.train)
        test_data = read_data(opts.test)
        super().__init__(train_data, train_data, test_data, opts.threshold, init_weight_scale=opts.init_weight_scale)

    def __call__(self):
        for i in range(opts.num_epochs):
            average_loss = self.model.fit(self.train_xdata, self.train_ydata, batch_size=opts.batch_size, learning_rate=opts.learning_rate)
            error_count, error_percentage = self.apply(self.train_xdata, self.train_ydata)
            print("Epoch: {:d}, average loss = {:>8.4f}, trianing error, # = {:>4d}, % = {:>8.4f}%.".format(i + 1, average_loss, error_count, error_percentage))

        error_count, error_percentage = self.apply(self.val_xdata, self.val_ydata)
        print("Test error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))


opts = Opts()
phrase = TestPhase(opts)
phrase()