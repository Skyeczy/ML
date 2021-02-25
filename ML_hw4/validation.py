from utils import Opts, read_data, split_train, Phase


class ValidationPhase(Phase):
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

        for i in range(opts.num_epochs):
            average_loss = self.model.fit(self.train_xdata, self.train_ydata, batch_size=opts.batch_size, learning_rate=opts.learning_rate)
            error_count, error_percentage = self.apply(self.train_xdata, self.train_ydata)
            print("Epoch: {:d}, average loss = {:>8.4f}, trianing error, # = {:>4d}, % = {:>8.4f}%.".format(i + 1, average_loss, error_count, error_percentage))

            error_count, error_percentage = self.apply(self.val_xdata, self.val_ydata)
            print("Epoch: {:d}, validation error, # = {:>4d}, % = {:>8.4f}%.".format(i + 1, error_count, error_percentage))


opts = Opts()
phrase = ValidationPhase(opts)
phrase()