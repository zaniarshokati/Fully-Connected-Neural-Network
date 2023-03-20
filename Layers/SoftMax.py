import numpy as np

class SoftMax:

    def __init__(self):
        self.y_hat = None

    def forward(self, input_tensor):
        max_row = np.max(input_tensor, axis=1, keepdims=True)
        input_tensor -= max_row

        ex_x = np.exp(input_tensor)
        sum_exp_x = np.sum(ex_x, axis=1, keepdims=True)
        self.y_hat = ex_x / sum_exp_x

        return self.y_hat

    def backward(self, error_tensor):
        error_tensor = self.y_hat * (error_tensor - np.sum(error_tensor * self.y_hat, axis=1, keepdims=True))
        return error_tensor
