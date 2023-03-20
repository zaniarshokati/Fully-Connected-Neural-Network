import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(0, 1, size=(input_size + 1, output_size))  # changed Weights includes biases as well changed
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.input_tensor = None
        self._gradient_weights = None

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer_value):
        self._optimizer = optimizer_value

    optimizer = property(get_optimizer, set_optimizer)

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def forward(self, input_tensor):
        temp = np.ones((np.shape(input_tensor)[0], 1))
        self.input_tensor = np.append(temp, input_tensor, axis=1)
        output = np.dot(self.input_tensor, self.weights)
        return output

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        error_tensor = np.dot(error_tensor, self.weights[1:, :].T)  # gradient_wrt_input

        return error_tensor
