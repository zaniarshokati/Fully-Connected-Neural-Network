import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.optimizer = optimizer
        self.label_tensor = None


    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_trainable_layer(self, layer):
        #make the deep copy and set it for the new layer that we want to append
        layer.optimizer = copy.deepcopy(self.optimizer)

        #append this layer to the list of the layers which hole the architecture
        self.layers.append(layer)


    def train(self, iterations=100):
        for it_ in range(iterations):
            loss_ = self.forward()
            self.loss.append(loss_)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor











