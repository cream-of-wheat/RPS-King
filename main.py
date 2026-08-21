from math import e
import random


class Predictor:
    def __init__(self, num_games, learning_rate = 0.1, num_hidden_layers=1):
        self.learning_rate = learning_rate
        self.num_games = num_games
        self.layers = []

        for i in range(num_hidden_layers):
            if not i:
                self.layers.append(Layer(32, num_games * 3))
            else:
                self.layers.append(Layer(32, 32))

        self.layers.append(Layer(3, 32, last_layer=True))

    def forward(self, inputs):
        new_inputs = []

        for game in inputs:
            for i in range(3):
                new_inputs.append(game[i])

        for layer in self.layers:
            new_inputs = layer.forward(new_inputs)

        return Predictor.softmax(new_inputs)

    @staticmethod
    def softmax(vector):
        max_vector = max(vector)
        stab_vector = [vector_comp - max_vector for vector_comp in vector]
        exp_vector = [e**stab_comp for stab_comp in stab_vector]
        exp_sum = sum(exp_vector)

        return [exp_comp/exp_sum for exp_comp in exp_vector]


class Layer:
    def __init__(self, num_neurons, num_inputs, last_layer=False): # last layer is different from the rest. All others need the ReLu
        self.last_layer = last_layer
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.inputs = []

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []

        for neuron in self.neurons:
            output = neuron.forward(inputs)

            if not self.last_layer:
                output = max(0.0, output)

            neuron.output = output

            outputs.append(output)

        return outputs

    def backward(self): # backpropagation research :)
        pass


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-0.2, 0.2) for _ in range(num_inputs)]
        self.bias = 0.0

        self.raw_output = 0.0
        self.output = 0.0
        self.delta = 0.0

    def forward(self, inputs):
        self.raw_output = self.bias

        for i in range(len(inputs)):
            self.raw_output += self.weights[i] * inputs[i]

        return self.raw_output


class Manager:
    def __init__(self):
        self.num_games = 10
        self.games = [[0, 0, 0] for _ in range(self.num_games)]
        self.predictor = Predictor(10)

    def add_new_round(self, choice):
        del self.games[0]
        self.games.append(choice)

    def predict(self):
        print(self.predictor.forward(self.games))


manager = Manager()
manager.predict()

