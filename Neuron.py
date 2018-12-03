import random
import math


class Neuron(object):

    def __init__(self, num_weights_in, num_weights_out, initialized_weights_in, neuron_type):
        if neuron_type == "bias":
            self.num_weights_out = num_weights_out
            self.weights_out = self.get_weights(num_weights_out)
            self.neuron_value = 1.0
            self.delta = 0
            self.change = num_weights_out * [0.0]
            self.weighted_sum = 0.0
        elif neuron_type == "from_outputs":
            self.num_weights_in = num_weights_in
            self.num_weights_out = num_weights_out
            self.weights_in = initialized_weights_in
            self.weights_out = self.get_weights(num_weights_out)
            self.delta = 0
            self.change = [0] * num_weights_out
            self.weighted_sum = 0
        else:
            self.num_weights_in = num_weights_in
            self.num_weights_out = num_weights_out
            self.weights_in = self.get_weights(num_weights_in)
            self.weights_out = self.get_weights(num_weights_out)
            self.neuron_value = 1.0
            self.delta = 0
            self.change = num_weights_out*[0.0]
            self.weighted_sum = 0.0

    def apply_sigmoid_function(self, weighted_sum):
        try:
            val = 1.0 / (1 + math.exp(-weighted_sum))
        except OverflowError:
            if weighted_sum > 0:
                val = 1
            else:
                val = 0
        if val > 0:
            self.neuron_value = val

    @staticmethod
    def get_weights(num):
        weights = num * [0]
        for num, weight in enumerate(weights):
            weights[num] = random.randint(0, 100) / 100.0
        return weights

