import random
import math


class Neuron:

    def __init__(self, num_weights_in, num_weights_out):
        self.num_weights_in = num_weights_in
        self.num_weights_out = num_weights_out
        self.weights_in = self.get_weights(num_weights_in)
        self.weights_out = self.get_weights(num_weights_out)
        self.neuron_value = 1.0
        self.delta = 0
        self.change = num_weights_out*[0.0]
        self.weighted_sum = 0.0

    @classmethod
    def new_bias_neuron(cls, num_weights_out):
        cls.num_weights_out = num_weights_out
        cls.weights_out = cls.get_weights(num_weights_out)
        cls.neuron_value = 1.0
        cls.delta = 0
        cls.change = num_weights_out*[0.0]
        cls.weighted_sum = 0.0

    @classmethod
    def new_neuron_from_outputs(cls, initialized_weights_in, num_weights_in, num_weights_out):
        cls.num_weights_in = num_weights_in
        cls.num_weights_out = num_weights_out
        cls.weights_in = initialized_weights_in
        cls.weights_out = cls.get_weights(num_weights_out)
        cls.delta = 0
        cls.change = [0] * num_weights_out
        cls.weighted_sum = 0

    def apply_sigmoid_function(self, weighted_sum):
        val = 1.0 / (1 + math.exp(-weighted_sum))
        if val > 0:
            self.neuron_value = val

    @staticmethod
    def get_weights(num):
        weights = num * [0]
        for num, weight in enumerate(weights):
            weights[num] = random.randint(0, 100) / 100.0
        return weights

