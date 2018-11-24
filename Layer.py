from Neuron import Neuron
import math


class Layer:

    def __init__(self, num_neurons, previous_layer, outputs_per_neuron, learning_rate, ml_lambda):
        from Neuron import Neuron
        self.neurons = []
        self.neurons.append(Neuron.new_bias_neuron(outputs_per_neuron))
        self.learning_rate = learning_rate
        self.ml_lambda = ml_lambda
        for x in range(len(num_neurons)):
            weights_in = []
            for neuron in previous_layer.neurons:
                weights_in.append(neuron.weights_out[x])
            self.neurons.append(Neuron.new_neuron_from_outputs(weights_in, len(weights_in), outputs_per_neuron))

    @classmethod
    def new_input_layer(cls, num_neurons, inputs_per_neuron, outputs_per_neuron):
        cls.neurons = []
        for x in range(1, num_neurons-1):
            cls.neurons.append(Neuron(inputs_per_neuron, outputs_per_neuron))

    @classmethod
    def new_output_layer(cls, num_neurons, previous_layer, outputs_per_neuron):
        cls.neurons = []
        for pointer in range(len(num_neurons)):
            weights_in = []
            for neuron in previous_layer.neurons:
                weights_in.append(neuron.weights_out[pointer])
            cls.neurons.append(Neuron.new_neuron_from_outputs(weights_in, len(weights_in), outputs_per_neuron))

    def set_layer_neuron_values(self, last_layer_values):
        start = 1
        if self.output_layer:
            start = 0
        for x in range(start, len(self.neurons)):
            final_value = 0
            for x2 in range(len(last_layer_values)):
                result = last_layer_values[x2] * self.neurons[x].weights_in[x2]
                if not math.isnan(result):
                    final_value += result
            if math.isnan(final_value):
                print("NaN detected")
            self.neurons[x].weighted_sum = final_value
            self.neurons[x].apply_sigmoid_function(final_value)

    def get_values(self):
        values = self.neurons
        return values

    def adjust_weights(self, sample_length):
        is_bias = True
        for neuron in self.neurons:
            if is_bias:
                for x in range(len(neuron.weights_out)):
                    cost = (1.0/sample_length) * neuron.change[x]
                    neuron.weights_out[x] = neuron.weights_out[x] - (self.learning_rate * cost)
                is_bias = False
            else:
                for x in range(len(neuron.weights_out)):
                    sign = -1 if neuron.change[x] > 0 else 1
                    cost = ((1.0/sample_length) * neuron.change[x]) + (self.ml_lambda * sign * neuron.weights_out[x])
                    neuron.weights_out[x] = neuron.weights_out[x] - (self.learning_rate * cost)

    def set_layer_deltas(self, errors):
        for neuron in self.neurons:
            for x in range(len(neuron.weights_out)):
                neuron.delta += neuron.weights_out[x] * errors[x]
                neuron.change += (neuron.neuron_value * errors[x])
            neuron.delta *= neuron.neuron_value * (1.0 - neuron.neuron_value)

    def reset_change_and_delta(self):
        for neuron in self.neurons:
            neuron.change = [0] * neuron.num_weights_out
            neuron.delta = 0

    def correct_inputs(self, previous_layer):
        if self.output_layer:
            pointer = 0
            weights_in = []
            while pointer < len(self.neurons):
                for neuron in previous_layer.neurons:
                    weights_in.append(neuron.weights_out[pointer])
                self.neurons[pointer+1].weights_in = weights_in
                pointer += 1
        else:
            pointer = 0
            while pointer < len(self.neurons)-1:
                weights_in = []
                for neuron in previous_layer.neurons:
                    weights_in.append(neuron.weights_out[pointer])
                self.neurons[pointer+1].weights_in = weights_in
                pointer += 1

