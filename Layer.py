from Neuron import Neuron
import math


class Layer(object):

    def __init__(self, num_neurons, previous_layer, inputs_per_neuron, outputs_per_neuron, learning_rate, ml_lambda,
                 layer_type):
        self.output_layer = False
        self.learning_rate = learning_rate
        self.ml_lambda = ml_lambda
        if layer_type == "input":
            self.neurons = []
            self.neurons.append(Neuron(0, outputs_per_neuron, 0, "bias"))
            for x in range(num_neurons):
                self.neurons.append(Neuron(inputs_per_neuron, outputs_per_neuron, 0, "input"))
        elif layer_type == "output":
            self.output_layer = True
            self.neurons = []
            for pointer in range(num_neurons):
                weights_in = []
                for neuron in previous_layer.neurons:
                    weights_in.append(neuron.weights_out[pointer])
                self.neurons.append(Neuron(len(weights_in), outputs_per_neuron, weights_in, "from_outputs"))
        else:
            # hidden layer
            self.neurons = []
            self.neurons.append(Neuron(0, outputs_per_neuron, 0, "bias"))
            for x in range(num_neurons):
                weights_in = []
                for neuron in previous_layer.neurons:
                    weights_in.append(neuron.weights_out[x])
                self.neurons.append(Neuron(len(weights_in), outputs_per_neuron, weights_in, "hidden"))

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
        values = []
        for i in range(len(self.neurons)):
            values.append(self.neurons[i].neuron_value)
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
        if self.learning_rate  > 0.3:
            self.learning_rate -= 0.001  # decrement learning rate by a small amount after each iteration

    # neuron deltas and change not shifting much???
    def set_layer_deltas_from_errors(self, errors):
        for neuron in self.neurons:
            for x in range(len(neuron.weights_out)):
                neuron.delta += neuron.weights_out[x] * errors[x]
                neuron.change[x] += (neuron.neuron_value * errors[x])
            neuron.delta *= (neuron.neuron_value * (1.0 - neuron.neuron_value))

    def set_layer_deltas_from_next_layer(self, next_layer):
        for neuron in self.neurons:
            for x in range(len(neuron.weights_out)):
                neuron.delta += (neuron.weights_out[x] * next_layer.neurons[x+1].delta)
                neuron.change[x] += (neuron.neuron_value * next_layer.neurons[x+1].delta)
            neuron.delta *= (neuron.neuron_value * (1.0 - neuron.neuron_value))

    def reset_change_and_delta(self):
        for neuron in self.neurons:
            neuron.change = [0] * len(neuron.weights_out)
            neuron.delta = 0

    def correct_inputs(self, previous_layer):
        if self.output_layer:
            pointer = 0
            while pointer < len(self.neurons):
                weights_in = []
                for neuron in previous_layer.neurons:
                    weights_in.append(neuron.weights_out[pointer])
                self.neurons[pointer].weights_in = weights_in
                pointer += 1
        else:
            pointer = 0
            while pointer < len(self.neurons)-1:
                weights_in = []
                for neuron in previous_layer.neurons:
                    weights_in.append(neuron.weights_out[pointer])
                self.neurons[pointer+1].weights_in = weights_in
                pointer += 1
