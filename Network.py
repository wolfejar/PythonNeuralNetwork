import numpy
import math
from TestNetwork import TestNetwork


class Network(object):

    def __init__(self, learning_rate, ml_lambda, input_data, targets, max_accepted_error, output_layer_size,
                 layer_size_arr):
        from Layer import Layer
        from TestNetwork import TestNetwork
        self.layers = []
        for x in range(len(layer_size_arr)):
            # num_neurons, previous_layer, inputs_per_neuron, outputs_per_neuron, learning_rate, ml_lambda, layer_type
            if x == 0:
                self.layers.append(Layer(layer_size_arr[x], 0, 1, layer_size_arr[x+1], learning_rate,
                                         ml_lambda, "input"))
            elif 0 < x < len(layer_size_arr)-1:
                self.layers.append(Layer(layer_size_arr[x], self.layers[x-1], 0, layer_size_arr[x+1], learning_rate,
                                         ml_lambda, "hidden"))
            else:
                self.layers.append(Layer(layer_size_arr[x], self.layers[x-1], 0, output_layer_size, learning_rate,
                                         ml_lambda, "output"))
        self.learning_rate = learning_rate
        self.ml_lambda = ml_lambda
        self.input_data = input_data
        self.targets = targets
        self.max_accepted_error = max_accepted_error
        self.correct_guesses = 0
        self.count = 0
        self.x_sample = self.get_x_sample(141)  # 141 size of train sample, total of 213 samples in JAFFE data set
        self.y_sample = self.get_y_sample(141)
        self.x_test_sample = self.get_test_x_sample(72)  # size of test sample
        self.y_test_sample = self.get_test_y_sample(72)
        self.test_percentage = [0]
        self.sample_correct_guesses = 0
        self.sample_count = len(targets)
        self.test_network = TestNetwork(self.layers, self.x_test_sample, self.y_test_sample, self.max_accepted_error)

    def get_results(self):
        final_network = TestNetwork(self.layers, self.x_test_sample, self.y_test_sample, self.max_accepted_error)
        return final_network.get_final_result()

    def train(self, iterations):
        for x in range(iterations):
            self.sample_count = 0
            self.sample_correct_guesses = 0
            print "\n\nTrain"
            for i in self.y_sample.keys():
                self.sample_count += 1
                self.forward_propagate(i)
                output_error = self.get_errors(i)
                if self.get_is_correct(i):
                    self.sample_correct_guesses += 1
                self.set_deltas(output_error)
                print self.layers[-1].get_values(), self.y_sample[i]
            self.back_propagate(x)
            print("\n\n\n\n")
        return

    def forward_propagate(self, i):
        for k in range(len(self.x_sample[i])):
            # set layer 1 values as inputs
            self.layers[0].neurons[k+1].neuron_value = self.x_sample[i][k]
        for k in range(len(self.layers)-1):
            # propagate values forward from each layer
            self.layers[k+1].set_layer_neuron_values(self.layers[k].get_values())

    def get_errors(self, i):
        predictions = self.layers[-1].get_values()
        actual = self.y_sample[i]
        result = numpy.subtract(predictions, actual)
        # print result
        return result

    def set_deltas(self, output_error):
        self.layers[-2].set_layer_deltas_from_errors(output_error)
        for k in range(len(self.layers)-2, -1, -1):
            if k > 0:
                self.layers[k-1].set_layer_deltas_from_next_layer(self.layers[k])

    def back_propagate(self, x):
        percentage = (float(self.sample_correct_guesses) / float(self.sample_count)) * 100.0

        self.test_network.update_network(self.layers)
        self.test_percentage = self.test_network.run_test()

        print(str(x) + " \tTrain: " + str(percentage) + "%\t\t" + "Test: " + str(self.test_percentage) + "%")
        '''if self.test_percentage > self.best_test_percentage:
            self.best_network = self.layers
            self.best_test_percentage = self.test_percentage'''

        for i, k in reversed(list(enumerate(self.layers))):
            if i > 0:
                self.layers[i-1].adjust_weights(len(self.y_sample))
        for i, k in reversed(list(enumerate(self.layers))):
            if i > 0:
                self.layers[i-1].reset_change_and_delta()
        for i, k in reversed(list(enumerate(self.layers))):
            if i > 0:
                self.layers[i].correct_inputs(self.layers[i-1])

        return percentage

    def get_is_correct(self, i):
        errors = numpy.subtract(self.layers[len(self.layers)-1].get_values(), self.targets[i])
        for error in errors:
            if math.fabs(error) > math.fabs(self.max_accepted_error):
                return False
        return True

    def get_x_sample(self, size):
        sample = {}
        for i in range(size):
            if i in self.input_data:
                sample[i] = self.input_data[i]
        return sample

    def get_y_sample(self, size):
        sample = {}
        for i in range(size):
            if i in self.input_data:
                sample[i] = self.targets[i]
        return sample

    def get_test_x_sample(self, size):
        test_sample = {}
        for i in range(size):
            if (len(self.targets)-1-i) in self.input_data:
                test_sample[i] = self.input_data[len(self.targets)-1-i]
        return test_sample

    def get_test_y_sample(self, size):
        test_sample = {}
        for i in range(size):
            if (len(self.targets)-1-i) in self.input_data:
                test_sample[i] = self.targets[(len(self.targets)-1-i)]
        return test_sample
