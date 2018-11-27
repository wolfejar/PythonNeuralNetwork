import numpy
import math


class Network(object):

    def __init__(self, learning_rate, ml_lambda, input_data, targets, max_accepted_error, output_layer_size,
                 layer_size_arr):
        from Layer import Layer
        self.layers = []
        for x in range(len(layer_size_arr)):
            print(self.layers)
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
        self.x_sample = self.get_x_sample(300)
        self.y_sample = self.get_y_sample(300)
        self.x_test_sample = [0]
        self.y_test_sample = [0]
        self.test_percentage = [0]
        self.test_network = 0 #set to TestNetwork.init()

    def get_results(self, network):
        result = ""
        x = len(self.y_sample)
        for i in range(len(self.input_data)):
            if i == x:
                result += "\n\n\n"
            self.count += 1
            for k in range(len(self.input_data[i])):
                network.layers[0].neurons[k+1].neuron_value = self.input_data[i][k]

    def train(self, iterations):
        for x in range(iterations):
            sample_count = 0
            sample_correct_guesses = 0
            for i in range(len(self.y_sample)):
                sample_count += 1
                self.forward_propagate(i)
                output_error = self.get_errors(i)
                if self.get_is_correct(i):
                    self.sample_correct_guesses += 1
                self.set_deltas(output_error)
            self.backpropagate(x)

            print("\n\n\n\n")
        #return self.best_test_percentage

    def forward_propagate(self, i):
        for k in range(len(self.x_sample[i])):
            self.layers[0].neurons[k+1].neuron_value = self.x_sample[i][k]
        for k in range(len(self.layers)-1):
            self.layers[k+1].set_layer_neuron_values(self.layers[k].get_values())

    def get_errors(self, i):
        predictions = self.layers[len(self.layers)-1].get_values()
        actual = self.y_sample[i]
        return numpy.subtract(predictions, actual)

    def set_deltas(self, output_error):
        self.layers[len(self.layers)-2].set_layer_deltas(output_error)
        for k in range(len(self.layers)-2, -1, -1):
            self.layers[k-1].set_layer_deltas(self.layers[k])

    def back_propagate(self, x):
        percentage = (self.sample_correct_guesses / self.sample_count) * 100
        print(str(x) + " \tTrain: " + "%\t\t")
        self.test_network.updateNetwork(self.layers)
        self.test_percentage = self.test_network.runTest()

        if self.test_percentage > self.best_test_percentage:
            self.best_network = self.layers
            self.best_test_percentage = self.test_percentage

        for k in range(len(self.layers)-1, -1, -1):
            self.layers[k-1].adjust_weights(len(self.y_sample))
        for k in range(len(self.layers)-1, -1, -1):
            self.layers[k-1].reset_change_and_delta()
        for k in range(len(self.layers) - 1, -1, -1):
            self.layers[k].correct_inputs(self.layers[k-1])

        return percentage

    def get_is_correct(self, i):
        errors = numpy.subtract(self.layers[len(self.layers)-1].get_values(), self.targets[i])
        for error in errors:
            if math.fabs(error) > math.fabs(self.max_accepted_error):
                return False
        return True

    def get_x_sample(self, size):
        sample = numpy.zeros((size, len(self.input_data[0])))
        for i in range(size):
            for k in range(len(sample[i])):
                sample[i][k] = self.input_data[i][k]
        return sample

    def get_y_sample(self, size):
        print(self.targets)
        sample = numpy.zeros((size, len(self.targets[0])))
        for i in range(size):
            sample[i] = self.targets[i]
        return sample

    def get_test_x_sample(self, size):
        test_sample = numpy.zeros((size, len(self.input_data[0])))
        for i in range(size):
            for k in range(len(test_sample[i])):
                test_sample[i, k] = self.input_data[len(self.input_data)-1-i, k]
        return test_sample

    def get_test_y_sample(self, size):
        test_sample = numpy.zeros((size, len(self.targets[0])))
        for i in range(size):
            test_sample[i] = self.targets[len(self.targets)-1-i]
        return test_sample
