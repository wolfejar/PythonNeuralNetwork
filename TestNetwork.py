import numpy
import math


class TestNetwork:

    def __init__(self, layers, x_sample, y_sample, max_accepted_error):
        self.layers = layers
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.max_accepted_error = max_accepted_error

    def update_network(self, layers):
        self.layers = layers

    def run_test(self):
        total = 0
        total_correct = 0
        print "\n\nTest"
        for i in self.y_sample.keys():
            total += 1
            self.forward_propagate(i)
            print self.layers[-1].get_values(), self.y_sample[i]
            if self.get_is_correct(i):
                total_correct += 1
        percentage = (float(total_correct) / float(total)) * 100
        return percentage

    def forward_propagate(self, i):
        for k in range(len(self.x_sample[i])):
            self.layers[0].neurons[k+1].neuron_value = self.x_sample[i][k]
        for k in range(len(self.layers)-1):
            self.layers[k+1].set_layer_neuron_values(self.layers[k].get_values())

    def get_is_correct(self, i):
        errors = numpy.subtract(self.layers[-1].get_values(), self.y_sample[i])
        for error in errors:
            if math.fabs(error) > math.fabs(self.max_accepted_error):
                return False
        return True
