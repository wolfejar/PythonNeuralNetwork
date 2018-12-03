import numpy
import math


class TestNetwork:

    def __init__(self, layers, x_sample, y_sample, max_accepted_error):
        self.layers = layers
        self.x_sample = x_sample
        self.y_sample = y_sample
        self.max_accepted_error = max_accepted_error
        self.threshold = 0.5

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
        # convert output to binary, above or below 0.5 threshold
        binary_output = []
        for val in self.layers[-1].get_values():
            if val < self.threshold:
                binary_output.append(0)
            else:
                binary_output.append(1)
        for k, output in enumerate(binary_output):  # if any of the outputs match the emotion, return true
            if output == 1 and output == self.y_sample[i][k]:
                return True
        return False
        '''errors = numpy.subtract(binary_output, self.y_sample[i])
        for error in errors:
            if math.fabs(error) > math.fabs(self.max_accepted_error):
                return False
        return True'''

    def get_final_result(self):
        total = 0
        total_correct = 0
        result_str = "RESULT\n"
        for i in self.y_sample.keys():
            if i in self.x_sample.keys():
                total += 1
                self.forward_propagate(i)
                result_str += str(self.layers[-1].get_values())
                result_str += "\t"
                result_str += str(self.y_sample[i])
                result_str += "\n"
                if self.get_is_correct(i):
                    total_correct += 1
        result_str += "Accuracy: "
        result_str += str((float(total_correct) / float(total)) * 100)
        return result_str
