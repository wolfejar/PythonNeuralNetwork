from Layer import Layer

class Network:

    def __init__(self, learning_rate, ml_lambda, input_data, targets, max_accepted_error, output_layer_size, layer_size_arr):
        self.layers = []
        for x in range(len(layer_size_arr)):
            if x == 0:
                self.layers.append(Layer.new_input_layer(layer_size_arr[x], 1, layer_size_arr[x+1]))
            elif 0 < x < len(layer_size_arr):
                self.layers.append(Layer(layer_size_arr[x], self.layers[x-1], layer_size_arr[x+1]))
            else:
                self.layers.append(Layer.new_output_layer(layer_size_arr[x], self.layers[x-1], output_layer_size))
        self.learning_rate = learning_rate
        self.ml_lambda = ml_lambda
        self.input_data = input_data
        self.targets = targets
        self.max_accepted_error = max_accepted_error
        self.correct_guesses = 0
        self.count = 0
        self.x_sample = [0] #set to get_x_sample(300)
        self.y_sample = [0] #set to get_y_sample(300)
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
