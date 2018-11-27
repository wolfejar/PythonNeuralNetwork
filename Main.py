from Network import Network

targets = []
input_data = [[]]
learning_rate = 0.3
ml_lambda = 0.01
saved_percentage = 0
max_accepted_error = 0.4
output_layer_size = 3

neural_network = Network(learning_rate, ml_lambda, input_data, targets, max_accepted_error, output_layer_size,
                         [30, 100, 50, 20, 5, 1])

