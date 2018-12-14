from Network import Network
import IO

labels = IO.labels
unordered_input_data = IO.encodings
targets = IO.targets
print "labels ", len(labels), " encodings ", len(unordered_input_data)
# print(len(unordered_input_data))
ordered_input_data = {}
learning_rate = 0.5  # best is about 0.7 ?
ml_lambda = 0.001  # lambda value for regularization, change to improve generalization
# saved_percentage = 0
max_accepted_error = 0.5  # change this if not using binary targets
output_layer_size = len(targets[0])
binary_targets = True

# reorganize encoding data to match with corresponding targets
for i, label in enumerate(labels):
    arr = label.split('.')
    index = int(arr[2])
    # print index
    ordered_input_data[index-1] = map(float, unordered_input_data[i])
print targets
print ordered_input_data

# normalize data
if binary_targets:
    for target in targets.items():
        max_target = max(target[1])
        for i, item in enumerate(target[1]):
            if target[1][i] < max_target:
                target[1][i] = 0
            else:
                target[1][i] = 1
else:
    for target in targets.items():
        for i, item in enumerate(target[1]):
            target[1][i] = item / 5.0
'''
for data in ordered_input_data.items():
    for i, item in enumerate(data[1]):
        data[1][i] = item * 0.1
'''

for target in targets.items():
    print target
for data in ordered_input_data.items():
    print data

neural_network = Network(learning_rate, ml_lambda, ordered_input_data, targets, max_accepted_error, output_layer_size,
                         [len(ordered_input_data[0]), 3, output_layer_size])

neural_network.train(5000)  # arg is number of iterations
out_file = open("results.txt", "w")
out_file.write(neural_network.get_results())

