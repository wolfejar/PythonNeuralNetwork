from Network import Network
import IO

labels = IO.labels
unordered_input_data = IO.encodings
targets = IO.targets
print "labels ", len(labels), " encodings ", len(unordered_input_data)
# print(len(unordered_input_data))
ordered_input_data = {}
learning_rate = 1
ml_lambda = 0
saved_percentage = 0
max_accepted_error = 0.4
output_layer_size = len(targets[0])


# reorganize data
for i, label in enumerate(labels):
    arr = label.split('.')
    index = int(arr[2])
    # print index
    ordered_input_data[index-1] = map(float, unordered_input_data[i])
print targets
print ordered_input_data

# normalize data
for target in targets.items():
    for i, item in enumerate(target[1]):
        target[1][i] = item / 5.0

'''for data in ordered_input_data.items():
    for i, item in enumerate(data[1]):
        data[1][i] = ((item + 1.0) / 2.0)'''

for target in targets.items():
    print target
for data in ordered_input_data.items():
    print data

neural_network = Network(learning_rate, ml_lambda, ordered_input_data, targets, max_accepted_error, output_layer_size,
                         [len(ordered_input_data[0]), 15, 10, output_layer_size])

neural_network.train(100)

