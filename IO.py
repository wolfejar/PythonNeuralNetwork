import pickle


data = dict(pickle.load(file("encodings.pickle")))
labels = data["names"]
encodings = data["encodings"]

for item in data.items():
    print item

unordered_targets = file("target_data.txt")
targets = {}
for line in unordered_targets:
    linearr = line.split(' ')
    if linearr[0].isdigit():
        targets[int(linearr[0])-1] = map(float, [linearr[1], linearr[2], linearr[3], linearr[4], linearr[5], linearr[6]])
