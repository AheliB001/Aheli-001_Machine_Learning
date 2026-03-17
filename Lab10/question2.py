import math
import pandas as pd

data = pd.read_csv('sonar_data.csv', header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

def entropy(labels):
    total = len(labels)

    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent

def information_gain(parent, left_child, right_child):
    total = len(parent)

    parent_entropy = entropy(parent)
    left_entropy = entropy(left_child)
    right_entropy = entropy(right_child)

    weighted_entropy = (
        (len(left_child) / total) * left_entropy +
        (len(right_child) / total) * right_entropy
    )

    return parent_entropy - weighted_entropy

# Split on feature 0
feature = X[0].values
threshold = feature.mean()

left_labels = y[feature <= threshold]
right_labels = y[feature > threshold]

ig_value = information_gain(list(y), list(left_labels), list(right_labels))

print("Information Gain:", ig_value)