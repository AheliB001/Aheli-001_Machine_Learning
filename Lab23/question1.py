import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, :2]
y = iris.target

#random noise
np.random.seed(42)
noise = np.random.normal(0, 0.3, X.shape)
X = X + noise

#convert continuous values into bins
disc = KBinsDiscretizer(
    n_bins=4,   #no. of bins
    encode='ordinal',
    strategy='uniform'
)

X = disc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

joint_prob = {}   #dict to store frequencies

for i in range(len(X_train)):
    sl = int(X_train[i][0])         #sepal length bin
    sw = int(X_train[i][1])         #sepal width bin
    cls = y_train[i]                #class label
    key = (sl, sw, cls)             #unique combination
    if key not in joint_prob:       #initialize count
        joint_prob[key] = 0
    joint_prob[key] += 1

def predict(sample):
    sl = int(sample[0])
    sw = int(sample[1])
    probs = []
    for cls in np.unique(y_train):      #check for all possible classes
        key = (sl, sw, cls)
        probs.append(joint_prob.get(key, 0))        #frequency of combination
    return np.argmax(probs)         #class with highest frequency

y_pred_joint = []
for sample in X_test:
    y_pred_joint.append(predict(sample))

acc_joint = accuracy_score(y_test, y_pred_joint)

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

acc_tree = accuracy_score(y_test, y_pred_tree)

print("Joint Probability Accuracy:", acc_joint)
print("Decision Tree Accuracy:", acc_tree)