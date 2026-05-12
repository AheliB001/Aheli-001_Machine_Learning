import pickle                                               #used to load CIFAR10 files
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#open training batch file, use rb as CIFAR10 batch files are stored in binary format : read binary
train_file = open("cifar-10-batches-py/data_batch_1", "rb")
train = pickle.load(train_file, encoding="bytes")            #converts the binary data back into Python dictionaries

#open test batch file in binary readible format
test_file = open("cifar-10-batches-py/test_batch", "rb")
test = pickle.load(test_file, encoding="bytes")

X_train = train[b'data'][:5000]
y_train = train[b'labels'][:5000]
X_test = test[b'data'][:1000]
y_test = test[b'labels'][:1000]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  
print("Accuracy:", accuracy_score(y_test, y_pred))