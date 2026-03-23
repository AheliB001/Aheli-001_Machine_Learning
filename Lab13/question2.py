import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#BAGGING
models = []          #store all models

#Training multiple models using bootstrap sampling

for i in range(10):
    idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_sample = X_train[idx]
    y_sample = y_train[idx]

    model = DecisionTreeRegressor()
    model.fit(X_sample, y_sample)

    models.append(model)

#Getting predictions from all models
predictions = []
for model in models:
    predictions.append(model.predict(X_test))

#Average predictions
final_pred = np.mean(predictions, axis=0)

#Calculate MSE
mse = np.mean((y_test - final_pred) ** 2)

print("Bagging MSE:", mse)