from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

model = RandomForestRegressor(n_estimators=100, random_state=999)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nDiabetes dataset")
print("RF Regressor MSE:", mean_squared_error(y_test, y_pred))
print("RF Regressor R2:", r2_score(y_test, y_pred))


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

model = RandomForestClassifier(n_estimators=100, random_state=999)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nIris dataset")
print("RF Classifier Accuracy:", accuracy_score(y_test, y_pred))