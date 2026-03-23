from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier()

bagging_model = BaggingClassifier(
    estimator=tree,     # base learner
    n_estimators=100,    # number of trees
    random_state=999
)

bagging_model.fit(X_train, y_train)

y_pred = bagging_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nIris Dataset")
print("Bagging Accuracy:", accuracy)

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=10),
    n_estimators=100,
    random_state=999
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nDiabetes Dataset")
print("Bagging Regressor MSE:", mean_squared_error(y_test, y_pred))
print("Bagging Regressor R2:", r2_score(y_test, y_pred))