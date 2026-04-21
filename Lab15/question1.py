from ISLP import load_data
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#----------------GradientBoostingRegressor---------------

boston = load_data('Boston')
X = boston.drop("medv", axis=1)
y = boston["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

model1 = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=999
)

model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Regression:")
print("MSE :", mse)
print("R2 score :", r2)


#---------------GradientBoostingClassifier---------------

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

weekly = load_data("Weekly")
X = weekly.drop("Direction", axis=1)
y = np.where(weekly["Direction"] == "Up", 1, 0)   #Convert target into numeric (Up = 1, Down = 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=999
)

model2 = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=999
)

model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

print("\nGradient Boosting Classification:")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score :", accuracy)


